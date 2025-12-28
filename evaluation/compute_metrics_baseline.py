#!/usr/bin/env python3
"""
Zero-shot QA evaluation using pure LLM baselines (e.g., LLaMA, Galactica, GPT).

Unlike CLaC which uses graph-text contrastive learning, this script evaluates
text-only LLMs that receive CIF structure descriptions as input.

Supports two modes:
1. Local HuggingFace models (default)
2. OpenAI API models (with --use-api flag)

Usage:
    # Local HuggingFace model
    python evaluation/compute_metrics_baseline.py \
        --model-name "facebook/galactica-125m" \
        --label structure_question_list \
        --batch-size 8 \
        --device cuda:0

    # OpenAI API (GPT-4o-mini, GPT-4o, etc.)
    # Option 1: Using .env file (recommended)
    # Create ../.env with: OPENAI_API_KEY=sk-...
    python evaluation/compute_metrics_baseline.py \
        --model-name "gpt-4o-mini" \
        --use-api \
        --label structure_question_list \
        --max-samples 100 \
        --requests-per-minute 100

    # Option 2: Using environment variable
    export OPENAI_API_KEY="sk-..."
    python evaluation/compute_metrics_baseline.py \
        --model-name "gpt-4o-mini" \
        --use-api \
        --label structure_question_list

    # Option 3: Using command line argument
    python evaluation/compute_metrics_baseline.py \
        --model-name "gpt-4o-mini" \
        --use-api \
        --api-key "sk-..." \
        --label structure_question_list
"""

from compute_metrics import bootstrap_confidence_interval
import argparse
from tqdm import tqdm
import torch
import numpy as np
import json
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import warnings
import os
import time

# Suppress transformers warnings
warnings.filterwarnings('ignore')
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

# OpenAI API (optional, for API-based models)
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    # Load from ../.env (parent directory)
    env_path = Path(__file__).parent.parent / '.env'
    load_dotenv(dotenv_path=env_path)
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False


def atoms_dict_to_text(atoms_dict):
    """
    Convert atoms dict to text by simple string conversion.

    Args:
        atoms_dict: Dictionary with structure information

    Returns:
        str: String representation of atoms dict
    """
    # Simple str() conversion - let LLM interpret the structure
    return f"Crystal Structure Data:\n{str(atoms_dict)}"


def convert_bytes_to_str(data):
    """Recursively convert bytes to str in nested structures."""
    if isinstance(data, bytes):
        return data.decode('utf-8')
    elif isinstance(data, dict):
        return {key: convert_bytes_to_str(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [convert_bytes_to_str(item) for item in data]
    elif isinstance(data, np.ndarray):
        if data.dtype == object:
            return np.array([convert_bytes_to_str(item) for item in data])
        return data
    else:
        return data


def get_structure_text(data: dict):
    """
    Extract structure information as readable text.

    Args:
        data: Dictionary containing 'cif' or 'atoms' key

    Returns:
        str: Human-readable structure description
    """
    if 'cif' in data and data['cif'] is not None:
        # Direct CIF string (may be bytes or str)
        cif_data = data['cif']
        if isinstance(cif_data, bytes):
            return cif_data.decode('utf-8')
        return cif_data
    elif 'atoms' in data and data['atoms'] is not None:
        # Convert atoms dict to readable text
        atoms_dict = convert_bytes_to_str(data['atoms'])
        return atoms_dict_to_text(atoms_dict)
    else:
        raise ValueError("Data must contain 'cif' or 'atoms' key")


def create_qa_prompt(structure_text: str, question: str, choices: list[str]) -> str:
    """
    Create a multiple-choice QA prompt with structure description.

    Args:
        structure_text: Structure description (CIF or readable text)
        question: Question text
        choices: List of choice texts

    Returns:
        str: Formatted prompt
    """
    # Create choice labels (A, B, C, D, ...)
    choice_labels = [chr(65 + i) for i in range(len(choices))]  # A, B, C, ...

    # Format choices
    choices_text = "\n".join([f"{label}) {choice}" for label, choice in zip(choice_labels, choices)])

    prompt = f"""Given the following crystal structure:

{structure_text}

Question: {question}

Choices:
{choices_text}

Answer with only the letter ({", ".join(choice_labels)}):
The answer is """

    return prompt


def parse_llm_answer(llm_output: str, num_choices: int) -> int:
    """
    Parse LLM output to extract answer choice index.

    Args:
        llm_output: LLM generated text
        num_choices: Number of available choices

    Returns:
        int: Predicted choice index (0-based), or -1 if parsing failed
    """
    import re

    # Valid choice letters
    valid_letters = [chr(65 + i) for i in range(num_choices)]  # A, B, C, ...

    llm_output_upper = llm_output.strip().upper()

    # Strategy 1: Look for standalone letter at the beginning (e.g., "A", "B)", "A.", "A,")
    for i, letter in enumerate(valid_letters):
        # Match letter at start, optionally followed by ), ., or ,
        pattern = r'^\s*' + letter + r'[\)\.,]?\s*(?:\n|$)'
        if re.match(pattern, llm_output_upper):
            return i

    # Strategy 2: Look for "answer is X" or "answer: X" pattern (more flexible)
    for i, letter in enumerate(valid_letters):
        # Matches: "answer is A", "answer: A", "the answer is A", etc.
        pattern = r'(?:the\s+)?(?:answer|choice|option)\s*(?:is|:|would be)?\s*' + letter + r'(?:\b|[\)\.,])'
        if re.search(pattern, llm_output_upper):
            return i

    # Strategy 3: Look for "X)" or "X." at start of line (choice format)
    for i, letter in enumerate(valid_letters):
        pattern = r'(?:^|\n)\s*' + letter + r'[\)\.]'
        if re.search(pattern, llm_output_upper):
            return i

    # Strategy 4: Look for standalone letter with word boundaries
    for i, letter in enumerate(valid_letters):
        pattern = r'\b' + letter + r'\b'
        if re.search(pattern, llm_output_upper):
            return i

    # Strategy 5: Fallback - first occurrence
    for i, letter in enumerate(valid_letters):
        if letter in llm_output_upper:
            return i

    # Parsing failed
    return -1


def evaluate_zero_shot_qa(
    model,
    tokenizer,
    dataset,
    max_new_tokens=10,
    label='structure_question_list',
    batch_size=1,
    max_length=None
):
    """
    Evaluate LLM on zero-shot QA task with CIF inputs.

    Args:
        model: HuggingFace causal LM model
        tokenizer: HuggingFace tokenizer
        dataset: HuggingFace dataset (test split)
        max_new_tokens: Max tokens to generate
        label: Question list label
        batch_size: Batch size for generation
        max_length: Max input sequence length

    Returns:
        dict: Evaluation results with bootstrap CI
    """
    model.eval()

    qa_outcomes = []  # Binary outcomes (1=correct, 0=incorrect)
    qa_num_choices = []  # Number of choices per question
    parsing_failures = 0
    total_samples = 0
    parsing_failure_examples = []  # Store first few parsing failures for debugging

    # Get model's device
    model_device = next(model.parameters()).device

    # Determine max_length
    if max_length is None:
        # Use model's max position embeddings if available
        if hasattr(model.config, 'max_position_embeddings'):
            max_length = model.config.max_position_embeddings
        elif hasattr(model.config, 'n_positions'):
            max_length = model.config.n_positions
        else:
            max_length = 2048  # Default fallback
            print(f"[Warning] Could not determine model's max length, using {max_length}")

    print(f"\n[Evaluation] Running zero-shot QA with LLM baseline...")
    print(f"[Evaluation] Label: {label}, Dataset size: {len(dataset)}")

    # Process in batches
    from torch.utils.data import DataLoader

    def collate_fn(batch):
        """Collate function to prepare batch data."""
        return batch

    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)

    with torch.no_grad():
        for batch_samples in tqdm(dataloader, desc="Evaluating"):
            batch_prompts = []
            batch_correct_answers = []
            batch_num_choices_list = []
            batch_valid_indices = []

            # Prepare batch
            for idx, sample in enumerate(batch_samples):
                try:
                    # Extract structure text
                    structure_text = get_structure_text(sample)

                    # Extract question list and correct answer
                    question_list = sample[label]
                    # Convert bytes to str if needed
                    question_list = [q.decode('utf-8') if isinstance(q, bytes) else q for q in question_list]
                    correct_answer_idx = sample['y'].index(max(sample['y']))

                    if len(question_list) < 2:
                        continue

                    num_choices = len(question_list)

                    # Create question text based on label
                    if label == 'structure_question_list':
                        question_text = "Which of the following best describes this material's crystal structure?"
                    elif label == 'composition_question_list':
                        question_text = "Which of the following best describes this material's composition?"
                    elif label == 'oxide_question_list':
                        question_text = "Which of the following statements about this material is true?"
                    else:
                        question_text = "Which of the following is correct about this material?"

                    prompt = create_qa_prompt(structure_text, question_text, question_list)

                    # Debug: Print first prompt (optional)
                    if True and total_samples == 0 and len(batch_prompts) == 0:
                        # Tokenize to check length
                        test_tokens = tokenizer(prompt, return_tensors='pt', truncation=False)
                        test_length = test_tokens['input_ids'].shape[1]

                        print(f"\n{'='*80}")
                        print(f"[DEBUG] First prompt example:")
                        print("=" * 80)
                        print(prompt[:500] + "..." if len(prompt) > 500 else prompt)  # Truncate if too long
                        print("=" * 80)
                        print(f"Prompt length: {len(prompt)} chars, {test_length} tokens")
                        print(f"Max length setting: {max_length}")
                        print(f"Will be truncated: {test_length > max_length}")
                        print(f"Correct answer index: {correct_answer_idx} ({question_list[correct_answer_idx]})")
                        print(f"Number of choices: {num_choices}")
                        print("=" * 80)

                    batch_prompts.append(prompt)
                    batch_correct_answers.append(correct_answer_idx)
                    batch_num_choices_list.append(num_choices)
                    batch_valid_indices.append(idx)
                    qa_num_choices.append(num_choices)

                except Exception as e:
                    # Print error for first few failed samples
                    if total_samples < 5:
                        print(f"\n[WARNING] Failed to process sample {total_samples}: {str(e)}")
                    continue

            if not batch_prompts:
                continue

            try:
                # Apply chat template to all prompts in batch
                formatted_prompts = []
                for prompt in batch_prompts:
                    if tokenizer.chat_template is not None:
                        messages = [{"role": "user", "content": prompt}]
                        formatted_prompt = tokenizer.apply_chat_template(
                            messages,
                            tokenize=False,
                            add_generation_prompt=True
                        )
                    else:
                        formatted_prompt = prompt
                    formatted_prompts.append(formatted_prompt)

                # Batch tokenization with LEFT padding
                inputs = tokenizer(
                    formatted_prompts,
                    return_tensors='pt',
                    truncation=True,
                    max_length=max_length,
                    padding=True,
                    return_token_type_ids=False
                )

                input_length = inputs['input_ids'].shape[1]
                inputs = {k: v.to(model_device) for k, v in inputs.items()}

                # Batch generation
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                    top_k=None
                )

                # Process each output in batch
                for idx, (output, correct_idx, num_choices) in enumerate(zip(outputs, batch_correct_answers, batch_num_choices_list)):
                    # Decode only the generated part
                    generated_text = tokenizer.decode(output[input_length:], skip_special_tokens=True)

                    # Parse answer
                    predicted_idx = parse_llm_answer(generated_text, num_choices)

                    # Debug: Print first 10 generated texts unconditionally
                    if total_samples < 10:
                        print(f"\n{'='*80}")
                        print(f"[DEBUG] Sample #{total_samples+1}")
                        print(f"{'='*80}")
                        print(f"Generated text: '{generated_text}'")
                        print(f"Predicted index: {predicted_idx}")
                        print(f"Correct index: {correct_idx}")
                        print(f"Number of choices: {num_choices}")
                        print(f"Parsing status: {'SUCCESS' if predicted_idx != -1 else 'FAILED'}")
                        print(f"{'='*80}")

                    if predicted_idx == -1:
                        # Parsing failed - mark as incorrect
                        parsing_failures += 1
                        qa_outcomes.append(0)
                        total_samples += 1
                        # Store more failures for debugging (increased from 5 to 20)
                        if len(parsing_failure_examples) < 20:
                            parsing_failure_examples.append({
                                'generated_text': generated_text,
                                'correct_idx': correct_idx,
                                'num_choices': num_choices
                            })
                    else:
                        # Check correctness
                        is_correct = int(predicted_idx == correct_idx)
                        qa_outcomes.append(is_correct)
                        total_samples += 1

            except Exception as e:
                # Print error details
                print(f"\n[ERROR] Exception during batch processing: {str(e)}")
                print(f"[ERROR] Batch size: {len(batch_prompts)}")
                import traceback
                traceback.print_exc()
                # Mark all samples in batch as incorrect
                for _ in batch_prompts:
                    parsing_failures += 1
                    qa_outcomes.append(0)
                    total_samples += 1

    # Compute bootstrap CI
    print(f"\n[Bootstrapping] Computing confidence intervals (n_bootstrap=1000)...")
    qa_stats = bootstrap_confidence_interval(qa_outcomes, n_bootstrap=1000)

    # Random baseline
    random_qa_outcomes = []
    for num_choices in qa_num_choices:
        random_correct = np.random.binomial(1, 1.0 / num_choices)
        random_qa_outcomes.append(random_correct)

    random_qa_stats = bootstrap_confidence_interval(random_qa_outcomes, n_bootstrap=1000)

    # Create results
    results = {
        'evaluation_method': 'zero-shot QA (LLM baseline)',
        'label': label,
        'model_name': model.config.name_or_path,
        'num_samples': total_samples,
        'parsing_failures': parsing_failures,
        'bootstrap_params': {
            'n_bootstrap': 1000,
            'confidence_level': 0.95,
            'random_seed': 42
        },
        'accuracy': qa_stats,
        'random_baseline': random_qa_stats
    }

    # Print results
    print("\n" + "="*80)
    print(f"ZERO-SHOT QA RESULTS (LLM Baseline) - Label: {label}")
    print("="*80)
    print(f"Model: {model.config.name_or_path}")
    print(f"Number of test samples: {total_samples}")
    if total_samples > 0:
        print(f"Parsing failures: {parsing_failures} ({parsing_failures/total_samples*100:.1f}%)")
    else:
        print(f"Parsing failures: {parsing_failures} (N/A)")

    # Print parsing failure examples
    if parsing_failure_examples:
        print("\n" + "-" * 80)
        print("PARSING FAILURE EXAMPLES:")
        print("-" * 80)
        for i, example in enumerate(parsing_failure_examples):
            print(f"\nExample {i+1}:")
            print(f"  Generated: '{example['generated_text']}'")
            print(f"  Expected: Choice {example['correct_idx']} (out of {example['num_choices']} choices)")
        print("-" * 80)

    print(f"Bootstrap iterations: {qa_stats['n_bootstrap']}")
    print("-" * 80)
    print("LLM Baseline:")
    print(f"  Accuracy: {qa_stats['mean']:.4f} ± {qa_stats['std']:.4f}")
    print(f"  95% CI: [{qa_stats['ci_lower']:.4f}, {qa_stats['ci_upper']:.4f}]")
    print("-" * 80)
    print("Random Baseline:")
    print(f"  Accuracy: {random_qa_stats['mean']:.4f} ± {random_qa_stats['std']:.4f}")
    print(f"  95% CI: [{random_qa_stats['ci_lower']:.4f}, {random_qa_stats['ci_upper']:.4f}]")
    print("="*80)

    return results


def evaluate_zero_shot_qa_api(
    client: 'OpenAI',
    model_name: str,
    dataset,
    label='structure_question_list',
    max_tokens=50,
    temperature=None,
    max_samples=None,
    requests_per_minute=100
):
    """
    Evaluate OpenAI API models on zero-shot QA task with CIF inputs.

    Args:
        client: OpenAI client instance
        model_name: OpenAI model name (e.g., "gpt-4o-mini", "gpt-4o")
        dataset: HuggingFace dataset (test split)
        label: Question list label
        max_tokens: Max tokens to generate
        temperature: Sampling temperature (None = use model default)
        max_samples: Max samples to evaluate (None = all)
        requests_per_minute: Rate limit for API calls

    Returns:
        dict: Evaluation results with bootstrap CI
    """
    qa_outcomes = []  # Binary outcomes (1=correct, 0=incorrect)
    qa_num_choices = []  # Number of choices per question
    parsing_failures = 0
    total_samples = 0
    parsing_failure_examples = []
    api_errors = 0

    # Rate limiting
    request_interval = 60.0 / requests_per_minute  # seconds between requests

    print(f"\n[Evaluation] Running zero-shot QA with OpenAI API...")
    print(f"[Evaluation] Model: {model_name}, Label: {label}, Dataset size: {len(dataset)}")
    print(f"[Evaluation] Rate limit: {requests_per_minute} requests/min ({request_interval:.2f}s interval)")

    # Subsample if requested
    if max_samples is not None:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
        print(f"[Evaluation] Using {len(dataset)} samples")

    with tqdm(total=len(dataset), desc="Evaluating") as pbar:
        for idx, sample in enumerate(dataset):
            try:
                # Extract structure text
                structure_text = get_structure_text(sample)

                # Extract question list and correct answer
                question_list = sample[label]
                question_list = [q.decode('utf-8') if isinstance(q, bytes) else q for q in question_list]
                correct_answer_idx = sample['y'].index(max(sample['y']))

                if len(question_list) < 2:
                    pbar.update(1)
                    continue

                num_choices = len(question_list)

                # Create question text based on label
                if label == 'structure_question_list':
                    question_text = "Which of the following best describes this material's crystal structure?"
                elif label == 'composition_question_list':
                    question_text = "Which of the following best describes this material's composition?"
                elif label == 'oxide_question_list':
                    question_text = "Which of the following statements about this material is true?"
                else:
                    question_text = "Which of the following is correct about this material?"

                prompt = create_qa_prompt(structure_text, question_text, question_list)

                # Debug: Print first prompt
                if total_samples == 0:
                    print(f"\n{'='*80}")
                    print(f"[DEBUG] First prompt example:")
                    print("=" * 80)
                    print(prompt[:500] + "..." if len(prompt) > 500 else prompt)
                    print("=" * 80)
                    print(f"Prompt length: {len(prompt)} chars")
                    print(f"Correct answer index: {correct_answer_idx} ({question_list[correct_answer_idx]})")
                    print(f"Number of choices: {num_choices}")
                    print("=" * 80)

                # Call OpenAI API
                try:
                    # Prepare API call parameters
                    api_params = {
                        "model": model_name,
                        "messages": [{"role": "user", "content": prompt}],
                        "max_completion_tokens": max_tokens,
                        "n": 1
                    }

                    # Only add temperature if explicitly set (some models don't support all values)
                    if temperature is not None:
                        api_params["temperature"] = temperature

                    response = client.chat.completions.create(**api_params)

                    # Extract generated text
                    generated_text = response.choices[0].message.content
                    if generated_text is None:
                        generated_text = ""
                    else:
                        generated_text = generated_text.strip()

                    # Debug: Print first 10 generated texts
                    if total_samples < 10:
                        print(f"\n{'='*80}")
                        print(f"[DEBUG] Sample #{total_samples+1}")
                        print(f"{'='*80}")
                        print(f"Generated text: '{generated_text}'")
                        print(f"Raw response content: {response.choices[0].message.content}")
                        print(f"Finish reason: {response.choices[0].finish_reason}")
                        print(f"Usage: {response.usage}")
                        print(f"Correct index: {correct_answer_idx}")
                        print(f"Number of choices: {num_choices}")
                        print(f"{'='*80}")

                    # Parse answer
                    predicted_idx = parse_llm_answer(generated_text, num_choices)

                    if predicted_idx == -1:
                        # Parsing failed
                        parsing_failures += 1
                        qa_outcomes.append(0)
                        if len(parsing_failure_examples) < 20:
                            parsing_failure_examples.append({
                                'generated_text': generated_text,
                                'correct_idx': correct_answer_idx,
                                'num_choices': num_choices
                            })
                    else:
                        # Check correctness
                        is_correct = int(predicted_idx == correct_answer_idx)
                        qa_outcomes.append(is_correct)

                    qa_num_choices.append(num_choices)
                    total_samples += 1

                    # Rate limiting
                    time.sleep(request_interval)

                except Exception as e:
                    print(f"\n[API Error] Sample {idx}: {str(e)}")
                    api_errors += 1
                    qa_outcomes.append(0)
                    qa_num_choices.append(num_choices)
                    total_samples += 1
                    parsing_failures += 1
                    time.sleep(request_interval * 2)  # Wait longer after error

            except Exception as e:
                # Sample processing error
                if total_samples < 5:
                    print(f"\n[WARNING] Failed to process sample {idx}: {str(e)}")

            pbar.update(1)

    # Compute bootstrap CI
    print(f"\n[Bootstrapping] Computing confidence intervals (n_bootstrap=1000)...")
    qa_stats = bootstrap_confidence_interval(qa_outcomes, n_bootstrap=1000)

    # Random baseline
    random_qa_outcomes = []
    for num_choices in qa_num_choices:
        random_correct = np.random.binomial(1, 1.0 / num_choices)
        random_qa_outcomes.append(random_correct)

    random_qa_stats = bootstrap_confidence_interval(random_qa_outcomes, n_bootstrap=1000)

    # Create results
    results = {
        'evaluation_method': 'zero-shot QA (OpenAI API)',
        'label': label,
        'model_name': model_name,
        'num_samples': total_samples,
        'parsing_failures': parsing_failures,
        'api_errors': api_errors,
        'bootstrap_params': {
            'n_bootstrap': 1000,
            'confidence_level': 0.95,
            'random_seed': 42
        },
        'accuracy': qa_stats,
        'random_baseline': random_qa_stats
    }

    # Print results
    print("\n" + "="*80)
    print(f"ZERO-SHOT QA RESULTS (OpenAI API) - Label: {label}")
    print("="*80)
    print(f"Model: {model_name}")
    print(f"Number of test samples: {total_samples}")
    if total_samples > 0:
        print(f"Parsing failures: {parsing_failures} ({parsing_failures/total_samples*100:.1f}%)")
        print(f"API errors: {api_errors} ({api_errors/total_samples*100:.1f}%)")
    else:
        print(f"Parsing failures: {parsing_failures} (N/A)")
        print(f"API errors: {api_errors} (N/A)")

    # Print parsing failure examples
    if parsing_failure_examples:
        print("\n" + "-" * 80)
        print("PARSING FAILURE EXAMPLES:")
        print("-" * 80)
        for i, example in enumerate(parsing_failure_examples[:5]):  # Limit to 5
            print(f"\nExample {i+1}:")
            print(f"  Generated: '{example['generated_text']}'")
            print(f"  Expected: Choice {example['correct_idx']} (out of {example['num_choices']} choices)")
        print("-" * 80)

    print(f"Bootstrap iterations: {qa_stats['n_bootstrap']}")
    print("-" * 80)
    print("OpenAI API Baseline:")
    print(f"  Accuracy: {qa_stats['mean']:.4f} ± {qa_stats['std']:.4f}")
    print(f"  95% CI: [{qa_stats['ci_lower']:.4f}, {qa_stats['ci_upper']:.4f}]")
    print("-" * 80)
    print("Random Baseline:")
    print(f"  Accuracy: {random_qa_stats['mean']:.4f} ± {random_qa_stats['std']:.4f}")
    print(f"  95% CI: [{random_qa_stats['ci_lower']:.4f}, {random_qa_stats['ci_upper']:.4f}]")
    print("="*80)

    return results


def main():
    parser = argparse.ArgumentParser(description='Zero-shot QA evaluation with LLM baseline')
    parser.add_argument('--model-name', type=str, required=True,
                       help='Model name (HuggingFace: facebook/galactica-125m, OpenAI: gpt-4o-mini)')
    parser.add_argument('--use-api', action='store_true',
                       help='Use OpenAI API instead of local HuggingFace model')
    parser.add_argument('--api-key', type=str, default=None,
                       help='OpenAI API key (or set OPENAI_API_KEY env variable)')
    parser.add_argument('--requests-per-minute', type=int, default=100,
                       help='API rate limit in requests per minute (for --use-api)')
    parser.add_argument('--temperature', type=float, default=None,
                       help='Sampling temperature for API calls (default: use model default, typically 1.0)')

    parser.add_argument('--data-path', type=str, default='/home/lucky/Projects/CLaC-revision/datafiles/mp_3d_2020_materials_graphs_gpt_questions',
                       help='Path to dataset parquet file (without _test.parquet suffix)')
    parser.add_argument('--label', type=str, default='structure_question_list',
                       choices=['composition_question_list', 'structure_question_list', 'oxide_question_list'])
    parser.add_argument('--batch-size', type=int, default=8,
                       help='Batch size for parallel generation (only for local models)')
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda:0',
                       help='Device for local models (ignored for --use-api)')
    parser.add_argument('--max-new-tokens', type=int, default=50,
                       help='Max tokens to generate for answer')
    parser.add_argument('--max-length', type=int, default=None,
                       help='Max input sequence length (only for local models)')
    parser.add_argument('--output-dir', type=str, default='outputs/zero_shot_qa_baseline',
                       help='Output directory for results')
    parser.add_argument('--max-samples', type=int, default=None,
                       help='Maximum number of samples to evaluate (for debugging)')
    parser.add_argument('--sample-fraction', type=float, default=None,
                       help='Fraction of dataset to evaluate (0.0-1.0, for debugging)')

    args = parser.parse_args()

    # Setup output directory
    model_safe_name = args.model_name.replace('/', '_').replace('-', '_')
    output_dir = Path(args.output_dir) / model_safe_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*80)
    print("ZERO-SHOT QA EVALUATION - LLM BASELINE")
    print("="*80)
    print(f"Model: {args.model_name}")
    print(f"Mode: {'OpenAI API' if args.use_api else 'Local HuggingFace'}")
    print(f"Label: {args.label}")
    if not args.use_api:
        print(f"Device: {args.device}")
    print("="*80 + "\n")

    # Load data
    print(f"[Loading] Loading data: {args.data_path}_test.parquet...")
    data_path_dict = {'test': args.data_path + '_test.parquet'}
    dataset = load_dataset('parquet', data_files=data_path_dict)
    test_dataset = dataset['test']
    print(f"[Loading] Data loaded: {len(test_dataset)} test samples")

    # Subsample dataset if requested (for debugging)
    if args.sample_fraction is not None:
        num_samples = int(len(test_dataset) * args.sample_fraction)
        test_dataset = test_dataset.select(range(num_samples))
        print(f"[Debug] Using {args.sample_fraction*100:.1f}% of data: {len(test_dataset)} samples")
    elif args.max_samples is not None:
        num_samples = min(args.max_samples, len(test_dataset))
        test_dataset = test_dataset.select(range(num_samples))
        print(f"[Debug] Using first {len(test_dataset)} samples")

    # Run evaluation based on mode
    if args.use_api:
        # OpenAI API mode
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI library not installed. Install with: pip install openai")

        # Get API key
        if DOTENV_AVAILABLE:
            print(f"[API] Loaded environment variables from ../.env")

        api_key = args.api_key or os.environ.get('OPENAI_API_KEY')
        if not api_key:
            error_msg = "OpenAI API key required. Options:\n"
            error_msg += "  1. Set --api-key argument\n"
            error_msg += "  2. Set OPENAI_API_KEY environment variable\n"
            error_msg += "  3. Create ../.env file with: OPENAI_API_KEY=sk-...\n"
            if not DOTENV_AVAILABLE:
                error_msg += "\nNote: python-dotenv not installed. Install with: pip install python-dotenv"
            raise ValueError(error_msg)

        # Initialize OpenAI client
        print(f"[API] Initializing OpenAI client...")
        client = OpenAI(api_key=api_key)

        # Run API evaluation
        results = evaluate_zero_shot_qa_api(
            client=client,
            model_name=args.model_name,
            dataset=test_dataset,
            label=args.label,
            max_tokens=args.max_new_tokens,
            temperature=args.temperature,
            max_samples=args.max_samples,
            requests_per_minute=args.requests_per_minute
        )
    else:
        # Local HuggingFace model mode
        print(f"[Loading] Loading model: {args.model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)

        # CRITICAL: Set padding side to 'left' for causal LM generation
        tokenizer.padding_side = 'left'

        # Set pad token if not exists - use a different token than EOS
        if tokenizer.pad_token is None:
            # Try to use unk_token first, fall back to eos_token
            if tokenizer.unk_token is not None:
                tokenizer.pad_token = tokenizer.unk_token
            else:
                tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            torch_dtype="auto",  # Use model's native dtype (bfloat16 for gpt-oss)
            device_map=args.device  # Automatically move to device
        )

        # Set pad_token_id to match tokenizer
        if model.config.pad_token_id != tokenizer.pad_token_id:
            model.config.pad_token_id = tokenizer.pad_token_id

        print(f"[Loading] Model loaded successfully")

        # Run local evaluation
        results = evaluate_zero_shot_qa(
            model=model,
            tokenizer=tokenizer,
            dataset=test_dataset,
            max_new_tokens=args.max_new_tokens,
            label=args.label,
            batch_size=args.batch_size,
            max_length=args.max_length
        )

    # Save results
    output_file = output_dir / f'qa_{args.label}.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Results saved to: {output_file}")
    print("\nDone!")


if __name__ == '__main__':
    main()
