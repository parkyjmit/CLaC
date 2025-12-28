from transformers import AutoModelForMaskedLM, AutoModelForCausalLM, AutoTokenizer, AutoConfig
from torch import nn


decoder_model_list = ['facebook/galactica-125m', 'facebook/opt-125m', 'gpt2', 'google/gemma-3-270m']
encoder_model_list = ['m3rg-iitd/matscibert', 'allenai/scibert_scivocab_uncased']


class TextEncoder(nn.Module):
    """
    A flexible text encoder that can load any model from the Hugging Face Hub
    supported by AutoModelForMaskedLM or AutoModelForCausalLM.
    """
    def __init__(self, pretrained_model_name_or_path: str, output_hidden_states: bool = True):
        super().__init__()
        self.pretrained_model_name_or_path = pretrained_model_name_or_path

        # Determine model type
        self.is_decoder_model = pretrained_model_name_or_path in decoder_model_list
        self.is_encoder_model = pretrained_model_name_or_path in encoder_model_list

        if self.is_decoder_model:
            self.model = AutoModelForCausalLM.from_pretrained(
                pretrained_model_name_or_path,
                output_hidden_states=output_hidden_states
            )
            self.feature_idx = -1
        if self.is_encoder_model:
            self.model = AutoModelForMaskedLM.from_pretrained(
                pretrained_model_name_or_path,
                output_hidden_states=output_hidden_states
            )
            self.feature_idx = 0
        # Expose the model's config so that parent modules can access it (e.g., for hidden_size)
        self.config = self.model.config

    def forward(self, **inputs):
        if self.pretrained_model_name_or_path in decoder_model_list:
            # For causal language models, we return the last hidden state
            outputs = self.model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                labels=inputs["input_ids"],  # computes next‐token loss
            )
            return outputs
        elif self.pretrained_model_name_or_path in encoder_model_list:
            # For masked language models, we return the hidden states
            outputs = self.model(
                input_ids=inputs['input_ids'], 
                attention_mask=inputs['attention_mask'],
                token_type_ids=inputs['token_type_ids'],
                labels=inputs['labels']
            )
            return outputs
        # return self.model(**inputs)


class CLaCTokenizer:
    """
    A tokenizer that wraps the Hugging Face tokenizer for the text encoder.
    """
    def __init__(self, pretrained_model_name_or_path: str):
        config = AutoConfig.from_pretrained(pretrained_model_name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)

        generative = config.is_encoder_decoder or getattr(config, "is_decoder_only", False)
        self.generative = generative or config.model_type in {"gpt2", "llama", "galactica", "opt"}

        if self.generative:
            self.tokenizer.add_eos_token = True
            self.tokenizer.padding_side = 'left'

            if pretrained_model_name_or_path == 'facebook/galactica-125m':
                self.tokenizer.pad_token_id = 1
                self.tokenizer.bos_token_id = 2
                self.tokenizer.eos_token_id = 2
        else:
            self.tokenizer.add_eos_token = False
            self.tokenizer.padding_side = 'right'

    @property
    def vocab_size(self):
        return len(self.tokenizer)

    @property
    def mask_token(self):
        return self.tokenizer.mask_token

    @property
    def mask_token_id(self):
        return self.tokenizer.mask_token_id

    def get_special_token_ids(self):
        """Returns a list of special token IDs (PAD, CLS, SEP, MASK, etc.)"""
        special_ids = []
        if self.tokenizer.pad_token_id is not None:
            special_ids.append(self.tokenizer.pad_token_id)
        if self.tokenizer.cls_token_id is not None:
            special_ids.append(self.tokenizer.cls_token_id)
        if self.tokenizer.sep_token_id is not None:
            special_ids.append(self.tokenizer.sep_token_id)
        if self.tokenizer.mask_token_id is not None:
            special_ids.append(self.tokenizer.mask_token_id)
        if self.tokenizer.bos_token_id is not None:
            special_ids.append(self.tokenizer.bos_token_id)
        if self.tokenizer.eos_token_id is not None:
            special_ids.append(self.tokenizer.eos_token_id)
        return special_ids
        # if pretrained_model_name_or_path == 'facebook/galactica-125m':
        #     self.tokenizer.pad_token_id = 1
        #     self.tokenizer.mask_token_id = 3
        #     self.tokenizer.padding_side = 'left'
        # elif pretrained_model_name_or_path == 'facebook/opt-125m':
        #     self.tokenizer.pad_token_id = 1
        #     self.tokenizer.mask_token_id = 3
        #     self.tokenizer.padding_side = 'left'
        # elif pretrained_model_name_or_path == 'gpt2':
        #     self.tokenizer.pad_token = self.tokenizer.eos_token
        #     self.tokenizer.padding_side = 'left'
        # elif pretrained_model_name_or_path in encoder_model_list:
        #     # self.tokenizer.pad_token = self.tokenizer.eos_token
        #     self.tokenizer.padding_side = 'right'

    def __call__(self, *args, **kwargs):
        return self.tokenizer(*args, **kwargs)

    @property
    def pad_token_id(self):
        return self.tokenizer.pad_token_id

    @property
    def eos_token_id(self):
        return self.tokenizer.eos_token_id