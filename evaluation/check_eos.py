from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("google/gemma-3-270m")  # 예: "bert-base-multilingual-cased", "meta-llama/Llama-3-8B"
print(tok.pad_token, tok.bos_token, tok.eos_token, tok.mask_token)
print(tok.pad_token_id, tok.bos_token_id, tok.eos_token_id, tok.mask_token_id)
