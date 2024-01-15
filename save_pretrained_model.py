from transformers import AutoTokenizer, AutoModel, utils, BertForMaskedLM, BertTokenizer
from model.clamp import CLaMPLite


# Trained Model loading
model_ckpt = 'outputs/2024-01-12/20-11-29/epoch=88-step=17622.ckpt'
clamp_model = CLaMPLite.load_from_checkpoint(model_ckpt, map_location={'cuda:0': 'cpu'})
model = clamp_model.text_encoder
model.eval()
tokenizer = clamp_model.tokenizer
# len(model(inputs))

model.save_pretrained("pretrained_model/clamp_gpt")
tokenizer.save_pretrained("pretrained_model/clamp_gpt")

# Pretrained Model loading
model = AutoModel.from_pretrained("pretrained_model/clamp_gpt")
original_config = AutoModel.from_pretrained("m3rg-iitd/matscibert").config
model.config = original_config
model.push_to_hub("clamp_gpt")
tokenizer.push_to_hub("clamp_gpt")
print(model.config)
print(type(model))