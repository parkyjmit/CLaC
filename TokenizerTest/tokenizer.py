import json
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModel
import pytorch_lightning as pl
import numpy as np
from torch.utils.data import DataLoader
import torch    

data = []
# from -1 to 1.0 gap 0.001
for i in np.arange(-1, 1.001, 0.001):
    data.append({'text' : f'{i:.3f}', 'num' : i})
dataset = Dataset.from_list(data)
def preprocess_function(examples):
    return tokenizer(examples['text'], padding=True, max_length=7)

tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-chat-hf')
tokenizer.pad_token = tokenizer.eos_token
dataset = dataset.shuffle()
dataset = dataset.map(preprocess_function)
dataset = dataset.remove_columns('text')
dataset.set_format(type='torch', columns=['num', 'input_ids', 'attention_mask'])

dataset = dataset.train_test_split(test_size=0.1)
train_dataset = dataset['train']
dataset = dataset['test'].train_test_split(test_size=0.5)
val_dataset = dataset['train']
test_dataset = dataset['test']

dataset_dict = DatasetDict({'train' : train_dataset, 'val' : val_dataset, 'test' : test_dataset})


class MyDataModule(pl.LightningDataModule):
    def __init__(self, dataset_dict, batch_size=16):
        super().__init__()
        self.dataset_dict = dataset_dict
        self.batch_size = batch_size
    
    def train_dataloader(self):
        return DataLoader(self.dataset_dict['train'], batch_size=self.batch_size)
    
    def val_dataloader(self):
        return DataLoader(self.dataset_dict['val'], batch_size=self.batch_size)
    
    def test_dataloader(self):
        return DataLoader(self.dataset_dict['test'], batch_size=self.batch_size)
    
class Regression(pl.LightningModule):
    def __init__(self, model_name):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name)
        self.mlp = torch.nn.Linear(768, 1)

    def forward(self, **inputs):
        inputs.pop('num')
        return self.model(**inputs)
    
    def training_step(self, batch, batch_idx):
        outputs = self(**batch)
        pred = self.mlp(outputs.last_hidden_state[:,-1])
        loss = torch.nn.functional.mse_loss(pred.squeeze(), batch['num'].squeeze())
        self.log('train_loss', loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        outputs = self(**batch)
        pred = self.mlp(outputs.last_hidden_state[:,-1])
        loss = torch.nn.functional.l1_loss(pred.squeeze(), batch['num'].squeeze())
        self.log('val_loss', loss, prog_bar=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        outputs = self(**batch)
        pred = self.mlp(outputs.last_hidden_state[:,-1])
        loss = torch.nn.functional.l1_loss(pred.squeeze(), batch['num'].squeeze())
        self.log('test_loss', loss, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=1e-4)
    
dm = MyDataModule(dataset_dict)
model = Regression('distilbert-base-uncased')
trainer = pl.Trainer(
    accelerator='gpu',
    devices=1,
    max_epochs=100
)
trainer.fit(model, dm)