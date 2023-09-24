import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from datasets import Dataset, DatasetDict, load_dataset
from transformers import AutoTokenizer

class LaMDataModule(pl.LightningDataModule):
    def __init__(
            self, 
            data_path: str, 
            model_name: str,
            batch_size: int = 16
        ):
        super().__init__()
        self.data_path = data_path
        self.model_name = model_name
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def setup(self, stage=None):
        dataset = load_dataset('json', data_files=self.data_path)

        dataset = dataset.shuffle()
        dataset = dataset.map(self.preprocess_function, batched=True, batch_size=self.batch_size, remove_columns=['text'])
        self.dataset = dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])

    def train_dataloader(self):
        return DataLoader(self.dataset['train'], batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.dataset['val'], batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.dataset['test'], batch_size=self.batch_size)
    
    def preprocess_function(self, examples):
        return self.tokenizer(examples['text'], padding=True, max_length=7)