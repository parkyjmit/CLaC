from typing import Any, Dict, List, Mapping
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from datasets import Dataset, DatasetDict, load_dataset
from transformers import AutoTokenizer
from transformers.data.data_collator import default_data_collator
from transformers.models.graphormer.collating_graphormer import preprocess_item, GraphormerDataCollator


class CLaMPBaseDataModule(pl.LightningDataModule):
    def __init__(
            self, 
            cfg,
            data_path: str, 
            batch_size: int = 16
        ):
        super().__init__()
        self.cfg = cfg
        self.data_path = data_path
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.model_link)
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def setup(self, stage=None):
        dataset = load_dataset('json', data_files=self.data_path)

        dataset = dataset.shuffle()
        dataset = dataset['train'].map(self.preprocess_function, batched=False, remove_columns=['y'])
        # split dataset
        dataset = dataset.train_test_split(test_size=0.2)
        train_dataset = dataset['train']
        dataset = dataset['test']
        dataset = dataset.train_test_split(test_size=0.5)
        val_dataset = dataset['val']
        test_dataset = dataset['test']
        self.dataset = DatasetDict({
            'train': train_dataset,
            'val': val_dataset,
            'test': test_dataset
        })
        self.dataset = dataset.set_format(type='torch', columns=['input_ids_1', 'attention_mask_1', 'input_ids_2', 'attention_mask_2'])
 
    def train_dataloader(self):
        return DataLoader(self.dataset['train'], batch_size=self.batch_size, collate_fn=self.collate_fn)

    def val_dataloader(self):
        return DataLoader(self.dataset['val'], batch_size=self.batch_size, collate_fn=self.collate_fn)

    def test_dataloader(self):
        return DataLoader(self.dataset['test'], batch_size=self.batch_size, collate_fn=self.collate_fn)
    

class CLaMPDataModule(CLaMPBaseDataModule):
    def __init__(
            self, 
            cfg,
            data_path: str, 
            batch_size: int = 16
        ):
        super().__init__(cfg, data_path, batch_size)
        self.collate_fn = CLaMPDataCollator()

    def preprocess_function(self, item, keep_features=True):
        entities = {}
        entities['graph'] = preprocess_item(item, keep_features)
        encoded = self.tokenizer(item['y'], padding=True, max_length=7)
        entities['text'] = encoded
        return entities
    

class CLaMPDataCollator:
    def __init__(self) -> None:
        self.graph_data_collator = GraphormerDataCollator()
        self.text_data_collator = default_data_collator

    def __call__(self, features: List[dict]) -> Dict[str, Any]:
        graph_batch = [f['graph'] for f in features]
        graph_batch = self.graph_data_collator(graph_batch)
        text_batch = [f['text'] for f in features]
        text_batch = self.text_data_collator(text_batch)
        return graph_batch, text_batch
    

class DeCLaMPDataModule(CLaMPBaseDataModule):
    def __init__(
            self, 
            cfg,
            data_path: str, 
            batch_size: int = 16
        ):
        super().__init__(cfg, data_path, batch_size)
        self.collate_fn = DeCLaMPDataCollator()

    def preprocess_function(self, item, keep_features=True):
        entities = {}
        entities['graph'] = preprocess_item(item, keep_features)
        texts = item['y'].split('\n')
        encoded = self.tokenizer(texts[0], padding=True, max_length=7)
        entities['text_1'] = encoded
        encoded = self.tokenizer(texts[1], padding=True, max_length=7)
        entities['text_2'] = encoded
        return entities
    
    
class DeCLaMPDataCollator:
    def __init__(self) -> None:
        self.graph_data_collator = GraphormerDataCollator()
        self.text_data_collator = default_data_collator

    def __call__(self, features: List[dict]) -> Dict[str, Any]:
        graph_batch = [f['graph'] for f in features]
        graph_batch = self.graph_data_collator(graph_batch)
        text_batch_1 = [f['text_1'] for f in features]
        text_batch_1 = self.text_data_collator(text_batch_1)
        text_batch_2 = [f['text_2'] for f in features]
        text_batch_2 = self.text_data_collator(text_batch_2)
        return graph_batch, text_batch_1, text_batch_2    