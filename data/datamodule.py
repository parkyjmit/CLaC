from typing import Any, Dict, List, Mapping
from omegaconf import DictConfig
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from datasets import Dataset, DatasetDict, load_dataset
from transformers import AutoTokenizer
from transformers.data.data_collator import default_data_collator
from transformers.models.graphormer.collating_graphormer import preprocess_item, GraphormerDataCollator
from data.utils import jarvis_atoms2graph
from torch_geometric.data import Data, Batch


class CLaMPBaseDataModule(pl.LightningDataModule):
    def __init__(
            self, 
            data_path: str, 
            batch_size: int = 16,
            num_workers: int = 12,
            tokenizer_model: str = 'bert-base-uncased',
            debug: bool = False,
            *args,
            **kwargs,
        ):
        super().__init__()
        self.data_path = {
            'train': data_path+'_train.parquet',
            'val': data_path+'_val.parquet',
            'test': data_path+'_test.parquet',
        }
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_model)
        self.debug = debug
        # if tokenizer_model == 'facebook/galactica-125m':
        #     self.tokenizer.pad_token_id = 1
        #     self.tokenizer.mask_token_id = 3
        # self.tokenizer.pad_token = self.tokenizer.eos_token

    def setup(self, stage=None):
        dataset = load_dataset('parquet', data_files=self.data_path)
        
        dataset['train'] = dataset['train'].train_test_split(test_size=0.995)['train'] if self.debug else dataset['train']
        dataset['train'] = dataset['train'].shuffle()
        self.dataset = dataset

        # dataset = dataset['train'].train_test_split(test_size=0.995) if self.debug else dataset

        # dataset = dataset.shuffle()
        # dataset = dataset['train']#.map(self.preprocess_function, batched=False, num_proc=self.num_workers)
        # # split dataset
        # dataset = dataset.train_test_split(test_size=0.2)
        # train_dataset = dataset['train']
        # dataset = dataset['test']
        # dataset = dataset.train_test_split(test_size=0.5)
        # val_dataset = dataset['train']
        # test_dataset = dataset['test']
        # self.dataset = DatasetDict({
        #     'train': train_dataset,
        #     'val': val_dataset,
        #     'test': test_dataset
        # })
        # # self.dataset = dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])
 
    def train_dataloader(self):
        return DataLoader(self.dataset['train'], batch_size=self.batch_size, collate_fn=self.collate_fn, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.dataset['val'], batch_size=self.batch_size, collate_fn=self.collate_fn, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.dataset['test'], batch_size=self.batch_size, collate_fn=self.collate_fn, num_workers=self.num_workers)
    

class CLaMPDataModule(CLaMPBaseDataModule):
    def __init__(
            self, 
            data_path: str, 
            batch_size: int = 16,
            num_workers: int = 12,
            tokenizer_model: str = 'bert-base-uncased',
            *args,
            **kwargs,
        ):
        super().__init__(data_path, batch_size, num_workers, tokenizer_model, *args, **kwargs)
        self.graph_data_collator = graph_data_collator
        self.text_data_collator = text_data_collator
        self.token_fn = lambda x: self.tokenizer(x, padding='max_length', truncation=True, max_length=512)

    def collate_fn(self, features: List[dict]) -> Dict[str, Any]:
        graph_batch = self.graph_data_collator(features)
        text_batch = self.text_data_collator(features, self.token_fn)
        return graph_batch, text_batch
    

def graph_data_collator(features: List[dict]) -> Dict[str, Any]:
    """
    """
    return Batch.from_data_list([Data(x=torch.tensor(f["node_feat"], dtype=torch.float32), 
                                      edge_index=torch.tensor(f['edge_index']), 
                                      edge_attr=torch.tensor(f['edge_attr'], dtype=torch.float32),
                                      y=torch.tensor(f['y'], dtype=torch.float32)) for f in features])


def text_data_collator(features: List[dict], token_fn) -> Dict[str, Any]:
    '''
    '''
    text_batch = [token_fn(f['text']) for f in features]
    return default_data_collator(text_batch)


class GraphSupervisedDataModule(CLaMPBaseDataModule):
    def __init__(
            self, 
            data_path: str, 
            batch_size: int = 16,
            num_workers: int = 12,
            tokenizer_model: str = 'bert-base-uncased',
            label: str = 'y',
            task: str = 'classification',
            *args,
            **kwargs,
        ):
        super().__init__(data_path, batch_size, num_workers, tokenizer_model, *args, **kwargs)
        self.graph_data_collator = graph_data_collator
        self.label = label
        self.task = task

    def setup(self, stage=None):
        dataset = load_dataset('parquet', data_files=self.data_path)
        if self.task == 'classification':
            from sklearn.preprocessing import LabelEncoder
            self.categories = dataset['train'].unique(self.label)
            self.label_encoder = LabelEncoder().fit(self.categories)
        
        dataset['train'] = dataset['train'].train_test_split(test_size=0.995)['train'] if self.debug else dataset['train']
        dataset['train'] = dataset['train'].shuffle()
        self.dataset = dataset

    def collate_fn(self, features: List[dict]) -> Dict[str, Any]:
        if self.task == 'classification':
            for f in features:
                f['y'] = self.label_encoder.transform([f[self.label]])[0]
        elif self.task == 'regression':
            for f in features:
                f['y'] = float(f[self.label])
        graph_batch = self.graph_data_collator(features)
        return graph_batch


class QuestionEvaluationDataModule(CLaMPBaseDataModule):
    def __init__(self, 
                 data_path: str, 
                 batch_size: int = 16, 
                 num_workers: int = 12, 
                 tokenizer_model: str = 'bert-base-uncased', 
                 debug: bool = False, 
                 label: str = 'structure_question_list',
                 *args, 
                 **kwargs):
        super().__init__(data_path, batch_size, num_workers, tokenizer_model, debug, *args, **kwargs)
        self.graph_data_collator = graph_data_collator
        self.text_data_collator = question_batch_data_collator
        self.label = label
        self.token_fn = lambda x: self.tokenizer(x, padding='max_length', truncation=True, max_length=512)

    def collate_fn(self, features: List[dict]) -> Dict[str, Any]:
        graph_batch = self.graph_data_collator(features)
        text_batch = self.text_data_collator(features, self.token_fn, self.label)
        return graph_batch, text_batch
    

def question_batch_data_collator(features: List[dict], token_fn, label) -> Dict[str, Any]:
    '''
    '''
    text_batch = [default_data_collator([token_fn(q) for q in f[label]]) for f in features]
    # return default_data_collator(text_batch)
    return text_batch[0]


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