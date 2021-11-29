import random
import os
import torch
import pytorch_lightning as pl
from transformers import AutoTokenizer
from torch.utils.data import DataLoader

from datasets import load_dataset
from . import yield_data_file
# from .example import DataLoader


# refer from: https://github.com/huggingface/transformers/blob/854260ca44080a13bbf1937c3c6ce3a2d17aba07/src/transformers/data/data_collator.py 
class DataCollator:
    def __init__(self):
        pass

    def __call__(self, examples):
        input_ids = torch.tensor([e['input_ids'] for e in examples], dtype=torch.long)
        attention_mask = torch.tensor([e['attention_mask'] for e in examples], dtype=torch.long)
        labels = torch.tensor([e['spam'] for e in examples], dtype=torch.long)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }



class DataModule(pl.LightningDataModule):
    def __init__(
            self,
            model_name_or_path: str='bert-base-uncased',
            max_seq_length: int = -1,
            train_batch_size: int = 32,
            eval_batch_size: int = 32,
            data_dir: str = '',
            num_workers: int = 1,
            **kwargs
        ):

        super().__init__()
        self.model_name_or_path = model_name_or_path
        self.max_seq_length = max_seq_length if max_seq_length > 0 else 512
        self.train_batch_size = train_batch_size
        self.eval_batch_size  = eval_batch_size
        self.data_dir = data_dir
        self.num_workers = num_workers
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)


    def load_dataset(self):
        train_file = os.path.join(self.data_dir, 'train.csv')
        self.raw_datasets = load_dataset('csv', data_files=[train_file])
        self.raw_datasets = self.raw_datasets['train'].train_test_split( # split data
            test_size=0.2, seed=42,
        )

    def prepare_dataset(self):
        def tokenize_function(examples):
            texts = []
            for i in range(len(examples['subject'])):
                text1 = examples['subject'][i] if examples['subject'][i]!=None else " "
                text2 = examples['email'][i] if examples['email'][i]!= None else " "
                texts.append(text1 + ' ' + text2) # concat email subject
            batch_encoding = self.tokenizer( # get token ids
                texts,
                padding='max_length',
                truncation=True,
                max_length=self.max_seq_length,
                # return_special_tokens_mask=True
            )
            return batch_encoding

        processed_datasets = self.raw_datasets.map(
            tokenize_function,
            batched=True,
            remove_columns=['subject', 'email'],
            load_from_cache_file=True,
            num_proc=8
        )

        print(processed_datasets)

        self.train_dataset = processed_datasets['train']
        self.eval_dataset  = processed_datasets['test']

    def get_dataloader(self, mode, batch_size, shuffle):

        dataset = self.train_dataset if mode == 'train' else self.eval_dataset

        dataloader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            collate_fn=DataCollator(),
        )
        print(mode, len(dataloader))
        return dataloader
        
    def train_dataloader(self):
        return self.get_dataloader('train', self.train_batch_size, shuffle=True)

    def val_dataloader(self):
        return self.get_dataloader("dev", self.eval_batch_size, shuffle=False)

    def test_dataloader(self):
        return self.get_dataloader("test", self.eval_batch_size, shuffle=False)
