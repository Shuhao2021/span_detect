import os
import torch
import logging
import argparse
import random
import numpy as np
import pickle

from sklearn.metrics import f1_score, precision_score, recall_score
import pytorch_lightning as pl 
import torch.nn.functional as F

pl.seed_everything(42)

from transformers.optimization import Adafactor, AdamW
from pytorch_lightning.utilities import rank_zero_info
from transformers import (
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_cosine_with_hard_restarts_schedule_with_warmup,
    get_polynomial_decay_schedule_with_warmup,
    get_constant_schedule_with_warmup
)

arg_to_scheduler = {
    'linear': get_linear_schedule_with_warmup,
    'cosine': get_cosine_schedule_with_warmup,
    'cosine_w_restarts': get_cosine_with_hard_restarts_schedule_with_warmup,
    'polynomial': get_polynomial_decay_schedule_with_warmup,
    'constant': get_constant_schedule_with_warmup,
}

from transformers import AutoTokenizer, BertConfig
from model.model import BertForSequenceClassification
from utils.datamodule import DataModule
from utils import mkdir_if_not_exist
logger = logging.getLogger(__name__)


class Training(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.config = BertConfig.from_pretrained(self.hparams.model_name_or_path, num_labels=2, problem_type=None)
        print(self.config)
        if self.hparams.do_train:
            self.model = BertForSequenceClassification.from_pretrained(self.hparams.model_name_or_path, config=self.config)
            self.tokenizer = AutoTokenizer.from_pretrained(self.hparams.model_name_or_path)
        else:
            self.model, self.tokenizer = self.load_model()
        self.f1_max = 0
        self.record = {'train_loss': [], 'valid_loss': [], 'f1':[], 'recall':[], 'precision':[]}
    @pl.utilities.rank_zero_only
    def save_model(self):
        dir_name = os.path.join(self.hparams.output_dir, 'model', f'step={self.global_step+1}')
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        print(f'## save model to {dir_name}')
        self.model.save_pretrained(dir_name)
        self.tokenizer.save_pretrained(dir_name)

    def load_model(self):
        def _get_best_file(file_list): # load best model
            max_step = -1
            best_file = ''
            for f in file_list:
                step = int(f.split('=')[-1])
                if step > max_step:
                    max_step = step
                    best_file = f
            return best_file
        source_dir_name = os.path.join(self.hparams.output_dir, 'model')
        file_name = _get_best_file(os.listdir(source_dir_name))
        dir_name = os.path.join(source_dir_name,file_name)
        print(f'## load model from {dir_name}')
        model = self.model_class.from_pretrained(dir_name, config=self.config)
        tokenizer = AutoTokenizer.from_pretrained(dir_name)
        return model, tokenizer

    def forward(self, **inputs):
        outputs = self.model(**inputs)
        return outputs
        
    def training_step(self, batch, batch_idx):
        # inputs = batch.pretrain_input()
        inputs = batch
        outputs = self(**inputs)
        loss = outputs[0]
        
        self.log('train_loss', loss)
        return loss
    
    def training_epoch_end(self, outputs):
        loss_lst = [x['loss'].item() for x in outputs]
        self.record['train_loss'].append(np.mean(loss_lst))

    def validation_step(self, batch, batch_idx):
        # inputs = batch.pretrain_input()
        inputs = batch
        outputs = self(**inputs)
        loss,logits = outputs[:2]
        self.log('valid_loss', loss)
        preds = self.model.decode(logits)
        labels = batch['labels']
        return {"valid_loss": loss, "preds": preds, "labels": labels}

    def validation_epoch_end(self, outputs):
        preds = torch.cat([x["preds"] for x in outputs]).detach().cpu().numpy().tolist()
        labels = torch.cat([x["labels"] for x in outputs]).detach().cpu().numpy().tolist()
        f1 = f1_score(labels, preds)
        precision = precision_score(labels, preds)
        recall = recall_score(labels, preds)
        if f1 > self.f1_max:
            self.f1_max = f1
            self.save_model()
        print({"f1": f1, "precision": precision, "recall": recall})

        loss_lst = [x['valid_loss'].item() for x in outputs]
        self.record['valid_loss'].append(np.mean(loss_lst))
        self.record['f1'].append(f1)
        self.record['precision'].append(precision)
        self.record['recall'].append(recall)

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def test_epoch_end(self, outputs):
        self.validation_epoch_end(outputs)

        

    def setup(self, stage):
        if stage == 'fit':
            ngpus = (len(self.hparams.gpus.split(',')) 
                     if type(self.hparams.gpus) is str else
                     self.hparams.gpus)

            effective_batch_size = self.hparams.train_batch_size * self.hparams.accumulate_grad_batches * ngpus
            dataset_size = len(self.train_dataloader().dataset)
            self.total_steps = (dataset_size / effective_batch_size) * self.hparams.max_epochs


    def get_lr_scheduler(self):
        get_scheduler_func = arg_to_scheduler[self.hparams.lr_scheduler]
        if self.hparams.lr_scheduler == 'constant':
            scheduler = get_scheduler_func(self.opt, num_warmup_steps=self.hparams.warmup_steps)
        else:
            scheduler = get_scheduler_func(self.opt, num_warmup_steps=self.hparams.warmup_steps, 
                                           num_training_steps=self.total_steps)
        scheduler = {'scheduler': scheduler, 'interval': 'step', 'frequency': 1}
        return scheduler

    def configure_optimizers(self):
        no_decay = ['bias', 'LayerNorm.weight']
    
        def has_keywords(n, keywords):
            return any(nd in n for nd in keywords)
        
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in self.model.named_parameters() if not has_keywords(n, no_decay)],
                'lr': self.hparams.learning_rate,
                'weight_decay': 0
            },
            {
                'params': [p for n, p in self.model.named_parameters() if has_keywords(n, no_decay)],
                'lr': self.hparams.learning_rate,
                'weight_decay': self.hparams.weight_decay
            }
        ]

        optimizer = (
            Adafactor(optimizer_grouped_parameters, scale_parameter=False, relative_step=False)
            if self.hparams.adafactor else
            AdamW(optimizer_grouped_parameters, eps=self.hparams.adam_epsilon)
        )
        self.opt = optimizer
        scheduler = self.get_lr_scheduler()

        return [optimizer], [scheduler]

    @staticmethod
    def add_model_specific_args(parser):
        parser.add_argument("--learning_rate", default=1e-5, type=float)
        parser.add_argument("--adam_epsilon", default=1e-8, type=float)
        parser.add_argument("--warmup_steps", default=0, type=int)
        parser.add_argument("--weight_decay", default=0., type=float)
        parser.add_argument("--lr_scheduler", type=str)
        parser.add_argument("--adafactor", action='store_true')

        parser.add_argument("--seed", default=42, type=int)
        parser.add_argument("--output_dir", type=str)
        parser.add_argument("--do_train", action='store_true')

        return parser

class LoggingCallback(pl.Callback):
    def on_validation_end(self, trainer, pl_module):
        pass


    # def on_train_end(self, trainier, pl_module):
    #     torch.cuda.empty_cache()



def main():
    parser = argparse.ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser = Training.add_model_specific_args(parser)
    parser = DataModule.add_argparse_args(parser)

    args = parser.parse_args()
    pl.seed_everything(args.seed)

    if args.learning_rate >= 1:
        args.learning_rate /= 1e5

    
    data_module = DataModule.from_argparse_args(args)
    data_module.load_dataset()
    data_module.prepare_dataset()
    model = Training(args)

    logging_callback = LoggingCallback()

    kwargs = {
        'weights_summary': None,
        'callbacks': [logging_callback],
        'logger': True,
        'checkpoint_callback': False,
        'num_sanity_val_steps': 5,
    }
    trainer = pl.Trainer.from_argparse_args(args, **kwargs)
    if args.do_train:
        trainer.fit(model, datamodule=data_module)
    else:
        trainer.test(model, datamodule=data_module)
    print(model.record)
    dir_name = os.path.join(args.output_dir, 'result', 'result.txt')
    mkdir_if_not_exist(dir_name)
    with open(dir_name,'wb') as f:
        pickle.dump(model.record, f)


if __name__ == '__main__':
    main()