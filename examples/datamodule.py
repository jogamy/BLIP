import os
from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from transformers import AutoTokenizer

DIR = os.path.dirname(os.path.realpath(__file__))

@dataclass
class BaseCollator:
    dec_tok: Any
    max_len : int = 100
    pad_id : int = 0
    label_pad_token_id : int = -100

    def __call__(self, features):
        for feature in features:
            # training set 결정하면 전처리 필요
            pass
        return {
            'public_id': [feature['public_id'] for feature in features],
            'pixel_values': torch.FloatTensor(np.stack([feature['pixel_values'] for feature in features]))
        }
                
    # def attention_mask(self, ids):
    #     attention_mask = np.concatenate([
    #         [True] * len(ids),
    #         [False] * (self.max_len - len(ids))]
    #     )
    #     return attention_mask

    # def padding(self, ids):
    #     ids = np.concatenate([
    #         ids,
    #         [self.pad_id] * (self.max_len - len(ids))
    #     ])
    #     return ids
        
    # def labeling(self, label):
    #     labels = np.concatenate([
    #         label,
    #         [self.label_pad_token_id] * (self.max_len - len(label))
    #     ])
    #     return labels

@dataclass
class BaseTestCollator(BaseCollator):
    def __call__(self, features):
        # train 때와 test때의 collator가 다를 수 있음
        return super().__call__(features)
    
        for feature in features:
            # training set 결정하면 전처리 필요
            pass
        return {
            'public_id': [feature['public_id'] for feature in features],
            'pixel_values': torch.FloatTensor(np.stack([feature['pixel_values'] for feature in features]))
        }
        

class BaseDataset(Dataset):
    def __init__(self, filepath, vis_processor, dec_tok, max_len, ignore_index=-100):
        self.filepath = filepath

        self.vis_processor = vis_processor
        self.dec_tok = dec_tok
        
        self.max_len = max_len

        self.ignore_index = ignore_index

        self.srcs = None
        self.tgts = None

    def __len__(self):
        return len(self.srcs)
    
    def __getitem__(self, index):
        raise NotImplementedError("Implement")
    

class BaseDataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()        

        self.dataset = args.dataset
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.max_len = args.max_len
        
        self.args = args

        from transformers import AutoProcessor
        self.vis_processor = AutoProcessor.from_pretrained(args.model_card)
        self.enc_tok = AutoTokenizer.from_pretrained(args.enc_tok)
        self.dec_tok = AutoTokenizer.from_pretrained(args.dec_tok)
    
    
    def setup(self, stage):
        raise NotImplementedError("Implement")
        
    def inference_setup(self):
        raise NotImplementedError("Implement")

    def train_dataloader(self):
        train = DataLoader(self.train, collate_fn=self.datacollator,
                           batch_size=self.batch_size,
                           num_workers=self.num_workers, shuffle=True)
        return train
        
    def val_dataloader(self):
        val = DataLoader(self.valid, collate_fn=self.datacollator,
                         batch_size=self.batch_size,
                         num_workers=self.num_workers, shuffle=False)
        return val

    def test_dataloader(self):
    #     self.datacollator = BaseCollator(self.lp_structure, 
    #                                         self.enc_tok, self.dec1_tok, self.dec2_tok,
    #                                         self.max_len, self.enc_tok.pad_token_id)
    #     test = DataLoader(self.test, collate_fn=self.datacollator,
    #                     #  batch_size=self.batch_size,
    #                     batch_size=1,
    #                      num_workers=self.num_workers, shuffle=False)
    #     return test
        raise NotImplementedError("Implement")
    