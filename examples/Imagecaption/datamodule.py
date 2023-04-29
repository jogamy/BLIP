import os
from dataclasses import dataclass
from typing import Any
import json

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..datamodule import BaseCollator, BaseDataModule, BaseTestCollator
from examples.Imagecaption.NICE.dataset import NICEtrainDataset
from transformers import AutoTokenizer

DIR = os.path.dirname(os.path.realpath(__file__))

@dataclass
class TestCollator(BaseTestCollator):

    def __call__(self, features):
        return super().__call__(features)
'''
self,
        pixel_values: torch.FloatTensor,
        input_ids: torch.FloatTensor,
        attention_mask: Optional[torch.LongTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        labels: Optional[torch.LongTensor] = None,
        return_dict: Optional[bool] = None,
'''
@dataclass
class ImageCaptionCollator(BaseCollator):
    
    def __call__(self, features):
        for feature in features:
            feature['decoder_input_ids'] = self.make_dec_ids(feature['caption'])
            feature['decoder_attention_mask'] = self.make_attn_mask(feature['decoder_input_ids'])
            feature['decoder_input_ids'] = self.padding(feature['decoder_input_ids'])
            assert len(feature['decoder_input_ids']) == self.max_len, f"{len(feature['decoder_input_ids'])}"
            feature['caption'] = self.labeling(feature['caption'])
            
        batch = {
            "pixel_values": torch.FloatTensor(np.stack([feature['pixel_values'] for feature in features])),
            "labels": torch.LongTensor(np.stack([feature['caption'] for feature in features])),
            "decoder_input_ids": torch.LongTensor(np.stack([feature['decoder_input_ids'] for feature in features])),
            "decoder_attention_mask": torch.LongTensor(np.stack([feature['decoder_attention_mask'] for feature in features])),                    
        }

        # batch = {
        #     "pixel_values": torch.HalfTensor(np.stack([feature['pixel_values'] for feature in features])),
        #     "labels": torch.LongTensor(np.stack([feature['caption'] for feature in features])),
        #     "decoder_input_ids": torch.LongTensor(np.stack([feature['decoder_input_ids'] for feature in features])),
        #     "decoder_attention_mask": torch.LongTensor(np.stack([feature['decoder_attention_mask'] for feature in features])),                    
        # }
   
        return batch
    
    def make_dec_ids(self, caption_ids):
        decoder_input_ids = np.concatenate([
            [1],    # 1이 bos던데?
            caption_ids,
        ])

        return decoder_input_ids
    
    def padding(self, dec_ids):
        dec_ids = np.concatenate([
            dec_ids,
            [0] * (self.max_len - len(dec_ids))
        ])
        return dec_ids

    def make_attn_mask(self, ids):
        attention_mask = np.concatenate([
            [1] * len(ids),
            [0] * (self.max_len - len(ids))]
        )
        return attention_mask

    def labeling(self, label):
        labels = np.concatenate([
            label,
            [self.label_pad_token_id] * (self.max_len - len(label))
        ])
        return labels
    
    
        

class ImageCaptionDataModule(BaseDataModule):
    def __init__(self, args):
        super().__init__(args)

        self.test_path = "/root/BLIP/examples/Imagecaption/NICE/test"

        self.datacollator = ImageCaptionCollator(self.dec_tok, self.max_len)
    
    def setup(self, stage):
        path = "/root/BLIP/examples/Imagecaption/NICE/cvpr-nice-val"
        if self.dataset == 'NICE':
            self.train = NICEtrainDataset(path, self.vis_processor, self.dec_tok, self.max_len)
            self.valid = NICEtrainDataset(path, self.vis_processor, self.dec_tok, self.max_len)


    def inference_setup(self):
        from examples.Imagecaption.NICE.dataset import NICEDataset
        self.test = NICEDataset(self.test_path, self.vis_processor, self.dec_tok, self.max_len)

    def test_dataloader(self):
        collat_fn = TestCollator(self.dec_tok,
                                self.max_len, self.enc_tok.pad_token_id)
        test = DataLoader(self.test, collate_fn=collat_fn,
                           batch_size=1,
                           num_workers=self.num_workers, shuffle=False)
        return test
        