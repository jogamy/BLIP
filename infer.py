import os
import csv
import argparse
import time
from tqdm import tqdm
import yaml
import json
import logging

import torch
import numpy as np
import pytorch_lightning as pl

from train import Module

DIR = os.path.dirname(os.path.realpath(__file__))

# from training
parser = argparse.ArgumentParser(description='Model')

class ArgBase():
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(
            parents=[parent_parser], add_help=False)

        parser.add_argument('--task',
                            type=str,
                            default=None,
                            help='choose task')

        parser.add_argument('--dataset',
                            type=str,
                            default=None,
                            help='choose dataset')

        parser.add_argument('--batch_size',
                            type=int,
                            default=32,
                            help='')

        parser.add_argument('--num_workers',
                            type=int,
                            default=5,
                            help='num of worker for dataloader')

        parser.add_argument('--lr',
                            type=float,
                            default=5e-4,
                            help='The initial learning rate')

        parser.add_argument('--warmup_ratio',
                            type=float,
                            default=0.05,
                            help='warmup ratio')

        parser.add_argument("--model_path", type=str, default=None, help="Model path")

        parser.add_argument("--model_card", type=str, default=None, help="Model card")

        
        
        return parser   

class ModelCommonArgs():
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(
            parents=[parent_parser], add_help=False)
        
        parser.add_argument('--n_heads', type=int, default=8)
        parser.add_argument('--d_model', type=int, default=512) 
        parser.add_argument('--dropout', type=float, default=0.1)
        parser.add_argument("--max_len", type=int, default=200, help="Maximum length of the output utterances")
        
        return parser   
    

class TokenizerArgs():
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(
            parents=[parent_parser], add_help=False)
        
        parser.add_argument('--enc_tok', type=str, default="custom")
        parser.add_argument('--dec_tok', type=str, default="custom")
        
        return parser   


if __name__ == '__main__':

    parser = ArgBase.add_model_specific_args(parser)
    parser = ModelCommonArgs.add_model_specific_args(parser)
    # parser = EncoderArgs.add_model_specific_args(parser)
    # parser = DecoderArgs.add_model_specific_args(parser)
    # parser = InformationPredictorArgs.add_model_specific_args(parser)
    parser = TokenizerArgs.add_model_specific_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    logging.info(args)


    # parser = argparse.ArgumentParser()
    # parser.add_argument("--hparams", default=None, type=str)
    # parser.add_argument("--model_path", default=None, type=str)
    # args = parser.parse_args()

    # with open(args.hparams) as f:
    #     hparams = yaml.load(f, Loader=yaml.FullLoader)
    #     hparams.update(vars(args))

    # args = argparse.Namespace(**hparams)

    from examples.Imagecaption.datamodule import ImageCaptionDataModule as DataModule
        
    datamodule = DataModule(args)
    datamodule.inference_setup()

    test_dataloader = datamodule.test_dataloader()

    enc_tok = datamodule.enc_tok
    dec_tok = datamodule.dec_tok 

    tokneizers = {
        'enc_tok': enc_tok,
        'dec_tok': dec_tok,
    }

    # if model 있으면
    #module = Module.load_from_checkpoint(args.model_path, args=args)
    # else:
    module = Module(args)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    module.model = module.model.to(device)
    module.model.eval()
        
    start_time = time.time()
    
    results = {
        'public_id': [],
        'caption': [],
    }

    for i, inp in enumerate(tqdm(test_dataloader)):

        out = module.generate(inp['pixel_values'].to(device), None, None, max_length=100)
        # 여기 top_p, top_k 디코딩 넣어도 됟듯
        seq = dec_tok.decode(out[0], skip_special_tokens=True)

        results['public_id'] += inp['public_id']
        results['caption'].append(seq.strip()) 

    end_time = time.time()

    rows = list(map(list, zip(*results.values())))
    with open(os.path.join(DIR, args.default_root_dir), 'w') as f:
        w = csv.writer(f)
        w.writerow(results.keys())
        w.writerows(rows)


