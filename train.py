import os
import math
import argparse
import logging
import json
import yaml

import torch
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR, LinearLR

from nn.model_templates import BLIPWrapper

parser = argparse.ArgumentParser(description='Model')

logger = logging.getLogger()
logger.setLevel(logging.INFO)

DIR = os.path.dirname(os.path.realpath(__file__))

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

class TokenizerArgs():
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(
            parents=[parent_parser], add_help=False)
        
        parser.add_argument('--enc_tok', type=str, default="custom")
        parser.add_argument('--dec_tok', type=str, default="custom")
        
        return parser   

class ModelCommonArgs():
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(
            parents=[parent_parser], add_help=False)
        
        parser.add_argument('--n_heads', type=int, default=8)
        parser.add_argument('--d_model', type=int, default=512) 
        parser.add_argument('--dropout', type=float, default=0.1)
        parser.add_argument("--max_len", type=int, default=50, help="Maximum length of the output utterances")
        
        return parser   
    
class EncoderArgs():
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(
            parents=[parent_parser], add_help=False)
        
        parser.add_argument('--enc_n_layers', type=int, default=6)
        parser.add_argument('--enc_plm', type=str, default=None)
        
        return parser   

class DecoderArgs():
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(
            parents=[parent_parser], add_help=False)
        
        parser.add_argument('--dec_n_layers', type=int, default=1)

        parser.add_argument("--dec_train_strategy", type=str, default=None, help="Non-autoregressive decoder training strategy")
        
        return parser   

class InformationPredictorArgs():
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(
            parents=[parent_parser], add_help=False)
        
        parser.add_argument('--ip_architecture', type=str, default=None, help="information predictor's architecture")
        parser.add_argument('--ip_information', type=str, default=None, help="what information")
        parser.add_argument("--ip_num_class", type=int, default=200, help="class of information")
        
        return parser   

class Module(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters(args)

        self.model = BLIPWrapper(args)

        # print("모듈 이ㅣㄴㅅ")
        # print(type(self.model.query_tokens))
        # print(type(self.model.vision_model))
        # print(type(self.model.qformer))
        # print(type(self.model.language_model))
        # print(type(self.model.language_projection))


        # self.model.query_tokens.eval()
        # self.model.query_tokens.requires_grad_(False)

        # self.model.vision_model.eval()
        # self.model.vision_model.requires_grad_(False)

        # self.model.qformer.eval()
        # self.model.qformer.requires_grad_(False)

        self.args = args
    
    def configure_optimizers(self):
        # Prepare optimizer     

        param_optimizer = list(self.model.named_parameters())
        
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(
                nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(
                nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters,lr=self.hparams.lr)
        num_workers = self.hparams.num_workers
        num_train_steps = self.trainer.estimated_stepping_batches
        logging.info(f'number of workers {num_workers}, num_train_steps : {num_train_steps}')
        num_warmup_steps = int(num_train_steps * self.hparams.warmup_ratio)
        logging.info(f'num_warmup_steps : {num_warmup_steps}')

        def lr_lambda(current_step):
            num_cycles = float(0.5)
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            progress = float(current_step - num_warmup_steps) / float(max(1, num_train_steps - num_warmup_steps))
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))
    
        def linear_lr_lambda(current_step):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            progress = float(current_step - num_warmup_steps) / float(max(1, num_train_steps - num_warmup_steps))
            return max(0.0, 1.0 - progress)

        # scheduler = LambdaLR(optimizer,lr_lambda=lr_lambda, last_epoch=self.current_epoch - 1)
        scheduler = LambdaLR(optimizer,lr_lambda=lr_lambda, last_epoch=self.current_epoch - 1)
        

        lr_scheduler = {'scheduler': scheduler,
                        'monitor': 'loss', 'interval': 'step',
                        'frequency': 1}

        return [optimizer], [lr_scheduler]
    
    @torch.no_grad()
    def generate(self, x, mask, attn_mask = None, **kwrags):

        return self.model.generate(x, mask, attn_mask, **kwrags)

    def forward(self, inputs):
        return self.model(**inputs) # = loss
        
    def training_step(self, batch):
        loss = self(batch)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self(batch)
        return (loss)

    def validation_epoch_end(self, outputs):
        losses = []
        for loss in outputs:
            losses.append(loss)
        val_loss_mean = torch.stack(losses).mean()
        self.log('val_loss', val_loss_mean, prog_bar=True, sync_dist=True)


if __name__=="__main__":
    parser = ArgBase.add_model_specific_args(parser)
    parser = ModelCommonArgs.add_model_specific_args(parser)
    # parser = EncoderArgs.add_model_specific_args(parser)
    # parser = DecoderArgs.add_model_specific_args(parser)
    # parser = InformationPredictorArgs.add_model_specific_args(parser)
    parser = TokenizerArgs.add_model_specific_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    logging.info(args)

    from examples import ImageCaptionDataModule as DataModule

    dm = DataModule(args)
    
    enc_tok = dm.enc_tok
    dec_tok = dm.dec_tok

    
    
    m = Module(args)
    
    dir_path = os.path.join(DIR, "model")
    checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor='val_loss',
                                                        dirpath=dir_path,
                                                        filename='{epoch:02d}-{val_loss:.3f}',
                                                        verbose=True,
                                                        save_top_k=3,
                                                        save_last=True,
                                                        mode='min')

    tb_logger = pl_loggers.TensorBoardLogger(os.path.join(dir_path, 'tb_logs'))
    lr_logger = pl.callbacks.LearningRateMonitor()

    trainer = pl.Trainer.from_argparse_args(args, accelerator='gpu', devices=args.devices, strategy="ddp", precision=16,
                                        logger=tb_logger, callbacks=[checkpoint_callback, lr_logger])
    
    trainer.fit(model=m, datamodule=dm)
