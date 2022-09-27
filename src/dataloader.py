import config
import pytorch_lightning as pl
from pytorch_lightning import LightningDataModule
import pandas as pd
from transformers import MT5Tokenizer
from dataset import MLQADataset
from torch.utils.data import DataLoader
import torch
import gc

class MultiQADataLoader(LightningDataModule):

    def __init__(
        self,
        ):
        super().__init__()
        self.batch_size = config.BATCH_SIZE
        self.train_df = pd.read_csv(config.TRAIN_DATA_PATH)
        self.test_df = pd.read_csv(config.TEST_DATA_PATH)
        self.valid_df = pd.read_csv(config.VALID_DATA_PATH)
        self.tokenizer = MT5Tokenizer.from_pretrained(config.MODEL_CKPT)
        self.src_max_token_len = config.QUESTION_MAX_LEN
        self.tgt_max_token_len = config.ANSWER_MAX_LEN

    def setup(self, stage):
        self.train_dataset = MLQADataset(
            self.train_df,
            self.tokenizer
        )

        self.test_dataset = MLQADataset(
            self.test_df,
            self.tokenizer
        )

        self.val_dataset = MLQADataset(
            self.valid_df,
            self.tokenizer
        )

    def train_dataloader(self):
        torch.cuda.empty_cache()
        gc.collect()
        return DataLoader(
            self.train_dataset,
            batch_size=1,
            shuffle=True,
            num_workers=4
        )

    def test_dataloader(self):
        torch.cuda.empty_cache()
        gc.collect()
        return DataLoader(
            self.test_dataset,
            batch_size=1,
            num_workers=4
        )
    
    def val_dataloader(self):
        torch.cuda.empty_cache()
        gc.collect()
        return DataLoader(
            self.val_dataset,
            batch_size=1,
            num_workers=4
        )