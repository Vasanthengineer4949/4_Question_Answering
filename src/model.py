import config
import pytorch_lightning as pl
from pytorch_lightning import LightningModule
from transformers import MT5ForConditionalGeneration, AdamW
import torch
import gc

class Model(LightningModule):

    def __init__(self):
        super().__init__()
        torch.cuda.empty_cache()
        gc.collect()
        self.model = MT5ForConditionalGeneration.from_pretrained(config.MODEL_CKPT)
        torch.cuda.empty_cache()
        gc.collect()

    def forward(self, input_ids, attention_mask, labels=None):
        torch.cuda.empty_cache()
        gc.collect()
        output = self.model(
            input_ids = input_ids,
            attention_mask = attention_mask,
            labels = labels
        )

        return output.loss, output.logits

    def training_step(self, batch, batch_idx):
        torch.cuda.empty_cache()
        gc.collect()
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        loss, outputs = self(input_ids, attention_mask, labels)
        self.log("train_loss", loss, prog_bar=True, logger=False)
        return loss

    def validation_step(self, batch, batch_idx):
        torch.cuda.empty_cache()
        gc.collect()
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        loss, outputs = self(input_ids, attention_mask, labels)
        self.log("val_loss", loss, prog_bar=True, logger=False)
        return loss

    def test_step(self, batch, batch_idx):
        torch.cuda.empty_cache()
        gc.collect()
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        loss, outputs = self(input_ids, attention_mask, labels)
        self.log("test_loss", loss, prog_bar=True, logger=False)
        return loss
    
    def configure_optimizers(self):
        torch.cuda.empty_cache()
        gc.collect()
        return AdamW(self.parameters(), lr=0.001)
        
        