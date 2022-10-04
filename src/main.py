import config
import pandas as pd
import pytorch_lightning as pl
from data_gen import DataGen
from dataloader import MultiQADataLoader
from model import Model
from hf_hub_lightning import HuggingFaceHubCallback
import torch
import gc

if __name__ == "__main__":

    print("Start")
    pl.seed_everything(0)
    
    datagen = DataGen()
    datagen.create_data_csv()

    torch.cuda.empty_cache()
    gc.collect()

    print("Data CSV Created")

    multiqadataloader = MultiQADataLoader()
    multiqadataloader.setup(stage="fit")

    torch.cuda.empty_cache()
    gc.collect()


    print("Data Loader Set")

    model = Model()

    torch.cuda.empty_cache()
    gc.collect()

    trainer = pl.Trainer(
        callbacks=[HuggingFaceHubCallback(config.MODEL_OUT)],
        max_epochs=config.NUM_EPOCHS,
        gpus=1
    )
    
    torch.cuda.empty_cache()
    gc.collect()

    trainer.fit(model, multiqadataloader)
    trainer.test()


