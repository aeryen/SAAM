import os

print(os.getcwd())
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)
print(os.getcwd())

os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
# os.environ['CUDA_VISIBLE_DEVICES'] = "1,2"

from typing import Any, Optional

import pandas as pd
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import transformers
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers.models.longformer.modeling_longformer import LongformerModel, LongformerPreTrainedModel

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from pytorch_lightning.utilities.distributed import rank_zero_only

# from src.SAAM_V3_Hotel.Hotel_Transformer_Classification import ReviewDataset, LongformerClassification, LightningLongformerBaseline
# from Hotel_Transformer_Classification import ReviewDataset, LongformerClassification, LightningLongformerBaseline
from Hotel_Transformer_Baseline import LightningLongformerBaseline

if __name__ == "__main__":
    train_config = {}
    train_config["cache_dir"] = "./cache/"
    train_config["epochs"] = 16
    train_config["batch_size"] = 4
    train_config["accumulate_grad_batches"] = 12
    train_config["gradient_clip_val"] = 1.5
    train_config["learning_rate"] = 2e-5


    # model = LightningLongformerBaseline.load_from_checkpoint(
    #             "./saam_hotel_longformer/25wkq0pm/checkpoints/epoch=10-step=857.ckpt",
    #             config=train_config)

    model = LightningLongformerBaseline.load_from_checkpoint(
                "./saam_hotel_longformer/1k5ejgm2/checkpoints/epoch=13-step=1052.ckpt",
                config=train_config)

    trainer = pl.Trainer(max_epochs=train_config["epochs"],
                accumulate_grad_batches=train_config["accumulate_grad_batches"],
                gradient_clip_val=train_config["gradient_clip_val"],

                gpus=[5],
                num_nodes=1,
                # distributed_backend='ddp',

                amp_backend='native',
                precision=16,

                val_check_interval=0.5,
                limit_val_batches=500,
                )

    trainer.test(model=model,
                test_dataloaders=model.val_dataloader() )