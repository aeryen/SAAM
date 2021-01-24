#%%
import os

# print(os.getcwd())
# abspath = os.path.abspath(__file__)
# dname = os.path.dirname(abspath)
# os.chdir(dname)
# print(os.getcwd())

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
from Hotel_Transformer_Classification import LightningLongformerCLS, ReviewDataset
# from Hotel_Transformer_Baseline import LightningLongformerBaseline

#%%
train_config = {}
train_config["cache_dir"] = "./cache/"
train_config["epochs"] = 16
train_config["batch_size"] = 10
train_config["accumulate_grad_batches"] = 12
train_config["gradient_clip_val"] = 1.5
train_config["learning_rate"] = 2e-5


model = LightningLongformerCLS.load_from_checkpoint(
            # "./saam_hotel_longformer/1ayk8022/checkpoints/epoch=4-step=841.ckpt",
            "./saam_hotel_longformer/1y4bg2wb/checkpoints/epoch=14-step=1409.ckpt",
            config=train_config)
model.eval()
model.freeze()

#%%
dataset_test = ReviewDataset("../../data/hotel_balance_LengthFix1_3000per/df_test.pickle")

#%%
input_id, mask, label = model.tokenCollate( [dataset_test[0]] )

#%%
(loss, logits, outputs, aspect_doc) = model(input_id, mask, label)

#%%
aspect_doc

#%%
dataset_test[0][0].split("xxPERIOD")

#%%
# model = LightningLongformerBaseline.load_from_checkpoint(
#             "./saam_hotel_longformer/1k5ejgm2/checkpoints/epoch=13-step=1052.ckpt",
#             config=train_config)

# trainer = pl.Trainer(max_epochs=train_config["epochs"],
#             accumulate_grad_batches=train_config["accumulate_grad_batches"],
#             gradient_clip_val=train_config["gradient_clip_val"],

#             gpus=[5],
#             num_nodes=1,
#             # distributed_backend='ddp',

#             amp_backend='native',
#             precision=16,

#             val_check_interval=0.5,
#             limit_val_batches=500,
#             )

# trainer.test(model=model,
#             test_dataloaders=model.test_dataloader() )

#%%
trainer.model.test_aspect_outputs[0].shape


#%%
aspect_dist = torch.cat(trainer.model.test_aspect_outputs, dim=0).detach().cpu().numpy()
aspect_dist.shape

# %%
def eval_hotel_asp(asp_pred, asp_true, asp_inc_overall):
    asp_to_id = {"value":0, "room":1, "location":2, "cleanliness":3, "service":4, "none":-1}
    asp_true = np.array( [asp_to_id[l] for l in asp_true] )
    print("total true: " + str(len(asp_true)) )
    print("total not none: " + str(sum(asp_true>0)) )
    
    asp_pred_index = []
    if asp_inc_overall:
        for i in range(1000):
            asp_pred_index.append( asp_pred[i,1:6].argsort() )
    else:
        for i in range(1000):
            asp_pred_index.append( asp_pred[i,0:5].argsort() )
    asp_pred_index = np.stack( asp_pred_index , axis=0)
    
    result_index = []
    for i,lbl in enumerate(asp_true):
        if(lbl==-1):
            result_index.append(-1)
        else:
            at = np.where(asp_pred_index[i,:] == lbl)
            result_index.append(at[0])
    result_index = np.array(result_index)
    
    print("Top 1 ACC:")
    print( sum(result_index>=4) / sum(result_index>=0) )
    print("Top 2 ACC:")
    print( sum(result_index>=3) / sum(result_index>=0) )

# %%
yifan_label = open("../../data/hotel_balance_LengthFix1_3000per/" + "test_aspect_0.yifanmarjan.aspect", "r").readlines()
yifan_label = [s.split()[0] for s in yifan_label]

# %%
len( yifan_label )

#%%
eval_hotel_asp(aspect_dist, yifan_label, asp_inc_overall=True)
# %%
