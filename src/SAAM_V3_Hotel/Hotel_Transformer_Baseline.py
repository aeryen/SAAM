
# %%
import os
from typing import Any, Optional

import pandas as pd
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import transformers
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers.modeling_longformer import LongformerModel, LongformerPreTrainedModel

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger


# %%
import wandb

# %%
print(os.getcwd())
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)
print(os.getcwd())

# %%
class ReviewDataset(Dataset):

    def __init__(self, df_path):
        self.df = pd.read_pickle(df_path)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return (self.df.iloc[idx,6], self.df.iloc[idx,0:6].to_numpy().astype(np.float) )


# %%
class LongformerBaseline(LongformerPreTrainedModel):

    authorized_unexpected_keys = [r"pooler"]

    def __init__(self, config):
        super().__init__(config)
        self.longformer = LongformerModel(config, add_pooling_layer=False)
        self.classifier = BaselineClasHead(config, num_aspect=6, num_rating=5)
        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        global_attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if global_attention_mask is None:
            global_attention_mask = torch.zeros_like(input_ids)
            # global attention on cls token
            global_attention_mask[:, 0] = 1

        outputs = self.longformer(
            input_ids,
            attention_mask=attention_mask,
            global_attention_mask=global_attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]
        logits = self.classifier(sequence_output)

        return logits


class BaselineClasHead(nn.Module):

    def __init__(self, config, num_aspect, num_rating):
        super().__init__()
        # self.ln1 = nn.LayerNorm(config.hidden_size)
        self.dp1 = nn.Dropout(0.4)
        self.dense1 = nn.Linear(config.hidden_size, 400)
        
        # self.ln2 = nn.LayerNorm(400)
        self.dp2 = nn.Dropout(0.4)
        self.dense2 = nn.Linear(400, num_aspect * num_rating)

    def forward(self, hidden_states, **kwargs):
        hidden_states = hidden_states[:, 0, :]  # take <s> token (equiv. to [CLS])
        
        # hidden_states = self.ln1(hidden_states)
        hidden_states = self.dp1(hidden_states)
        hidden_states = self.dense1(hidden_states)
        
        hidden_states = torch.tanh(hidden_states)
        
        # hidden_states = self.ln2(hidden_states)
        hidden_states = self.dp2(hidden_states)
        hidden_states = self.dense2(hidden_states)
        
        return hidden_states.view(-1, 6, 5)


# %%
class TokenizerCollate:
    def __init__(self):
        self.tkz = AutoTokenizer.from_pretrained("allenai/longformer-base-4096")
    
    def __call__(self, batch):
        batch_split = list(zip(*batch))
        seqs, targs= batch_split[0], batch_split[1]
        encode = self.tkz(seqs, padding="longest")
        return torch.tensor(encode["input_ids"]), torch.tensor(encode["attention_mask"]), torch.tensor(targs)
    
class MultiLabelCEL(nn.CrossEntropyLoss):
    def forward(self, input, target, nasp=6):
        target = target.long()
        loss = 0
        for i in range(nasp):
            loss = loss + super(MultiLabelCEL, self).forward(input[:,i,:], target[:,i])
        
        return loss
    
class AspectACC(pl.metrics.metric.Metric):
    def __init__(self, aspect: int,
                compute_on_step: bool = True,
                ddp_sync_on_step: bool = False,
                process_group: Optional[Any] = None,):
        super().__init__(
            compute_on_step=compute_on_step,
            ddp_sync_on_step=ddp_sync_on_step,
            process_group=process_group,)
        
        self.aspect = aspect
        self.add_state("correct", default=torch.tensor(0).cuda(), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0).cuda(), dist_reduce_fx="sum")
    
    def update(self, preds: torch.Tensor, target: torch.Tensor):
        preds = torch.argmax(preds, dim=2)
        assert preds.shape == target.shape
        
        target = target.contiguous().long()
        
        self.correct += torch.sum( preds[:, self.aspect]==target[:, self.aspect] )
        self.total += target[:, self.aspect].numel()
        
    def compute(self):
        return self.correct.float() / self.total


# %%
class LightningLongformerBaseline(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.train_config = config
        self.longformer = LongformerBaseline.from_pretrained('allenai/longformer-base-4096',
                                                             cache_dir=self.train_config["cache_dir"],
                                                            #  gradient_checkpointing=True
                                                               )
        for param in self.longformer.longformer.embeddings.parameters():
            param.requires_grad = False
        for param in self.longformer.longformer.encoder.parameters():
            param.requires_grad = False

        self.lossfunc = MultiLabelCEL()
        self.metrics = [AspectACC(aspect=i) for i in range(6)]

    def configure_optimizers(self):
        # optimizer = torch.optim.AdamW(self.parameters(), lr=self.train_config["learning_rate"])
        optimizer = transformers.AdamW(model.parameters(), lr=train_config["learning_rate"], weight_decay=0.01)
        return optimizer

    def train_dataloader(self):
        self.dataset_train = ReviewDataset("../../data/hotel_balance_LengthFix1_3000per/df_train.pickle")
        self.loader_train = DataLoader(self.dataset_train,
                                        batch_size=train_config["batch_size"],
                                        collate_fn=TokenizerCollate(),
                                        num_workers=0,
                                        pin_memory=True, drop_last=False, shuffle=False)
        return self.loader_train

    def val_dataloader(self):
        self.dataset_val = ReviewDataset("../../data/hotel_balance_LengthFix1_3000per/df_test.pickle")
        self.loader_val = DataLoader(self.dataset_val,
                                        batch_size=train_config["batch_size"],
                                        collate_fn=TokenizerCollate(),
                                        num_workers=0,
                                        pin_memory=True, drop_last=False, shuffle=False)
        return self.loader_val
    
#     @autocast()
    def forward(self, input_ids, attention_mask, labels):
        logits = self.longformer(input_ids=input_ids, attention_mask=attention_mask)
        loss = self.lossfunc(logits, labels)

        return (loss, logits)
    
    def training_step(self, batch, batch_idx):
        input_ids, mask, label  = batch[0].type(torch.int64), batch[1].type(torch.int64), batch[2].type(torch.int64)
        
        loss, logits = self(input_ids=input_ids, attention_mask=mask, labels=label)
        
        self.log("train_loss", loss)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        input_ids, mask, label  = batch[0].type(torch.int64), batch[1].type(torch.int64), batch[2].type(torch.int64)
        
        loss, logits = self(input_ids=input_ids, attention_mask=mask, labels=label)
        
        self.log('val_loss', loss, on_step=False, on_epoch=True, reduce_fx=torch.mean, prog_bar=False)
        accs = [m(logits, label) for m in self.metrics]  # update metric counters
        
        return loss
    
    def validation_epoch_end(self, validation_step_outputs):
        for i,m in enumerate(self.metrics):
            self.log('acc'+str(i), m.compute())


# %%
train_config = {}
train_config["cache_dir"] = "./cache/"
train_config["epochs"] = 6
train_config["batch_size"] = 6
train_config["accumulate_grad_batches"] = 10
train_config["gradient_clip_val"] = 1.2
train_config["learning_rate"] = 3e-5

# %%
wandb_logger = WandbLogger(name='baseline_accumu',project='saam_hotel_longformer')
wandb_logger.log_hyperparams(train_config)


# %%
model = LightningLongformerBaseline(train_config)


# %%
trainer = pl.Trainer(max_epochs=train_config["epochs"],
                     accumulate_grad_batches=train_config["accumulate_grad_batches"],
                    #  gradient_clip_val=train_config["gradient_clip_val"],
                     gpus=1, num_nodes=1,
                     logger=wandb_logger,
                     log_every_n_steps=5)


# %%
trainer.fit(model)

# %%



