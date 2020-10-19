
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
from pytorch_lightning.callbacks import Callback, ModelCheckpoint

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
        # self.classifier = BaselineClasHead(config, num_aspect=6, num_rating=5)
        self.classifier = AvgClasHead(config, num_aspect=6, num_rating=5, average=True)
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
        # logits = self.classifier(sequence_output)
        logits = self.classifier(sequence_output, attention_mask)

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


class AvgClasHead(torch.nn.Module):

    def __init__(self, config, num_aspect, num_rating, average:bool=True):
        super().__init__()
        self.average = average

        self.ln1 = nn.LayerNorm(config.hidden_size)
        self.dp1 = nn.Dropout(0.5)
        self.dense1 = nn.Linear(config.hidden_size, 300)

        self.ln2 = nn.LayerNorm(300)
        self.dp2 = nn.Dropout(0.4)
        self.dense2 = nn.Linear(300, num_aspect * num_rating)

    def forward(self, embedding: torch.Tensor, mask: torch.Tensor):
        embedding = embedding * mask.unsqueeze(-1).float()
        embedding = embedding.sum(1)

        if self.average:
            lengths = mask.long().sum(-1)
            length_mask = (lengths > 0)
            # Set any length 0 to 1, to avoid dividing by zero.
            lengths = torch.max(lengths, lengths.new_ones(1))
            # normalize by length
            embedding = embedding / lengths.unsqueeze(-1).float()
            # set those with 0 mask to all zeros, i think
            embedding = embedding * (length_mask > 0).float().unsqueeze(-1)

        embedding = self.ln1(embedding)
        embedding = self.dp1(embedding)
        embedding = self.dense1(embedding)

        embedding = self.ln2(embedding)
        embedding = self.dp2(embedding)
        logits = self.dense2(embedding)

        return logits.view(-1, 6, 5)


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
                dist_sync_on_step: bool = False,
                process_group: Optional[Any] = None,):
        super().__init__(
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
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
        # for param in self.longformer.longformer.embeddings.parameters():
        #     param.requires_grad = False
        # for param in self.longformer.longformer.encoder.parameters():
        #     param.requires_grad = False

        self.lossfunc = MultiLabelCEL()
        self.metrics = [AspectACC(aspect=i) for i in range(6)]

    def configure_optimizers(self):
        # optimizer = torch.optim.AdamW(self.parameters(), lr=self.train_config["learning_rate"])
        optimizer = transformers.AdamW(model.parameters(), lr=train_config["learning_rate"], weight_decay=0.01)
        scheduler = transformers.get_cosine_with_hard_restarts_schedule_with_warmup(optimizer,
                                                                                    num_warmup_steps=500,
                                                                                    num_training_steps=1000,
                                                                                    num_cycles=20)
        schedulers = [    
        {
         'scheduler': scheduler,
         'interval': 'step',
         'frequency': 1
        }]
        return [optimizer], schedulers

    def train_dataloader(self):
        self.dataset_train = ReviewDataset("../../data/hotel_balance_LengthFix1_3000per/df_train.pickle")
        self.loader_train = DataLoader(self.dataset_train,
                                        batch_size=train_config["batch_size"],
                                        collate_fn=TokenizerCollate(),
                                        num_workers=0,
                                        pin_memory=True, drop_last=False, shuffle=True)
        return self.loader_train

    def val_dataloader(self):
        self.dataset_val = ReviewDataset("../../data/hotel_balance_LengthFix1_3000per/df_test.pickle")
        self.loader_val = DataLoader(self.dataset_val,
                                        batch_size=train_config["batch_size"],
                                        collate_fn=TokenizerCollate(),
                                        num_workers=0,
                                        pin_memory=True, drop_last=False, shuffle=True)
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

    def on_after_backward(self):
        if (self.trainer.batch_idx + 1) % self.trainer.accumulate_grad_batches == 0:
            self.log("learning_rate", self.trainer.optimizers[0].param_groups[0]['lr'] )
            with torch.no_grad():
                if (model.longformer.longformer.embeddings.word_embeddings.weight.grad is not None):
                    norm_value = model.longformer.longformer.embeddings.word_embeddings.weight.grad.detach().norm(2).item()
                    # assert not np.isnan(norm_value)
                    self.log('NORMS/embedding norm', norm_value)

                for i in [0, 4, 8, 11]:
                    if (model.longformer.longformer.encoder.layer[i].output.dense.weight.grad is not None):
                        norm_value = model.longformer.longformer.encoder.layer[i].output.dense.weight.grad.detach().norm(2).item()
                        # assert not np.isnan(norm_value)
                        self.log('NORMS/encoder %d output norm' % i, norm_value)

                if (self.longformer.classifier.dense1.weight.grad is not None):
                    norm_value = self.longformer.classifier.dense1.weight.grad.detach().norm(2).item()
                    # assert not np.isnan(norm_value)
                    self.log("dense1_norm2", norm_value)

                if (self.longformer.classifier.dense2.weight.grad is not None):
                    norm_value = self.longformer.classifier.dense2.weight.grad.detach().norm(2).item()
                    # assert not np.isnan(norm_value)
                    self.log("dense2_norm2", norm_value)

    def validation_step(self, batch, batch_idx):
        input_ids, mask, label  = batch[0].type(torch.int64), batch[1].type(torch.int64), batch[2].type(torch.int64)
        
        loss, logits = self(input_ids=input_ids, attention_mask=mask, labels=label)
        
        # self.log('val_loss', loss, on_step=False, on_epoch=True, reduce_fx=torch.mean, prog_bar=False)
        accs = [m(logits, label) for m in self.metrics]  # update metric counters
        
        return {"val_loss": loss}
    
    def validation_epoch_end(self, validation_step_outputs):
        avg_loss = torch.stack([x['val_loss'] for x in validation_step_outputs]).mean()
        self.log("val_loss", avg_loss)

        for i,m in enumerate(self.metrics):
            self.log('acc'+str(i), m.compute())

    def test_step(self, batch, batch_idx):
        input_ids, mask, label  = batch[0].type(torch.int64), batch[1].type(torch.int64), batch[2].type(torch.int64)
        
        loss, logits = self(input_ids=input_ids, attention_mask=mask, labels=label)
        accs = [m(logits, label) for m in self.metrics]  # update metric counters
        
        return loss

    def on_test_epoch_end(self):
        for i,m in enumerate(self.metrics):
            print('acc'+str(i), m.compute())

# %%
train_config = {}
train_config["cache_dir"] = "./cache/"
train_config["epochs"] = 15
train_config["batch_size"] = 4
train_config["accumulate_grad_batches"] = 15
train_config["gradient_clip_val"] = 1.5
train_config["learning_rate"] = 3e-5

# %%
wandb_logger = WandbLogger(name='baseline_fulltrain',project='saam_hotel_longformer')
wandb_logger.log_hyperparams(train_config)
wandb.save('./Hotel_Transformer_Baseline.py')

# %%
# model = LightningLongformerBaseline(train_config)
model = LightningLongformerBaseline.load_from_checkpoint("./saam_hotel_longformer/2q1t5ns9/checkpoints/epoch=10.ckpt", config=train_config)


# %%
cp_valloss = ModelCheckpoint(filepath=wandb.run.dir+'{epoch:02d}-{val_loss:.2f}', save_top_k=5, monitor='val_loss', mode='min')
cp_acc0 = ModelCheckpoint(filepath=wandb.run.dir+'{epoch:02d}-{acc0:.2f}', save_top_k=5, monitor='acc0', mode='max')
cp_acc3 = ModelCheckpoint(filepath=wandb.run.dir+'{epoch:02d}-{acc3:.2f}', save_top_k=1, monitor='acc3', mode='max')

# %%
pl.seed_everything(42)

trainer = pl.Trainer(max_epochs=train_config["epochs"],
                     accumulate_grad_batches=train_config["accumulate_grad_batches"],
                     gradient_clip_val=train_config["gradient_clip_val"],

                     gpus=1, num_nodes=1,

                     amp_backend='native',
                     precision=16,

                     logger=wandb_logger,
                     log_every_n_steps=1,

                     val_check_interval=0.5,
                     limit_val_batches=500,

                     checkpoint_callback=cp_acc0,

                    #  resume_from_checkpoint='./saam_hotel_longformer/2q1t5ns9/checkpoints/epoch=10.ckpt'
                     )


# %%
trainer.fit(model)

# %%
# model = LightningLongformerBaseline.load_from_checkpoint("./saam_hotel_longformer/2q1t5ns9/checkpoints/epoch=10.ckpt", config=train_config)
# trainer.test(
#                 model=model,
#                 test_dataloaders=model.val_dataloader(),
#                 )

# %%



