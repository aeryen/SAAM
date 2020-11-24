
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

import wandb

# %%
class ReviewDataset(Dataset):

    def __init__(self, df_path):
        self.df = pd.read_pickle(df_path)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return (self.df.iloc[idx,5], self.df.iloc[idx,0:5].to_numpy().astype(np.float) )


# %%
class LongformerBaseline(LongformerPreTrainedModel):

    authorized_unexpected_keys = [r"pooler"]

    def __init__(self, config):
        super().__init__(config)
        self.longformer = LongformerModel(config, add_pooling_layer=False)
        self.classifier = AvgClasHead(config, num_aspect=5, num_rating=5, average=True)
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
        logits = self.classifier(sequence_output, attention_mask)

        return logits


class BnDropLin(nn.Sequential):
    "Module grouping `BatchNorm1d`, `Dropout` and `Linear` layers"
    def __init__(self, n_in, n_out, bn=True, p=0., act=None):
        layers = [nn.LayerNorm(n_in)] if bn else []
        if p != 0: layers.append(nn.Dropout(p))
        lin = [nn.Linear(n_in, n_out, bias=not bn)]
        if act is not None: lin.append(act)
        layers = layers+lin
        super().__init__(*layers)
        self.lin = lin


class AvgClasHead(torch.nn.Module):

    def __init__(self, config, num_aspect, num_rating, average:bool=True):
        super().__init__()
        self.average = average

        self.lbd1 = BnDropLin(n_in=config.hidden_size, n_out=256, p=0.5, act=nn.ReLU(inplace=True))
        self.lbd2 = BnDropLin(n_in=256, n_out=num_aspect*num_rating, p=0.4, act=None)

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

        embedding = self.lbd1(embedding)
        logits    = self.lbd2(embedding)

        return logits.view(-1, 5, 5)


# %%
class TokenizerCollate:
    def __init__(self):
        self.tkz = AutoTokenizer.from_pretrained("allenai/longformer-base-4096")
    
    def __call__(self, batch):
        batch_split = list(zip(*batch))
        seqs, targs= batch_split[0], batch_split[1]
        encode = self.tkz(list(seqs), padding="longest")
        return torch.tensor(encode["input_ids"]), torch.tensor(encode["attention_mask"]), torch.tensor(targs)
    
class MultiLabelCEL(nn.CrossEntropyLoss):
    def forward(self, input, target, nasp=5):
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
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")
    
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
        self.metrics = torch.nn.ModuleList( [AspectACC(aspect=i) for i in range(5)] )

    def configure_optimizers(self):
        # optimizer = torch.optim.AdamW(self.parameters(), lr=self.train_config["learning_rate"])
        optimizer = transformers.AdamW(model.parameters(), lr=train_config["learning_rate"]) #, weight_decay=0.01
        scheduler = transformers.get_cosine_with_hard_restarts_schedule_with_warmup(optimizer,
                                                                                    num_warmup_steps=350,
                                                                                    num_training_steps=9000,
                                                                                    num_cycles=2)
        schedulers = [    
        {
         'scheduler': scheduler,
         'interval': 'step',
         'frequency': 1
        }]
        return [optimizer], schedulers

    def train_dataloader(self):
        self.dataset_train = ReviewDataset("../../data/beer_100k/df_train.pickle")
        self.loader_train = DataLoader(self.dataset_train,
                                        batch_size=train_config["batch_size"],
                                        collate_fn=TokenizerCollate(),
                                        num_workers=2,
                                        pin_memory=True, drop_last=False, shuffle=True)
        return self.loader_train

    def val_dataloader(self):
        self.dataset_val = ReviewDataset("../../data/beer_100k/df_test.pickle")
        self.loader_val = DataLoader(self.dataset_val,
                                        batch_size=train_config["batch_size"],
                                        collate_fn=TokenizerCollate(),
                                        num_workers=2,
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

            # if (self.longformer.classifier.lbd1[2].weight.grad is not None):
            #     norm_value = self.longformer.classifier.lbd1[2].weight.grad.detach().norm(2).item()
            #     # assert not np.isnan(norm_value)
            #     self.log("dense1_norm2", norm_value)

            # if (self.longformer.classifier.lbd2[2].weight.grad is not None):
            #     norm_value = self.longformer.classifier.lbd2[2].weight.grad.detach().norm(2).item()
            #     # assert not np.isnan(norm_value)
            #     self.log("dense2_norm2", norm_value)

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

@rank_zero_only
def wandb_save(wandb_logger):
    wandb_logger.log_hyperparams(train_config)
    wandb_logger.experiment.save('./Beer_Transformer_Baseline.py', policy="now")

if __name__ == "__main__":
    train_config = {}
    train_config["cache_dir"] = "./cache/"
    train_config["epochs"] = 15
    train_config["batch_size"] = 4
    train_config["accumulate_grad_batches"] = 12
    train_config["gradient_clip_val"] = 1.0
    train_config["learning_rate"] = 2e-5

    pl.seed_everything(42)

    wandb_logger = WandbLogger(name='baseline_fromScrt',project='saam_beer_longformer')
    wandb_save(wandb_logger)

    model = LightningLongformerBaseline(train_config)
    # model = LightningLongformerBaseline.load_from_checkpoint("./saam_hotel_longformer/2q1t5ns9/checkpoints/epoch=10.ckpt", config=train_config)

    cp_valloss = ModelCheckpoint(
        # filepath=wandb.run.dir+'{epoch:02d}-{val_loss:.2f}',
                                save_top_k=5, monitor='val_loss', mode='min')
    cp_acc0 = ModelCheckpoint(
        # filepath=wandb.run.dir+'{epoch:02d}-{acc0:.2f}',
                                save_top_k=5, monitor='acc0', mode='max')
    cp_acc3 = ModelCheckpoint(
        # filepath=wandb.run.dir+'{epoch:02d}-{acc3:.2f}',
                                save_top_k=1, monitor='acc3', mode='max')

    trainer = pl.Trainer(max_epochs=train_config["epochs"],
                        accumulate_grad_batches=train_config["accumulate_grad_batches"],
                        gradient_clip_val=train_config["gradient_clip_val"],

                        gpus=[0,1,3],
                        num_nodes=1,
                        distributed_backend='ddp',

                        amp_backend='native',
                        precision=16,

                        logger=wandb_logger,
                        log_every_n_steps=1,

                        val_check_interval=0.5,
                        limit_val_batches=500,

                        checkpoint_callback=cp_acc0,
                        )

    trainer.fit(model)

    # model = LightningLongformerBaseline.load_from_checkpoint("./saam_hotel_longformer/2q1t5ns9/checkpoints/epoch=10.ckpt", config=train_config)
    # trainer.test(
    #                 model=model,
    #                 test_dataloaders=model.val_dataloader(),
    #                 )




