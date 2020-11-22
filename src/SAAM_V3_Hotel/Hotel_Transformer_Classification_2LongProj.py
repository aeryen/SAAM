
# %%
import os

print(os.getcwd())
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)
print(os.getcwd())

os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
# os.environ['CUDA_VISIBLE_DEVICES'] = "1,2"

# %%
from typing import Any, Optional
import copy 

import pandas as pd
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import transformers
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers.models.longformer.modeling_longformer import LongformerLayer, LongformerModel, LongformerPreTrainedModel

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
        return (self.df.iloc[idx,6], self.df.iloc[idx,0:6].to_numpy().astype(np.float) )


# %%
class LongformerLfmProjCLS(LongformerPreTrainedModel):

    authorized_unexpected_keys = [r"pooler"]

    def __init__(self, config, tkz):
        super().__init__(config)
        self.longformer = LongformerModel(config, add_pooling_layer=False)
        self.classifier = SAAMv3CLS(config, n_asp=6, n_rat=5)
        self.init_weights()
        self.tkz = tkz

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
            return_dict=True,
        )

        p_index = (input_ids == self.tkz.get_vocab()["xxPERIOD"])

        results = self.classifier(outputs, p_index)

        return results


class BnDropLin(nn.Sequential):
    "Module grouping `BatchNorm1d`, `Dropout` and `Linear` layers"
    def __init__(self, n_in, n_out, bn=True, p=0., act=None):
        layers = [nn.LayerNorm(n_in)] if bn else []
        if p != 0: layers.append(nn.Dropout(p))
        lin = [nn.Linear(n_in, n_out, bias=not bn)]
        if act is not None: lin.append(act)
        layers = layers+lin
        super().__init__(*layers)


class SAAMv3CLS(nn.Module):

    def __init__(self, config, n_asp:int, n_rat:int):
        super().__init__()
        print("CLS init")
        print("Num Aspect: "+str(n_asp) )
        print("Num Rating: "+str(n_rat) )
        self.config = config
        self.proj_config = copy.deepcopy(config)

        self.n_asp = n_asp
        self.n_rat = n_rat
        
        self.proj_dim = 768
        self.proj_config.attention_window = [32]
        self.proj_config.final_hidden_size = self.proj_dim
        
        self.aspect_projector = LongformerLayer(self.proj_config, layer_id=0)
        self.senti_projector = LongformerLayer(self.proj_config, layer_id=0)
        
        # aspect projector, with additional 1 aspect for throw out
        self.aspect = BnDropLin(n_in=self.proj_dim, n_out=self.n_asp+1, p=0.5, act=nn.Softmax(dim=1))
        # self.aspect = BnDropLin(n_in=self.proj_dim, n_out=self.n_asp+1, p=0.5, act=nn.Sigmoid())
        self.sentiments = nn.ModuleList( [BnDropLin(n_in=self.proj_dim, n_out=self.n_rat, p=0.5, act=None)] * self.n_asp )

    def average_emb(self, output, start, end):
        avg_pool = output[start:end, :].mean(dim=0)
        return avg_pool

    def sentence_avgpool(self, output, p_index):
        batch = []
        for doci in range(0,output.shape[0]):
            pi = p_index[doci,:].nonzero(as_tuple=True)[0].int()
            doc = []
            for senti in range( len(pi) ):
                if senti==0:
                    # from start of doc to end of first sent
                    doc.append( self.average_emb(output[doci,:,:], 0, pi[senti]) )
                else:
                    # from previous period to next
                    doc.append( self.average_emb(output[doci,:,:], pi[senti-1]+1, pi[senti]) )
                
            batch.append( torch.stack(doc, 0) )

        return batch

    def forward(self, long_outputs, p_index):

        outputs = long_outputs.orig_last_hidden_state
        extended_attention_mask = long_outputs.mask

        is_index_masked = extended_attention_mask  < 0
        is_index_global_attn = torch.zeros_like(extended_attention_mask).bool()
        is_global_attn = False

        aspect_projection = self.aspect_projector(outputs,
                                                    attention_mask=extended_attention_mask,
                                                    is_index_masked=is_index_masked,
                                                    is_index_global_attn=is_index_global_attn,
                                                    is_global_attn=is_global_attn,)
        senti_projection  = self.senti_projector(outputs,
                                                    attention_mask=extended_attention_mask,
                                                    is_index_masked=is_index_masked,
                                                    is_index_global_attn=is_index_global_attn,
                                                    is_global_attn=is_global_attn,)

        aspect_projection = aspect_projection[0]
        senti_projection = senti_projection[0]

        if long_outputs.padding_len > 0:
            aspect_projection = aspect_projection[:, :-long_outputs.padding_len]
            senti_projection = senti_projection[:, :-long_outputs.padding_len]
        
        aspect_batch = self.sentence_avgpool(aspect_projection, p_index)
        senti_batch  = self.sentence_avgpool(senti_projection, p_index)
        
        allsent_aspect_proj = torch.cat(aspect_batch, dim=0)        # [n_sentence, 256]
        allsent_senti_proj = torch.cat(senti_batch, dim=0)          # [n_sentence, 256]
        
        aspect_dist = self.aspect(allsent_aspect_proj)         # [n_sentence, aspect6]

        sent_bmm = torch.bmm(aspect_dist[:,0:self.n_asp].unsqueeze(2), allsent_senti_proj.unsqueeze(1))  # [319, 6, 256]
        
        all_doc_emb = []
        aspect_doc = []
        sentim_doc = []
        cur = 0
        for doci in range(0, len(aspect_batch)):
            sn = aspect_batch[doci].shape[0]
            doc_emb_avg = torch.sum(sent_bmm[cur:(cur+sn), :, : ], dim=0, keepdim=True) # [1, 6, 400]
            asp_w_sum = torch.sum(aspect_dist[cur:(cur+sn),0:self.n_asp], dim=0, keepdim=True)     # [1, 6]
            doc_emb_avg = doc_emb_avg / asp_w_sum[:,:,None]                             # [1, 6, 400]
            all_doc_emb.append( doc_emb_avg )
            aspect_doc.append( aspect_dist[cur:(cur+sn), :] )
            
            cur = cur + sn

        all_doc_emb = torch.cat( all_doc_emb, dim=0 )          # [batch, asp, 400]
        
        result_senti = [ self.sentiments[aspi]( all_doc_emb[:,aspi,:] ) for aspi in range(0,self.n_asp)] # [batch, ra]
        
        result = torch.stack(result_senti, dim=1)  # [batch, asp, sentiment5]
        
        return result,outputs,aspect_doc


# %%
class TokenizerCollate:
    def __init__(self):
        self.tkz = AutoTokenizer.from_pretrained("allenai/longformer-base-4096")
        self.tkz.add_tokens("xxPERIOD")
    
    def __call__(self, batch):
        batch_split = list(zip(*batch))
        seqs, targs= batch_split[0], batch_split[1]
        encode = self.tkz(list(seqs), padding="longest")
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
        self.tokenCollate = TokenizerCollate()
        self.longformer = LongformerLfmProjCLS.from_pretrained('allenai/longformer-base-4096',
                                                             cache_dir=self.train_config["cache_dir"],
                                                            #  gradient_checkpointing=True,
                                                            tkz=self.tokenCollate.tkz
                                                               )
        self.longformer.longformer.resize_token_embeddings(len(self.tokenCollate.tkz))

        # for param in self.longformer.longformer.embeddings.parameters():
        #     param.requires_grad = False
        # for param in self.longformer.longformer.encoder.parameters():
        #     param.requires_grad = False

        self.lossfunc = MultiLabelCEL()
        self.metrics = torch.nn.ModuleList( [AspectACC(aspect=i) for i in range(6)] )

    def configure_optimizers(self):
        # optimizer = torch.optim.AdamW(self.parameters(), lr=self.train_config["learning_rate"])
        optimizer = transformers.AdamW(model.parameters(), lr=train_config["learning_rate"], weight_decay=0.01)
        scheduler = transformers.get_cosine_with_hard_restarts_schedule_with_warmup(optimizer,
                                                                                    num_warmup_steps=500,
                                                                                    num_training_steps=4000,
                                                                                    num_cycles=1)
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
                                        collate_fn=self.tokenCollate,
                                        num_workers=2,
                                        pin_memory=True, drop_last=False, shuffle=True)
        return self.loader_train

    def val_dataloader(self):
        self.dataset_val = ReviewDataset("../../data/hotel_balance_LengthFix1_3000per/df_test.pickle")
        self.loader_val = DataLoader(self.dataset_val,
                                        batch_size=train_config["batch_size"],
                                        collate_fn=self.tokenCollate,
                                        num_workers=2,
                                        pin_memory=True, drop_last=False, shuffle=True)
        return self.loader_val
    
#     @autocast()
    def forward(self, input_ids, attention_mask, labels):
        logits,outputs,aspect_doc = self.longformer(input_ids=input_ids, attention_mask=attention_mask)
        loss = self.lossfunc(logits, labels)

        return (loss, logits, outputs, aspect_doc)
    
    def training_step(self, batch, batch_idx):
        input_ids, mask, label  = batch[0].type(torch.int64), batch[1].type(torch.int64), batch[2].type(torch.int64)
        
        loss, logits, outputs, aspect_doc = self(input_ids=input_ids, attention_mask=mask, labels=label)
        
        self.log("train_loss", loss)
        
        return loss

    def on_after_backward(self):
        # if (self.trainer.batch_idx + 1) % self.trainer.accumulate_grad_batches == 0:
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

            if (self.longformer.classifier.aspect_projector.output.dense.weight.grad is not None):
                norm_value = self.longformer.classifier.aspect_projector.output.dense.weight.grad.detach().norm(2).item()
                # assert not np.isnan(norm_value)
                self.log("NORMS/aspect_projector", norm_value)

            if (self.longformer.classifier.senti_projector.output.dense.weight.grad is not None):
                norm_value = self.longformer.classifier.senti_projector.output.dense.weight.grad.detach().norm(2).item()
                # assert not np.isnan(norm_value)
                self.log("NORMS/senti_projector", norm_value)

            if (self.longformer.classifier.aspect[2].weight.grad is not None):
                norm_value = self.longformer.classifier.aspect[2].weight.grad.detach().norm(2).item()
                # assert not np.isnan(norm_value)
                self.log("NORMS/aspect", norm_value)

            if (self.longformer.classifier.sentiments[0][2].weight.grad is not None):
                norm_value = self.longformer.classifier.sentiments[0][2].weight.grad.detach().norm(2).item()
                # assert not np.isnan(norm_value)
                self.log("NORMS/sentiments", norm_value)

    def validation_step(self, batch, batch_idx):
        input_ids, mask, label  = batch[0].type(torch.int64), batch[1].type(torch.int64), batch[2].type(torch.int64)
        
        loss, logits, outputs, aspect_doc = self(input_ids=input_ids, attention_mask=mask, labels=label)
        
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
        
        loss, logits, outputs, aspect_doc = self(input_ids=input_ids, attention_mask=mask, labels=label)
        accs = [m(logits, label) for m in self.metrics]  # update metric counters
        
        return loss

    def on_test_epoch_end(self):
        for i,m in enumerate(self.metrics):
            print('acc'+str(i), m.compute())

@rank_zero_only
def wandb_save(wandb_logger):
    wandb_logger.log_hyperparams(train_config)
    wandb_logger.experiment.save('./Hotel_Transformer_Classification_2LongProj.py', policy="now")

if __name__ == "__main__":
    train_config = {}
    train_config["cache_dir"] = "./cache/"
    train_config["epochs"] = 15
    train_config["batch_size"] = 4
    train_config["accumulate_grad_batches"] = 10
    train_config["gradient_clip_val"] = 1.0
    train_config["learning_rate"] = 3e-5

    pl.seed_everything(42)

    wandb_logger = WandbLogger(name='clasV3_longProj_aspSigm',project='saam_hotel_longformer')
    wandb_save(wandb_logger)

    model = LightningLongformerBaseline(train_config)

    model = LightningLongformerBaseline.load_from_checkpoint(
                                "./saam_hotel_longformer/14odv0tz/checkpoints/epoch=8-step=1598.ckpt",
                                config=train_config)

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

                        # gpus=[0,1,3],
                        gpus=[0],
                        num_nodes=1,
                        # distributed_backend='ddp',

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




