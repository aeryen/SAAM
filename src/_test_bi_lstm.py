# %%
from fastai.text import *
from data_helpers.Data import *
from fastai.text.transform import *

#%%
hyper_params = {
    "max_sequence_length": 20*70,
    "batch_size": 32,
    "num_epochs1": 12,
    "num_epochs2": 15,
    "num_aspect": 5,
    "num_rating": 5,
}

#%%
cls_db = load_data("./data/", "beer_clas_databunch_rint.TraValTes")
cls_db.batch_size = hyper_params["batch_size"]
cls_db.batch_size

#%%
def sentence_pool_400(output:Tensor, mask, p_index):
    batch = []
    for doci in range(0, output.shape[0]):
        doc = output[doci, p_index[doci, :], :]
        batch.append(doc)

    return batch

# %%
class SentenceEncoder(Module):
    "Create an encoder over `module` that can process a full sentence."
    
    def __init__(self, bptt:int, max_len:int, module:nn.Module, vocab, pad_idx:int=1):
        print("Encoder initing")
        self.max_len,self.bptt,self.module,self.pad_idx = max_len,bptt,module,pad_idx
        print("max len " + str(self.max_len))
        print("bptt " + str(self.bptt))
        print("pad_idx " + str(self.pad_idx))
        self.vocab = vocab
        self.period_index = self.vocab.stoi["xxperiod"]
        print("period index " + str(self.period_index))

    def concat(self, arrs:Collection[Tensor])->Tensor:
        "Concatenate the `arrs` along the batch dimension."
        return [torch.cat([l[si] for l in arrs], dim=1) for si in range_of(arrs[0])]

    def reset(self):
        if hasattr(self.module, 'reset'): self.module.reset()

    def forward(self, input:LongTensor)->Tuple[Tensor,Tensor]:
        bs,sl = input.size()
        self.reset()
        raw_outputs,outputs,masks = [],[],[]
        p_index = []
        for i in range(0, sl, self.bptt):
            r, o = self.module(input[:,i: min(i+self.bptt, sl)])
            if i>(sl-self.max_len):
                masks.append(input[:,i: min(i+self.bptt, sl)] == self.pad_idx)
                raw_outputs.append(r)
                outputs.append(o)
                p_index.append( input[:,i: min(i+self.bptt, sl)] == self.period_index )

#         print("number of sentences in docs:")
#         n_sent = torch.sum( x==self.vocab.stoi["xxperiod"] , dim=1)
#         print(n_sent)
        
        period_index = torch.cat(p_index,dim=1)
        
        return self.concat(raw_outputs),self.concat(outputs), \
               torch.cat(masks,dim=1),period_index

# %%
class BI_AWD_LSTM(AWD_LSTM):
    def __init__(self, vocab_sz:int, emb_sz:int, n_hid:int, pad_token:int=1, hidden_p:float=0.2,
                 input_p:float=0.6, embed_p:float=0.1, weight_p:float=0.5, bidir:bool=False):
        self.bs,self.emb_sz,self.n_hid = 1,emb_sz,n_hid
        self.n_dir = 2 if bidir else 1
        self.encoder = nn.Embedding(vocab_sz, emb_sz, padding_idx=pad_token)
        self.encoder_dp = EmbeddingDropout(self.encoder, embed_p)
 
        self.rnns = [
            nn.LSTM(emb_sz, n_hid//self.n_dir, 1,
                 batch_first=True, bidirectional=bidir),
            nn.LSTM(n_hid, n_hid//self.n_dir, 1,
                 batch_first=True, bidirectional=bidir)
        ]
        
        self.rnns.extend( [ nn.LSTM(n_hid, emb_sz//self.n_dir, 1,
                                    batch_first=True, bidirectional=bidir) ] * 2 )
        
        for i in range(len(self.rnns)):
            self.rnns[i] = WeightDropout(self.rnns[i], weight_p)
        
        self.rnns = nn.ModuleList(self.rnns)
        self.encoder.weight.data.uniform_(-self.initrange, self.initrange)
        if self.encoder.padding_idx is not None:
            self.encoder.weight.data[self.encoder.padding_idx] = 0.
        self.input_dp = RNNDropout(input_p)
        self.hidden_dps = nn.ModuleList([RNNDropout(hidden_p) for l in range(len(self.rnns))])
        
    def forward(self, input:Tensor, from_embeddings:bool=False)->Tuple[List[Tensor],List[Tensor]]:
        if from_embeddings:
            bs,sl,es = input.size()  #  batchsize, seqlen, embsize
        else:
            bs,sl = input.size()     #  batchsize, seqlen
        if bs!=self.bs:
            self.bs=bs
            self.reset()
        raw_output = self.input_dp(input if from_embeddings else self.encoder_dp(input))
        
        new_hidden,raw_outputs,outputs = [],[],[]
        for l, (rnn,hid_dp) in enumerate(zip(self.rnns[:-2], self.hidden_dps[:-2])):  #  go through rnn and its dp
            raw_output, new_h = rnn(raw_output, self.hidden[l])  #  use previous hidden state
            new_hidden.append(new_h)                    #  store hidden for next batch
            raw_outputs.append(raw_output)              #  raw_outputs = lstm out before drop
            raw_output = hid_dp(raw_output)             #  drop outputs for next layer use
            outputs.append(raw_output)                  #  outputs = after drop
        
        raw_output_1, hidden_1 = self.rnns[-2](raw_output, self.hidden[-2])  #  work on dropped 2nd layer
        new_hidden.append(hidden_1)
        raw_outputs.append(raw_output_1)
        outputs.append(raw_output_1)
        
        raw_output_2, hidden_2 = self.rnns[-1](raw_output, self.hidden[-1])  #  work on dropped 2nd layer
        new_hidden.append(hidden_2)
        raw_outputs.append(raw_output_2)
        outputs.append(raw_output_2)
        
        self.hidden = to_detach(new_hidden, cpu=False)  #  store state for stateful lstm
        return raw_outputs, outputs
    
    def _one_hidden(self, l:int) -> Tensor:
        "Return one hidden state."
        nh_list = [self.n_hid, self.n_hid, self.emb_sz, self.emb_sz]  #  1152, 1152, 400, 400
        nh = nh_list[l] // self.n_dir
        return one_param(self).new(self.n_dir, self.bs, nh).zero_()

    def select_hidden(self, idxs):
        self.hidden = [(h[0][:,idxs,:],h[1][:,idxs,:]) for h in self.hidden]
        self.bs = len(idxs)

    def reset(self):
        "Reset the hidden states."
        [r.reset() for r in self.rnns if hasattr(r, 'reset')]
        #  (torch.Size([1, 32, 1152]), torch.Size([1, 32, 1152]))
        self.hidden = [(self._one_hidden(l), self._one_hidden(l)) for l in range( len(self.rnns) )]

# %%
# LSTM ATTENTION
class Cls02ATT_BILSTM(Module):
    "Create a linear classifier with pooling."
    def __init__(self, n_asp:int, n_rat:int):
        print("CLS init")
        print("Num Aspect: "+str(n_asp) )
        print("Num Rating: "+str(n_rat) )
        self.n_asp = n_asp + 1
        self.n_rat = n_rat
        
        self.asp_hidden = 100
        mod_layers = []
        mod_layers += bn_drop_lin( 400, self.asp_hidden, p=0.5, actn=nn.ReLU(inplace=True) )
        mod_layers += bn_drop_lin( self.asp_hidden, self.n_asp, p=0.1, actn=nn.Sigmoid() ) # actn=torch.nn.ReLU(dim=1)
        self.aspect = nn.Sequential(*mod_layers)
        
        self.smt_hidden = 20
        self.s0 = nn.Sequential(* (bn_drop_lin( 400, self.smt_hidden, p=0.5, actn=nn.ReLU(inplace=True) ) + 
                                   bn_drop_lin( self.smt_hidden, self.n_rat, p=0.1, actn=None ) ) )
        self.s1 = nn.Sequential(* (bn_drop_lin( 400, self.smt_hidden, p=0.5, actn=nn.ReLU(inplace=True) ) + 
                                   bn_drop_lin( self.smt_hidden, self.n_rat, p=0.1, actn=None ) ) )
        self.s2 = nn.Sequential(* (bn_drop_lin( 400, self.smt_hidden, p=0.5, actn=nn.ReLU(inplace=True) ) + 
                                   bn_drop_lin( self.smt_hidden, self.n_rat, p=0.1, actn=None ) ) )
        self.s3 = nn.Sequential(* (bn_drop_lin( 400, self.smt_hidden, p=0.5, actn=nn.ReLU(inplace=True) ) + 
                                   bn_drop_lin( self.smt_hidden, self.n_rat, p=0.1, actn=None ) ) )
        self.s4 = nn.Sequential(* (bn_drop_lin( 400, self.smt_hidden, p=0.5, actn=nn.ReLU(inplace=True) ) + 
                                   bn_drop_lin( self.smt_hidden, self.n_rat, p=0.1, actn=None ) ) )
        self.s5 = nn.Sequential(* (bn_drop_lin( 400, self.smt_hidden, p=0.5, actn=nn.ReLU(inplace=True) ) + 
                                   bn_drop_lin( self.smt_hidden, self.n_rat, p=0.1, actn=None ) ) )

        self.sentiments = []
        self.sentiments.append( self.s0 )
        self.sentiments.append( self.s1 )
        self.sentiments.append( self.s2 )
        self.sentiments.append( self.s3 )
        self.sentiments.append( self.s4 )
        self.sentiments.append( self.s5 )

    def forward(self, input:Tuple[Tensor,Tensor,Tensor,Tensor])->Tuple[Tensor,Tensor,Tensor]:
        raw_outputs,outputs,mask,p_index = input
        
        batch_sent_emb_asp = sentence_pool_400(outputs[3], mask, p_index)  #  list of size batch, each [n_sent, emb]
        batch_sent_emb_smt = sentence_pool_400(outputs[2], mask, p_index)  #  list of size batch, each [n_sent, emb]
        
        sent_emb_asp = torch.cat(batch_sent_emb_asp, dim=0)          # aspects [n_sentence, emb400]
        sent_dist_asp = self.aspect(sent_emb_asp)                    # [n_sentence, aspect6]
        
        sent_emb_smt = torch.cat(batch_sent_emb_smt, dim=0)          # sentiments [n_sentence, emb400]

        sent_bmm = torch.bmm(sent_dist_asp.unsqueeze(2), sent_emb_smt.unsqueeze(1))  # [n_sentence, asp, emb400]
        
        all_doc_emb = []
        aspect_doc = []
        sentim_doc = []
        cur = 0
        for doci in range(0, len(batch_sent_emb_asp)):
            sn = batch_sent_emb_asp[doci].shape[0]                                       #  number of sent in this doc
            doc_emb_avg = torch.sum(sent_bmm[cur:(cur+sn), :, : ], dim=0, keepdim=True)  #  [1, 7, 400]
            asp_w_sum = torch.sum(sent_dist_asp[cur:(cur+sn),:], dim=0, keepdim=True)      #  [1, 7]
            doc_emb_avg = doc_emb_avg / asp_w_sum[:,:,None]                              #  [1, 7, 400]
            all_doc_emb.append( doc_emb_avg )
            aspect_doc.append( sent_dist_asp[cur:(cur+sn), :] )
            
            cur = cur + sn

        all_doc_emb = torch.cat( all_doc_emb, dim=0 )          # [batch, asp, 400]
        
        result_senti = [ self.sentiments[aspi]( all_doc_emb[:,aspi,:] ) for aspi in range(0,self.n_asp) ] # [batch, ra]
        
        result = torch.stack(result_senti, dim=1)  # [batch, asp, sentiment5]
        
        return result,raw_outputs,outputs,aspect_doc

# %%
def get_model(vocab_sz:int, vocab, n_class:int, bptt:int=70, max_len:int=20*70, config:dict=None,
                        drop_mult:float=1., lin_ftrs:Collection[int]=None, ps:Collection[float]=None,
                        pad_idx:int=1) -> nn.Module:
    print("Creating Custom Model")
    meta = text.learner._model_meta[AWD_LSTM]
    # if we specified config then we dont use default
    config = ifnone(config, meta['config_clas']).copy()
    config.pop("output_p")
    config.pop("qrnn")
    config.pop("n_layers")
    print(config)
    # Drop multiplier
    for k in config.keys():
        if k.endswith('_p'): config[k] *= drop_mult
    init = config.pop('init') if 'init' in config else None
    
#     encoder = SentenceEncoder(bptt, max_len, AWD_LSTM(vocab_sz, **config), vocab, pad_idx=pad_idx)
#     cls_layer = Cls02ATT400(n_asp=hyper_params["num_aspect"], n_rat=hyper_params["num_rating"])
    
    encoder = SentenceEncoder(bptt, max_len, BI_AWD_LSTM(vocab_sz, **config), vocab, pad_idx=pad_idx)
    cls_layer = Cls02ATT_BILSTM(n_asp=hyper_params["num_aspect"], n_rat=hyper_params["num_rating"])

    model = SequentialRNN(encoder, cls_layer)
    return model if init is None else model.apply(init)

def load_pretrained(learn, wgts_fname:str, itos_fname:str, strict:bool=True):
    "Load a pretrained model and adapts it to the data vocabulary."
    old_itos = pickle.load(open(itos_fname, 'rb'))
    old_stoi = {v:k for k,v in enumerate(old_itos)}
    wgts = torch.load(wgts_fname, map_location=lambda storage, loc: storage)
    if 'model' in wgts: wgts = wgts['model']
    wgts = convert_weights(wgts, old_stoi, learn.data.train_ds.vocab.itos)
    
    wkeys = list( wgts.keys() )                                           #  for BI LSTM
    for wkey in wkeys:
        if wkey.startswith("0.rnns.2"):
            wgts["0.rnns.3"+wkey[len("0.rnns.3"):]] = wgts[wkey].clone()  #  for BI LSTM

    print("Loading Pre_Trained")
    learn.model.load_state_dict(wgts, strict=strict)
    return learn

def text_classifier_learner(data:DataBunch, bptt:int=70, max_len:int=20*70, config:dict=None,
                            pretrained:bool=True, drop_mult:float=1., lin_ftrs:Collection[int]=None,
                            ps:Collection[float]=None, **learn_kwargs) -> 'TextClassifierLearner':
    "Create a `Learner` with a text classifier from `data` and `arch`."
    model = get_model(len(data.vocab.itos), data.vocab, data.c, bptt=bptt, max_len=max_len,
                                config=config, drop_mult=drop_mult, lin_ftrs=lin_ftrs, ps=ps)
    meta = text.learner._model_meta[AWD_LSTM]
    learn = RNNLearner(data, model, split_func=meta['split_clas'], **learn_kwargs)
    if pretrained:
        if 'url' not in meta:
            warn("There are no pretrained weights for that architecture yet!")
            return learn
        model_path = untar_data(meta['url'], data=False)
        fnames = [list(model_path.glob(f'*.{ext}'))[0] for ext in ['pth', 'pkl']]
        learn = load_pretrained(learn, *fnames, strict=False)
        learn.freeze()
    return learn

# %%
class MultiLabelCEL(nn.CrossEntropyLoss):
    def forward(self, input, target, nasp=5):
        target = target.long()
        loss = 0
        
        for i in range(nasp):
            loss = loss + super(MultiLabelCEL, self).forward(input[:,i,:], target[:,i])
        
        return loss

def multi_acc(preds, targs, nasp=hyper_params["num_aspect"], nrat=5):
    preds = preds[:,0:nasp,:]
    preds = preds.contiguous().view(-1, nrat)
    preds = torch.max(preds, dim=1)[1]
    targs = targs.contiguous().view(-1).long()
    return (preds==targs).float().mean()

def get_clas_acc(asp_index):
    def asp_acc(preds, targs):
        preds = torch.max(preds, dim=2)[1]
        targs = targs.contiguous().long()
        return (preds[:,asp_index]==targs[:,asp_index]).float().mean()
    return asp_acc

def get_clas_mse(asp_index):
    def asp_mse(preds, targs):
        preds = torch.max(preds, dim=2)[1].float()[:,asp_index]
        targs = targs.contiguous().float()[:,asp_index]
        return torch.nn.functional.mse_loss(preds, targs)
    return asp_mse

# %%
macc = [get_clas_acc(ai) for ai in range(hyper_params["num_aspect"])]
for ai in range(hyper_params["num_aspect"]):
    macc[ai].__name__ = "clas_acc_"+str(ai)
mmse = [get_clas_mse(ai) for ai in range(hyper_params["num_aspect"])]
for ai in range(hyper_params["num_aspect"]):
    mmse[ai].__name__ = "clas_mse_"+str(ai)

# %%
mloss = MultiLabelCEL()
cls_learn = text_classifier_learner(cls_db,
                                    drop_mult=1.1,
                                    loss_func=mloss,
                                    metrics=[multi_acc]+macc+mmse,
                                    bptt=70,
                                    max_len=hyper_params["max_sequence_length"])

# %%
x, y = cls_db.one_batch()
print(x.shape)
print(y.shape)
x = x.cuda()
y = y.cuda()

# %%
result,raw_outputs,outputs,aspect_doc = cls_learn.model(x)

# %%
result.shape

# %%
