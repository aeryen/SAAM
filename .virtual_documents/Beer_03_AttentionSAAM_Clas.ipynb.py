get_ipython().run_line_magic("reload_ext", " autoreload")
get_ipython().run_line_magic("autoreload", " 2")
get_ipython().run_line_magic("matplotlib", " inline")


import comet_ml
experiment = comet_ml.Experiment(project_name="2019nn_beer")


# get_ipython().run_line_magic("%", "")
from fastai.text import *
from data_helpers.Data import *
from fastai.text.transform import *


hyper_params = {
    "max_sequence_length": 20*70,
    "batch_size": 32,
    "num_epochs1": 12,
    "num_epochs2": 15,
    "num_aspect": 5,
    "num_rating": 5,
}


experiment.log_parameters(hyper_params)


torch.cuda.set_device('cuda:0')


# lm_db = load_data("./data/", "hotel_lm_databunch.1001")
# lm_learn = language_model_learner(lm_db, AWD_LSTM)
# lm_learn = lm_learn.load("lang_model_hotel")


# lm_learn.save_encoder('lang_model_hotel_enc')


cls_db = load_data("./data/", "beer_clas_databunch_rint.TraValTes")
cls_db.batch_size = hyper_params["batch_size"]
cls_db.batch_size


cls_db.show_batch()


x,y = cls_db.one_batch()


x.shape


def pool_combo(output, start, end):
    avg_pool = output[start:end, :].mean(dim=0)
    max_pool = output[start:end, :].max(dim=0)[0]
    x = torch.cat([output[-1, :], max_pool, avg_pool], 0)
    return x


def sentence_pool_1200(outputs, mask, p_index):
    output = outputs[-1]
    seq_max = output.size(1)
    doc_start = mask.int().sum(dim=1)

    batch = []
    for doci in range(0, output.shape[0]):
        pi = p_index[doci, :].nonzero(as_tuple=True)[0].int()
        doc = []
        for senti in range(len(pi)):
            if senti == 0:
                doc.append(pool_combo(output[doci, :, :], doc_start[doci], pi[senti]))
            else:
                doc.append(pool_combo(output[doci, :, :], pi[senti - 1] + 1, pi[senti]))

        batch.append(torch.stack(doc, 0))

    return batch


def sentence_pool_400(output:Tensor, mask, p_index):
    batch = []
    for doci in range(0, output.shape[0]):
        doc = output[doci, p_index[doci, :], :]
        batch.append(doc)

    return batch


def masked_concat_pool(outputs, mask):
    "Pool MultiBatchEncoder outputs into one vector [last_hidden, max_pool, avg_pool]."
    output = outputs[-1]
    avg_pool = output.masked_fill(mask[:, :, None], 0).mean(dim=1)
    avg_pool *= (
        output.size(1) / (output.size(1) - mask.type(avg_pool.dtype).sum(dim=1))[:, None]
    )
    max_pool = output.masked_fill(mask[:, :, None], -float("inf")).max(dim=1)[0]
    x = torch.cat([output[:, -1], max_pool, avg_pool], 1)
    return x


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
        
        bptt = self.bptt
        if np.random.random() > 0.95:
            bptt = 50
        
        raw_outputs,outputs,masks = [],[],[]
        p_index = []
        for i in range(0, sl, bptt):
            r, o = self.module(input[:,i: min(i+bptt, sl)])
            if i>(sl-self.max_len):
                masks.append(input[:,i: min(i+bptt, sl)] == self.pad_idx)
                raw_outputs.append(r)
                outputs.append(o)
                p_index.append( input[:,i: min(i+bptt, sl)] == self.period_index )

#         print("number of sentences in docs:")
#         n_sent = torch.sum( x==self.vocab.stoi["xxperiod"] , dim=1)
#         print(n_sent)
        
        period_index = torch.cat(p_index,dim=1)
        
        return self.concat(raw_outputs),self.concat(outputs), \
               torch.cat(masks,dim=1),period_index


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
        if bsget_ipython().getoutput("=self.bs:")
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


meta = text.learner._model_meta[AWD_LSTM].copy()
config = meta["config_clas"].copy()
print(config)
config.pop("output_p")
config.pop("qrnn")
config.pop("n_layers")
print(config)


vocab_sz = len(cls_db.vocab.itos)
m = BI_AWD_LSTM(vocab_sz, **config)
m


x, y = cls_db.one_batch()
print(x.shape)
print(y.shape)


raw_outputs, outputs = m(x)


outputs[2].shape, outputs[3].shape


encoder = SentenceEncoder(70, hyper_params["max_sequence_length"], m, cls_db.vocab, pad_idx=1)


raw_outputs,outputs,mask,p_index = encoder( x )


for o in outputs:
    print(o.shape)


# experiment.add_tag("CLAS02")
# experiment.add_tag("FULLIND400")

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
            asp_w_sum = torch.sum(sent_dist_asp[cur:(cur+sn),:], dim=0, keepdim=True)    #  [1, 7]
            doc_emb_avg = doc_emb_avg / asp_w_sum[:,:,None]                              #  [1, 7, 400]
            all_doc_emb.append( doc_emb_avg )
            aspect_doc.append( sent_dist_asp[cur:(cur+sn), :] )
            
            cur = cur + sn

        all_doc_emb = torch.cat( all_doc_emb, dim=0 )          # [batch, asp, 400]
        
        result_senti = [ self.sentiments[aspi]( all_doc_emb[:,aspi,:] ) for aspi in range(0,self.n_asp) ]  #  [batch, ra]
        
        result = torch.stack(result_senti, dim=1)  # [batch, asp, sentiment5]
        
        return result,raw_outputs,outputs,aspect_doc


experiment.add_tag("CLAS02")
experiment.add_tag("LIATLI")

# ATTENTIONAL AVERAGING, COMPLETELY INDEPENDENT SENTI OUT
class Cls02_LIATLI(Module):
    "Create a linear classifier with pooling."
    def __init__(self, n_asp:int, n_rat:int):
        print("CLS init")
        print("Num Aspect: "+str(n_asp) )
        print("Num Rating: "+str(n_rat) )
        self.n_asp = n_asp + 1
        self.n_rat = n_rat
        
        self.doc_mapper = nn.Sequential(*mod_layers)
        
        self.asp_hidden = 40
        mod_layers = []
        mod_layers += bn_drop_lin( 800, self.asp_hidden, p=0.5, actn=nn.LeakyReLU(inplace=True) )  #  inplace=True
        mod_layers += bn_drop_lin( self.asp_hidden, self.n_asp, p=0.15, actn=nn.Sigmoid() )  #  actn=nn.Softmax(dim=1)
        self.aspect = nn.Sequential(*mod_layers)
        
        self.smt_hidden = 300
        self.first_fn = nn.Sequential( * ( bn_drop_lin( 400, self.smt_hidden * self.n_asp, p=0.5,
                                                       actn=nn.LeakyReLU(inplace=True) ) ) )
#         self.second_fn = nn.ModuleList(
#             [ nn.Sequential( * ( bn_drop_lin( self.smt_hidden, self.n_rat, p=0.3, actn=None) ) )
#              for i in range(self.n_asp) ]
#         )
        self.second_fn = nn.Sequential(
            * ( bn_drop_lin( self.smt_hidden, 80, p=0.4, actn=nn.LeakyReLU(inplace=True)) +
                bn_drop_lin( 80, self.n_rat, p=0.2, actn=None) )
        )

    def forward(self, input:Tuple[Tensor,Tensor,Tensor,Tensor]) -> Tuple[Tensor,Tensor,Tensor]:
        raw_outputs,outputs,mask,p_index = input
        
        batch = sentence_pool_400(outputs[-1], mask, p_index)  #  list of size batch, each [n_sent, emb]
        
        temp1 = batch[0].mean(dim=0)
        temp1 = temp1.unsqueeze(0)
        temp1 = temp1.expand(10,-1)
        
        doc_emb = [doc.mean(dim=0).unsqueeze(0).expand(doc.shape[0],-1) for doc in batch]
        doc_emb = torch.cat(doc_emb, dim=0)
        
        allsent_emb = torch.cat(batch, dim=0)          #  [n_sentence, emb400]
        asp_emb = torch.cat([allsent_emb,doc_emb], dim=1)
        aspect_dist = self.aspect(asp_emb)         #  [n_sentence, aspect6]
        sentim_dist = self.first_fn(allsent_emb)
        sentim_dist = sentim_dist.view(-1, self.n_asp, self.smt_hidden)  #  [n_sentence, aspect6, emb100]
        
        sent_bmm = sentim_dist * aspect_dist.unsqueeze(2)                #  [n_sentence, asp, emb100]
        
        all_doc_emb = []
        aspect_doc = []
        sentim_doc = []
        cur = 0
        for doci in range(0, len(batch)):
            sn = batch[doci].shape[0]
            doc_emb_avg = torch.sum(sent_bmm[cur:(cur+sn), :, : ], dim=0, keepdim=True)  #  [1, 7, 400]
            asp_w_sum = torch.sum(aspect_dist[cur:(cur+sn),:], dim=0, keepdim=True)      # [1, 7]
            doc_emb_avg = doc_emb_avg / asp_w_sum[:,:,None]                                  # [1, 7, 400]
#             doc_emb_max = torch.max(sent_bmm[cur:(cur+sn), :, : ], dim=0, keepdim=True)[0] # [1, 7, 400]
#             all_doc_emb.append( torch.cat( [doc_emb_avg, doc_emb_max], dim=2 ) )
            all_doc_emb.append( doc_emb_avg )
            aspect_doc.append( aspect_dist[cur:(cur+sn), :] )
            
            cur = cur + sn

        all_doc_emb = torch.cat( all_doc_emb, dim=0 )          # [batch, asp, 100]
        
#         all_doc_emb = nn.functional.elu_( all_doc_emb )
        
#         result_senti = [ self.second_fn[aspi]( all_doc_emb[:,aspi,:] ) for aspi in range(0,self.n_asp)] # [batch, ra]
        result_senti = [ self.second_fn( all_doc_emb[:,aspi,:] ) for aspi in range(0,self.n_asp)] # [batch, ra]
        
        result = torch.stack(result_senti, dim=1)  # [batch, asp, sentiment5]
        
        return result,raw_outputs,outputs,aspect_doc


# experiment.add_tag("CLAS02")
# experiment.add_tag("FULLIND400")

# ATTENTIONAL AVERAGING, COMPLETELY INDEPENDENT SENTI OUT

class Cls02ATT400(Module):
    "Create a linear classifier with pooling."
    def __init__(self, n_asp:int, n_rat:int):
        print("CLS init")
        print("Num Aspect: "+str(n_asp) )
        print("Num Rating: "+str(n_rat) )
        self.n_asp = n_asp + 1
        self.n_rat = n_rat
        
        self.asp_hidden = 40
        mod_layers = []
        mod_layers += bn_drop_lin( 400, self.asp_hidden, p=0.5, actn=nn.Tanh() )  #  inplace=True
        mod_layers += bn_drop_lin( self.asp_hidden, self.n_asp, p=0.15, actn=nn.Softmax(dim=1) )  #  actn=torch.nn.ReLU(dim=1)
        self.aspect = nn.Sequential(*mod_layers)
        
        self.sent_hidden = 20
#         self.senti_base = nn.Sequential(*bn_drop_lin( 400, 50, p=0.5, actn=nn.ReLU(inplace=True) ) )
        self.s0 = nn.Sequential(* (bn_drop_lin( 400, self.sent_hidden, p=0.5, actn=nn.ReLU(inplace=True) ) +
                                   bn_drop_lin( self.sent_hidden, self.n_rat, p=0.1, actn=None ) ) )
        self.s1 = nn.Sequential(* (bn_drop_lin( 400, self.sent_hidden, p=0.5, actn=nn.ReLU(inplace=True) ) +
                                   bn_drop_lin( self.sent_hidden, self.n_rat, p=0.1, actn=None ) ) )
        self.s2 = nn.Sequential(* (bn_drop_lin( 400, self.sent_hidden, p=0.5, actn=nn.ReLU(inplace=True) ) +
                                   bn_drop_lin( self.sent_hidden, self.n_rat, p=0.1, actn=None ) ) )
        self.s3 = nn.Sequential(* (bn_drop_lin( 400, self.sent_hidden, p=0.5, actn=nn.ReLU(inplace=True) ) +
                                   bn_drop_lin( self.sent_hidden, self.n_rat, p=0.1, actn=None ) ) )
        self.s4 = nn.Sequential(* (bn_drop_lin( 400, self.sent_hidden, p=0.5, actn=nn.ReLU(inplace=True) ) +
                                   bn_drop_lin( self.sent_hidden, self.n_rat, p=0.1, actn=None ) ) )
        self.s5 = nn.Sequential(* (bn_drop_lin( 400, self.sent_hidden, p=0.5, actn=nn.ReLU(inplace=True) ) +
                                   bn_drop_lin( self.sent_hidden, self.n_rat, p=0.1, actn=None ) ) )

        self.sentiments = []
        self.sentiments.append( self.s0 )
        self.sentiments.append( self.s1 )
        self.sentiments.append( self.s2 )
        self.sentiments.append( self.s3 )
        self.sentiments.append( self.s4 )
        self.sentiments.append( self.s5 )

    def forward(self, input:Tuple[Tensor,Tensor,Tensor,Tensor])->Tuple[Tensor,Tensor,Tensor]:
        raw_outputs,outputs,mask,p_index = input
        
        batch = sentence_pool_400(outputs[-1], mask, p_index)
        
        allsent_emb = torch.cat(batch, dim=0)          # [n_sentence, emb400]
        aspect_dist = self.aspect(allsent_emb)         # [n_sentence, aspect6]

        sent_bmm = torch.bmm(aspect_dist.unsqueeze(2), allsent_emb.unsqueeze(1))  # [319, 7, 400]
        
        all_doc_emb = []
        aspect_doc = []
        sentim_doc = []
        cur = 0
        for doci in range(0, len(batch)):
            sn = batch[doci].shape[0]
            doc_emb_avg = torch.sum(sent_bmm[cur:(cur+sn), :, : ], dim=0, keepdim=True)  #  [1, 7, 400]
            asp_w_sum = torch.sum(aspect_dist[cur:(cur+sn),:], dim=0, keepdim=True) # [1, 7]
            doc_emb_avg = doc_emb_avg / asp_w_sum[:,:,None]                                 # [1, 7, 400]
#             doc_emb_max = torch.max(sent_bmm[cur:(cur+sn), :, : ], dim=0, keepdim=True)[0] # [1, 7, 400]
#             all_doc_emb.append( torch.cat( [doc_emb_avg, doc_emb_max], dim=2 ) )
            all_doc_emb.append( doc_emb_avg )
            aspect_doc.append( aspect_dist[cur:(cur+sn), :] )
            
            cur = cur + sn

        all_doc_emb = torch.cat( all_doc_emb, dim=0 )          # [batch, asp, 400]
        
#         result_senti_base = self.senti_base( all_doc_emb.view(-1, 400) ) # [batch*asp, 50]
#         result_senti_base = result_senti_base.view(-1, self.n_asp, 50)    # [batch, asp, 50]
        
        result_senti = [ self.sentiments[aspi]( all_doc_emb[:,aspi,:] ) for aspi in range(0,self.n_asp)] # [batch, ra]
        
        result = torch.stack(result_senti, dim=1)  # [batch, asp, sentiment5]
        
        return result,raw_outputs,outputs,aspect_doc


def get_model(vocab_sz:int, vocab, n_class:int, bptt:int=70, max_len:int=20*70, config:dict=None,
                        drop_mult:float=1., lin_ftrs:Collection[int]=None, ps:Collection[float]=None,
                        pad_idx:int=1) -> nn.Module:
    print("Creating Custom Model")
    meta = text.learner._model_meta[AWD_LSTM]
    # if we specified config then we dont use default
    config = ifnone(config, meta['config_clas']).copy()
    config.pop("output_p")
#     config.pop("qrnn")
#     config.pop("n_layers")
#     print(config)
    # Drop multiplier
    for k in config.keys():
        if k.endswith('_p'): config[k] *= drop_mult
    init = config.pop('init') if 'init' in config else None
    
    encoder = SentenceEncoder(bptt, max_len, AWD_LSTM(vocab_sz, **config), vocab, pad_idx=pad_idx)
#     cls_layer = Cls02ATT400(n_asp=hyper_params["num_aspect"], n_rat=hyper_params["num_rating"])
    cls_layer = Cls02_LIATLI(n_asp=hyper_params["num_aspect"], n_rat=hyper_params["num_rating"])
    
#     encoder = SentenceEncoder(bptt, max_len, BI_AWD_LSTM(vocab_sz, **config), vocab, pad_idx=pad_idx)
#     cls_layer = Cls02ATT_BILSTM(n_asp=hyper_params["num_aspect"], n_rat=hyper_params["num_rating"])

    model = SequentialRNN(encoder, cls_layer)
    return model if init is None else model.apply(init)

def load_pretrained(learn, wgts_fname:str, itos_fname:str, strict:bool=True):
    "Load a pretrained model and adapts it to the data vocabulary."
    old_itos = pickle.load(open(itos_fname, 'rb'))
    old_stoi = {v:k for k,v in enumerate(old_itos)}
    wgts = torch.load(wgts_fname, map_location=lambda storage, loc: storage)
    if 'model' in wgts: wgts = wgts['model']
    wgts = convert_weights(wgts, old_stoi, learn.data.train_ds.vocab.itos)
    
#     wkeys = list( wgts.keys() )                                           #  for BI LSTM
#     for wkey in wkeys:
#         if wkey.startswith("0.rnns.2"):
#             wgts["0.rnns.3"+wkey[len("0.rnns.3"):]] = wgts[wkey].clone()  #  for BI LSTM

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
            warn("There are no pretrained weights for that architecture yetget_ipython().getoutput("")")
            return learn
        model_path = untar_data(meta['url'], data=False)
        fnames = [list(model_path.glob(f'*.{ext}'))[0] for ext in ['pth', 'pkl']]
        learn = load_pretrained(learn, *fnames, strict=False)
        learn.freeze()
    return learn


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


macc = [get_clas_acc(ai) for ai in range(hyper_params["num_aspect"])]
for ai in range(hyper_params["num_aspect"]):
    macc[ai].__name__ = "clas_acc_"+str(ai)
mmse = [get_clas_mse(ai) for ai in range(hyper_params["num_aspect"])]
for ai in range(hyper_params["num_aspect"]):
    mmse[ai].__name__ = "clas_mse_"+str(ai)


mloss = MultiLabelCEL()
cls_learn = text_classifier_learner(cls_db,
                                    drop_mult=1.1,
                                    loss_func=mloss,
                                    metrics=[multi_acc]+macc+mmse,
                                    bptt=70,
                                    max_len=hyper_params["max_sequence_length"])


for i in range(len(cls_learn.layer_groups)):
    print("group " + str(i) + " =====")
    print(cls_learn.layer_groups[i])


from fastai.callbacks.tracker import TrackerCallback


def fast_validate(model:nn.Module, dl:DataLoader, loss_func:OptLossFunc=None, n_batch:Optional[int]=None)->Iterator[Tuple[Union[Tensor,int],...]]:
    
    with torch.no_grad():
        val_losses,nums = [],[]

        for xb,yb in dl:
            out = model(xb)[0]
            val_loss = loss_func(out, yb)
            val_loss = val_loss.detach().cpu()
            
            val_losses.append(val_loss)
            if not is_listy(yb): yb = [yb]
            nums.append(first_el(yb).shape[0])
            if n_batch and (len(nums)>=n_batch): break
            
        nums = np.array(nums, dtype=np.float32)
        return (to_np(torch.stack(val_losses)) * nums).sum() / nums.sum()



class SaveBestStepModel(TrackerCallback):
    "A `TrackerCallback` that saves the model when monitored quantity is best."
    def __init__(self, learn:Learner, monitor:str='valid_loss', mode:str='auto', every:int=50, name:str='bestmodel'):
        super().__init__(learn, monitor=monitor, mode=mode)
        self.every, self.name = every, name
        self.step = 0
        self.records = []
    
    def on_train_begin(self, **kwargs:Any)->None:
        super().on_train_begin(**kwargs)
        self.step = 0
        
    def on_batch_end(self, **kwargs:Any)->None:
        self.step += 1

        if self.step % self.every == 0:
            self.learn.model.eval()
            current = fast_validate(self.learn.model, self.learn.data.valid_dl, self.learn.loss_func, n_batch=50)
            self.learn.model.train()
            
            if isinstance(current, Tensor): current = current.cpu()
            self.records.append(current)
            
            if current is not None and self.operator(current, self.best):
                print(f'Better model found at step {self.step} with {self.monitor} value: {current}.')
                self.best = current
                self.learn.save(f'{self.name}')

    def on_train_end(self, **kwargs):
        pass
#         "Load the best model."
#         if self.every=="improvement" and os.path.isfile(self.path/self.model_dir/f'{self.name}.pth'):
#             self.learn.load(f'{self.name}', purge=False)
            
    


cls_learn.callback_fns = [cls_learn.callback_fns[0]]


cls_learn.callback_fns


cls_learn.callback_fns += [ partial(SaveBestStepModel, monitor="valid_loss", mode="min", every=100, name='beer.clas.attfullind400.best.learner') ]


weight_file = "lm_enc_beer.1115"

encoder = cls_learn.model[0]
if hasattr(encoder, 'module'):
    encoder = encoder.module
distrib_barrier()
wgts = torch.load(cls_learn.path/cls_learn.model_dir/f'{weight_file}.pth', map_location=cls_learn.data.device)

# wkeys = list( wgts.keys() )
# print(wkeys)
# for wkey in wkeys:
#     if wkey.startswith("rnns.2"):
#         wgts["rnns.3"+wkey[len("rnns.3"):]] = wgts[wkey].clone()
        
encoder.load_state_dict(wgts)
cls_learn.freeze()





x, y = cls_db.one_batch()
print(x.shape)
print(y.shape)


cls_learn.model(x.cuda())


with experiment.train():
    cls_learn.fit_one_cycle( 6, max_lr=slice(1e-3,2e-2) )


#  With redist first
with experiment.train():
    cls_learn.fit_one_cycle( 7, max_lr=slice(1e-3,2e-2) )


fig = cls_learn.recorder.plot_losses()
experiment.log_figure(figure_name="train loss 01", figure=fig)


cls_learn.save('beer.clas.LIATLI.1.learner')


cls_learn.load('beer.clas.attfullind400.1.learner')


cls_learn.unfreeze()


# FULL INDIPENDENT
with experiment.train():
    cls_learn.fit_one_cycle(10)


# LIATLI
with experiment.train():
    cls_learn.fit_one_cycle(12)


fig = cls_learn.recorder.plot_losses()
experiment.log_figure(figure_name="train loss 02", figure=fig)


cls_learn.save('beer.clas.LIATLI.2.learner')


cls_learn.load('beer.clas.attfullind400.2.learner')


experiment.end()


sent_num_file = ["train.count", "test.count"]
rating_file = ["train.rating", "test.rating"]
content_file = ["train.txt", "test.txt"]

dataset_dir = "./data/beer_100k/"


def concat_to_doc(sent_list, sent_count):
    start_index = 0
    docs = []
    for s in sent_count:
#         doc = " xxPERIOD ".join(sent_list[start_index:start_index + s])
#         doc = doc + " xxPERIOD "
        docs.append(sent_list[start_index:start_index + s])
        start_index = start_index + s
    return docs


# TRAIN_DATA = 0
TEST_DATA = 1


# # Load Count
sent_count_test = list(open(dataset_dir + sent_num_file[TEST_DATA], "r").readlines())
sent_count_test = [int(s) for s in sent_count_test if (len(s) > 0 and s get_ipython().getoutput("= "\n")]")
print( sent_count_test[0:5] )

# Load Ratings
aspect_rating_test = list(open(dataset_dir + rating_file[TEST_DATA], "r").readlines())
aspect_rating_test = [s for s in aspect_rating_test if (len(s) > 0 and s get_ipython().getoutput("= "\n")]")

aspect_rating_test = [s.split(" ") for s in aspect_rating_test]
aspect_rating_test = np.array(aspect_rating_test)[:, :]
aspect_rating_test = aspect_rating_test.astype(np.float) - 1
aspect_rating_test = np.rint(aspect_rating_test).astype(int)  # ROUND TO INTEGER =================
aspect_rating_test = pd.DataFrame(aspect_rating_test)
print( aspect_rating_test.head() )

# Load Sents
sents_test = list(open(dataset_dir + content_file[TEST_DATA], "r").readlines())
sents_test = [s.strip() for s in sents_test]
sents_test = [s[:-1] for s in sents_test if s.endswith(".")]
print( sents_test[0:5] )

# Sents to Doc
docs_test = concat_to_doc(sents_test, sent_count_test)
docs_test = pd.DataFrame({doc:docs_test})


df_test = pd.concat( [aspect_rating_test, docs_test], axis=1, ignore_index=True )
df_test.head()


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


def get_preds(self,
              ds_type:DatasetType,
              activ:nn.Module=None,
              with_loss:bool=False,
              n_batch:Optional[int]=None,
              pbar:Optional[PBar]=None,
              ordered:bool=False) -> List[Tensor]:
    "Return predictions and targets on the valid, train, or test set, depending on `ds_type`."
    self.model.reset()
    if ordered: np.random.seed(42)
    
    with torch.no_grad():
        outs = []
        asps = []
        for xb,yb in progress_bar(cls_learn.dl(ds_type)):
            out,raw_enc,enc,asp = cls_learn.model(xb)
            outs.append(out)
            for doc in asp:
                asps.append( to_float(doc.cpu()))

    outs = to_float(torch.cat(outs).cpu())
    
    if ordered and hasattr(self.dl(ds_type), 'sampler'):
        np.random.seed(42)
        sampler = [i for i in self.dl(ds_type).sampler]
        reverse_sampler = np.argsort(sampler)
        
        outs = outs[reverse_sampler]
        asps = [asps[i] for i in reverse_sampler]
    return (outs,asps)


outs,asps = get_preds(self=cls_learn, ds_type=DatasetType.Test, ordered=True)


outs.shape


target = torch.tensor( aspect_rating_test.values )
target


mloss = MultiLabelCEL()
mloss.forward(outs, target)


pd.DataFrame.from_dict( {"ASP"+str(ai):[get_clas_acc(ai)(outs, target).item()] for ai in range(5)} )


pd.DataFrame.from_dict( {"ASP"+str(ai):[get_clas_mse(ai)(outs, target).item()] for ai in range(5)} )


asp_inc_overall = True
if not asp_inc_overall: 
    nasp_analysis = hyper_params["num_aspect"] - 1
else:
    nasp_analysis = hyper_params["num_aspect"]
    
np.set_printoptions(precision=3)
asp_name = ["Overall", "Appearance", "Taste", "Palate", "Aroma"]
for i in range(10):
    print("truth:")
    print(df_test.iloc[i,0:5].values.flatten().tolist() )
    print("prediction:")
    print( torch.argmax(outs[i][0:5],dim=1) )
    print("doc:")
    dasp = torch.argmax(asps[i][:,0:nasp_analysis],dim=1).numpy()
    if asp_inc_overall: dasp_noall = torch.argmax(asps[i][:,1:6],dim=1).numpy()
#     dasp_dist = torch.nn.functional.softmax(asps[i][:,0:nasp_analysis], dim=1).numpy()
    dasp_dist = asps[i][:,0:nasp_analysis].numpy()
    for senti,s in enumerate(df_test.iloc[i,-1]):
        print(s)
        if asp_inc_overall:
            print("          +++ "+ asp_name[dasp[senti]] + " +++ " + str(dasp_dist[senti]) )
        else:
            print("          +++ "+ asp_name[dasp[senti]+1] + " +++ " + str(dasp_dist[senti]) )
    print("===========")


def eval_hotel_asp(asp_pred, asp_true, asp_inc_overall):
    asp_to_id = {"appearance":0, "taste":1, "palate":2, "aroma":3, "none":-1}
    asp_true = np.array( [asp_to_id[l] for l in asp_true] )
    print("total true: " + str(len(asp_true)) )
    print("total not none: " + str(sum(asp_true>0)) )
    
    asp_pred_index = []
    if asp_inc_overall:
        for i in range(1000):
            asp_pred_index.append( asp_pred[i][:,1:6].numpy().argsort() )
    else:
        for i in range(1000):
            asp_pred_index.append( asp_pred[i][:,0:5].numpy().argsort() )
    asp_pred_index = np.concatenate( asp_pred_index , axis=0)
    
    result_index = []
    for i,lbl in enumerate(asp_true):
        if(lbl==-1):
            result_index.append(-1)
        else:
            at = np.where(asp_pred_index[i,] == lbl)
            result_index.append(at[0][0])
    result_index = np.array(result_index)
    
    print("Top 1 ACC:")
    print( sum(result_index>=4) / sum(result_index>=0) )
    print("Top 2 ACC:")
    print( sum(result_index>=3) / sum(result_index>=0) )


dataset_dir


yifan_label = open(dataset_dir + "test_aspect_0.yifanmarjan.aspect", "r").readlines()
yifan_label = [s.split()[0] for s in yifan_label]


eval_hotel_asp(asps, yifan_label, asp_inc_overall=True)





fan_label = open(dataset_dir + "test_aspect_0.fan.aspect", "r").readlines()
fan_label = [s.split()[0] for s in fan_label]


eval_hotel_asp(asps, fan_label, asp_inc_overall=True)






