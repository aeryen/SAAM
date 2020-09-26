# %%
from fastai.text import *
from data_helpers.Data import *
from fastai.text.transform import *
# %%
hyper_params = {
    "max_sequence_length": 40*70,
    "batch_size": 72,
    "num_epochs1": 12,
    "num_epochs2": 15,
    "num_aspect": 6,
    "num_rating": 5,
}
# %%
cls_db = load_data("./data/", "hotel_clas_databunch_2020.TraVal")
#%%
cls_db.batch_size=16
cls_db.batch_size
# %%
class SentenceEncoder(Module):
    "Create an encoder over `module` that can process a full sentence."
    def __init__(self, bptt:int, max_len:int, module:nn.Module, vocab, pad_idx:int=1):
        print("Encoder init")
        self.max_len,self.bptt,self.module,self.pad_idx = max_len,bptt,module,pad_idx
        self.vocab = vocab
        self.period_index = self.vocab.stoi["xxperiod"]

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

                
        # print("number of sentences in docs:")
#         n_sent = torch.sum( x==self.vocab.stoi["xxperiod"] , dim=1)
        # print(n_sent)
        
        # print("locating period marks")
        period_index = torch.cat(p_index,dim=1)
        
        return self.concat(raw_outputs),self.concat(outputs), \
               torch.cat(masks,dim=1),period_index

# %%
# ATTENTIONAL AVERAGING, COMPLETELY INDEPENDENT SENTI OUT

class Cls02ATT400(Module):
    "Create a linear classifier with pooling."
    def __init__(self, n_asp:int, n_rat:int, layers:Collection[int], drops:Collection[float]):
        print("CLS init")
        print("Num Aspect: "+str(n_asp) )
        print("Num Rating: "+str(n_rat) )
        self.n_asp = n_asp
        self.n_rat = n_rat
        
        self.hid_size = 128
        
        mod_layers = []
        mod_layers += bn_drop_lin( 400, self.hid_size, p=0.5, actn=nn.GELU() )
        mod_layers += bn_drop_lin( self.hid_size, self.n_asp, p=0.2, actn=torch.nn.Softmax(dim=1) )
        self.aspect = nn.Sequential(*mod_layers)
        
        self.s0 = nn.Sequential(* (bn_drop_lin( 400, self.hid_size, p=0.5, actn=nn.GELU() ) + 
                                   bn_drop_lin( self.hid_size, self.n_rat, p=0.2, actn=None ) ) )
        self.s1 = nn.Sequential(* (bn_drop_lin( 400, self.hid_size, p=0.5, actn=nn.GELU() ) + 
                                   bn_drop_lin( self.hid_size, self.n_rat, p=0.2, actn=None ) ) )
        self.s2 = nn.Sequential(* (bn_drop_lin( 400, self.hid_size, p=0.5, actn=nn.GELU() ) + 
                                   bn_drop_lin( self.hid_size, self.n_rat, p=0.2, actn=None ) ) )
        self.s3 = nn.Sequential(* (bn_drop_lin( 400, self.hid_size, p=0.5, actn=nn.GELU() ) + 
                                   bn_drop_lin( self.hid_size, self.n_rat, p=0.2, actn=None ) ) )
        self.s4 = nn.Sequential(* (bn_drop_lin( 400, self.hid_size, p=0.5, actn=nn.GELU() ) + 
                                   bn_drop_lin( self.hid_size, self.n_rat, p=0.2, actn=None ) ) )
        self.s5 = nn.Sequential(* (bn_drop_lin( 400, self.hid_size, p=0.5, actn=nn.GELU() ) + 
                                   bn_drop_lin( self.hid_size, self.n_rat, p=0.2, actn=None ) ) )
#         self.s6 = nn.Sequential(* (bn_drop_lin( 400, self.hid_size, p=0.5, actn=nn.GeLU(inplace=True) ) + 
#                                    bn_drop_lin( self.hid_size, self.n_rat, p=0.2, actn=None ) ) )
        self.sentiments = []
        self.sentiments.append( self.s0 )
        self.sentiments.append( self.s1 )
        self.sentiments.append( self.s2 )
        self.sentiments.append( self.s3 )
        self.sentiments.append( self.s4 )
        self.sentiments.append( self.s5 )
#         self.sentiments.append( self.s6 )

    def forward(self, input:Tuple[Tensor,Tensor,Tensor,Tensor])->Tuple[Tensor,Tensor,Tensor]:
        raw_outputs,outputs,mask,p_index = input
        
        batch = sentence_pool_400(outputs, mask, p_index)
        
        allsent_emb = torch.cat(batch, dim=0)          # [n_sentence, emb400]
        aspect_dist = self.aspect(allsent_emb)         # [n_sentence, aspect6]

        sent_bmm = torch.bmm(aspect_dist.unsqueeze(2), allsent_emb.unsqueeze(1))  # [319, 7, 400]
        
        all_doc_emb = []
        aspect_doc = []
        sentim_doc = []
        cur = 0
        for doci in range(0, len(batch)):
            sn = batch[doci].shape[0]
            doc_emb_avg = torch.sum(sent_bmm[cur:(cur+sn), :, : ], dim=0, keepdim=True) # [1, 7, 400]
            asp_w_sum = torch.sum(aspect_dist[cur:(cur+sn),:], dim=0, keepdim=True) # [1, 7]
            doc_emb_avg = doc_emb_avg / asp_w_sum[:,:,None]                                 # [1, 7, 400]
            all_doc_emb.append( doc_emb_avg )
            aspect_doc.append( aspect_dist[cur:(cur+sn), :] )
            
            cur = cur + sn

        all_doc_emb = torch.cat( all_doc_emb, dim=0 )          # [batch, asp, 400]
        
        result_senti = [ self.sentiments[aspi]( all_doc_emb[:,aspi,:] ) for aspi in range(0,self.n_asp)] # [batch, ra]
        
        result = torch.stack(result_senti, dim=1)  # [batch, asp, sentiment5]
        
        return result,raw_outputs,outputs,aspect_doc

# %%
def get_text_classifier(arch:Callable, vocab_sz:int, vocab, n_class:int, bptt:int=70, max_len:int=20*70, config:dict=None,
                        drop_mult:float=1., lin_ftrs:Collection[int]=None, ps:Collection[float]=None,
                        pad_idx:int=1) -> nn.Module:
    "Create a text classifier from `arch` and its `config`, maybe `pretrained`."
    print("CUSTOM DEFINED CLASSIFIER")
    meta = text.learner._model_meta[arch]
    config = ifnone(config, meta['config_clas']).copy()
    for k in config.keys():
        if k.endswith('_p'): config[k] *= drop_mult
    if lin_ftrs is None: lin_ftrs = [50]
    if ps is None:  ps = [0.1]*len(lin_ftrs)
    layers = [config[meta['hid_name']] * 3] + lin_ftrs + [n_class]
    ps = [config.pop('output_p')] + ps
    init = config.pop('init') if 'init' in config else None
    encoder = SentenceEncoder(bptt, max_len, arch(vocab_sz, **config), vocab, pad_idx=pad_idx)
    cls_layer = Cls02ATT400(n_asp=hyper_params["num_aspect"], n_rat=hyper_params["num_rating"], layers=layers, drops=ps)
    model = SequentialRNN(encoder, cls_layer)
    return model if init is None else model.apply(init)

# %%
model = get_text_classifier(AWD_LSTM, len(cls_db.vocab.itos), cls_db.vocab, cls_db.c,
                             bptt=70, max_len=2800,
                             config=None,
                             drop_mult=1.1, lin_ftrs=None, ps=None)

# %%
x,y = cls_db.one_batch()
# %%
raw_outputs,outputs,mask,p_index = model[0].cuda()(x.cuda())
# %%
def average_emb(output, start, end):
    avg_pool = output[start:end, :].mean(dim=0)
    return avg_pool

def sentence_pool_1200(outputs, mask, p_index):
    output = outputs[-1]
    doc_start = mask.int().sum(dim=1)
    
    batch = []
    for doci in range(0,output.shape[0]):
        pi = p_index[doci,:].nonzero(as_tuple=True)[0].int()
        doc = []
        for senti in range( len(pi) ):
            if senti==0:
                # from start of doc to end of first sent
                doc.append( average_emb(output[doci,:,:], doc_start[doci], pi[senti]) )
            else:
                # from previous period to next
                doc.append( average_emb(output[doci,:,:], pi[senti-1]+1, pi[senti]) )
            
        batch.append( torch.stack(doc, 0) )

    return batch
# %%
print("1")
sentence_pool_1200(outputs, mask, p_index)

# %%

