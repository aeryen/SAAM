# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'


# %% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
# ms-python.python added
import os
try:
	os.chdir('/home/jupyter/2019NN/')
	print(os.getcwd())
except:
	pass

# %%
from IPython import get_ipython

# %%
get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')


# %%
from fastai.text import *
from data_helpers.Data import *
from fastai.text.transform import *

# %% [markdown]
# ##  -- load LM Databunch and LM Learner

# %%
lm_db = load_data("./data/", "hotel_lm_databunch.1001")


# %%
lm_learn = language_model_learner(lm_db, AWD_LSTM)
lm_learn = lm_learn.load("lang_model_hotel")

# %% [markdown]
# ## -- load databunch

# %%
cls_db = load_data("./data/", "hotel_cls_databunch.aspect_only")


# %%
cls_db.batch_size=2


# %%
cls_db.c

# %% [markdown]
# # MODEL DEFINE

# %%
lm_learn.predict("the rooms are very", n_words=20)


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
class ClsModule(Module):
    "Create a linear classifier with pooling."
    def __init__(self, layers:Collection[int], drops:Collection[float]):
        print("CLS init")
        self.sentiment = torch.nn.Linear(400, 5)
        self.sentiment_sm = torch.nn.Softmax(dim=1)
        self.aspect = torch.nn.Linear(400, 5)
        self.aspect_sm = torch.nn.Softmax(dim=1)

    def forward(self, input:Tuple[Tensor,Tensor,Tensor,Tensor])->Tuple[Tensor,Tensor,Tensor]:
        raw_outputs,outputs,mask,p_index = input

        output = outputs[-1] # [batch, seq_len, emb_size]

        # flatten doc length dimension
        # doc_enc = outputs.contiguous().view(-1, 400)  # [batch_size * doc_length, embedding400]

        # print("number of sentences in docs:")
        n_sent = torch.sum( p_index , dim=1)  # [batch_size]

        # selecting only the encoder output at period marks
        sent_output = output[p_index, :]  # [total n_sentence, embedding]

        sentiment_dist = self.sentiment(sent_output)   # [total n_sentence, embedding]
        sentiment_dist = self.sentiment_sm(sentiment_dist)
        aspect_dist = self.aspect(sent_output)         # [total n_sentence, embedding]
        aspect_dist = self.aspect_sm(aspect_dist)

        sent_bmm = torch.bmm(sentiment_dist.unsqueeze(2), aspect_dist.unsqueeze(1))
        
        cur = 0
        result = []
        for i in n_sent :
            doc = sent_bmm[(cur):(cur+i), :, :]
            doc = torch.mean(doc, dim=0, keepdim=True)
            result.append(doc)
        
        result = torch.cat( result, dim=0 )
        
        return sent_output


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
    cls_layer = ClsModule(layers, ps)
    model = SequentialRNN(encoder, cls_layer)
    return model if init is None else model.apply(init)

# %% [markdown]
# # create Learner

# %%
def text_classifier_learner(data:DataBunch, arch:Callable, bptt:int=70, max_len:int=40*70, config:dict=None,
                            pretrained:bool=True, drop_mult:float=1., lin_ftrs:Collection[int]=None,
                            ps:Collection[float]=None, **learn_kwargs) -> 'TextClassifierLearner':
    "Create a `Learner` with a text classifier from `data` and `arch`."
    model = get_text_classifier(arch, len(data.vocab.itos), data.vocab, data.c, bptt=bptt, max_len=max_len,
                                config=config, drop_mult=drop_mult, lin_ftrs=lin_ftrs, ps=ps)
    meta = text.learner._model_meta[arch]
    learn = RNNLearner(data, model, split_func=meta['split_clas'], **learn_kwargs)
    if pretrained:
        if 'url' not in meta:
            warn("There are no pretrained weights for that architecture yet!")
            return learn
        model_path = untar_data(meta['url'], data=False)
        fnames = [list(model_path.glob(f'*.{ext}'))[0] for ext in ['pth', 'pkl']]
        learn = learn.load_pretrained(*fnames, strict=False)
        learn.freeze()
    return learn


# %%
cls_learn = text_classifier_learner(cls_db, AWD_LSTM)


# %%
x,y = cls_db.one_batch()
input = x[0:2]
label = y[0:2]
label


# %%
encoder = cls_learn.model[0]
encoder

# %%
cls_layer = cls_learn.model[1]
cls_layer

# %%
raw, out, mask, pindex = encoder(x.cuda())


# %%
out[-1].shape


# %%
cls_out = cls_layer( (raw, out, mask, pindex) )
cls_out

# %%
class MultiLabelCEL(nn.CrossEntropyLoss):
    def forward(self, input, target):
        # print("in multi label cel")
#         print(input.shape)
#         print(target.shape)
        i = input.view(-1, 5)    # flatten the aspect dimension, [batch*aspect, sentiment]
        t = target.contiguous().view(-1).long()  # flatten the aspect dimension
#         print(i.shape)
#         print(t.shape)
        loss = super(MultiLabelCEL, self).forward(i, t)
        return loss

# %% 
loss = MultiLabelCEL()

# %%
!nvidia-smi

# %%
cls_learn.layer_groups[4]

# %%
cls_learn.unfreeze()

# %%
moms = (0.8, 0.8, 0.8, 0.8, 0.7)
cls_learn.fit_one_cycle(4, moms)

# %%
