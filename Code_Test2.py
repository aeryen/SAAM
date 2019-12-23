from fastai.text import *
# from fastai.text.transform import *

lm_db = load_data("./data/", "hotel_lm_databunch.1001")

lm_learn = language_model_learner(lm_db, AWD_LSTM)
lm_learn = lm_learn.load("lang_model_hotel")

cls_db = load_data("./data/", "hotel_cls_databunch.aspect_only")

cls_db.batch_size=2




class MultiBatchEncoder(Module):
    "Create an encoder over `module` that can process a full sentence."
    def __init__(self, bptt:int, max_len:int, module:nn.Module, vocab, pad_idx:int=1):
        print("Encoder init")
        self.max_len,self.bptt,self.module,self.pad_idx = max_len,bptt,module,pad_idx
        self.vocab = vocab

    def concat(self, arrs:Collection[Tensor])->Tensor:
        "Concatenate the `arrs` along the batch dimension."
        return [torch.cat([l[si] for l in arrs], dim=1) for si in range_of(arrs[0])]

    def reset(self):
        if hasattr(self.module, 'reset'): self.module.reset()

    def forward(self, input:LongTensor)->Tuple[Tensor,Tensor,Tensor]:
        bs,sl = input.size()
        self.reset()
        raw_outputs,outputs,masks = [],[],[]
        for i in range(0, sl, self.bptt):
            r, o = self.module(input[:,i: min(i+self.bptt, sl)])
            if i>(sl-self.max_len):
                masks.append(input[:,i: min(i+self.bptt, sl)] == self.pad_idx)
                raw_outputs.append(r)
                outputs.append(o)
                
        # print("number of sentences in docs:")
#         n_sent = torch.sum( x==self.vocab.stoi["xxperiod"] , dim=1)
        # print(n_sent)
        
        # print("locating period marks")
        period_index = x.view(-1)==self.vocab.stoi["xxperiod"]
        
        return self.concat(raw_outputs),self.concat(outputs),torch.cat(masks,dim=1)

class ClsModule(Module):
    "Create a linear classifier with pooling."
    def __init__(self, layers:Collection[int], drops:Collection[float]):
        print("CLS init")
        self.sentiment = torch.nn.Linear(1200, 5)
        self.sentiment_sm = torch.nn.Softmax(dim=1)
        self.aspect = torch.nn.Linear(1200, 5)
        self.aspect_sm = torch.nn.Softmax(dim=1)

    def forward(self, input:Tuple[Tensor,Tensor, Tensor])->Tuple[Tensor,Tensor,Tensor]:
        raw_outputs,outputs,mask = input
        
        # flatten doc length dimension
        doc_enc = outputs.contiguous().view(-1, 400)  # [batch_size * doc_length, embedding400]
        # print("number of sentences in docs:")
        n_sent = torch.sum( input==self.vocab.stoi["xxperiod"] , dim=1)
        # print("locating period marks")
        period_index = input.view(-1)==self.vocab.stoi["xxperiod"]
        # selecting only the encoder output at period marks
        sent_output = doc_enc[period_index, :]  # [total n_sentence, embedding]
        
        return x, raw_outputs, outputs

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
    encoder = MultiBatchEncoder(bptt, max_len, arch(vocab_sz, **config), vocab, pad_idx=pad_idx)
    cls_layer = PoolingLinearClassifier(layers, ps)
    model = SequentialRNN(encoder, cls_layer)
    return model if init is None else model.apply(init)


def text_classifier_learner(data:DataBunch, arch:Callable, bptt:int=70, max_len:int=70*20, config:dict=None,
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


cls_learn = text_classifier_learner(cls_db, AWD_LSTM)

