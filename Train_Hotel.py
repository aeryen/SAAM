from fastai.text import *
from data_helpers.Data import *
from fastai.text.transform import *


class ClsNet(torch.nn.Module):
    def __init__(self, encoder, vocab):
        super(ClsNet, self).__init__()
        self.vocab = vocab
        self.p_index = vocab
        self.enc = encoder
        self.sentiment = torch.nn.Linear(400, 5)
        self.sentiment_sm = torch.nn.Softmax(dim=1)
        self.aspect = torch.nn.Linear(400, 5)
        self.aspect_sm = torch.nn.Softmax(dim=1)
        
    def forward(self, x):
        # encode whole doc
        enc_result = self.enc(x)
        doc_enc = enc_result[0][2]  # [batch_size, doc_length, embedding]
        # print("shape of doc encoder:")
        # print( doc_enc.shape )
        # print("doc_enc grad_fn:")
        # print(doc_enc.grad_fn)
        
        # flatten doc length dimension
        doc_enc = doc_enc.contiguous().view(-1, 400)  # [batch_size * doc_length]
        # print("doc_enc grad_fn:")
        # print(doc_enc.grad_fn)
        
        # print("number of sentences in docs:")
        n_sent = torch.sum( x==self.vocab.stoi["xxperiod"] , dim=1)
        # print(n_sent)
        
        # print("locating period marks")
        period_index = x.view(-1)==self.vocab.stoi["xxperiod"]
        # print(period_index.shape)
        # print(torch.sum(period_index))

        # selecting only the encoder output at period marks
        sent_output = doc_enc[period_index, :]  # [total n_sentence, embedding]
        # print(sent_output)
        # print(sent_output.shape)
        
        sentiment_dist = self.sentiment(sent_output)   # [total n_sentence, embedding]
        sentiment_dist = self.sentiment_sm(sentiment_dist)
        aspect_dist = self.aspect(sent_output)         # [total n_sentence, embedding]
        aspect_dist = self.aspect_sm(aspect_dist)
        # print("sentiment dist weight:")
        # print(sentiment_dist.grad_fn)
        # print("aspect dist weight:")
        # print(aspect_dist.grad_fn)
        
        sent_bmm = torch.bmm(sentiment_dist.unsqueeze(2), aspect_dist.unsqueeze(1))
        # print("sent bmm:")
        # print(sent_bmm.dtype)
        # print(sent_bmm.shape)  # [total n_sentence, sentiment, aspect]
        # print(sent_bmm.grad_fn)
        
        cur = 0
        result = []
        for n_sent in n_sent :
            # print("-----")
            doc = sent_bmm[(cur):(cur+n_sent), :, :]
            # print(doc.shape)
            doc = torch.mean(doc, dim=0, keepdim=True)
            # print(doc.shape)
            result.append(doc)
        
        result = torch.cat( result, dim=0 )
        # print(result.dtype)
        
        return result

class MultiLabelCEL(nn.CrossEntropyLoss):
    def forward(self, input, target):
        # print("in multi label cel")
        i = input.view(-1, 5)    # flatten the aspect dimension, [batch*aspect, sentiment]
        t = target.contiguous().view(-1).long()  # flatten the aspect dimension
        loss = super(MultiLabelCEL, self).forward(i, t)
        return loss

lmdb = load_data("./data/", "hotel_lm_databunch.1001")

learn = language_model_learner(lmdb, AWD_LSTM)
learn.unfreeze()
learn = learn.load("lang_model_hotel")

clas_db = load_data("./data/", "hotel_cls_databunch.aspect_only")

clas_db.batch_size=10

encoder = learn.model[0]

net = ClsNet(encoder, lmdb.vocab)
net.train()

l = MultiLabelCEL()

net.enc.reset()

moms = (0.8,0.7)

my_learner = Learner(clas_db,
                    net,
                    opt_func=torch.optim.Adam,
                    loss_func=l)
my_learner.fit_one_cycle(1, moms=moms)

my_learner.save('stage1-clas-4epo')

my_learner.unfreeze()
my_learner.fit_one_cycle(1, moms=moms)

my_learner.save('stage2-clas-8epo')