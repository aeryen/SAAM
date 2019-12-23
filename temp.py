class ClsModule(Module):
    "Create a linear classifier with pooling."
    def __init__(self, n_asp:int, n_rat:int, layers:Collection[int], drops:Collection[float]):
        print("CLS init")
        print("Num Aspect: "+str(n_asp) )
        print("Num Rating: "+str(n_rat) )
        self.n_asp = n_asp
        self.n_rat = n_rat

        self.aspect = torch.nn.Linear(400, self.n_asp+1)
        self.aspect_sm = torch.nn.Softmax(dim=1)

        self.sentiment = torch.nn.Linear(400, self.n_rat)
        self.sentiment_sm = torch.nn.Softmax(dim=1)


    def forward(self, input:Tuple[Tensor,Tensor,Tensor,Tensor])->Tuple[Tensor,Tensor,Tensor]:
        raw_outputs,outputs,mask,p_index = input

        output = outputs[-1] # [batch, seq_len, emb_size]

        # print("number of sentences in docs:")
        n_sent = torch.sum( p_index , dim=1)

        result = []
        for bati in range(0,output.shape[0]):
            sent_output = output[bati, p_index[bati,:], :]

            sentiment_dist = self.sentiment(sent_output)   # [n_sentence, embedding]
            sentiment_dist = self.sentiment_sm(sentiment_dist)
            aspect_dist = self.aspect(sent_output)         # [n_sentence, embedding]
            aspect_dist = self.aspect_sm(aspect_dist)

            sent_bmm = torch.bmm(sentiment_dist.unsqueeze(2), aspect_dist.unsqueeze(1))
            doc = torch.mean(sent_bmm, dim=0, keepdim=True)
            result.append(doc)
        
        result = torch.cat( result, dim=0 )
        result = result[:,0:6,:]
        
        return result,raw_outputs,outputs