from fastai.text import *

clas_db = load_data("./data/", "hotel_cls_databunch.aspect_only")
clas_db.batch_size=2

x,y = clas_db.one_batch()
x=x.cuda()
y=y.cuda()

cls_learn = text_classifier_learner(clas_db, AWD_LSTM)

result = cls_learn.pred_batch(batch=(x,y))

print(result)

