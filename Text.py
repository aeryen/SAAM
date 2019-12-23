from fastai.text import *   # Quick access to NLP functionality

path = untar_data(URLs.IMDB_SAMPLE)
path

df = pd.read_csv(path/'texts.csv')

data_lm = TextLMDataBunch.from_csv(path, 'texts.csv')
data_clas = TextClasDataBunch.from_csv(path, 'texts.csv', vocab=data_lm.train_ds.vocab, bs=42)

moms = (0.8,0.7)

learn = language_model_learner(data_lm, AWD_LSTM)
learn.unfreeze()
learn.fit_one_cycle(4, slice(1e-2), moms=moms)

learn.save_encoder('enc')

learn = text_classifier_learner(data_clas, AWD_LSTM)
learn.load_encoder('enc')
learn.fit_one_cycle(4, moms=moms)