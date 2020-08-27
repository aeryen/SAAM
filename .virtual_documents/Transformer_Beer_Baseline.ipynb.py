import pandas as pd
from data_helpers.Data import *





def concat_to_doc(sent_list, sent_count):
    start_index = 0
    docs = []
    for s in sent_count:
        doc = " xxPERIOD ".join(sent_list[start_index:start_index + s])
        doc = doc + " xxPERIOD "
        docs.append(doc)
        start_index = start_index + s
    return docs


sent_num_file = ["train.count", "test.count"]
rating_file = ["train.rating", "test.rating"]
content_file = ["train.txt", "test.txt"]

dataset_dir = "./data/beer_100k/"

TRAIN_DATA = 0
TEST_DATA = 1

def load_data(data_index) :
    # Load Count
    sent_count = list(open(dataset_dir + sent_num_file[data_index], "r").readlines())
    sent_count = [int(s) for s in sent_count if (len(s) > 0 and s get_ipython().getoutput("= "\n")]")
    print( sent_count[0:5] )

    # Load Ratings
    aspect_rating = list(open(dataset_dir + rating_file[data_index], "r").readlines())
    aspect_rating = [s for s in aspect_rating if (len(s) > 0 and s get_ipython().getoutput("= "\n")]")

    aspect_rating = [s.split(" ") for s in aspect_rating]
    aspect_rating = np.array(aspect_rating)[:, :]
    aspect_rating = aspect_rating.astype(np.float) - 1
    aspect_rating = np.rint(aspect_rating).astype(int)  # ROUND TO INTEGER =================
    aspect_rating = pd.DataFrame(aspect_rating)
    print( aspect_rating.head() )

    # Load Sents
    sents = list(open(dataset_dir + content_file[data_index], "r").readlines())
    sents = [s.strip() for s in sents]
    sents = [s[:-1] for s in sents if s.endswith(".")]
    print( sents[0:5] )
    
    return sent_count, aspect_rating, sents


sent_count_train, aspect_rating_train, sents_train = load_data(TRAIN_DATA)


# Concate sentences to doc
docs_train = concat_to_doc(sents_train, sent_count_train)
docs_train = pd.DataFrame(docs_train)
docs_train.head()


len(docs_train)


df_train = pd.DataFrame( {
    'id': list(range(len(aspect_rating_train))),
    'label': aspect_rating_train[0],
    'alpha': ['a']*len(aspect_rating_train),
    'text': docs_train[0]
    })
df_train.head()


# remove all the negative ratingsget_ipython().getoutput("!")
df_train = df_train[df_train["label"] >= 0]
df_train.shape


np.random.seed(42)
msk = np.random.rand(len(df_train)) < 0.85
df_train08 = df_train[msk]
df_valid02 = df_train[~msk]
len(df_train08), len(df_valid02)


df_train08.to_csv('data/transformer_beer_train.tsv', sep='\t', index=False, header=False)
df_valid02.to_csv('data/transformer_beer_valid.tsv', sep='\t', index=False, header=False)


sent_count_test, aspect_rating_test, sents_test = load_data(TEST_DATA)


# Concate sentences to doc
docs_test = concat_to_doc(sents_test, sent_count_test)
docs_test = pd.DataFrame(docs_test)
docs_test.head()


len(docs_test)


df_test = pd.DataFrame( {
    'id': list(range(len(aspect_rating_test))),
    'label': aspect_rating_test[0],
    'alpha': ['a']*len(aspect_rating_test),
    'text': docs_test[0]
    })
df_test.head()


len(df_test)


# ACTUALLY NO DIFFERENCE
df_test = df_test[df_test["label"] >= 0]
df_test.shape


df_test.to_csv('data/transformer_beer_test.tsv', sep='\t', index=False, header=False)


import os
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0"


from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer, EvalPrediction


from transformers.data.processors.utils import *


from transformers import Trainer, TrainingArguments


from typing import Dict, Optional


import torch

torch.cuda.is_available()


train_ds = SingleSentenceClassificationProcessor.create_from_csv(file_name="./data/transformer_beer_train.tsv", split_name="train", column_id=0, column_label=1, column_text=3)
valid_ds = SingleSentenceClassificationProcessor.create_from_csv(file_name="./data/transformer_beer_valid.tsv", split_name="train", column_id=0, column_label=1, column_text=3)


tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
tokenizer


cache_name = "cached_{}_{}_{}_{}".format(
    "train",
    tokenizer.__class__.__name__,
    str(512),
    "beer",
)
cache_name


feat_train = train_ds.get_features(tokenizer)
torch.save(feat_train, cache_name)


cache_name = "cached_{}_{}_{}_{}".format(
    "valid",
    tokenizer.__class__.__name__,
    str(512),
    "beer",
)
cache_name


feat_valid = valid_ds.get_features(tokenizer)
torch.save(feat_valid, cache_name)


tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
tokenizer


cache_name = "cached_{}_{}_{}_{}".format(
    "train",
    tokenizer.__class__.__name__,
    str(512),
    "beer",
)
feat_train = torch.load(cache_name)


cache_name = "cached_{}_{}_{}_{}".format(
    "valid",
    tokenizer.__class__.__name__,
    str(512),
    "beer",
)
feat_valid = torch.load(cache_name)


config = AutoConfig.from_pretrained("bert-base-cased", num_labels=5)
config


model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", config=config)
model


def simple_accuracy(preds, labels):
        return (preds == labels).mean()

def compute_metrics(p: EvalPrediction) -> Dict:
    preds = np.argmax(p.predictions, axis=1)
    return {"acc": simple_accuracy( preds, p.label_ids ) }


train_args = TrainingArguments(
    output_dir = "./trfm_out/BEERTEST/",
    do_train = True,
    do_eval = True,
    evaluate_during_training = True,
    
    per_gpu_train_batch_size = 16,
    per_gpu_eval_batch_size = 16,
    
    learning_rate = 2e-5,
    num_train_epochs = 5,
    
    logging_dir = "./trfm_out/BEERTEST/tblog/",
    logging_steps = 1000
)

# train_args.device = torch.device("cuda:0")
train_args.device


trainer = Trainer(
    model=model,
    args=train_args,
    train_dataset=feat_train,
    eval_dataset=feat_valid,
    compute_metrics=compute_metrics,
)


trainer.train()



