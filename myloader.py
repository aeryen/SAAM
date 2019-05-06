from fastai.text import *
from data_helpers.Data import *

from data_helpers.DataHelperHotelOne import DataHelperHotelOne
from data_helpers.DataHelperBeer import DataHelperBeer

sent_num_file = ["aspect_0.count", "test_aspect_0.count"]
rating_file = ["aspect_0.rating", "test_aspect_0.rating"]
content_file = ["aspect_0.txt", "test_aspect_0.txt"]

dataset_dir = "./data/hotel_balance_LengthFix1_3000per/"


@staticmethod
def concat_to_doc(sent_list, sent_count):
    start_index = 0
    docs = []
    for s in sent_count:
        doc = " <LB> ".join(sent_list[start_index:start_index + s])
        docs.append(doc)
        start_index = start_index + s
    return docs


def load_files(load_test) -> DataObject:
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.

    aspect_id should be 0 1 2 3 4 5
    """
    sent_count = list(open(dataset_dir + sent_num_file[load_test], "r").readlines())
    sent_count = [int(s) for s in sent_count if (len(s) > 0 and s != "\n")]

    aspect_rating = list(open(dataset_dir + rating_file[load_test], "r").readlines())
    aspect_rating = [s for s in aspect_rating if (len(s) > 0 and s != "\n")]

    y = [s.split(" ") for s in aspect_rating]
    y = np.array(y)[:, 0:-1]
    y = y.astype(np.int) - 1

    content = list(open(dataset_dir + content_file[load_test], "r").readlines())
    content = [s.strip() for s in content]

    x_text = concat_to_doc(sent_list=content, sent_count=sent_count)

    s_len = np.array([len(x) for x in x_text], dtype=np.int32)

    data = DataObject("hotel", len(y))
    data.raw = x_text
    data.sentence_len = s_len
    data.label_doc = y
    data.doc_size = sent_count

    return data


# dater = DataHelperHotelOne(target_doc_len=100, target_sent_len=64,
#                            aspect_id=None, doc_as_sent=False, doc_level=True)

load_files(0)

TextLMDataBunch.from_csv()

