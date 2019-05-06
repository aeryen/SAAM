import numpy as np
import logging

from data_helpers.DataHelpers import DataHelper
from data_helpers.Data import DataObject


class DataHelperHotelOne(DataHelper):
    sent_num_file = ["aspect_0.count", "test_aspect_0.count"]
    rating_file = ["aspect_0.rating", "test_aspect_0.rating"]
    content_file = ["aspect_0.txt", "test_aspect_0.txt"]

    def __init__(self, target_doc_len, target_sent_len, aspect_id, doc_as_sent=False, doc_level=True):
        super(DataHelperHotelOne, self).__init__(problem_name="TripAdvisor",
                                                 target_doc_len=target_doc_len, target_sent_len=target_sent_len)

        logging.info("setting: %s is %s", "aspect_id", aspect_id)
        self.aspect_id = aspect_id
        logging.info("setting: %s is %s", "doc_as_sent", doc_as_sent)
        self.doc_as_sent = doc_as_sent
        logging.info("setting: %s is %s", "sent_list_doc", doc_level)
        self.doc_level = doc_level

        self.dataset_dir = self.data_dir_path + 'hotel_balance_LengthFix1_3000per/'
        self.num_classes = 5
        self.num_aspects = 6

        self.load_all_data()

    def print_rating_distribution(self, y):
        for aspect_index in range(y.shape[1]):
            for score in range(5):
                print(str(sum(y[:, aspect_index] == score)), end="\t")
            print("")

    def load_files(self, load_test) -> DataObject:
        """
        Loads MR polarity data from files, splits the data into words and generates labels.
        Returns split sentences and labels.

        aspect_id should be 0 1 2 3 4 5
        """
        sent_count = list(open(self.dataset_dir + self.sent_num_file[load_test], "r").readlines())
        sent_count = [int(s) for s in sent_count if (len(s) > 0 and s != "\n")]

        aspect_rating = list(open(self.dataset_dir + self.rating_file[load_test], "r").readlines())
        aspect_rating = [s for s in aspect_rating if (len(s) > 0 and s != "\n")]
        if self.aspect_id is None:
            y = [s.split(" ") for s in aspect_rating]
            y = np.array(y)[:, 0:-1]
            y = y.astype(np.int) - 1
            # self.print_rating_distribution(y)
            y_onehot = self.to_onehot_3d(y, self.num_classes)
        else:
            y = [s.split(" ")[self.aspect_id] for s in aspect_rating]
            y = np.array(list(map(float, y)), dtype=np.int) - 1
            y_onehot = self.to_onehot(y, self.num_classes)

        content = list(open(self.dataset_dir + self.content_file[load_test], "r").readlines())
        content = [s.strip() for s in content]
        # Split by words
        x_text = [self.old_clean_str(sent) for sent in content]

        if self.doc_as_sent:
            x_text = DataHelperHotelOne.concat_to_doc(sent_list=x_text, sent_count=sent_count)

        x_text = [x.split() for x in x_text]
        s_len = np.array([len(x) for x in x_text], dtype=np.int32)

        data = DataObject(self.problem_name, len(y))
        data.raw = x_text
        data.sentence_len = s_len
        data.label_doc = y_onehot
        data.doc_size = sent_count

        return data

    def load_all_data(self):
        train_data = self.load_files(0)
        self.vocab, self.vocab_inv = self.build_vocab([train_data], self.vocabulary_size)
        self.init_embedding = self.build_glove_embedding(self.vocab_inv)
        train_data = self.build_content_vector(train_data)
        train_data = self.pad_sentences(train_data)

        if self.doc_level:
            train_data = self.to_list_of_sent(train_data)
            DataHelper.pad_document(data=train_data,
                                    target_doc_len=self.target_doc_len, target_sent_len=self.target_sent_len)

        self.train_data = train_data
        self.train_data.target_doc_len = self.target_doc_len
        self.train_data.target_sent_len = self.target_sent_len
        self.train_data.num_aspects = self.num_aspects
        self.train_data.num_classes = self.num_classes
        self.train_data.init_embedding = self.init_embedding
        self.train_data.vocab = self.vocab
        self.train_data.vocab_inv = self.vocab_inv
        self.train_data.label_instance = self.train_data.label_doc

        test_data = self.load_files(1)
        test_data = self.build_content_vector(test_data)
        test_data = self.pad_sentences(test_data)

        if self.doc_level:
            test_data = self.to_list_of_sent(test_data)
            DataHelper.pad_document(data=test_data,
                                    target_doc_len=self.target_doc_len, target_sent_len=self.target_sent_len)

        self.test_data = test_data
        self.test_data.target_doc_len = self.target_doc_len
        self.test_data.target_sent_len = self.target_sent_len
        self.test_data.num_aspects = self.num_aspects
        self.test_data.num_classes = self.num_classes
        self.test_data.init_embedding = self.init_embedding
        self.test_data.vocab = self.vocab
        self.test_data.vocab_inv = self.vocab_inv
        self.test_data.label_instance = self.test_data.label_doc


if __name__ == "__main__":
    a = DataHelperHotelOne(embed_dim=300, target_doc_len=64, target_sent_len=65, aspect_id=None,
                           doc_as_sent=False, doc_level=True)
