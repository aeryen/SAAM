import numpy as np
import logging

from data_helpers.DataHelpers import DataHelper
from data_helpers.Data import DataObject

sent_length_target = 65
sent_length_hard = 67


class DataHelperBeer(DataHelper):
    sent_num_file = ["train.count", "test.count"]
    rating_file = ["train.rating", "test.rating"]
    content_file = ["train.txt", "test.txt"]

    def __init__(self, embed_dim, target_doc_len, target_sent_len, aspect_id, doc_as_sent=False, doc_level=True):
        super(DataHelperBeer, self).__init__(problem_name="BeerAdvocate", embed_dim=embed_dim,
                                             target_doc_len=target_doc_len, target_sent_len=target_sent_len)

        logging.info("setting: %s is %s", "aspect_id", aspect_id)
        self.aspect_id = aspect_id
        logging.info("setting: %s is %s", "doc_as_sent", doc_as_sent)
        self.doc_as_sent = doc_as_sent
        logging.info("setting: %s is %s", "sent_list_doc", doc_level)
        self.doc_level = doc_level

        self.dataset_dir = self.data_dir_path + 'beer_100k/'
        self.num_classes = 9
        self.num_aspects = 5

        self.load_all_data()

    def load_files(self, load_test):
        """
        Loads training data from file
        """
        # Load data from files
        s_count = list(open(self.dataset_dir + self.sent_num_file[load_test], "r").readlines())
        s_count = [int(s) for s in s_count if (len(s) > 0 and s != "\n")]

        aspect_rating = list(open(self.dataset_dir + self.rating_file[load_test], "r").readlines())
        aspect_rating = [s for s in aspect_rating if (len(s) > 0 and s != "\n")]
        if self.aspect_id is None:
            y = [s.split(" ") for s in aspect_rating]
            y = np.array(y)
            y = y.astype(np.float32) - 1
            y[y < 0] = 0
            y = y * 2  # Scale 0-4 to 0-8 to remove
            y = y.astype(np.int32)
            # self.print_rating_distribution(y)
            y_onehot = self.to_onehot_3d(y, self.num_classes)
        else:
            y = [s.split(" ")[self.aspect_id] for s in aspect_rating]
            y = np.array(y)
            y = y.astype(np.float32) - 1
            y[y < 0] = 0
            y = y * 2  # Scale 0-4 to 0-8 to remove
            y = y.astype(np.int32)
            y_onehot = self.to_onehot(y, self.num_classes)

        content = list(open(self.dataset_dir + self.content_file[load_test], "r").readlines())
        content = [s.strip() for s in content]
        # Split by words
        x_text = [self.old_clean_str(sent) for sent in content]

        if self.doc_as_sent:
            x_text = DataHelperBeer.concat_to_doc(sent_list=x_text, sent_count=s_count)

        x_text = [x.split() for x in x_text]
        s_len = np.array([len(x) for x in x_text], dtype=np.int32)

        data = DataObject(self.problem_name, len(y))
        data.raw = x_text
        data.sentence_len = s_len
        data.doc_size = s_count
        data.label_doc = y_onehot

        return data

    def remove_negative_instances(self, data: DataObject):
        y_neg_index = np.logical_not(np.logical_and.reduce(data.label_doc < 0, axis=1))
        data.value = data.value[y_neg_index]
        # data.raw = data.raw[y_neg_index]
        # data.sentence_len = data.sentence_len[y_neg_index]
        data.sentence_len_trim = data.sentence_len_trim[y_neg_index]
        data.doc_size = data.doc_size[y_neg_index]
        # data.doc_size_trim = data.doc_size_trim[y_neg_index]
        data.label_doc = data.label_doc[y_neg_index]
        return data

    def load_all_data(self):
        train_data = self.load_files(0)
        self.vocab, self.vocab_inv = self.build_vocab([train_data], self.vocabulary_size)
        self.init_embedding = self.build_glove_embedding(self.vocab_inv)
        train_data = self.build_content_vector(train_data)
        train_data = self.pad_sentences(train_data)

        if self.doc_level:
            train_data = self.to_list_of_sent(train_data)
            # train_data = self.remove_negative_instances(train_data)
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
            # test_data = self.remove_negative_instances(test_data)
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
    a = DataHelperBeer(embed_dim=300, target_doc_len=64, target_sent_len=64, aspect_id=None,
                       doc_as_sent=False, doc_level=True)
    t = a.get_train_data()

    print()
