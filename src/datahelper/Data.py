from enum import Enum


class DataObject:
    def __init__(self, name, size):
        self.name = name
        self.num_instance = size
        self.file_id = None

        self.raw = None
        self.value = None

        self.label_doc = None
        self.label_instance = None  # this is for sentences or comb or paragraph

        self.sentence_len = None
        self.sentence_len_trim = None

        self.doc_size = None
        self.doc_size_trim = None

        self.vocab = None
        self.vocab_inv = None
        self.init_embedding = None
        self.init_embedding_w2v = None

        self.target_doc_len = None
        self.target_sent_len = None
        self.num_aspects = None
        self.num_classes = None

    def init_empty_list(self):
        self.file_id = []
        self.raw = []
        self.value = []
        self.label_doc = []
        self.label_instance = []
        self.doc_size = []
        self.doc_size_trim = []


class LoadMethod(Enum):
    DOC = 1
    COMB = 2
    SENT = 3
