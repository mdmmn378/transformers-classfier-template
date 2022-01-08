import re

import numpy as np
import pandas as pd
import torch
from sklearn import preprocessing
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

from .emo import EMOTICONS_EMO

DEFAULT_PIPELINE = [
    "remove_uppercase",
    "remove_emoji",
    "remove_punctuation",
    "remove_newline_tabs",
    "remove_digits",
    "remove_whitespace",
]


class TextNormalizer(object):
    """
    Some of the methods were taken from
    https://www.kaggle.com/sudalairajkumar/getting-started-with-text-preprocessing
    """

    def __init__(self):
        pass

    def remove_uppercase(self, text):
        return text.lower()

    def remove_punctuation(self, text):
        text = text.replace(".", " ").replace(",", " ")
        text = text.translate(
            str.maketrans("", "", """!"#$%&\'()*+,-./:;<=>?@[\\]^_{|}~`""")
        )
        text = text.replace("\xa0", " ")
        return text

    def remove_emoji(self, string):
        emoji_pattern = re.compile(
            "["
            u"\U0001F600-\U0001F64F"  # emoticons
            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
            u"\U0001F680-\U0001F6FF"  # transport & map symbols
            u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
            u"\U00002702-\U000027B0"
            u"\U000024C2-\U0001F251"
            "]+",
            flags=re.UNICODE,
        )
        return emoji_pattern.sub(r"", string)

    # def remove_emoticons(self, text):
    #     emoticon_pattern = re.compile(u'(' + u'|'.join(k for k in EMOTICONS_EMO) + u')')
    #     return emoticon_pattern.sub(r'', text)
    # TODO:
    # 1. Fix "error: unbalanced parenthesis at position 7"

    def remove_whitespace(self, string):
        return re.sub(" +", " ", string)

    def remove_digits(self, string):
        return "".join([i for i in string if not i.isdigit()])

    def remove_newline_tabs(self, string):
        string = string.replace("\n", " ").replace("\t", " ")
        return string

    def apply(self, lines, pipeline=DEFAULT_PIPELINE):
        for text in lines:
            for step in pipeline:
                fun = getattr(self, step)
                text = fun(text)
            yield text


class VocabExtractor(object):
    def __init__(self, data):
        self.tokenizer = get_tokenizer(None)
        self.labels = set()
        self.vocab = build_vocab_from_iterator(
            self.yield_tokens(data), specials=["<unk>"]
        )
        self.vocab.set_default_index(self.vocab["<unk>"])
        self.label_enc = preprocessing.LabelEncoder()
        self.label_enc.fit(list(self.labels))

    def __len__(self):
        return len(self.vocab)

    def text_pipeline(self, text):
        return self.vocab(self.tokenizer(text))

    def label_pipeline(self, label):
        return self.label_enc.transform([label])

    def yield_tokens(self, data_iter):
        for sample in data_iter:
            self.labels.add(int(sample["label"]))
            yield self.tokenizer(sample["text"])
