import json
import torch
import numpy as np
from torchtext.vocab import vocab
from .utils import VocabExtractor
from transformers import RobertaTokenizer
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight


class GenericDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


class GenericDatasetBuilder(object):
    def __init__(self, dset_path, tokenizer_class, tokenizer_name, train_size=0.7, val_size=0.5, device="cuda"):
        self.device = device
        self.data = json.load(open(dset_path))
        self.data = shuffle(self.data)
        self.vocab = VocabExtractor(self.data)
        self.tokenizer = tokenizer_class.from_pretrained(tokenizer_name)
        self.train, self.test = train_test_split(self.data, train_size=train_size)
        self.test, self.val = train_test_split(self.test, test_size=val_size)

    def _extract_texts_labels(self, dataset):
        texts = []
        labels = []
        for i in dataset:
            text, label = i["text"], i["label"]
            label = self.vocab.label_enc.transform([int(label)])[0]
            texts.append(text)
            labels.append(label)
        return texts, labels

    def build(self):
        train_texts, train_labels = self._extract_texts_labels(self.train)
        test_texts, test_labels = self._extract_texts_labels(self.test)
        val_texts, val_labels = self._extract_texts_labels(self.val)
        train_texts = self.tokenizer(train_texts, truncation=True, padding=True)
        val_texts = self.tokenizer(val_texts, truncation=True, padding=True)
        test_texts = self.tokenizer(test_texts, truncation=True, padding=True)
        train_dataset = GenericDataset(train_texts, train_labels)
        val_dataset = GenericDataset(val_texts, val_labels)
        test_dataset = GenericDataset(test_texts, test_labels)
        self.class_weight = torch.Tensor(
            compute_class_weight(
                "balanced", classes=np.unique(train_labels), y=train_labels
            )
        ).to(self.device)
        return train_dataset, val_dataset, test_dataset

    def get_number_of_classes(self):
        return len(self.vocab.labels)

    def get_class_weights(self):
        return self.class_weight
