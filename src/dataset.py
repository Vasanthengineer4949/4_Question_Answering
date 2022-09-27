import config
import pandas as pd
from torch.utils.data import Dataset
from transformers import MT5Tokenizer
import pytorch_lightning as pl


class MLQADataset(Dataset):

    def __init__(
        self,
        data: pd.DataFrame,
        tokenizer: MT5Tokenizer,
        question_max_token_len = config.QUESTION_MAX_LEN,
        answer_max_token_len = config.ANSWER_MAX_LEN
                    ):
        self.tokenizer = tokenizer
        self.data = data
        self.max_qns_len = question_max_token_len
        self.max_ans_len = answer_max_token_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data_idx = self.data.iloc[index]

        question_encoding = self.tokenizer(
            data_idx["question"],
            data_idx["context"],
            max_length=config.QUESTION_MAX_LEN,
            padding="max_length",
            truncation="only_second",
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt"   
        )

        answer_encoding = self.tokenizer(
            data_idx["answers"],
            max_length=config.ANSWER_MAX_LEN,
            padding="max_length",
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt"   
        )

        labels = answer_encoding["input_ids"]
        labels[labels==0] = -100

        return dict (
            context = data_idx["context"],
            question = data_idx["question"],
            answer = data_idx["answers"],
            input_ids = question_encoding["input_ids"].flatten(),
            attention_mask = question_encoding["attention_mask"].flatten(),
            labels = labels.flatten()
        )