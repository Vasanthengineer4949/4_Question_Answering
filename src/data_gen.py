import config
from datasets import load_dataset
import pandas as pd

class DataGen:
    
    def __init__(self):
        self.dataset_id = config.DATASET_ID
        self.subset_id = config.SUBSET_ID
    
    def load_data(self):
        train_test_data = load_dataset(self.dataset_id, self.subset_id, split="test")
        valid_data = load_dataset(self.dataset_id, self.subset_id, split="validation")
        return train_test_data[:4400], train_test_data[4400:], valid_data
        
    def data_df(self):
        train_data, test_data, valid_data = self.load_data()
        self.df_train = pd.DataFrame(train_data)
        self.df_test = pd.DataFrame(test_data)
        self.df_valid = pd.DataFrame(valid_data)
    
    def final_data_prep(self):
        self.df_train = self.df_train.drop("id", axis=1)
        self.df_train["answers"] = self.df_train["answers"].apply(lambda x: x["text"][0])
        self.df_test = self.df_test.drop("id", axis=1)
        self.df_test["answers"] = self.df_test["answers"].apply(lambda x: x["text"][0])
        self.df_valid = self.df_valid.drop("id", axis=1)
        self.df_valid["answers"] = self.df_valid["answers"].apply(lambda x: x["text"][0])
    
    def create_data_csv(self):
        self.data_df()
        self.final_data_prep()
        self.df_train.to_csv(config.TRAIN_DATA_PATH, index=False)
        self.df_test.to_csv(config.TEST_DATA_PATH, index=False)
        self.df_valid.to_csv(config.VALID_DATA_PATH, index=False)

