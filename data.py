import os
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import torch


class Data:
    def __init__(self, train_data_dir, validation_data_dir, test_data_dir, seq_length, batch_size):
        self.train_data_dir = train_data_dir
        self.validation_data_dir = validation_data_dir
        self.test_data_dir = test_data_dir
        self.SEQ_LENGTH = seq_length
        self.batch_size = batch_size
        self.data_splits = self._preprocess_data()

    def create_sequences(self, features, targets):
        feature_sequences, target_sequences, sequence_ids = [], [], []

        for i in range(len(features) - self.SEQ_LENGTH):
            feature_seq = features[i:i + self.SEQ_LENGTH]
            target_seq = targets[i + self.SEQ_LENGTH]

            if target_seq != 0:
                if target_seq == -1:
                    target_seq = 0
                feature_sequences.append(feature_seq)
                target_sequences.append(target_seq)
                sequence_ids.append(i)  # Store the starting index of the sequence

        return (torch.tensor(np.array(feature_sequences, dtype=np.float32)),
                torch.tensor(np.array(target_sequences, dtype=np.int64)),
                torch.tensor(np.array(sequence_ids, dtype=np.int64)))

    def get_loaders(self):
        features = ['c_open', 'c_high', 'c_low', 'n_close', 'n_adj_close', 'Normalized_MA_5', 'Normalized_MA_10',
                    'Normalized_MA_15', 'Normalized_MA_20', 'Normalized_MA_25', 'Normalized_MA_30']
        loaders = {"train": {}, "test": {}, "validation": {}}

        for split, data_dict in self.data_splits.items():
            for ticker, data in data_dict.items():
                features_data, targets_data, seq_ids = self.create_sequences(data[features].values,
                                                                             data['Class'].values)
                dataset = TensorDataset(features_data, targets_data, seq_ids)
                loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
                loaders[split][ticker] = loader

        return loaders["train"], loaders["test"], loaders["validation"]

    def _load_individual_stock(self, filepath):
        return pd.read_csv(filepath).sort_values(by='Date', ascending=True).reset_index(drop=True)

    def _preprocess_data(self):
        data_splits = {"train": {}, "test": {}, "validation": {}}
        data_splits["train"] = self._process_data_dir(self.train_data_dir)
        data_splits["test"] = self._process_data_dir(self.validation_data_dir)
        data_splits["validation"] = self._process_data_dir(self.test_data_dir)

        return data_splits

    def _process_data_dir(self, directory):
        processed_data = {}
        for file in os.listdir(directory):
            if file.endswith('.csv'):
                filepath = os.path.join(directory, file)
                df = self._load_individual_stock(filepath)
                ticker_name = os.path.basename(filepath).split('.')[0]
                processed_data[ticker_name] = df
        return processed_data
