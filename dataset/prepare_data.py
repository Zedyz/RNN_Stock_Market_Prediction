import os
import pandas as pd
import matplotlib.pyplot as plt


class PrepareData:
    def __init__(self, directory='price/raw/'):

        self.directory = directory
        self.file_paths = self._get_all_csv_files()
        self.tickers = [os.path.basename(fp).split('.')[0] for fp in self.file_paths]
        self.class_distribution = {-1: 0, 0: 0, 1: 0}

    def _get_all_csv_files(self):
        return [os.path.join(self.directory, file) for file in os.listdir(self.directory) if file.endswith('.csv')]

    def calculate_normalized_moving_averages(self, df, windows=[5, 10, 15, 20, 25, 30]):
        for window in windows:
            moving_avg = df['Adj Close'].rolling(window=window).mean()
            normalized_moving_avg = (moving_avg / df['Adj Close']) - 1
            df[f'Normalized_MA_{window}'] = normalized_moving_avg

    def calculate_percentage_changes(self, df):
        df['c_open'] = (df['Open'] / df['Close'] - 1) * 100
        df['c_high'] = (df['High'] / df['Close'] - 1) * 100
        df['c_low'] = (df['Low'] / df['Close'] - 1) * 100
        df['n_close'] = (df['Close'] / df['Close'].shift(1) - 1) * 100
        df['n_adj_close'] = (df['Adj Close'] / df['Adj Close'].shift(1) - 1) * 100

        # Convert all price and volume columns to numeric
        for col in ['c_open', 'c_high', 'c_low', 'n_close', 'n_adj_close']:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    def classify_tops_bottoms(self, file_path):
        df_original = pd.read_csv(file_path, names=["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"],
                                  header=None)
        df = df_original.copy()

        for col in ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        self.calculate_percentage_changes(df)
        self.calculate_normalized_moving_averages(df)
        train_end_date = '2016-01-16'
        validation_end_date = '2016-04-24'
        test_end_date = '2017-01-01'
        df = df[(df['Date'] >= '2014-01-01') & (df['Date'] <= test_end_date)]

        ticker = os.path.basename(file_path).split('.')[0]
        df['Ticker'] = ticker

        df['Pct Change'] = df['Adj Close'].pct_change(periods=1) * 100

        df['Class'] = 0
        df.loc[df['Pct Change'] >= 0.57, 'Class'] = 1  # positive, up class
        df.loc[df['Pct Change'] <= -0.5, 'Class'] = -1  # negative, down class
        df.dropna(inplace=True)

        self.class_distribution[0] += len(df[df['Class'] == 0])
        self.class_distribution[1] += len(df[df['Class'] == 1])
        self.class_distribution[-1] += len(df[df['Class'] == -1])

        print(f"Class distribution for {ticker}: {self.class_distribution}")

        train_df = df[df['Date'] <= train_end_date]
        validation_df = df[(df['Date'] > train_end_date) & (df['Date'] <= validation_end_date)]
        test_df = df[df['Date'] > validation_end_date]

        if len(validation_df) < 7:
            return None, None, None, None, None, None
        if len(test_df) < 7:
            return None, None, None, None, None, None

        class_distribution_train = {-1: 0, 0: 0, 1: 0}
        class_distribution_validation = {-1: 0, 0: 0, 1: 0}
        class_distribution_test = {-1: 0, 0: 0, 1: 0}

        for cls in [-1, 0, 1]:
            class_distribution_train[cls] = len(train_df[train_df['Class'] == cls])
            class_distribution_validation[cls] = len(validation_df[validation_df['Class'] == cls])
            class_distribution_test[cls] = len(test_df[test_df['Class'] == cls])

        print(
            f"Class distribution for {ticker} - Train: {class_distribution_train}, Validation: {class_distribution_validation}, Test: {class_distribution_test}")

        self.plot_with_classes(train_df, ticker)
        return train_df, validation_df, test_df, class_distribution_train, class_distribution_validation, class_distribution_test

    def plot_with_classes(self, df, stock_name):
        positive = df[df['Class'] == 1]
        negative = df[df['Class'] == -1]

        plt.figure(figsize=(15, 7))
        plt.plot(df['Date'], df['Adj Close'], label='Price', color='blue')
        plt.scatter(positive['Date'], positive['Adj Close'], color='green', label='Rise (Positive)', zorder=5)
        plt.scatter(negative['Date'], negative['Adj Close'], color='red', label='Fall (Negative)', zorder=5)

        plt.title(f'{stock_name} Stock Price with Indicators')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.xticks(rotation=45)
        plt.legend()
        plt.tight_layout()
        plt.show()

    def analyze_stocks(self):
        total_class_distribution_train = {-1: 0, 0: 0, 1: 0}
        total_class_distribution_validation = {-1: 0, 0: 0, 1: 0}
        total_class_distribution_test = {-1: 0, 0: 0, 1: 0}
        all_dates = set()
        file_count = 0

        for file_path in self.file_paths:
            result = self.classify_tops_bottoms(file_path)
            if result is None or any(part is None for part in result):
                continue

            train_df, val_df, test_df, dist_train, dist_val, dist_test = result

            for cls in [-1, 0, 1]:
                total_class_distribution_train[cls] += dist_train[cls]
                total_class_distribution_validation[cls] += dist_val[cls]
                total_class_distribution_test[cls] += dist_test[cls]

            all_dates.update(train_df['Date'].tolist())
            all_dates.update(val_df['Date'].tolist())

            columns_to_keep = ['c_open', 'c_high', 'c_low', 'n_close', 'n_adj_close', 'Adj Close']
            columns_to_scale = ['Normalized_MA_5', 'Normalized_MA_10', 'Normalized_MA_15',
                                'Normalized_MA_20', 'Normalized_MA_25', 'Normalized_MA_30']

            # save the datasets
            for split, data in {'train': train_df, 'test': test_df, 'validation': val_df}.items():
                output_path = file_path.replace('raw', split)
                data[columns_to_scale] = data[columns_to_scale] * 100
                columns_to_keep_ext = ['Ticker', 'Date'] + columns_to_keep + columns_to_scale + ['Class']
                data_to_save = data[columns_to_keep_ext]
                data_to_save.to_csv(output_path, index=False)

            file_count += 1

        print(f"Total class distribution before alignment: {self.class_distribution}")
        print(
            f"Total class distribution - Train: {total_class_distribution_train}, Validation: {total_class_distribution_validation}, Test: {total_class_distribution_test}")


if __name__ == "__main__":
    data = PrepareData()
    data.analyze_stocks()
