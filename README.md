# Stock Market Prediction using RNNs

## Project Overview
This project aims to predict stock market movements by leveraging historical price data and a variety of indicators including: `c_open`, `c_high`, `c_low`, `n_close`, `n_adj_close`, `Adj Close`, `Normalized_MA_5`, `Normalized_MA_10`, `Normalized_MA_15`, `Normalized_MA_20`, `Normalized_MA_25`, and `Normalized_MA_30`. We utilize advanced machine learning models such as LSTM (Long Short-Term Memory) and GRU (Gated Recurrent Units) RNNs (Recurrent Neural Networks) to analyze trends and predict future stock prices.

### Data Preparation
Should you experience any issues with the dataset, it may be necessary to clear the contents of the `train`, `validation`, and `test` directories. Afterwards, execute the `prepare_data.py` script found in the `dataset/price/raw` subfolder. Ensure that the raw price data is correctly placed within this folder prior to running the script.

### Ongoing Development
This project is under development. As such, the current implementation may have limitations and is subject to future improvements and updates.

## Citing Our Data Source
The dataset utilized in this project is inspired by and extends the research described in the following publication:

> Sawhney, Ramit, Shivam Agarwal, Arnav Wadhwa, and Rajiv Ratn Shah. "Deep Attentive Learning for Stock Movement Prediction From Social Media Text and Company Correlations." In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 8415–8426. Association for Computational Linguistics, November 2020. [Read the paper](https://www.aclweb.org/anthology/2020.emnlp-main.676). DOI: 10.18653/v1/2020.emnlp-main.676
