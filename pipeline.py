import random
import numpy as np
import torch
import itertools
from models.GRU import GRU
from models.LSTM_v1 import LSTMv1
from models.random import RandomGuess
from models.LSTM_v2 import LSTMv2
from data import Data
from predictor import Predictor

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

seed_value = 42
random.seed(seed_value)
np.random.seed(seed_value)
torch.manual_seed(seed_value)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)


def pipeline(model_class, seq_length, lr, batch_size, epochs, train_dir, val_dir, test_dir, input_size, output_size,
             dropout_prob, grid_params):
    best_val_accuracy = 0
    best_test_accuracy = 0
    best_mcc = 0
    best_params = None

    for weight_decay, clip_value, hidden_size, depth, width, num_layers in itertools.product(*grid_params.values()):
        print(
            f"Testing with weight decay = {weight_decay}, clip value = {clip_value}, hidden_size={hidden_size}, depth={depth}, width={width}, num_layers={num_layers}")

        model = model_class(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, depth=depth,
                            width=width, num_classes=output_size, dropout_prob=dropout_prob).to(device)
        data = Data(train_dir, val_dir, test_dir, seq_length, batch_size)
        _, val_loaders, test_loaders = data.get_loaders()

        predictor = Predictor(seq_length, model, lr, batch_size, epochs, train_dir, val_dir, test_dir, weight_decay,
                              clip_value)

        # Train the model once for all epochs
        best_train_loss, best_val_metrics, best_test_metrics = predictor.train()  # Updated to capture best metrics

        # Check if the current parameters yield the best results
        if best_val_metrics[1] >= best_val_accuracy and best_test_metrics[1] >= best_test_accuracy and \
                best_test_metrics[2] >= best_mcc:
            best_val_accuracy = best_val_metrics[1]
            best_test_accuracy = best_test_metrics[1]
            best_mcc = best_test_metrics[2]
            best_params = (hidden_size, depth, width, num_layers)

    if best_params:
        print(
            f"Best Parameters: Hidden Size: {best_params[0]}, Depth: {best_params[1]}, Width: {best_params[2]}, Num Layers: {best_params[3]}")
        print(
            f"Best Validation Accuracy: {best_val_accuracy}, Best Test Accuracy: {best_test_accuracy}, Best MCC: {best_mcc}")
    else:
        print("No best parameters found.")


if __name__ == '__main__':
    seq_length = 5
    lr = 0.00001
    batch_size = 1024
    epochs = 500
    input_size = 11
    output_size = 1
    dropout_prob = 0.2

    # for grid search
    # grid_params = {
    #     'weight_decay': [0.001, 0.01, 0.1, 1.0],
    #     'clip_value': [0.01, 0.1, 1.0],
    #     'hidden_size': [5, 6, 7, 8],
    #     'depth': [1, 2],
    #     'width': [5, 6, 7, 8],
    #     'num_layers': [3, 4]
    # }

    #GRU
    # grid_params = {
    #     'weight_decay': [0.001],
    #     'clip_value': [1.0],
    #     'hidden_size': [17],
    #     'depth': [1],
    #     'width': [19],
    #     'num_layers': [1]
    # }

    #LSTM
    grid_params = {
        'weight_decay': [0.01],
        'clip_value': [1.0],
        'hidden_size': [8],
        'depth': [1],
        'width': [14],
        'num_layers': [1]
    }

    # lstm attention
    # grid_params = {
    #     'weight_decay': [0.001],
    #     'clip_value': [1.0],
    #     'hidden_size': [20],
    #     'depth': [1],
    #     'width': [22],
    #     'num_layers': [1]
    # }

    train_dir = "dataset/price/train"
    test_dir = "dataset/price/test"
    val_dir = "dataset/price/validation"
    model_path = 'trained_model.pth'

    # Models: LSTMv1, LSTMv2, GRU
    # Uncommend whatever grid params are related to the model before running

    pipeline(LSTMv1, seq_length, lr, batch_size, epochs, train_dir, val_dir, test_dir, input_size,
             output_size, dropout_prob, grid_params)
