from sklearn.metrics import f1_score, matthews_corrcoef, confusion_matrix
import torch.nn as nn
import random
import numpy as np
import torch
from data import Data

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


class Predictor:
    def __init__(self, seq_length, model, lr, batch_size, epochs, train_data_dir, validation_data_dir,
                 test_data_dir, weight_decay, clip_value):
        self.data = Data(train_data_dir, validation_data_dir, test_data_dir, seq_length, batch_size)
        self.model = model.to(device)
        self.lr = lr
        self.epochs = epochs
        self.best_val_accuracy = 0.0
        self.weight_decay = weight_decay
        self.clip_value = clip_value

    def train(self):
        criterion = nn.BCELoss()
        train_loaders, val_loaders, test_loaders = self.data.get_loaders()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        best_train_loss = float('inf')
        best_val_metrics = (0.0, 0.0, 0.0)
        best_test_metrics = (0.0, 0.0, 0.0)
        best_train_cm, best_val_cm, best_test_cm = None, None, None
        best_train_epoch, best_val_epoch, best_test_epoch = -1, -1, -1

        for epoch in range(self.epochs):
            self.model.train()
            total_train_loss = 0.0
            total_train_f1 = 0.0
            total_train_accuracy = 0.0
            total_train_mcc = 0.0
            all_train_targets = []
            all_train_preds = []

            for ticker, loader in train_loaders.items():
                for features, targets, _ in loader:
                    features, targets = features.to(device), targets.to(device).float().unsqueeze(1)
                    optimizer.zero_grad()
                    outputs = self.model(features)

                    loss = criterion(outputs, targets)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_value)
                    optimizer.step()

                    total_train_loss += loss.item()

                    preds = (outputs >= 0.5).float()
                    all_train_targets.extend(targets.cpu().numpy())
                    all_train_preds.extend(preds.cpu().numpy())

                    f1 = f1_score(targets.cpu(), preds.cpu(), average='macro')
                    mcc = matthews_corrcoef(targets.cpu(), preds.cpu())
                    accuracy = (preds == targets).float().mean()

                    total_train_f1 += f1
                    total_train_accuracy += accuracy
                    total_train_mcc += mcc

            train_cm = confusion_matrix(all_train_targets, all_train_preds)
            val_cm = self.compute_cm(val_loaders)
            test_cm = self.compute_cm(test_loaders)

            if self.is_better_cm(train_cm, best_train_cm):
                best_train_cm, best_train_epoch = train_cm, epoch
            if self.is_better_cm(val_cm, best_val_cm):
                best_val_cm, best_val_epoch = val_cm, epoch
            if self.is_better_cm(test_cm, best_test_cm):
                best_test_cm, best_test_epoch = test_cm, epoch

            avg_train_loss = total_train_loss / len(train_loaders)
            avg_train_f1 = total_train_f1 / len(train_loaders)
            avg_train_accuracy = total_train_accuracy / len(train_loaders)
            avg_train_mcc = total_train_mcc / len(train_loaders)
            # Compute the confusion matrix for training data
            # train_cm = confusion_matrix(all_train_targets, all_train_preds)
            # print(f"Training Confusion Matrix:\n{train_cm}")

            if total_train_loss < best_train_loss:
                best_train_loss = total_train_loss

            val_loss, val_f1, val_accuracy, val_mcc = self.validation(val_loaders)

            if (val_accuracy > best_val_metrics[1]) and (val_mcc > best_val_metrics[2]):
                best_val_metrics = (val_f1, val_accuracy, val_mcc)

            test_loss, test_f1, test_accuracy, test_mcc = self.test(test_loaders)

            if (test_accuracy > best_test_metrics[1]) and (test_mcc > best_test_metrics[2]):
                best_test_metrics = (test_f1, test_accuracy, test_mcc)

            print(f"Epoch {epoch + 1}/{self.epochs}, "
                  f"Train Loss: {avg_train_loss:.4f}, "
                  f"Validation Loss: {val_loss:.4f}, "
                  f"Validation F1: {val_f1:.4f}, "
                  f"Validation Accuracy: {val_accuracy:.4f}, "
                  f"Validation MCC: {val_mcc:.4f}, "
                  f"Test F1: {test_f1:.4f}, "
                  f"Test Accuracy: {test_accuracy:.4f}, "
                  f"Test MCC: {test_mcc:.4f}")

        print(f"Best Train CM at Epoch {best_train_epoch}:\n{best_train_cm}")
        print(f"Best Validation CM at Epoch {best_val_epoch}:\n{best_val_cm}")
        print(f"Best Test CM at Epoch {best_test_epoch}:\n{best_test_cm}")

        return best_train_loss, best_val_metrics, best_test_metrics

    def validation(self, val_loaders):
        self.model.eval()
        total_val_loss = 0.0
        total_val_f1 = 0.0
        total_val_accuracy = 0.0
        total_val_mcc = 0.0
        all_targets = []
        all_preds = []
        criterion = nn.BCELoss()

        with torch.no_grad():
            for ticker, loader in val_loaders.items():
                for features, targets, _ in loader:
                    features, targets = features.to(device), targets.to(device).float().unsqueeze(1)
                    outputs = self.model(features)
                    loss = criterion(outputs, targets)
                    preds = (outputs >= 0.5).float()

                    all_targets.extend(targets.cpu().numpy())
                    all_preds.extend(preds.cpu().numpy())

                    f1 = f1_score(targets.cpu(), preds.cpu(), average='macro')
                    mcc = matthews_corrcoef(targets.cpu(), preds.cpu())
                    accuracy = (preds == targets).float().mean()

                    total_val_loss += loss.item()
                    total_val_f1 += f1
                    total_val_accuracy += accuracy
                    total_val_mcc += mcc

        avg_val_loss = total_val_loss / len(val_loaders)
        avg_val_f1 = total_val_f1 / len(val_loaders)
        avg_val_accuracy = total_val_accuracy / len(val_loaders)
        avg_val_mcc = total_val_mcc / len(val_loaders)

        cm = confusion_matrix(all_targets, all_preds)
        print(f"Validation Confusion Matrix:\n{cm}")

        return avg_val_loss, avg_val_f1, avg_val_accuracy, avg_val_mcc

    def test(self, test_loaders):
        self.model.eval()
        total_test_loss = 0.0
        total_test_f1 = 0.0
        total_test_accuracy = 0.0
        total_test_mcc = 0.0
        criterion = nn.BCELoss()

        for ticker, loader in test_loaders.items():
            all_test_identifiers = []
            all_test_preds = []
            all_test_targets = []

            with torch.no_grad():
                for features, targets, seq_id in loader:
                    features, targets = features.to(device), targets.to(device).float().unsqueeze(1)
                    outputs = self.model(features)
                    loss = criterion(outputs, targets)
                    total_test_loss += loss.item()

                    preds = (outputs >= 0.5).float()
                    all_test_targets.extend(targets.cpu().numpy())
                    all_test_preds.extend(preds.cpu().numpy())
                    all_test_identifiers.extend(seq_id.cpu().numpy())

                    f1 = f1_score(targets.cpu(), preds.cpu(), average='macro')
                    mcc = matthews_corrcoef(targets.cpu(), preds.cpu())
                    accuracy = (preds == targets).float().mean()
                    print(accuracy)
                    total_test_f1 += f1
                    total_test_accuracy += accuracy
                    total_test_mcc += mcc

            # sort predictions after processing each ticker
            sorted_predictions = sorted(zip(all_test_identifiers, all_test_preds), key=lambda x: x[0])
            sorted_preds = [pred.item() for _, pred in sorted_predictions]

            print(f"{ticker} Sorted Predictions: ", sorted_preds)

        avg_test_loss = total_test_loss / len(test_loaders)
        avg_test_f1 = total_test_f1 / len(test_loaders)
        avg_test_accuracy = total_test_accuracy / len(test_loaders)
        avg_test_mcc = total_test_mcc / len(test_loaders)

        return avg_test_loss, avg_test_f1, avg_test_accuracy, avg_test_mcc

    def print_model_weights(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(name, param.data)

    def is_better_cm(self, current_cm, best_cm):
        if best_cm is None:
            return True
        current_ratio = min((current_cm[0, 0] / max(current_cm[0, 1], 1),
                             current_cm[1, 1] / max(current_cm[1, 0], 1)))
        best_ratio = min((best_cm[0, 0] / max(best_cm[0, 1], 1),
                          best_cm[1, 1] / max(best_cm[1, 0], 1)))
        return current_ratio > best_ratio

    def compute_cm(self, loaders):
        self.model.eval()
        all_targets = []
        all_preds = []
        with torch.no_grad():
            for loader in loaders.values():
                for features, targets, _ in loader:
                    features, targets = features.to(device), targets.to(device).float().unsqueeze(1)
                    outputs = self.model(features)
                    preds = (outputs >= 0.5).float()
                    all_targets.extend(targets.cpu().numpy())
                    all_preds.extend(preds.cpu().numpy())
        return confusion_matrix(all_targets, all_preds)
