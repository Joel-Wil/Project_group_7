import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.model_selection import StratifiedShuffleSplit
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import f1_score


# === Configuration ===
SEQUENCE_LENGTH = 90  # Number of (magnitude, latency, vector short, vector short angle, vector long, vector long angle, x, y, t) points
FEATURE_SIZE = 8      # Features per point: (x, y, t)
NUM_CLASSES = 2       # Number of output classes: bird (0), drone (1)
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 0.01


# === Load Data ===
data = pd.read_csv("data_csv/all_train_val.csv")  # Adjust path if necessary
'''
Data format:
'ID_'
'Diff_Avg_Magnitude_'
'Latency_'
'Avg_Vector_Short_Magnitude_'
'Avg_Vector_Short_Angle_'
'Avg_Vector_Long_Magnitude_'
'Avg_Vector_Long_Angle_'
'X_'
'Y_'
'Timestamp_'
'''
exclude_keywords = ['ID_', 'Latency_']#, 'Avg_Vector_Short_Magnitude_',
                   # 'Avg_Vector_Short_Angle_', 'Avg_Vector_Long_Magnitude_', 'Avg_Vector_Long_Angle_']

X = data.loc[:, ~data.columns.str.contains('|'.join(exclude_keywords))]
y = X.pop('Label').values
#X = data.iloc[:, :-1].values  # All columns except the last (features)
#y = data.iloc[:, -1].values   # Last column (labels)

# Normalize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Reshape X into (samples, sequence_length, feature_size)
X = X.reshape(-1, SEQUENCE_LENGTH, FEATURE_SIZE)

# Train-test split
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
for train_index, test_index in sss.split(X,y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)


# === Define LSTM Model ===
class TrajectoryLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(TrajectoryLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=False, dropout=0.2)
        self.norm = nn.LayerNorm(hidden_size)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        # LSTM forward pass
        out, _ = self.lstm(x)
        # Use the last hidden state for classification
        out = self.norm(out[:, -1, :])   
        out = self.relu(out)
        out = self.fc(out) 
        return out

# Initialize model
model = TrajectoryLSTM(input_size=FEATURE_SIZE, hidden_size=128, num_layers=2, num_classes=NUM_CLASSES)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
#scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1) 

# === Training ===
def train_model(model, X_train, y_train, X_test, y_test, epochs, batch_size):
    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    best_f1 = 0.0
    train_losses = []
    val_losses = []
    test_accuracies = []
    f1_scores = []

    class_confidences = defaultdict(list)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # === Evaluation ===
        model.eval()
        with torch.no_grad():
            outputs = model(X_test)
            probs = torch.softmax(outputs, dim=1)
            confidences, predictions = torch.max(probs, 1)

            val_loss = criterion(outputs, y_test).item()
            val_losses.append(val_loss)

            f1 = f1_score(y_test.cpu().numpy(), predictions.cpu().numpy(), average='macro')
            f1_scores.append(f1)
            accuracy = (predictions == y_test).float().mean().item()

            if f1 > best_f1:
                best_f1 = f1
                best_model_state = model.state_dict()
                torch.save(best_model_state, "best_model.pt")
                print(f"Saved new best model with F1: {f1:.4f}")

            for cls in [0, 1]:
                cls_mask = predictions == cls
                if cls_mask.sum() > 0:
                    avg_conf = confidences[cls_mask].mean().item()
                else:
                    avg_conf = 0.0
                class_confidences[cls].append(avg_conf)

        test_accuracies.append(accuracy)
        print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {accuracy:.4f}, F1: {f1:.4f}")

    print('Best F1-score:', best_f1)
    # === Plot Loss ===
    epochs_range = list(range(1, epochs + 1))
    plt.figure(figsize=(12, 5))

    plt.plot(epochs_range, train_losses, label='Training Loss', marker='.')
    plt.plot(epochs_range, val_losses, label='Validation Loss', marker='.')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training & Validation Loss Over Epochs')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # === Accuracy Plot ===
    plt.figure(figsize=(8, 5))
    plt.plot(epochs_range, test_accuracies, label='Validation Accuracy', color='green', marker='.')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Validation Accuracy Over Epochs')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # === Confidence Plot ===
    plt.figure(figsize=(8, 6))
    plt.plot(epochs_range, class_confidences[0], label='Confidence for Class 0 (Bird)', marker='.', color='blue')
    plt.plot(epochs_range, class_confidences[1], label='Confidence for Class 1 (Drone)', marker='.', color='red')
    plt.axhline(0.5, color='gray', linestyle='--', linewidth=0.8)
    plt.xlabel('Epoch')
    plt.ylabel('Average Confidence')
    plt.title('Model Confidence Over Epochs')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # === F1 Score Plot ===
    plt.figure(figsize=(8, 5))
    plt.plot(epochs_range, f1_scores, label='Validation F1 Score', color='purple', marker='.')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.title('Validation F1 Score Over Epochs')
    plt.grid(True)
    plt.tight_layout()
    plt.legend()
    plt.show()



# Train the model
if __name__ == "__main__":
    train_model(model, X_train, y_train, X_test, y_test, EPOCHS, BATCH_SIZE)