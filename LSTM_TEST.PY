import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from joblib import load
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler


# === Configuration ===
SEQUENCE_LENGTH = 90  # Number of (magnitude, latency, vector short, vector short angle, vector long, vector long angle, x, y, t) points
FEATURE_SIZE = 8      # Features per point: (x, y, t)
NUM_CLASSES = 2       # Number of output classes: bird (0), drone (1)


# === Load Data ===
data = pd.read_csv("data_csv/all_test.csv")  # Adjust path if necessary
'''
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
exclude_keywords = ['ID_', 'Latency_']

X = data.loc[:, ~data.columns.str.contains('|'.join(exclude_keywords))]
y = X.pop('Label').values

# Normalize features
scaler = load('finalModel/scaler.joblib')
X = scaler.transform(X)

# Reshape X into (samples, sequence_length, feature_size)
X = X.reshape(-1, SEQUENCE_LENGTH, FEATURE_SIZE)

X_test = torch.tensor(X, dtype=torch.float32)
y_test = torch.tensor(y, dtype=torch.long)

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
        out, _ = self.lstm(x)
        out = self.norm(out[:, -1, :])   
        out = self.relu(out)
        out = self.fc(out) 
        return out


def plot_confusion_matrix(y_true, y_pred, labels, filename="confusion_matrix.png"):
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels)
    
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    print(f"Confusion matrix saved as: {filename}")
    plt.close()


def testModel(model, X_test, y_test):
    from sklearn.metrics import classification_report, confusion_matrix

    model.eval()
    with torch.no_grad():
        outputs = model(X_test)
        probs = torch.softmax(outputs, dim=1)
        confidences, predictions = torch.max(probs, 1)

        val_loss = nn.CrossEntropyLoss()(outputs, y_test).item()
        f1 = f1_score(y_test.cpu().numpy(), predictions.cpu().numpy(), average='macro')
        accuracy = (predictions == y_test).float().mean().item()

        print(f"\n--- Evaluation Results ---")
        print(f"Validation Loss : {val_loss:.4f}")
        print(f"Accuracy        : {accuracy:.4f}")
        print(f"F1 Score        : {f1:.4f}\n")

        print("Classification Report:")
        print(classification_report(y_test.cpu().numpy(), predictions.cpu().numpy(), target_names=["Bird", "Drone"]))

        plot_confusion_matrix(
        y_true=y_test.cpu().numpy(),
        y_pred=predictions.cpu().numpy(),
        labels=["Bird", "Drone"],
        filename="confusion_matrix.png"
)

        # Optional: Plot class confidence
        class_confidences = {}
        for cls in [0, 1]:
            cls_mask = predictions == cls
            if cls_mask.sum() > 0:
                avg_conf = confidences[cls_mask].mean().item()
            else:
                avg_conf = 0.0
            class_confidences[cls] = avg_conf

        plt.figure(figsize=(6, 4))
        plt.bar(["Bird", "Drone"], [class_confidences[0], class_confidences[1]], color=['blue', 'red'])
        plt.title("Average Confidence per Class")
        plt.ylabel("Confidence")
        plt.ylim(0, 1)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.show()



if __name__ == "__main__":
    model = TrajectoryLSTM(input_size=8, hidden_size=128, num_layers=2, num_classes=2)
    model.load_state_dict(torch.load("finalModel/weights.pt"))
    model.eval()

    testModel(model, X_test, y_test)
