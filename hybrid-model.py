# %% [markdown]
# ### 1- Modules installation

# %%
%pip install librosa torch matplotlib scikit-learn tqdm

# %%
import os
import numpy as np
import librosa
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, roc_curve
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import Counter

# %% [markdown]
# ### 2- Configuration

# %%
class Config:
    # Audio processing
    sample_rate = 16000
    n_mfcc = 40                     # number of MFCC coefficients
    n_fft = 512
    hop_length = 160
    time_frames = 200               # pad/trim all features to 200 time steps

    # Data augmentation (reduced strength)
    noise_factor = 0.001            # reduced from 0.002
    pitch_shift_steps = 1           # reduced from ±2 to ±1
    # time_stretch removed (can alter emotion cues)

    # Model (increased capacity)
    lstm_hidden = 384              # increased from 128
    cnn_channels = [32, 64, 128, 256]    # three conv blocks
    dropout = 0.3                   # reduced from 0.5 (underfitting)
    batch_size = 32                 # increased from 16 (more stable)
    learning_rate = 1e-3
    weight_decay = 1e-4             # L2 regularisation
    epochs = 70
    early_stopping_patience = 10

    # Paths (CHANGE THIS to your dataset location)
    data_root = "/Volumes/workspace/default/speech_emotion_recognition"
    model_save_path = "/Volumes/workspace/default/speech_emotion_recognition/ser_model_best.pth"

# %% [markdown]
# ### 3- Feature Extraction: MFCC + Delta + Delta-Delta
# (More discriminative than Mel spectrograms for SER)

# %%
def augment_audio(audio, sr, config):
    """Apply mild random augmentations."""
    if np.random.rand() < 0.5:
        # Add Gaussian noise
        noise = np.random.randn(len(audio)) * config.noise_factor
        audio = audio + noise

    if np.random.rand() < 0.5:
        # Pitch shift (mild)
        steps = np.random.randint(-config.pitch_shift_steps, config.pitch_shift_steps+1)
        audio = librosa.effects.pitch_shift(audio, sr=sr, n_steps=steps)

    # Time stretching removed – can distort rhythmic emotion cues
    return audio

def extract_mfcc_features(file_path, config, augment=False):
    """Extract MFCC + delta + delta-delta features."""
    audio, sr = librosa.load(file_path, sr=config.sample_rate)

    if augment:
        audio = augment_audio(audio, sr, config)

    # MFCC
    mfcc = librosa.feature.mfcc(
        y=audio, sr=sr, n_mfcc=config.n_mfcc,
        n_fft=config.n_fft, hop_length=config.hop_length
    )
    # Delta and delta-delta
    mfcc_delta = librosa.feature.delta(mfcc)
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)

    # Stack: shape (120, time_frames) because n_mfcc=40, 3*40=120
    features = np.concatenate([mfcc, mfcc_delta, mfcc_delta2], axis=0)

    # Pad or truncate to fixed time frames
    if features.shape[1] < config.time_frames:
        pad_width = config.time_frames - features.shape[1]
        features = np.pad(features, ((0,0),(0,pad_width)), mode='constant')
    else:
        features = features[:, :config.time_frames]

    return features

# %% [markdown]
# ### 4- Dataset Class

# %%
class EmotionDataset(Dataset):
    def __init__(self, file_paths, labels, config, augment=False):
        self.file_paths = file_paths
        self.labels = labels
        self.config = config
        self.augment = augment

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        features = extract_mfcc_features(self.file_paths[idx], self.config, augment=self.augment)
        # Add channel dimension: (1, 120, time_frames) for CNN
        features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return features_tensor, label

# %% [markdown]
# ### 5- Load Dataset (CREMA-D style, flat folder structure)

# %%
def load_cremad_dataset(data_root):
    """
    Assumes CREMA-D structure: all .wav files in data_root/AudioWAV/
    Filenames contain emotion codes: ANG, DIS, FEA, HAP, NEU, SAD
    Speaker ID is extracted from the first part of the filename.
    """
    emotion_map = {
        'ANG': 0,   # angry
        'DIS': 1,   # disgust
        'FEA': 2,   # fear
        'HAP': 3,   # happy
        'NEU': 4,   # neutral
        'SAD': 5    # sad
    }
    file_paths = []
    emotions = []
    speakers = []

    audio_dir = os.path.join(data_root, 'AudioWAV')
    if not os.path.isdir(audio_dir):
        raise ValueError(f"AudioWAV directory not found at {audio_dir}")

    for fname in os.listdir(audio_dir):
        if fname.lower().endswith('.wav'):
            # Example: 1001_DFA_ANG_XX.wav
            parts = fname.split('_')
            if len(parts) >= 3:
                speaker_id = parts[0]
                emotion_code = parts[2].upper()
                if emotion_code in emotion_map:
                    file_paths.append(os.path.join(audio_dir, fname))
                    emotions.append(emotion_map[emotion_code])
                    speakers.append(speaker_id)

    print(f"Total files found: {len(file_paths)}")
    return file_paths, emotions, speakers

# %% [markdown]
# ### 6- Improved Model Architecture (increased capacity)

# %%
class EnhancedEmotionCNN_BiLSTM(nn.Module):
    def __init__(self, config, n_features=120):
        super().__init__()

        # Three convolutional blocks with increasing channels
        self.conv = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d((2,2)),

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((2,2)),

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d((2,2)),

            # Block 4 (new)
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((None, 1))
        )

        # LSTM (input size = 128, matches final CNN channels)
        self.lstm = nn.LSTM(
            input_size=256,
            hidden_size=config.lstm_hidden,
            num_layers=3,
            batch_first=True,
            bidirectional=True,
            dropout=config.dropout
        )

        # Attention pooling
        self.attention = nn.Sequential(
            nn.Linear(config.lstm_hidden * 2, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(config.lstm_hidden * 2, 128),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(128, config.num_classes)
        )

    def forward(self, x):
        # x: [batch, 1, n_features, time_frames]
        x = self.conv(x)               # [B, 256, mel', 1]
        x = x.squeeze(-1)              # [B, 256, mel']
        x = x.permute(0, 2, 1)         # [B, mel', 256]   (time, features)

        lstm_out, _ = self.lstm(x)     # [B, mel', hidden*2]

        # Attention weights
        attn_weights = self.attention(lstm_out)          # [B, mel', 1]
        attn_weights = torch.softmax(attn_weights.squeeze(-1), dim=1)  # [B, mel']
        context = torch.sum(lstm_out * attn_weights.unsqueeze(-1), dim=1)  # [B, hidden*2]

        out = self.classifier(context)
        return out

# %% [markdown]
# ### 7- Evaluation with AUC

# %%
def evaluate_with_auc(model, dataloader, criterion, device, class_names):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for features, labels in tqdm(dataloader, desc="Evaluating (AUC)"):
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    accuracy = 100.0 * correct / total

    try:
        macro_auc = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='macro')
        per_class_auc = roc_auc_score(all_labels, all_probs, multi_class='ovr', average=None)
    except ValueError as e:
        print(f"AUC could not be computed: {e}")
        macro_auc = None
        per_class_auc = None

    return avg_loss, accuracy, macro_auc, per_class_auc, all_labels, all_probs

# %% [markdown]
# ### 8- Training Functions with Learning Rate Scheduler

# %%
def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for features, labels in tqdm(dataloader, desc="Training"):
        features, labels = features.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    return total_loss / len(dataloader), 100.0 * correct / total

# %% [markdown]
# ### 9- Main Script

# %%
def main():
    config = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load dataset
    print("Loading dataset...")
    file_paths, emotion_ids, speakers = load_cremad_dataset(config.data_root)
    le = LabelEncoder()
    labels = le.fit_transform(emotion_ids)
    config.num_classes = len(le.classes_)
    print(f"Classes: {le.classes_}")

    # Speaker-independent split
    unique_speakers = list(set(speakers))
    train_speakers, temp_speakers = train_test_split(unique_speakers, test_size=0.4, random_state=42)
    val_speakers, test_speakers = train_test_split(temp_speakers, test_size=0.5, random_state=42)

    train_idx = [i for i, spk in enumerate(speakers) if spk in train_speakers]
    val_idx = [i for i, spk in enumerate(speakers) if spk in val_speakers]
    test_idx = [i for i, spk in enumerate(speakers) if spk in test_speakers]

    X_train = [file_paths[i] for i in train_idx]
    y_train = [labels[i] for i in train_idx]
    X_val = [file_paths[i] for i in val_idx]
    y_val = [labels[i] for i in val_idx]
    X_test = [file_paths[i] for i in test_idx]
    y_test = [labels[i] for i in test_idx]

    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # Class weights for imbalance
    class_counts = Counter(y_train)
    total_samples = len(y_train)
    class_weights = [total_samples / (config.num_classes * class_counts[i]) for i in range(config.num_classes)]
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

    # Datasets and loaders
    train_dataset = EmotionDataset(X_train, y_train, config, augment=True)
    val_dataset = EmotionDataset(X_val, y_val, config, augment=False)
    test_dataset = EmotionDataset(X_test, y_test, config, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=2)

    # Model, optimizer, loss
    model = EnhancedEmotionCNN_BiLSTM(config).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Training loop
    best_val_acc = 0
    patience_counter = 0
    history = {'train_loss': [], 'val_acc': [], 'val_auc': []}

    for epoch in range(config.epochs):
        print(f"\nEpoch {epoch+1}/{config.epochs}")
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, val_auc, _, _, _ = evaluate_with_auc(model, val_loader, criterion, device, le.classes_)

        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, Val AUC: {val_auc:.4f}")

        history['train_loss'].append(train_loss)
        history['val_acc'].append(val_acc)
        if val_auc:
            history['val_auc'].append(val_auc)

        # Learning rate scheduling based on validation loss
        scheduler.step(val_loss)

        # Early stopping and model checkpoint
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), config.model_save_path)
            print("Best model saved!")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config.early_stopping_patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    # Final test evaluation
    print("\n--- Final Evaluation on Test Set ---")
    model.load_state_dict(torch.load(config.model_save_path))
    test_loss, test_acc, test_macro_auc, per_class_auc, all_labels, all_probs = evaluate_with_auc(
        model, test_loader, criterion, device, le.classes_
    )
    print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")
    print(f"Test Macro AUC: {test_macro_auc:.4f}")
    if per_class_auc is not None:
        print("\nPer-class AUC:")
        for i, auc_val in enumerate(per_class_auc):
            print(f"  {le.classes_[i]}: {auc_val:.4f}")

    # Plot ROC curves
    plt.figure(figsize=(8,6))
    for i, class_name in enumerate(le.classes_):
        fpr, tpr, _ = roc_curve(all_labels, [p[i] for p in all_probs], pos_label=i)
        auc_val = per_class_auc[i] if per_class_auc is not None else 0
        plt.plot(fpr, tpr, label=f'{class_name} (AUC={auc_val:.3f})')
    plt.plot([0,1], [0,1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Multi-class ROC Curves (One-vs-Rest)')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.show()

    # Plot training curves
    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.subplot(1,2,2)
    plt.plot(history['val_acc'], label='Val Accuracy')
    if history['val_auc']:
        plt.plot(history['val_auc'], label='Val AUC')
    plt.title('Validation Metrics')
    plt.xlabel('Epoch')
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
