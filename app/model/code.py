import torch
import torch.nn as nn
import math

# Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=10):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :].to(x.device)
        return x

# Transformer
class TransformerClassifier(nn.Module):
    def __init__(self, input_dim=11, seq_len=6, d_model=256, nhead=8, num_layers=4, num_classes=2, dropout=0.2):
        super().__init__()
        self.input_linear = nn.Linear(input_dim, d_model)
        self.layer_norm = nn.LayerNorm(d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=seq_len)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, x, mask=None):
        x = self.input_linear(x)
        x = self.layer_norm(x)
        x = self.pos_encoder(x)
        if mask is not None:
            x = self.transformer_encoder(x, src_key_padding_mask=mask)
        else:
            x = self.transformer_encoder(x)
        x = x.mean(dim=1)  # Global average pooling
        return self.classifier(x)

# import pandas as pd
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.utils.class_weight import compute_class_weight
# from torch.utils.data import DataLoader, TensorDataset
# from tqdm import tqdm
# import os, pickle
# import math

# os.makedirs("./models", exist_ok=True)
# best_model_path_pth = "./models/best_model.pth"
# best_model_path_pkl = "./models/best_model.pkl"
# best_val_loss = float('inf')

# # Load dataset
# # df = pd.read_csv('/content/drive/MyDrive/Capstone-2025/Datasets/sequences.csv')
# df = pd.read_csv('./sequences.csv')

# # Features
# feature_cols = [
#     'VV_July', 'VH_July', 'VH_VV_July', 'NDVI_July', 'EVI_July', 'GNDVI_July', 'SAVI_July', 'NDWI_July', 'NDMI_July', 'RENDVI_July',
#     'VV_August', 'VH_August', 'VH_VV_August', 'NDVI_August', 'EVI_August', 'GNDVI_August', 'SAVI_August', 'NDWI_August', 'NDMI_August', 'RENDVI_August',
#     'VV_September', 'VH_September', 'VH_VV_September', 'NDVI_September', 'EVI_September', 'GNDVI_September', 'SAVI_September', 'NDWI_September', 'NDMI_September', 'RENDVI_September',
#     'VV_October', 'VH_October', 'VH_VV_October', 'NDVI_October', 'EVI_October', 'GNDVI_October', 'SAVI_October', 'NDWI_October', 'NDMI_October', 'RENDVI_October',
#     'VV_November', 'VH_November', 'VH_VV_November', 'NDVI_November', 'EVI_November', 'GNDVI_November', 'SAVI_November', 'NDWI_November', 'NDMI_November', 'RENDVI_November',
#     'VV_December', 'VH_December', 'VH_VV_December', 'NDVI_December', 'EVI_December', 'GNDVI_December', 'SAVI_December', 'NDWI_December', 'NDMI_December', 'RENDVI_December'
# ]

# X = df[feature_cols].values
# y = df['Output'].values

# # Handle missing S2 data (add missing flag)
# X = X.reshape(-1, 6, 10)
# s2_missing = np.all(X[:, :, 3:10] == 0, axis=2)  # True if all S2 features are 0
# X_new = np.zeros((X.shape[0], 6, 11))  # 10 features + 1 missing flag
# X_new[:, :, :10] = X
# X_new[:, :, 10] = s2_missing.astype(float)
# X = X_new

# # Normalize
# scaler = StandardScaler()
# X_flat = X.reshape(X.shape[0], -1)
# X_flat = scaler.fit_transform(X_flat)
# X = X_flat.reshape(X.shape[0], 6, 11)

# # Split
# X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
# X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
# y_train_tensor = torch.tensor(y_train, dtype=torch.long)
# X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
# y_val_tensor = torch.tensor(y_val, dtype=torch.long)

# train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
# val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
# train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=128)

# # Positional Encoding
# class PositionalEncoding(nn.Module):
#     def __init__(self, d_model, max_len=10):
#         super().__init__()
#         pe = torch.zeros(max_len, d_model)
#         position = torch.arange(0, max_len).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
#         pe[:, 0::2] = torch.sin(position * div_term)
#         pe[:, 1::2] = torch.cos(position * div_term)
#         self.pe = pe.unsqueeze(0)

#     def forward(self, x):
#         x = x + self.pe[:, :x.size(1), :].to(x.device)
#         return x

# # Label Smoothing Loss
# class LabelSmoothingCrossEntropy(nn.Module):
#     def __init__(self, smoothing=0.1):
#         super().__init__()
#         self.smoothing = smoothing

#     def forward(self, pred, target):
#         log_preds = nn.functional.log_softmax(pred, dim=1)
#         n_classes = pred.size(1)
#         true_dist = torch.zeros_like(log_preds).scatter_(1, target.unsqueeze(1), 1)
#         true_dist = true_dist * (1 - self.smoothing) + self.smoothing / n_classes
#         return torch.mean(torch.sum(-true_dist * log_preds, dim=1))

# # Transformer
# class TransformerClassifier(nn.Module):
#     def __init__(self, input_dim=11, seq_len=6, d_model=256, nhead=8, num_layers=4, num_classes=2, dropout=0.2):
#         super().__init__()
#         self.input_linear = nn.Linear(input_dim, d_model)
#         self.layer_norm = nn.LayerNorm(d_model)
#         self.pos_encoder = PositionalEncoding(d_model, max_len=seq_len)
#         encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True)
#         self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
#         self.classifier = nn.Sequential(
#             nn.Linear(d_model, 64),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(64, num_classes)
#         )

#     def forward(self, x, mask=None):
#         x = self.input_linear(x)
#         x = self.layer_norm(x)
#         x = self.pos_encoder(x)
#         if mask is not None:
#             x = self.transformer_encoder(x, src_key_padding_mask=mask)
#         else:
#             x = self.transformer_encoder(x)
#         x = x.mean(dim=1)  # Global average pooling
#         return self.classifier(x)

# # Setup
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = TransformerClassifier(input_dim=11).to(device)

# # Class weights
# class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
# class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
# criterion = LabelSmoothingCrossEntropy(smoothing=0.1)
# optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.1)
# scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

# # Early Stopping
# class EarlyStopping:
#     def __init__(self, patience=10):
#         super().__init__()
#         self.patience = patience
#         self.counter = 0
#         self.best_loss = float('inf')
#         self.early_stop = False

#     def step(self, val_loss):
#         if val_loss < self.best_loss:
#             self.best_loss = val_loss
#             self.counter = 0
#         else:
#             self.counter += 1
#             if self.counter >= self.patience:
#                 self.early_stop = True

# early_stopper = EarlyStopping(patience=10)

# # Training loop
# epochs = 100
# for epoch in tqdm(range(1, epochs + 1), desc="Training epochs"):
#     model.train()
#     train_loss, train_correct, total_train = 0.0, 0, 0
#     for inputs, targets in train_loader:
#         inputs, targets = inputs.to(device), targets.to(device)
#         # Create mask for missing S2 data
#         s2_missing = inputs[:, :, 10] == 1  # Missing flag
#         mask = s2_missing.to(device) if s2_missing.any() else None

#         optimizer.zero_grad()
#         outputs = model(inputs, mask=mask)
#         loss = criterion(outputs, targets)
#         loss.backward()
#         torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
#         optimizer.step()

#         train_loss += loss.item() * inputs.size(0)
#         _, predicted = torch.max(outputs, 1)
#         train_correct += (predicted == targets).sum().item()
#         total_train += targets.size(0)

#     train_acc = train_correct / total_train
#     train_loss = train_loss / total_train

#     model.eval()
#     val_loss, val_correct, total_val = 0.0, 0, 0
#     with torch.no_grad():
#         for inputs, targets in val_loader:
#             inputs, targets = inputs.to(device), targets.to(device)
#             s2_missing = inputs[:, :, 10] == 1
#             mask = s2_missing.to(device) if s2_missing.any() else None

#             outputs = model(inputs, mask=mask)
#             loss = criterion(outputs, targets)

#             val_loss += loss.item() * inputs.size(0)
#             _, predicted = torch.max(outputs, 1)
#             val_correct += (predicted == targets).sum().item()
#             total_val += targets.size(0)

#     val_acc = val_correct / total_val
#     val_loss = val_loss / total_val

#     print(f"Epoch {epoch}/{epochs} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

#     if val_loss < best_val_loss:
#         best_val_loss = val_loss
#         torch.save(model.state_dict(), best_model_path_pth)
#         with open(best_model_path_pkl, 'wb') as f:
#             pickle.dump(model, f)
#         print(f"✅ Saved best model at epoch {epoch} with Val Loss: {val_loss:.4f}")

#     scheduler.step(val_loss)
#     early_stopper.step(val_loss)
#     if early_stopper.early_stop:
#         print("⏹️ Early stopping triggered.")
#         break

# # Evaluate
# from sklearn.metrics import classification_report, confusion_matrix
# model.eval()
# y_pred = []
# y_true = []
# with torch.no_grad():
#     for inputs, targets in val_loader:
#         inputs, targets = inputs.to(device), targets.to(device)
#         s2_missing = inputs[:, :, 10] == 1
#         mask = s2_missing.to(device) if s2_missing.any() else None
#         outputs = model(inputs, mask=mask)
#         _, predicted = torch.max(outputs, 1)
#         y_pred.extend(predicted.cpu().numpy())
#         y_true.extend(targets.cpu().numpy())

# print("\nClassification Report:")
# print(classification_report(y_true, y_pred))
# print("\nConfusion Matrix:")
# print(confusion_matrix(y_true, y_pred))