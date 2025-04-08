import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Input, Concatenate, Lambda
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import csv

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Configure TensorFlow to use GPU with CUDA and mixed precision
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        print(f"GPU {physical_devices[0]} is available and configured for use.")
        from tensorflow.keras.mixed_precision import set_global_policy
        set_global_policy('mixed_float16')
        print("Mixed precision training enabled.")
    except RuntimeError as e:
        print(f"Error configuring GPU: {e}")
else:
    print("No GPU found. Training will use CPU.")

# Create logging directories
os.makedirs('logs', exist_ok=True)
os.makedirs('models/best_model', exist_ok=True)
os.makedirs('models/weights', exist_ok=True)

def load_data_in_chunks(file_path, chunk_size=1000000):
    """Load large dataset in chunks to manage memory."""
    chunks = []
    for chunk in pd.read_csv(file_path, chunksize=chunk_size):
        chunks.append(chunk)
    df = pd.concat(chunks, ignore_index=True)
    return df

def handle_cloud_masked_data(X):
    """Properly handle Sentinel-2 data that is unavailable due to cloud masking."""
    s1_indices = [0, 1, 2]  # VV, VH, VH_VV
    s2_indices = [3, 4, 5, 6, 7, 8, 9]  # NDVI, EVI, GNDVI, SAVI, NDWI, NDMI, RENDVI
    
    # Identify samples where all S2 values are zeros (cloud masked)
    s2_data = X[:, s2_indices]
    cloud_mask = np.all(s2_data == 0, axis=1)
    
    print(f"Found {np.sum(cloud_mask)} samples with cloud-masked Sentinel-2 data out of {X.shape[0]} total samples.")
    
    # Option 1: For cloudy pixels, replace S2 with mean values from non-cloudy pixels
    if np.sum(cloud_mask) > 0 and np.sum(~cloud_mask) > 0:
        # Calculate mean values from non-cloudy pixels
        s2_means = np.mean(X[~cloud_mask][:, s2_indices], axis=0)
        
        # Apply these means to cloudy pixels
        for i, idx in enumerate(s2_indices):
            X[cloud_mask, idx] = s2_means[i]
    
    return X, cloud_mask

def preprocess_data(file_path):
    """Load and preprocess the dataset with validation split."""
    df = load_data_in_chunks(file_path)
    feature_columns = ['VV', 'VH', 'VH_VV', 'NDVI', 'EVI', 'GNDVI', 'SAVI', 'NDWI', 'NDMI', 'RENDVI']
    target_column = 'Output'
    
    X = df[feature_columns].values
    y = df[target_column].values.reshape(-1, 1)
    
    # Handle cloud-masked data
    X, cloud_mask = handle_cloud_masked_data(X)
    
    # Split into train+val and test
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Split train+val into train and validation
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.1, random_state=42)
    
    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    
    return X_train, X_val, X_test, y_train, y_val, y_test, scaler, cloud_mask

def create_ann_model(input_dim):
    """Create an improved ANN model with better handling of the different sensor data."""
    # Standard model for direct input
    inputs = Input(shape=(input_dim,))
    
    # First layer processes all data
    x = Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001))(inputs)
    x = Dropout(0.2)(x)

    x = Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.001))(x)
    x = Dropout(0.2)(x)
    
    # Output layer
    outputs = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    return model

def create_dual_input_model():
    """Create a model with separate pathways for S1 and S2 data."""
    # Sentinel-1 pathway (VV, VH, VH_VV)
    s1_input = Input(shape=(3,), name='sentinel1_input')
    s1_dense = Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.001))(s1_input)
    s1_dense = Dropout(0.2)(s1_dense)
    
    # Sentinel-2 pathway (NDVI, EVI, GNDVI, SAVI, NDWI, NDMI, RENDVI)
    s2_input = Input(shape=(7,), name='sentinel2_input')
    s2_dense = Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001))(s2_input)
    s2_dense = Dropout(0.2)(s2_dense)
    
    # Merge pathways
    merged = Concatenate()([s1_dense, s2_dense])
    
    # Continue with regular layers
    x = Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001))(merged)
    x = Dropout(0.2)(x)
    x = Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.001))(x)
    x = Dropout(0.2)(x)
    output = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=[s1_input, s2_input], outputs=output)
    return model

def train_model(X_train, y_train, X_val, y_val, X_test, y_test, input_dim, 
                batch_size=10000, max_epochs=100):
    """Train model with constant learning rate and Keras callbacks."""
    # File paths
    best_model_file = 'models/best_model/best_model.keras'
    weights_file = 'models/weights/model_weights.weights.h5'
    log_file = 'logs/training_metrics.csv'
    
    # Initialize model
    model = create_ann_model(input_dim)
    optimizer = Adam(learning_rate=0.001)  # Fixed learning rate
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    
    # Define callbacks
    csv_logger = tf.keras.callbacks.CSVLogger(log_file, append=True)
    checkpoint = ModelCheckpoint(
        filepath=weights_file,
        save_weights_only=True,
        save_freq='epoch',
        verbose=1
    )
    best_model_checkpoint = ModelCheckpoint(
        filepath=best_model_file,
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    )
    # Removed ReduceLROnPlateau
    
    # Train the model
    history = model.fit(
        X_train, y_train,
        batch_size=batch_size,
        epochs=max_epochs,
        validation_data=(X_val, y_val),
        callbacks=[csv_logger, checkpoint, best_model_checkpoint, early_stopping],
        verbose=1
    )
    
    # Final evaluation on test set
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=1)
    print(f"Final Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
    
    # Save final metrics
    with open('logs/final_metrics.txt', 'w') as f:
        f.write(f"test_loss: {test_loss}\n")
        f.write(f"test_accuracy: {test_accuracy}\n")
        f.write(f"epochs_trained: {len(history.history['loss'])}\n")
    
    return model, test_accuracy, history

if __name__ == "__main__":
    file_path = './balanced_dataset.csv'  # Your dataset path
    X_train, X_val, X_test, y_train, y_val, y_test, scaler, cloud_mask = preprocess_data(file_path)
    print(f"Dataset loaded: {X_train.shape[0]} training rows, {X_val.shape[0]} validation rows")
    
    input_dim = X_train.shape[1]
    model, test_accuracy, history = train_model(
        X_train, y_train, X_val, y_val, X_test, y_test, input_dim,
        batch_size=10000, max_epochs=20
    )