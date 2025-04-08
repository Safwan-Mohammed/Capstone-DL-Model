import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from core import handle_cloud_masked_data, preprocess_data, create_ann_model  # Import necessary functions

# Set random seed for reproducibility
tf.random.set_seed(42)

# Configure TensorFlow to use GPU with CUDA
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        print(f"GPU {physical_devices[0]} is available and configured for use.")
        from tensorflow.keras.mixed_precision import set_global_policy
        set_global_policy('mixed_float16')
        print("Mixed precision enabled for predictions.")
    except RuntimeError as e:
        print(f"Error configuring GPU: {e}")
else:
    print("No GPU found. Predictions will use CPU.")

def predict_with_edge_case_handling(model, X_data, cloud_mask):
    """
    Make predictions with special handling for cloud-masked data.
    For cloud-masked data (all Sentinel-2 values are 0), rely only on Sentinel-1 features.
    """
    predictions = model.predict(X_data, batch_size=64)
    
    masked_predictions = predictions[cloud_mask]
    non_masked_predictions = predictions[~cloud_mask]
    
    print(f"\nPrediction statistics:")
    print(f"Cloud-masked samples: {len(masked_predictions)}, "
          f"Mean prediction: {np.mean(masked_predictions):.4f}")
    print(f"Non-masked samples: {len(non_masked_predictions)}, "
          f"Mean prediction: {np.mean(non_masked_predictions):.4f}")
    
    return predictions

def plot_learning_curves(history):
    """Plot and save learning curves from a provided history object."""
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('learning_curves.png')
    plt.close()
    
    print("Learning curves saved as 'learning_curves.png'")

def process_new_data(new_data, scaler, model):
    """
    Process new data for prediction, including edge case handling.
    """
    processed_data, cloud_mask = handle_cloud_masked_data(new_data)
    scaled_data = scaler.transform(processed_data)
    predictions = model.predict(scaled_data, batch_size=64)
    return predictions, cloud_mask

if __name__ == "__main__":
    # Load training data to get X_test, y_test, scaler, and cloud_mask
    file_path = './balanced_dataset.csv'  # Use your actual dataset path
    X_train, X_val, X_test, y_train, y_val, y_test, scaler, cloud_mask = preprocess_data(file_path)
    print(f"Dataset loaded: {X_train.shape[0]} training rows, {X_val.shape[0]} validation rows, {X_test.shape[0]} test rows")
    
    # Load the model architecture and weights
    input_dim = X_test.shape[1]
    model = create_ann_model(input_dim)  # Use the same architecture from training
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    # Load weights from checkpoint (choose one of the following)
    weights_path = 'models/weights/model_weights.weights.h5'  # Last epoch weights
    best_model_path = 'models/best_model/best_model.keras'   # Best model with architecture
    
    try:
        # Option 1: Load weights only (requires model architecture defined)
        model.load_weights(weights_path)
        print(f"Loaded weights from {weights_path}")
    except:
        # Option 2: Load full model (if weights-only fails or architecture is in .keras file)
        model = load_model(best_model_path)
        print(f"Loaded full model from {best_model_path}")
    
    # Make predictions on test data
    test_cloud_mask = cloud_mask[len(X_train) + len(X_val):]  # Slice cloud_mask for test set
    predictions = predict_with_edge_case_handling(model, X_test, test_cloud_mask)
    
    # Evaluate model on test data
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=1)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
    
    # Load training history from CSV (assumes it was saved during training)
    try:
        history_df = pd.read_csv('logs/training_metrics.csv')
        history = {
            'loss': history_df['loss'].values,
            'val_loss': history_df['val_loss'].values,
            'accuracy': history_df['accuracy'].values,
            'val_accuracy': history_df['val_accuracy'].values
        }
        plot_learning_curves(history)
    except FileNotFoundError:
        print("Training metrics CSV not found. Skipping learning curves.")
    
    # Example of processing new data
    new_data = np.random.rand(500, 10)  # Replace with actual new data if available
    predictions_new, _ = process_new_data(new_data, scaler, model)
    print(f"New data predictions shape: {predictions_new.shape}")