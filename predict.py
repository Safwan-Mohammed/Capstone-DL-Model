import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
from core import handle_cloud_masked_data, create_ann_model  # Import necessary functions

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

def load_scaler(scaler_path='scaler.pkl'):
    """Load the scaler used during training."""
    import pickle
    try:
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        print(f"Loaded scaler from {scaler_path}")
        return scaler
    except FileNotFoundError:
        print(f"Scaler file {scaler_path} not found. Please provide a trained scaler.")
        raise

def process_and_predict(model, input_data, scaler):
    """
    Process a single input (10 values) and predict classification output.
    """
    # Ensure input is a numpy array with shape (1, 10)
    input_data = np.array(input_data).reshape(1, -1)
    if input_data.shape[1] != 10:
        raise ValueError("Input must contain exactly 10 values (VV, VH, VH_VV, NDVI, EVI, GNDVI, SAVI, NDWI, NDMI, RENDVI)")
    
    # Handle cloud-masked data
    processed_data, cloud_mask = handle_cloud_masked_data(input_data)
    if cloud_mask[0]:
        print("Warning: Input is cloud-masked (all Sentinel-2 values are 0). Prediction relies on Sentinel-1 data only.")
    
    # Scale the input using the trained scaler
    scaled_data = scaler.transform(processed_data)
    
    # Predict
    prediction = model.predict(scaled_data, batch_size=1)[0][0]  # Get single probability
    class_output = 1 if prediction >= 0.5 else 0  # Binary classification threshold
    
    print(f"\nPrediction details:")
    print(f"Raw probability: {prediction:.4f}")
    print(f"Classification output: {class_output} ({'Positive' if class_output == 1 else 'Negative'})")
    
    return prediction, class_output

if __name__ == "__main__":
    # Load the model architecture and weights
    input_dim = 10  # Fixed for your 10 features
    model = create_ann_model(input_dim)  # Use the same architecture from training
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    # Load weights from checkpoint
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
    
    # Load the scaler used during training
    scaler = load_scaler('scaler.pkl')  # Save this during training if not already done
    
    # Example input: 10 values representing VV, VH, VH_VV, NDVI, EVI, GNDVI, SAVI, NDWI, NDMI, RENDVI
    example_input = [0.1, 0.2, 0.05, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]  # Replace with your input
    print(f"Input values: {example_input}")
    
    # Predict
    prediction, class_output = process_and_predict(model, example_input, scaler)