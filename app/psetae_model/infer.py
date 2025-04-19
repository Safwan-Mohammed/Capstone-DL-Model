import torch
import torch.utils.data as data
import numpy as np
import os, shutil
from typing import List, Dict

from app.psetae_model.stclassifier import PseTae
from app.psetae_model.dataset import PixelSetData

def recursive_todevice(x, device):
    if isinstance(x, torch.Tensor):
        return x.to(device)
    else:
        return [recursive_todevice(c, device) for c in x]

def prepare_model_and_loader(config, dataset_folder: str):
    
    mean = np.array([-9.495785, -16.39869, 1.788546, 0.51690346, 1.705192,
                     0.5024654, 0.76597273, -0.5024654, 0.10062386, 0.23882234], dtype=np.float32)
    std = np.array([1.9506195, 2.1658468, 8.339499, 0.14740352, 37.201633,
                    0.09887332, 0.22141898, 0.09887332, 0.12670133, 0.09622738], dtype=np.float32)
    norm = (mean, std)
    dt = PixelSetData(dataset_folder, npixel=config['npixel'], norm=norm, extra_feature=None)
    dl = data.DataLoader(dt, batch_size=config['batch_size'], num_workers=config['num_workers'])

    model_config = dict(
        input_dim=config['input_dim'],
        mlp1=config['mlp1'],
        pooling=config['pooling'],
        mlp2=config['mlp2'],
        n_head=config['n_head'],
        d_k=config['d_k'],
        mlp3=config['mlp3'],
        dropout=config['dropout'],
        T=config['T'],
        len_max_seq=config['lms'],
        positions=None,
        mlp4=config['mlp4'],
        with_extra=False,
        extra_size=None
    )

    model = PseTae(**model_config)
    model = model.to(config['device'])

    weight_path = os.path.join(os.path.dirname(__file__), 'model.pth.tar')
    if not os.path.exists(weight_path):
        raise FileNotFoundError(f"Weight file not found: {weight_path}")
    checkpoint = torch.load(weight_path, map_location=config['device'])
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    return model, dl

def predict(model, loader, config):
    predictions = []
    device = torch.device(config['device'])

    for x in loader:
        x = recursive_todevice(x, device)
        with torch.no_grad():
            prediction = model(x)
        # Convert int64 to Python int
        y_p = [int(y) for y in prediction.argmax(dim=1).cpu().numpy()]
        predictions.extend(y_p)

    return predictions

def get_model_prediction(data: List[Dict]) -> List[Dict]:
    config = {
        'dataset_folder': os.path.join(os.path.dirname(__file__), '..', 'PRED_DATA'),
        'num_workers': 0,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'batch_size': 512,
        'npixel': 1,
        'input_dim': 10,
        'mlp1': [10, 32, 64],
        'pooling': 'mean_std',
        'mlp2': [128, 128],
        'n_head': 4,
        'd_k': 32,
        'mlp3': [512, 128, 128],
        'T': 1000,
        'lms': None,
        'dropout': 0.2,
        'mlp4': [128, 64, 32, 2]
    }

    npy_files = []
    results = []
    for row in data:
        all_keys = list(row.keys())
        if len(all_keys) < 62:  # 60 features + 2 non-feature keys
            raise ValueError(f"Expected at least 62 keys (2 non-features + 60 features), got {len(all_keys)}")
        
        col1_key, col2_key = all_keys[0], all_keys[1]
        col1_value, col2_value = str(row[col1_key]), str(row[col2_key])
        col1_value = col1_value.replace(",", "_").replace("/", "_").replace("\\", "_")
        col2_value = col2_value.replace(",", "_").replace("/", "_").replace("\\", "_")
        file_name = f"{col1_value}_{col2_value}.npy"
        file_path = os.path.join(config['dataset_folder'], 'DATA', file_name)
        npy_files.append(file_path)
        
        results.append({
            col1_key: col1_value,
            col2_key: col2_value,
            "prediction": None
        })
    model, loader = prepare_model_and_loader(config, config['dataset_folder'])
    predictions = predict(model, loader, config)

    # Assign predictions to results
    for idx, prediction in enumerate(predictions):
        results[idx]["prediction"] = int(prediction) 
    
    try:
        for subfolder in ['DATA', 'META']:
            folder_path = os.path.join(config['dataset_folder'], subfolder)
            if os.path.exists(folder_path):
                shutil.rmtree(folder_path)
    except Exception as e:
        print(f"Warning: Failed to delete folders: {e}")
    
    return results