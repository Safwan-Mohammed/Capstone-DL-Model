import torch
import torch.utils.data as data
import numpy as np
import os
import pickle as pkl

from app.psetae_model.stclassifier import PseTae
from app.psetae_model.dataset import PixelSetData

def recursive_todevice(x, device):
    if isinstance(x, torch.Tensor):
        return x.to(device)
    else:
        return [recursive_todevice(c, device) for c in x]

def prepare_model_and_loader(config):
    mean = np.array([-9.495785, -16.39869, 1.788546, 0.51690346, 1.705192,
                     0.5024654, 0.76597273, -0.5024654, 0.10062386, 0.23882234], dtype=np.float32)
    std = np.array([1.9506195, 2.1658468, 8.339499, 0.14740352, 37.201633,
                    0.09887332, 0.22141898, 0.09887332, 0.12670133, 0.09622738], dtype=np.float32)
    norm = (mean, std)

    dt = PixelSetData(config['dataset_folder'], labels='labels', npixel=config['npixel'],
                      norm=norm, extra_feature=None)
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

    weight_path = os.path.join(os.path.dirname(__file__), f'Fold_1', 'model.pth.tar')
    if not os.path.exists(weight_path):
        raise FileNotFoundError(f"Weight file not found: {weight_path}")
    checkpoint = torch.load(weight_path, map_location=config['device'])
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    return model, dl

def predict(model, loader, config):
    record = []
    device = torch.device(config['device'])

    for x, y in loader:
        y_true = list(map(int, y))
        x = recursive_todevice(x, device)
        with torch.no_grad():
            prediction = model(x)
        y_p = list(prediction.argmax(dim=1).cpu().numpy())
        record.append(np.stack([y_true, y_p], axis=1))

    record = np.concatenate(record, axis=0)
    os.makedirs(config['res_dir'], exist_ok=True)
    output_path = os.path.join(config['res_dir'], 'predictions_ytrue_ypred.npy')
    np.save(output_path, record)
    return record

def main(config):
    model, loader = prepare_model_and_loader(config)
    record = predict(model, loader, config)
    return record

if __name__ == '__main__':
    config = {
        'dataset_folder': os.path.join(os.path.dirname(__file__), 'PRED_DATA'),
        'res_dir': os.path.join(os.path.dirname(__file__), 'PRED_DATA'),
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
    main(config)