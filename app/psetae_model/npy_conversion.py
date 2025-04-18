import numpy as np
import os
import json
from typing import List, Dict

def convert_to_npy(data: List[Dict]) -> str:

    data_dir = os.path.join(os.path.dirname(__file__), '..', 'PRED_DATA', 'DATA')
    meta_dir = os.path.join(os.path.dirname(__file__), '..', 'PRED_DATA', "META")

    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(meta_dir, exist_ok=True)

    channels = ["VV", "VH", "VH_VV", "NDVI", "EVI", "GNDVI", "SAVI", "NDWI", "NDMI", "RENDVI"]
    dates = ["2019-07-01", "2019-08-01", "2019-09-01", "2019-10-01", "2019-11-01", "2019-12-01"]

    dates_dict = {}

    for idx, row in enumerate(data):
        all_keys = list(row.keys())
        if len(all_keys) < 62:  
            raise ValueError(f"Expected at least 62 keys (2 non-features + 60 features), got {len(all_keys)}")
        
        col1, col2 = all_keys[0], all_keys[1]
        col1_value, col2_value = str(row[col1]), str(row[col2])
        
        col1_value = col1_value.replace(",", "_").replace("/", "_").replace("\\", "_")
        col2_value = col2_value.replace(",", "_").replace("/", "_").replace("\\", "_")
        sample_id = f"{col1_value}_{col2_value}"

        feature_keys = all_keys[2:62]
        if len(feature_keys) != 60:
            raise ValueError(f"Expected 60 feature keys, got {len(feature_keys)}")

        features = [row[k] for k in feature_keys]

        pixel_array = np.zeros((6, 10, 1), dtype=np.float32)
        for t in range(6):
            for c in range(10):
                feature_idx = t * 10 + c
                pixel_array[t, c, 0] = features[feature_idx]

        npy_path = os.path.join(data_dir, f"{sample_id}.npy")
        np.save(npy_path, pixel_array)

        dates_dict[sample_id] = dates

    with open(os.path.join(meta_dir, 'dates.json'), 'w') as f:
        json.dump(dates_dict, f, indent=4)
