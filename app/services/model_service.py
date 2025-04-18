from typing import List, Dict
from app.model.code import TransformerClassifier
import torch
import os
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path_pth = os.path.join(BASE_DIR, "../model/best_model.pth")

SCALER_MEAN = np.array([-9.839126792510282, -17.510648052429946, 1.837298725589949, 0.15274922776906527, 0.48816525750933715, 0.160986327243752, 0.22909554448392827, -0.160986327243752, 0.017381617424810727, 0.07010029670965835, 0.6269868659355704, -9.598363844967507, -16.854010864436948, 1.82316538702239, 0.17898137791958327, 0.6085305142004614, 0.17810046833996204, 0.2684377547395131, -0.17810046833996204, 0.031093134910432085, 0.08254912082093149, 0.612363038280355, -9.691781752456366, -16.455823828029494, 1.76427386446364, 0.45326338092983753, 1.428182805720329, 0.4350914031933361, 0.6798106780473133, -0.4350914031933361, 0.10019184958982862, 0.2065684131726553, 0.17585480750490803, -9.181006568563891, -15.55350511020221, 1.7557808471696084, 0.5658775894594584, 2.0915021695916285, 0.5140627293323577, 0.8487056897940831, -0.5140627293323577, 0.17761998770447968, 0.271062632966961, 0.07208096249922755, -9.08255351368742, -15.403223812234186, 1.765097327830467, 0.4642638357020022, 1.7032608816408523, 0.427470707140536, 0.6962973635308856, -0.427470707140536, 0.12400318314155141, 0.22154137600718402, 0.18908489368680748, -9.619612518669985, -16.09367518395693, 1.7158121245532227, 0.504677678953463, 1.2508877684647082, 0.5286352842169136, 0.7568981286337474, -0.5286352842169136, 0.022933140818149476, 0.233074286249691, 0.004972215488023426])  # Paste the 66 mean values here
SCALER_STD = np.array([2.1686930003835516, 2.627307794860598, 5.232871529802508, 0.2290347735222058, 28.92452705023297, 0.22291123430730975, 0.34350897543480313, 0.22291123430730975, 0.09156387512173914, 0.1152015646689817, 0.4836055581541931, 1.9823272301543517, 2.179477530956449, 7.54106517482163, 0.2480032768591597, 28.83754723076872, 0.23571259565146885, 0.3719565109613614, 0.23571259565146885, 0.09082709490352428, 0.12363981089290499, 0.4872109888247903, 1.9611007841350936, 1.9939694243159547, 8.569114535432035, 0.2500891614436016, 28.229554912827172, 0.22098674939051602, 0.375086088763201, 0.22098674939051602, 0.12201034096851056, 0.13403132854504862, 0.38069659071856404, 1.8262952082031452, 1.7074023353117382, 5.948316186350825, 0.20592923364232482, 61.554863405095006, 0.17170043977113397, 0.3088531190131438, 0.17170043977113397, 0.13080243978528233, 0.12368880084569701, 0.2586219196880232, 1.7660058870553106, 1.7465701010249866, 8.301815950975877, 0.2578665182188327, 41.49115989139571, 0.22750652835902654, 0.3867437043728367, 0.22750652835902654, 0.12696934378737637, 0.14158651713119524, 0.39157604199728596, 1.899388197703793, 1.876715600896326, 12.488285138306207, 0.15411494290564587, 18.349278662116173, 0.09556442085615781, 0.23112722068200497, 0.09556442085615781, 0.14193448314166726, 0.10234478943776813, 0.07033841454846287])   # Paste the 66 std values here

if any(SCALER_STD == 0):
    SCALER_STD = np.where(SCALER_STD == 0, 1e-10, SCALER_STD)

if SCALER_MEAN.shape != (66,) or SCALER_STD.shape != (66,):
    raise ValueError(f"Expected 66 features for mean and std, got {SCALER_MEAN.shape} and {SCALER_STD.shape}")

def manual_scale(X, mean, std):
    return (X - mean) / std

if not os.path.exists(model_path_pth):
    raise FileNotFoundError("Model file not found.")

model = TransformerClassifier(input_dim=11).to(device)
model.load_state_dict(torch.load(model_path_pth, map_location=device))
model.eval()

def get_model_prediction(data: List[Dict]) -> List[Dict]:
    try:
        all_keys = list(data[0].keys())
        lon_key = all_keys[0]
        lat_key = all_keys[1]
        feature_keys = [key for key in all_keys[2:]]
        expected_feature_count = 60

        if len(feature_keys) != expected_feature_count:
            raise ValueError(f"Expected {expected_feature_count} features, got {len(feature_keys)}")
        
        X = []
        lon_lat = []
        for sample in data:
            sample_values = [sample.get(col, 0.0) for col in feature_keys]
            X.append(sample_values)
            lon = sample.get(lon_key)
            lat = sample.get(lat_key)
            lon_lat.append((lon, lat))

        X = np.array(X, dtype=np.float32)

        if X.shape[1] != 60:
            raise ValueError(f"Each sample must have exactly 60 features, got {X.shape[1]}")
        X = X.reshape(-1, 6, 10)

        s2_missing = np.all(X[:, :, 3:10] == 0, axis=2)
        X_new = np.zeros((X.shape[0], 6, 11))
        X_new[:, :, :10] = X
        X_new[:, :, 10] = s2_missing.astype(float)

        X_flat = X_new.reshape(X_new.shape[0], -1)
        X_flat = manual_scale(X_flat, SCALER_MEAN, SCALER_STD)
        X_new = X_flat.reshape(X_new.shape[0], 6, 11)

        X_tensor = torch.tensor(X_new, dtype=torch.float32).to(device)
        mask = torch.tensor(s2_missing, dtype=torch.bool).to(device) if s2_missing.any() else None

        with torch.no_grad():
            outputs = model(X_tensor, mask=mask)
            _, predicted = torch.max(outputs, 1)
            predictions = predicted.cpu().numpy().tolist()

        result = [
            {"lon": float(lon), "lat": float(lat), "prediction": pred}
            for (lon, lat), pred in zip(lon_lat, predictions)
        ]

        with open('./predict_result.txt', 'w') as f:
            for item in result:
                f.write(f"lon: {item['lon']}, lat: {item['lat']}, prediction: {item['prediction']}\n")

        return result

    except Exception as e:
        raise ValueError(f"Error processing input data: {str(e)}")