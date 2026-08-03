# UIDAI Single-Finger App

This branch contains the original single-finger Flutter app and backend.

Backend default:
- http://<YOUR_LAN_IP>:5002

## Prerequisites
- Flutter SDK (stable)
- Android SDK / adb
- Python 3.10+ (recommended) for the backend

## Run Backend (local)

```bash
cd backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
PORT=5002 python app.py
```

Health check:
- http://127.0.0.1:5002/health

## Run Flutter App

```bash
flutter pub get
flutter run
```

In the app:
- Settings → set backend URL
  - Single: `http://<YOUR_LAN_IP>:5002`

## Models (Large Files)
The backend model files are intentionally not committed to git.
You must place them under:
- `backend/models/`

Expected model files:
- `best-new.pt`
- `bright_spot_detection.pt`
- `best_f1.pth`
- `liveness_model_v3.pt`
- `zero_dce_model.h5`
- `u2net_320x320_float32.tflite`

## Repo Branches (High Level)
- `feature/uidai-app` (this branch): original single-finger app
- `feature/uidai-unified`: unified app (single + slap)
- `feature/slap-preprocessing`: slap preprocessing snapshot
