# Gestura: AI-Powered Sign Language Translator (MVP)

## Run Backend

```bash
uvicorn backend.app:app --reload
```

## Install Dependencies

```bash
venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Environment Variables

Set these before running the backend:

- `JWT_SECRET`
- `JWT_ALGORITHM` (default: `HS256`)
- `ACCESS_TOKEN_EXPIRE_MINUTES` (default: `60`)
- `SMTP_HOST`
- `SMTP_PORT`
- `SMTP_USER`
- `SMTP_PASS`
- `SMTP_USE_TLS` (default: `true`)
- `DELETE_CONFIRM_BASE_URL` (default: `http://127.0.0.1:8000/user/delete/confirm`)
- `DELETE_TOKEN_EXPIRE_MINUTES` (default: `30`)
- `CORS_ORIGINS` (comma-separated)
- `ACTIVATION_HOLD_FRAMES` (default: `6`)
- `NO_HAND_COOLDOWN_FRAMES` (default: `12`)
- `COOLDOWN_SECONDS` (default: `1.5`)
- `HAND_MAX_NUM` (default: `2`)
- `HAND_DETECTION_CONFIDENCE` (default: `0.55`)
- `HAND_TRACKING_CONFIDENCE` (default: `0.50`)
- `FACE_DETECTION_CONFIDENCE` (default: `0.60`)
- `FACE_TRACKING_CONFIDENCE` (default: `0.50`)
- `MEDIAPIPE_MODEL_COMPLEXITY` (default: `1`)
- `RECORD_ENCRYPTION_KEY` (Fernet key for encrypted record uploads)
- `SECURE_TEMP_DIR` (optional temp directory for encrypted/decrypted transient files)
- `SECURE_DELETE_PASSES` (default: `1`)

Use [.env.example](c:/Users/nabiya/Downloads/Gestura-Model%20-%20Copy%20-%20Copy/.env.example) as the template.
The backend also auto-loads a local `.env` file from the repo root.

## Primary API

- `POST /auth/signup`
- `POST /auth/login`
- `GET /auth/profile`
- `POST /auth/logout`
- `GET /mode`
- `POST /mode`
- `POST /inference/frame`
- `POST /inference/video`
- `POST /user/delete`
- `POST /user/delete/request` (backward-compatible alias)

## Frontend

Open:

- `frontend/login.html`
- `frontend/dashboard.html`

For best browser permissions/CORS behavior, serve `frontend/` via a local static server (for example Live Server in VS Code).

## Run Tests

```bash
venv\Scripts\python.exe -m pytest
```

## Train Dynamic Model

```bash
python -m ai_engine.recognition.train_dynamic_model
```

Train static CNN model:

```bash
python -m ai_engine.recognition.train_static_cnn
```

Collect new dynamic sequences (per gesture label):

```bash
set GESTURE_LABEL=hello
python experiments/live_capture_prototype.py
```

Or collect all dynamic labels in one guided session (recommended):

```bash
python -m ai_engine.recognition.collect_dynamic_sequences --sequences-per-label 25 --min-hand-frames 20
```

Optional training environment variables:

- `DYNAMIC_EPOCHS` (default: `40`)
- `DYNAMIC_BATCH_SIZE` (default: `16`)
- `DYNAMIC_LEARNING_RATE` (default: `0.001`)
- `DYNAMIC_VAL_SPLIT` (default: `0.2`)
- `DYNAMIC_MIN_HAND_FRAMES_RATIO` (default: `0.4`)
- `ENABLE_LOW_LIGHT_HAND_FALLBACK` (default: `true`)
- `HAND_MIN_SPAN` (default: `0.06`)

## Verify SMTP Delivery

```bash
venv\Scripts\python.exe scripts\verify_smtp.py --to your_email@example.com
```
