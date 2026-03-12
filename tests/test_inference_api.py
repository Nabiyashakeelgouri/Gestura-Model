import numpy as np
from fastapi.testclient import TestClient


def _auth_headers(client: TestClient, user_payload: dict) -> dict:
    signup = client.post("/auth/signup", json=user_payload)
    assert signup.status_code == 201, signup.text

    login = client.post(
        "/auth/login",
        json={"email": user_payload["email"], "password": user_payload["password"]},
    )
    assert login.status_code == 200, login.text
    token = login.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


def test_frame_inference_contract(client: TestClient, unique_user_payload: dict, monkeypatch):
    headers = _auth_headers(client, unique_user_payload)

    monkeypatch.setattr(
        "backend.routes.inference._decode_frame",
        lambda _: np.zeros((64, 64, 3), dtype=np.uint8),
    )

    def fake_process_frame(**_kwargs):
        return {
            "status": "analyzing",
            "prediction": "hello",
            "confidence": 0.91,
            "sentence": "Hello.",
            "speak_text": "Hello.",
            "errors": [],
        }

    monkeypatch.setattr("backend.routes.inference.process_frame", fake_process_frame)
    monkeypatch.setattr(
        "backend.routes.inference.synthesize_to_audio_url",
        lambda _text: "/media/audio/mock.mp3",
    )

    response = client.post(
        "/inference/frame",
        headers=headers,
        json={"image": "data:image/jpeg;base64,ZmFrZQ==", "session_id": "session-test"},
    )
    assert response.status_code == 200, response.text

    payload = response.json()
    assert payload["mode"] in {"live", "record"}
    assert payload["status"] == "analyzing"
    assert payload["prediction"] == "hello"
    assert isinstance(payload["confidence"], float)
    assert payload["sentence"] == "Hello."
    assert payload["audio_url"] == "/media/audio/mock.mp3"
    assert "errors" in payload


def test_trial_frame_inference_contract_without_auth(client: TestClient, monkeypatch):
    monkeypatch.setattr(
        "backend.routes.inference._decode_frame",
        lambda _: np.zeros((64, 64, 3), dtype=np.uint8),
    )

    def fake_process_frame(**_kwargs):
        return {
            "status": "analyzing",
            "prediction": "hello",
            "confidence": 0.91,
            "sentence": "Hello.",
            "speak_text": "Hello.",
            "errors": [],
        }

    monkeypatch.setattr("backend.routes.inference.process_frame", fake_process_frame)
    monkeypatch.setattr(
        "backend.routes.inference.synthesize_to_audio_url",
        lambda _text: "/media/audio/mock.mp3",
    )

    response = client.post(
        "/inference/trial/frame",
        json={"image": "data:image/jpeg;base64,ZmFrZQ==", "session_id": "trial-session"},
    )
    assert response.status_code == 200, response.text

    payload = response.json()
    assert payload["mode"] == "live"
    assert payload["status"] == "analyzing"
    assert payload["prediction"] == "hello"
    assert isinstance(payload["confidence"], float)
    assert payload["sentence"] == "Hello."
    assert payload["audio_url"] == "/media/audio/mock.mp3"
    assert "errors" in payload


def test_frame_inference_rejects_bad_base64(
    client: TestClient, unique_user_payload: dict
):
    headers = _auth_headers(client, unique_user_payload)
    response = client.post(
        "/inference/frame",
        headers=headers,
        json={"image": "this-is-not-base64", "session_id": "session-test"},
    )
    assert response.status_code == 400


def test_video_inference_contract(client: TestClient, unique_user_payload: dict, monkeypatch):
    headers = _auth_headers(client, unique_user_payload)

    monkeypatch.setattr(
        "backend.routes.inference.write_encrypted_upload",
        lambda _upload_bytes, suffix: ("C:\\nonexistent\\encrypted" + suffix, object()),
    )
    monkeypatch.setattr(
        "backend.routes.inference.decrypt_to_temp_video",
        lambda _encrypted, _fernet, suffix: ("C:\\nonexistent\\plain" + suffix),
    )
    monkeypatch.setattr(
        "backend.routes.inference.process_video",
        lambda _path: {
            "status": "completed",
            "prediction": "yes",
            "confidence": 0.88,
            "sentence": "Yes.",
            "frames_processed": 42,
            "errors": [],
        },
    )
    monkeypatch.setattr(
        "backend.routes.inference.synthesize_to_audio_url",
        lambda _text: "/media/audio/mock-video.mp3",
    )

    response = client.post(
        "/inference/video",
        headers=headers,
        files={"file": ("demo.mp4", b"fake-bytes", "video/mp4")},
    )
    assert response.status_code == 200, response.text

    payload = response.json()
    assert payload["status"] == "completed"
    assert payload["prediction"] == "yes"
    assert payload["sentence"] == "Yes."
    assert payload["frames_processed"] == 42
    assert isinstance(payload["processing_ms"], int)
