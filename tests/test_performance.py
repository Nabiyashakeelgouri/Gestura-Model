import time

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


def test_frame_endpoint_latency_under_threshold(
    client: TestClient, unique_user_payload: dict, monkeypatch
):
    headers = _auth_headers(client, unique_user_payload)

    monkeypatch.setattr(
        "backend.routes.inference._decode_frame",
        lambda _: np.zeros((64, 64, 3), dtype=np.uint8),
    )
    monkeypatch.setattr(
        "backend.routes.inference.process_frame",
        lambda **_kwargs: {
            "status": "standby",
            "prediction": "A",
            "confidence": 0.99,
            "sentence": "",
            "speak_text": "",
            "errors": [],
        },
    )
    monkeypatch.setattr("backend.routes.inference.synthesize_to_audio_url", lambda _text: None)

    started = time.perf_counter()
    response = client.post(
        "/inference/frame",
        headers=headers,
        json={"image": "data:image/jpeg;base64,ZmFrZQ==", "session_id": "perf-session"},
    )
    elapsed_ms = (time.perf_counter() - started) * 1000

    assert response.status_code == 200, response.text
    assert elapsed_ms < 500
