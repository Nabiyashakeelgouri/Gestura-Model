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


def test_auth_and_mode_flow(client: TestClient, unique_user_payload: dict):
    headers = _auth_headers(client, unique_user_payload)

    profile = client.get("/auth/profile", headers=headers)
    assert profile.status_code == 200, profile.text
    assert profile.json()["email"] == unique_user_payload["email"]

    mode = client.get("/mode", headers=headers)
    assert mode.status_code == 200, mode.text
    assert mode.json()["mode"] in {"live", "record"}

    switch = client.post("/mode", headers=headers, json={"mode": "record"})
    assert switch.status_code == 200, switch.text
    assert switch.json() == {"mode": "record"}

    back = client.post("/mode", headers=headers, json={"mode": "live"})
    assert back.status_code == 200, back.text
    assert back.json() == {"mode": "live"}


def test_mode_requires_auth(client: TestClient):
    response = client.get("/mode")
    assert response.status_code == 401
