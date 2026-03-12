from fastapi.testclient import TestClient


def test_account_delete_confirmation_flow(
    client: TestClient, unique_user_payload: dict, monkeypatch
):
    signup = client.post("/auth/signup", json=unique_user_payload)
    assert signup.status_code == 201, signup.text

    login = client.post(
        "/auth/login",
        json={"email": unique_user_payload["email"], "password": unique_user_payload["password"]},
    )
    assert login.status_code == 200, login.text
    token = login.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    request_delete = client.post("/user/delete", headers=headers)
    assert request_delete.status_code == 200, request_delete.text
    assert request_delete.json()["status"] == "deleted"

    relogin = client.post(
        "/auth/login",
        json={"email": unique_user_payload["email"], "password": unique_user_payload["password"]},
    )
    assert relogin.status_code == 404


def test_account_delete_request_alias_also_deletes(client: TestClient, unique_user_payload: dict):
    signup = client.post("/auth/signup", json=unique_user_payload)
    assert signup.status_code == 201, signup.text

    login = client.post(
        "/auth/login",
        json={"email": unique_user_payload["email"], "password": unique_user_payload["password"]},
    )
    assert login.status_code == 200, login.text
    token = login.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    request_delete = client.post("/user/delete/request", headers=headers)
    assert request_delete.status_code == 200, request_delete.text
    payload = request_delete.json()
    assert payload["status"] == "deleted"
