import uuid
from typing import Dict

import pytest
from fastapi.testclient import TestClient

from backend.app import app


@pytest.fixture(scope="session")
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture()
def unique_user_payload() -> Dict[str, str]:
    suffix = uuid.uuid4().hex[:10]
    return {
        "username": f"user_{suffix}",
        "email": f"user_{suffix}@example.com",
        "password": "test12345",
    }

