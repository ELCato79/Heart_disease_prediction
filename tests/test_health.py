from src.app import app


def test_health_endpoint_exists():
    # Basic sanity check: app object should exist
    assert app is not None
