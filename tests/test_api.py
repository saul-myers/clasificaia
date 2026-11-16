"""Pruebas automáticas de la API (se ejecutan con: pytest -q).
TestClient simula peticiones HTTP sin levantar un servidor. GitHub Actions las
ejecuta en cada push: si una falla, la imagen no se publica.
"""
from fastapi.testclient import TestClient
from app.main import app

cliente = TestClient(app)


def test_health():
    """El servicio responde correctamente."""
    r = cliente.get("/health")

    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_predict_vivienda():
    """Prueba una predicción válida."""

    r = cliente.post(
        "/predict",
        json={
            "MedInc": 8.3252,
            "HouseAge": 41,
            "AveRooms": 6.98,
            "AveBedrms": 1.02,
            "Population": 322,
            "AveOccup": 2.55,
            "Latitude": 37.88,
            "Longitude": -122.23
        }
    )

    assert r.status_code == 200
    assert "precio_estimado" in r.json()


def test_predict_invalido():
    """Datos inválidos deben devolver 422."""

    r = cliente.post(
        "/predict",
        json={
            "MedInc": -1,
            "HouseAge": 41,
            "AveRooms": 6.98,
            "AveBedrms": 1.02,
            "Population": 322,
            "AveOccup": 2.55,
            "Latitude": 37.88,
            "Longitude": -122.23
        }
    )

    assert r.status_code == 422


def test_calidad_minima():
    """Comprueba la calidad mínima del modelo."""

    datos = cliente.get("/health").json()

    assert datos["mae"] <= 0.55
    assert datos["r2"] >= 0.59


def test_llave_requerida(monkeypatch):

    import app.main as m

    monkeypatch.setattr(m, "API_KEY", "secreto-de-prueba")

    vivienda = {
        "MedInc": 8.3252,
        "HouseAge": 41,
        "AveRooms": 6.98,
        "AveBedrms": 1.02,
        "Population": 322,
        "AveOccup": 2.55,
        "Latitude": 37.88,
        "Longitude": -122.23
    }

    assert cliente.post(
        "/predict",
        json=vivienda
    ).status_code == 401

    r = cliente.post(
        "/predict",
        json=vivienda,
        headers={"x-api-key": "secreto-de-prueba"}
    )

    assert r.status_code == 200