"""ClasificaIA: API de clasificación de flores Iris con FastAPI (proyecto base).
Endpoints:
GET / página web con el formulario (app/index.html)
GET /health estado del servicio y versión del modelo
POST /predict recibe 4 medidas y devuelve la especie y la confianza
GET /docs documentación automática de FastAPI
El modelo se entrena antes con modelo/entrenar.py y se lee de app/modelo.joblib.
"""
import hmac
import os
from pathlib import Path
import joblib
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
# ---------------------------------------------------------------- Modelo
CARPETA = Path(__file__).parent
# Se carga UNA sola vez al iniciar el servidor (no en cada petición: sería lento).
ARTEFACTO = joblib.load(os.getenv("MODELO_PATH", CARPETA / "modelo.joblib"))
# ---------------------------------------------------------------- Seguridad (Hito 4)
# MOSTRAR_DOCS=0 en producción oculta /docs (menos información para un atacante).
MOSTRAR_DOCS = os.getenv("MOSTRAR_DOCS", "1") == "1"
# API_KEY: si se define (variable de entorno en Render, nunca en el código), /predict
# exige la cabecera x-api-key con ese valor. Vacía = API abierta (Hitos 1 a 3).
API_KEY = os.getenv("API_KEY", "")
app = FastAPI(title="ClasificaIA", version=ARTEFACTO["version"],
docs_url="/docs" if MOSTRAR_DOCS else None,
redoc_url=None, openapi_url="/openapi.json" if MOSTRAR_DOCS else None)

def requiere_llave(x_api_key: str = Header(default="")):
    """Dependencia de FastAPI: 401 si falta la llave y 403 si es incorrecta."""
    if not API_KEY:
        return # sin llave configurada: acceso libre
    if not x_api_key:
        raise HTTPException(401, "Falta la cabecera x-api-key")
    if not hmac.compare_digest(x_api_key, API_KEY): # comparación en tiempo constante
        raise HTTPException(403, "Llave inválida")
# ---------------------------------------------------------------- Entrada
class Vivienda(BaseModel):
    """Variables del conjunto California Housing."""

    MedInc: float = Field(gt=0)
    HouseAge: float = Field(ge=0)
    AveRooms: float = Field(gt=0)
    AveBedrms: float = Field(gt=0)
    Population: float = Field(gt=0)
    AveOccup: float = Field(gt=0)
    Latitude: float = Field(ge=32, le=42)
    Longitude: float = Field(ge=-125, le=-114)
# ---------------------------------------------------------------- Endpoints
@app.get("/health")
def salud():
    """Lo consultan Docker, Kubernetes y Render para saber si el servicio está vivo."""
    return {
        "status": "ok",
        "version": ARTEFACTO["version"],
        "mae": ARTEFACTO["mae"],
        "r2": ARTEFACTO["r2"]
    }

@app.post("/predict", dependencies=[Depends(requiere_llave)])
def predecir(v: Vivienda):
    """Devuelve la clase más probable y su probabilidad (200), o 422 si los datos no
    son válidos."""
    # El modelo espera una lista de filas; aquí hay una sola fila con 4 columnas
    # en el MISMO orden que en el entrenamiento.
    x = [[
    v.MedInc,
    v.HouseAge,
    v.AveRooms,
    v.AveBedrms,
    v.Population,
    v.AveOccup,
    v.Latitude,
    v.Longitude
]]
    precio = float(ARTEFACTO["modelo"].predict(x)[0])

    return {
    "precio_estimado": round(precio, 3),
    "unidad": ARTEFACTO["unidad"]
    }
@app.get("/", response_class=HTMLResponse)
def inicio():
    """Sirve la página web; esta y la API comparten origen, así que no hay CORS."""
    return (CARPETA / "index.html").read_text(encoding="utf-8")