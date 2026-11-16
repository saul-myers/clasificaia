"""Entrena el modelo de ClasificaIA y lo guarda en un archivo (proyecto base).
ClasificaIA clasifica flores Iris con KNN (k vecinos más cercanos). Es el EJEMPLO
del proyecto: cada equipo cambia los datos y el algoritmo por los suyos.
Uso: python modelo/entrenar.py [archivo_salida]
"""
import sys
from datetime import datetime, timezone
import joblib
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
# ---------------------------------------------------------------- Configuración
# Archivo donde se guarda el modelo; la API lo lee al iniciar.
SALIDA = sys.argv[1] if len(sys.argv) > 1 else "app/modelo.joblib"
SEMILLA = 42 # misma semilla = mismo resultado (reproducible)
# ---------------------------------------------------------------- 1. Datos
# load_iris trae 150 flores con 4 medidas (cm) y su especie (0, 1 o 2).
# >>> En su proyecto: aquí cargan SUS datos (CSV, dataset de scikit-learn, etc.).
datos = fetch_california_housing()

X = datos.data
y = datos.target

NOMBRES_VARIABLES = list(datos.feature_names)

# 70 % para entrenar y 30 % para evaluar. stratify mantiene la proporción de
# cada clase en ambos conjuntos; el modelo nunca ve los datos de prueba.
X_tr, X_te, y_tr, y_te = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=SEMILLA
)
# ---------------------------------------------------------------- 2. Modelo
# Pipeline: primero escala las medidas (media 0, desviación 1) y luego aplica
# KNN. Escalar importa en KNN porque compara distancias entre puntos.
# >>> En su proyecto: cambian KNeighborsClassifier por su algoritmo.
modelo = make_pipeline(
    StandardScaler(),
    LinearRegression()
)
modelo.fit(X_tr, y_tr)
# ---------------------------------------------------------------- 3. Evaluación
predicciones = modelo.predict(X_te)

mae = mean_absolute_error(y_te, predicciones)
r2 = r2_score(y_te, predicciones)

print(f"MAE = {mae:.3f}")
print(f"R2 = {r2:.3f}")
# ---------------------------------------------------------------- 4. Guardar
# Se guarda un diccionario: el modelo y los datos que la API necesita mostrar.
joblib.dump({
    "modelo": modelo,
    "variables": NOMBRES_VARIABLES,
    "mae": round(float(mae), 4),
    "r2": round(float(r2), 4),
    "unidad": "cientos de miles de dólares",
    "version": datetime.now(timezone.utc).strftime("%Y %m %d %H %M"),
}, SALIDA)
print(f"MAE={mae:.3f}")
print(f"R2={r2:.3f}")
print(f"guardado en {SALIDA}")