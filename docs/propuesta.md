# Hito 1: Propuesta de Proyecto — Estimador de Precio de Vivienda

**Materia:** Cómputo en la Nube – Unidad 3 – Proyecto de la materia  
**Fecha:** 11 de noviembre de 2026  
**Equipo:** Integrante A (Saúl GonzalezZz), Integrante B [Adiel.f1], Integrante C [Maria Jose]  

---

## 1. Problema y Usuarios
El modelo resuelve el problema de estimar el valor mediano de mercado de una propiedad inmobiliaria a partir de indicadores demográficos y geográficos de su zona. Ayuda a **agentes inmobiliarios, compradores particulares y analistas de inversión** a tomar decisiones fundamentadas sobre fijación de precios de venta, evaluación de ofertas de compra y detección de zonas subvaloradas sin depender únicamente de avalúos manuales lentos.

## 2. Datos
* **Fuente:** Conjunto de datos *California Housing* integrado en `scikit-learn` mediante la función `fetch_california_housing()` (`https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html`), el cual se descarga automáticamente la primera vez que se ejecuta y queda almacenado en caché local. Proviene originalmente del repositorio StatLib de Carnegie Mellon (`http://lib.stat.cmu.edu/datasets/`) con datos del Censo de EE. UU. de 1990 (no es de la región local).
* **Licencia:** Dominio público (datos gubernamentales del U.S. Census Bureau; librería bajo licencia BSD 3-Clause).
* **Número de registros:** 20,640 instancias (distritos censales) y 8 atributos.
* **Variables de entrada ($X$):** `MedInc` (ingreso mediano), `HouseAge` (antigüedad mediana), `AveRooms` (promedio de habitaciones), `AveBedrms` (promedio de dormitorios), `Population` (población), `AveOccup` (ocupación promedio), `Latitude` (latitud) y `Longitude` (longitud).
* **Variable a predecir ($y$):** `MedHouseVal` (valor mediano de la vivienda en el distrito, expresado en cientos de miles de dólares).

## 3. Algoritmo
Se empleará **Regresión Lineal Múltiple** (`LinearRegression` de `scikit-learn`). Es adecuado porque la variable objetivo (`MedHouseVal`) es continua y numérica, lo que define una tarea clásica de **regresión supervisada**. Además, ofrece un tiempo de entrenamiento instantáneo, baja huella de memoria y alta interpretabilidad para medir el impacto directo de cada variable sobre el precio.

## 4. Métrica Objetivo
* **Métrica 1 — MAE (Error Absoluto Medio):** **MAE $\le 0.55$** (equivalente a un error promedio máximo de $\$55,000$ dólares por vivienda en el conjunto de prueba).
* **Métrica 2 — $R^2$ (Coeficiente de Determinación):** **$R^2 \ge 0.60$** (el modelo debe explicar al menos el 60% de la variabilidad del precio).

## 5. Recursos de Cómputo
* **Medición local (Laptop / Codespace):** En una ejecución de referencia sobre un equipo local (CPU Intel/AMD multicore, 8–16 GB RAM), la descarga inicial toma unos segundos (posteriormente lee de caché), el entrenamiento del modelo toma **$< 0.05$ segundos** y el consumo máximo de memoria RAM es de **$\approx 45\text{ MB}$** (el archivo serializado `.joblib` pesa $< 5\text{ KB}$).
* **Máquina necesaria para servir:** Al ser un modelo ligero sin requerimientos de GPU, basta con una instancia mínima en la nube (por ejemplo, **1 vCPU y 512 MB de RAM**, como el plan gratuito/básico de Render, Railway o un contenedor Docker ligero en Kubernetes) para atender peticiones REST en milisegundos. *(Nota: Para servir en producción no se vuelve a llamar a `fetch_california_housing()`, solo se carga el modelo entrenado).*

## 6. Tres Riesgos Iniciales
1. **Relaciones espaciales no lineales:** La latitud y longitud no guardan una relación lineal directa con el precio (ej. zonas costeras vs. interior), lo que puede limitar el valor de $R^2$ en el modelo lineal.
2. **Datos censurados y valores atípicos (*outliers*):** El precio objetivo está topado artificialmente en $5.0$ ($\$500,000$), y existen registros extremos en `AveRooms` y `AveOccup` que pueden elevar el MAE y sesgar los coeficientes.
3. **Multicolinealidad entre variables:** Atributos como `AveRooms` y `AveBedrms` están fuertemente correlacionados, lo cual puede provocar inestabilidad en los pesos calculados por mínimos cuadrados.

| **Hito 1: Propuesta** | Redacción de propuesta (`docs/propuesta.md`), métricas ($MAE$, $R^2$) y recursos de cómputo. | Definición de problema, usuarios y análisis de riesgos iniciales. | Documentación de fuente de datos (`fetch_california_housing`), licencias y repositorio. |