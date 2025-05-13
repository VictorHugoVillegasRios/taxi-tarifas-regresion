# 🚖 Predicción de Tarifas de Taxi en Nueva York con Regresión Lineal

Este proyecto desarrolla un modelo de regresión lineal para predecir el **monto total de un viaje en taxi** en Nueva York, utilizando variables operacionales como la **distancia** y la **duración** del viaje.

---

## 📌 Objetivo

Predecir el valor de `total_amount` (tarifa total pagada) en un viaje de taxi, a partir de:

- `trip_distance`: distancia recorrida (en millas)
- `trip_duration`: duración del viaje (en minutos)

---

## 🧰 Herramientas utilizadas

- Python 3.x  
- Bibliotecas: `pandas`, `scikit-learn`, `matplotlib`, `seaborn`, `joblib`
- Jupyter Notebook / VSCode

---

## 🛠️ Flujo del Proyecto

1. **Carga del dataset limpio**
2. **Selección de variables predictoras**
3. **División en entrenamiento y prueba**
4. **Entrenamiento del modelo de regresión lineal**
5. **Evaluación del modelo** con MAE, MSE y R²
6. **Visualización de errores**
7. **Exportación de predicciones** y guardado del modelo

---

## 📈 Resultados del modelo

- Coeficientes:
  - `trip_distance`: +1.81 USD
  - `trip_duration`: +0.44 USD
- Intercepto: 3.80
- R²: 0.91 → excelente capacidad explicativa
- MAE: 1.04 | MSE: 1.77

---

## 📂 Archivos incluidos

- `modelo_regresion_taxi.py`: script con todo el código.
- `modelo_regresion_taxi.pkl`: modelo entrenado guardado con `joblib`.
- `predicciones_taxi.csv`: archivo con todas las predicciones generadas.
- `informe_tecnico_regresion_taxi.pdf`: informe técnico del proceso.
- `Informe_Ejecutivo_Prediccion_Taxi.pptx`: presentación ejecutiva (opcional).

---

## 🧠 Conclusiones

Este modelo demuestra que, con variables básicas, es posible predecir de forma eficiente las tarifas de taxi. El flujo implementado es fácilmente escalable para aplicaciones reales.

---

## 📌 Autor

**Nombre**: (tu nombre aquí)  
**Contacto**: (email o LinkedIn)

---

