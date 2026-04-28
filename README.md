# 🚖 Predicción de Tarifas de Taxi en Nueva York

Modelo de regresión lineal desarrollado en Python para predecir el costo total de viajes en taxi en Nueva York, utilizando variables operacionales clave como distancia y duración del viaje.

---

## 📄 Documentación Ejecutiva

Este proyecto incluye documentación técnica y ejecutiva que resume el proceso de modelado y los principales resultados obtenidos.

📥 Acceso directo:

* 👉 [Informe Técnico](./docs/informe_tecnico_regresion_taxi.pdf)
* 👉 [Presentación Ejecutiva](./docs/Informe_Ejecutivo_Prediccion_Taxi.pptx)

---

## 🎯 Objetivo

Predecir el valor de **total_amount** (tarifa total del viaje) a partir de:

* 📏 `trip_distance`: distancia recorrida (millas)
* ⏱️ `trip_duration`: duración del viaje (minutos)

---

## 🛠️ Tecnologías Utilizadas

* **Python 3.x**

  * Pandas
  * Scikit-learn
  * Matplotlib
  * Seaborn
  * Joblib

* **Entorno de desarrollo**

  * Jupyter Notebook / VSCode

---

## ⚙️ Flujo del Proyecto

1. Carga y preparación del dataset
2. Selección de variables predictoras
3. División en conjuntos de entrenamiento y prueba
4. Entrenamiento del modelo de regresión lineal
5. Evaluación del modelo (MAE, MSE, R²)
6. Visualización de resultados
7. Exportación de predicciones y modelo

---

## 📊 Resultados del Modelo

* **Coeficientes:**

  * trip_distance: +1.81 USD
  * trip_duration: +0.44 USD

* **Intercepto:** 3.80

* **Métricas:**

  * R²: 0.91
  * MAE: 1.04
  * MSE: 1.77

📌 El modelo presenta una alta capacidad explicativa, demostrando que variables operacionales simples pueden capturar gran parte del comportamiento de las tarifas.

---

## 📂 Estructura del Repositorio

* `/data` → Datos utilizados
* `/modelos` → Modelo entrenado y predicciones
* `/docs` → Documentación (PDF / PPT)
* `modelo_regresion_taxi.py` → Script principal

---

## 🧠 Conclusiones

El modelo desarrollado demuestra que es posible predecir con alta precisión el costo de un viaje de taxi utilizando variables básicas.

Este enfoque es escalable y puede ser aplicado en:

* Sistemas de predicción de tarifas
* Optimización de rutas
* Plataformas de movilidad

---

## 🧑‍💻 Autor

**Víctor Hugo Villegas Ríos**
Consultor Freelance en Análisis y Ciencia de Datos

🔗 LinkedIn:
https://www.linkedin.com/in/victorhugovillegasrios/

---

## 🎯 Contexto del Proyecto

Este proyecto fue desarrollado como parte del proceso de formación en el programa de Google Data Analytics, representando un caso práctico de aplicación de modelos predictivos en ciencia de datos.

---


