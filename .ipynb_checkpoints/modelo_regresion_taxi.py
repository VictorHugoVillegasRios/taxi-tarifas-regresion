
# ========================================
# 1. Importación de librerías
# ========================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# ========================================
# 2. Cargar datos
# ========================================
df = pd.read_csv("C:/Users/victo/OneDrive/Data_Science/Python/taxi_trip_data_clean.csv")

# ========================================
# 3. Definir variables predictoras y objetivo
# ========================================
X = df[['trip_distance', 'trip_duration']]
y = df['total_amount']

# ========================================
# 4. Dividir los datos
# ========================================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ========================================
# 5. Entrenamiento del modelo
# ========================================
modelo = LinearRegression()
modelo.fit(X_train, y_train)

# ========================================
# 6. Resultados del modelo
# ========================================
print("Coeficientes:", modelo.coef_)
print("Intercepto:", modelo.intercept_)

# ========================================
# 7. Predicciones
# ========================================
y_pred = modelo.predict(X_test)

# ========================================
# 8. Evaluación del modelo
# ========================================
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MAE: {mae:.2f}")
print(f"MSE: {mse:.2f}")
print(f"R²: {r2:.2f}")

# ========================================
# 9. Visualización: Histograma de errores
# ========================================
errores = y_test - y_pred
plt.figure(figsize=(8, 5))
sns.histplot(errores, bins=30, kde=True, color="skyblue")
plt.title("Distribución de errores (y_test - y_pred)")
plt.xlabel("Error")
plt.ylabel("Frecuencia")
plt.tight_layout()
plt.savefig("histograma_errores.png")
plt.close()

# ========================================
# 10. Guardar modelo entrenado
# ========================================
joblib.dump(modelo, "modelo_regresion_taxi.pkl")
print("✅ Modelo guardado como 'modelo_regresion_taxi.pkl'")

# ========================================
# 11. Cargar modelo y hacer nuevas predicciones
# ========================================
modelo_cargado = joblib.load("modelo_regresion_taxi.pkl")
nuevas_predicciones = modelo_cargado.predict(X_test[:5])
reales = y_test[:5].values

for i in range(len(nuevas_predicciones)):
    print(f"Registro {i+1}:")
    print(f"  🔮 Predicción: {nuevas_predicciones[i]:.2f}")
    print(f"  ✅ Real:       {reales[i]:.2f}")
    print("-" * 30)

# ========================================
# 12. Guardar todas las predicciones
# ========================================
df_predicciones = pd.DataFrame({
    'trip_distance': X_test['trip_distance'],
    'trip_duration': X_test['trip_duration'],
    'total_amount_real': y_test,
    'total_amount_predicho': y_pred
})
df_predicciones.to_csv("predicciones_taxi.csv", index=False)
print("✅ Predicciones guardadas como 'predicciones_taxi.csv'")
