# ==============================================================================
# SCRIPT DE EVALUACIÓN DEL MODELO (complemento del laboratorio)
# Clase: ISOF CLASS | Docente: Dra. Aiblis Susej Herrera Palafox
#
# Mide qué tan bueno es 'modelo_riesgo.pkl' antes de publicarlo en la nube.
# ==============================================================================

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, train_test_split

df = pd.read_csv('alumnos.csv', sep=',')
df.columns = df.columns.str.strip().str.lower()
df = df.dropna()

X = df[['asistencia', 'promedio', 'horas_estudio']]
y = df['riesgo']

modelo = joblib.load('modelo_riesgo.pkl')

# 1. Desempeño sobre la misma partición usada al entrenar (80/20, semilla 42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
y_pred = modelo.predict(X_test)

print("=" * 62)
print("EVALUACIÓN SOBRE EL CONJUNTO DE PRUEBA (20% de los datos)")
print("=" * 62)
print(f"Registros de prueba: {len(y_test)}")
print(f"Exactitud (accuracy): {accuracy_score(y_test, y_pred) * 100:.1f}%")
print("\nReporte de clasificación:")
print(classification_report(y_test, y_pred, target_names=['Bajo riesgo (0)', 'Alto riesgo (1)'], zero_division=0))
print("Matriz de confusión (filas = real, columnas = predicho):")
print(confusion_matrix(y_test, y_pred))

# 2. Validación cruzada (más confiable con pocos datos que una sola partición)
puntajes = cross_val_score(modelo, X, y, cv=5)
print("\n" + "=" * 62)
print("VALIDACIÓN CRUZADA (5 pliegues sobre los 20 registros)")
print("=" * 62)
print(f"Exactitud por pliegue: {[round(float(p) * 100, 1) for p in puntajes]}")
print(f"Exactitud promedio: {puntajes.mean() * 100:.1f}% (+/- {puntajes.std() * 100:.1f}%)")

# 3. Importancia de las variables: ¿qué pesa más en la decisión del modelo?
print("\n" + "=" * 62)
print("IMPORTANCIA DE LAS VARIABLES (Random Forest)")
print("=" * 62)
importancias = sorted(zip(X.columns, modelo.feature_importances_), key=lambda t: -t[1])
for nombre, valor in importancias:
    barra = '█' * int(valor * 40)
    print(f"{nombre:<15} {valor * 100:5.1f}%  {barra}")

# 4. Predicciones de ejemplo con el modelo ya guardado
print("\n" + "=" * 62)
print("EJEMPLOS DE PREDICCIÓN (entrada -> salida del modelo)")
print("=" * 62)
casos = pd.DataFrame([
    {'asistencia': 95, 'promedio': 9.2, 'horas_estudio': 18},
    {'asistencia': 80, 'promedio': 7.5, 'horas_estudio': 8},
    {'asistencia': 45, 'promedio': 4.8, 'horas_estudio': 2},
])
etiquetas = {0: 'BAJO RIESGO', 1: 'ALTO RIESGO'}
for _, fila in casos.iterrows():
    datos = pd.DataFrame([fila])
    pred = modelo.predict(datos)[0]
    prob = modelo.predict_proba(datos)[0][pred] * 100
    print(f"Asist {fila['asistencia']:>3}% | Prom {fila['promedio']:.1f} | "
          f"Estudio {fila['horas_estudio']:>2} h  ->  {etiquetas[pred]:<11} (confianza {prob:.1f}%)")
