# 📝 Cuestionario de Reflexión Pedagógica — Respuestas

**Laboratorio 1 · ISOF CLASS · Carga de Datos CSV + ML + Groq API**
Docente: Dra. Aiblis Susej Herrera Palafox

---

## 1. Funcionamiento del Algoritmo

> ¿Por qué el modelo de Machine Learning puede predecir el riesgo de un alumno sin necesidad de escribir sentencias `if / else` manuales?

**Respuesta corta:** porque el algoritmo **descubre las reglas a partir de los ejemplos** en lugar de recibirlas escritas por el programador. Esas reglas quedan codificadas como números (los parámetros del modelo), no como condiciones del lenguaje.

**Explicación.** En un programa tradicional, la lógica vive en el código: *si asistencia < 60 y promedio < 6 entonces riesgo = 1*. Ese enfoque obliga a que una persona conozca de antemano todos los umbrales y todas las combinaciones posibles; con 3 variables aún es manejable, pero con 30 se vuelve imposible y frágil. En Machine Learning ocurre lo contrario: se le entrega al algoritmo un conjunto de ejemplos ya etiquetados —el archivo `alumnos.csv`, donde cada fila dice *"con estas tres características, el desenlace fue este"*— y él busca la combinación de umbrales y relaciones que mejor separa una clase de la otra.

Concretamente, `RandomForestClassifier` no genera un árbol de decisión único, sino **100 árboles** (`n_estimators=100`); cada árbol se entrena con una muestra aleatoria distinta de filas y de columnas, y cada uno elige sus cortes minimizando la impureza (índice Gini). Al final, los 100 árboles **votan** y la clase más votada es la predicción. Esta aleatoriedad hace que el bosque sea más robusto que un árbol suelto: un solo árbol puede memorizar los 20 ejemplos, mientras que el bosque solo se queda con los patrones que se repiten en muchas muestras.

Por eso en la práctica se puede clasificar a un alumno nuevo (asistencia 45 %, promedio 4.8, 2 h de estudio) **sin haber escrito una sola condición**: el modelo ya "aprendió" que ese perfil se parece geométricamente a los casos etiquetados como riesgo alto y lo sitúa en esa región del espacio de características. Además, `predict_proba()` devuelve el porcentaje de árboles que votaron por cada clase, lo cual se traduce en la probabilidad que la app muestra al usuario. Lo que el modelo **no** hace es explicar causalidad: aprende correlaciones presentes en el dataset, y si los datos de entrenamiento están sesgados o son pocos, sus reglas heredarán ese sesgo —de ahí que después evaluemos con datos que no vio durante el entrenamiento.

---

## 2. Limpieza de Datos (`dropna`)

> ¿Por qué es importante ejecutar `df.dropna()` al leer un `.csv` antes de iniciar el entrenamiento del modelo?

**Respuesta corta:** porque los valores faltantes (`NaN`) rompen el cálculo matemático del entrenamiento y, si se rellenan mal, enseñan al modelo patrones falsos.

**Explicación.** `dropna()` elimina las filas que contienen al menos un valor ausente. Hay tres razones para hacerlo antes de `modelo.fit()`:

1. **El algoritmo no acepta huecos.** Los cortes de los árboles se calculan comparando valores numéricos; un `NaN` no se puede comparar, y scikit-learn lanza un error del tipo *"Input contains NaN, infinity or a value too large"*. En el caso del bosque aleatorio actual la excepción sería explícita, pero en otros modelos el fallo aparece como resultados sin sentido.
2. **Un hueco suele ser un dato mal exportado.** Cuando `pandas` lee un CSV con comillas de más, filas con comas extra o celdas vacías, produce `NaN` o agrupa toda la fila en una sola columna. En este laboratorio `entrenar.py` incluye una validación que detecta ese caso (una única columna con comas) y vuelve a dividir el contenido; `dropna()` es la segunda red de seguridad. Sin esa limpieza, el modelo podría entrenarse con filas donde `riesgo` quedó vacío y aprender una clase inexistente.
3. **Calidad del aprendizaje.** Cada fila incompleta aporta ruido: si `horas_estudio` falta y se rellena con 0, el algoritmo interpreta "este alumno no estudia nada" cuando en realidad el dato se perdió. Eso sesga la frontera de decisión. `dropna()` es la opción conservadora: prefiere perder 2 de 200 registros antes que entrenar con información inventada.

En un dataset grande, descartar filas no afecta; en uno pequeño como `alumnos.csv` (20 filas) conviene **imprimirlas** antes de eliminarlas para detectar la causa raíz. Si el porcentaje de faltantes fuera alto y las filas valiosas, se justificaría imputar (media, mediana o un modelo de imputación) en lugar de borrar; pero para este laboratorio, `dropna()` deja el conjunto limpio y consistente, y garantiza que las 20 filas leídas correspondan exactamente a 20 registros completos.

---

## 3. Persistencia del Modelo (`.pkl`)

> Una vez generado `modelo_riesgo.pkl`, ¿por qué la aplicación web no necesita volver a entrenar el modelo cada vez que un usuario presiona el botón?

**Respuesta corta:** porque el archivo `.pkl` guarda el modelo **ya entrenado** (sus 100 árboles con todos sus cortes y umbrales), y cargarlo por `joblib.load()` solo reconstruye un objeto en memoria; predecir es únicamente recorrer esos árboles, no aprender de nuevo.

**Explicación.** Entrenar y predecir son dos operaciones de costo muy distinto. Entrenar implica leer el dataset, dividirlo, construir 100 árboles evaluando cientos de divisiones candidatas y votar: es un proceso **de una sola vez**, de segundos o minutos, que además necesita `alumnos.csv`. Predecir, en cambio, solo exige recorrer los árboles ya construidos con los tres valores de entrada: es un cálculo de milisegundos que no toca el dataset.

`joblib.dump(modelo, 'modelo_riesgo.pkl')` **serializa** el objeto Python completo: la lista de `DecisionTreeClassifier` hijos, sus nodos, umbrales, importancias y clases. Al arrancar, `joblib.load()` lo reconstruye tal cual estaba, sin volver a llamar a `fit()`. Si la app reentrenara en cada clic:

- La respuesta tardaría segundos o minutos en lugar de milisegundos.
- Cada clic consumiría CPU innecesariamente (en Streamlit Community Cloud, recursos limitados y compartidos).
- Sería **no determinista**: reentrenar con `random_state` distinto daría un modelo ligeramente diferente en cada petición, y el mismo alumno podría recibir diagnósticos distintos al presionar el botón dos veces. Congelar el modelo garantiza **reproducibilidad y auditoría**.
- Exigiría empaquetar `alumnos.csv` en el servidor y volver a procesarlo, cuando el modelo ya es un artefacto independiente.

De ahí el `@st.cache_resource` en `app.py`: la primera ejecución carga el `.pkl` y las siguientes reutilizan el mismo objeto en memoria, así que ni siquiera se relee el archivo en cada interacción. Es la lógica de **separar entrenamiento (offline) de inferencia (online)** que se usa en producción: se entrena, se versiona el artefacto y el servicio solo lo consume. Cuando los datos cambien, se reentrena, se genera un nuevo `.pkl` y se vuelve a desplegar; eso es lo que en la industria se llama *retraining* programado y es un proceso deliberado, no algo que ocurra por cada clic del usuario.

---

## 4. Rol de Groq API

> ¿Qué función cumple el modelo predictivo de `scikit-learn` y qué función cumple la API de Groq en esta aplicación híbrida?

**Respuesta corta:** el modelo de `scikit-learn` **decide** (dice si hay riesgo y con qué probabilidad, de forma determinista y basada en los datos); la API de Groq **comunica y aconseja** (convierte ese número en un dictamen pedagógico redactado en lenguaje natural).

**Explicación.** La aplicación es híbrida porque combina dos tipos de inteligencia con fortalezas complementarias:

| Aspecto | `scikit-learn` (Random Forest) | Groq API (LLM) |
|---|---|---|
| Tarea | Clasificación binaria | Generación de lenguaje |
| Entrada | 3 números (asistencia, promedio, horas) | El prompt con esos números + el diagnóstico |
| Salida | Clase 0/1 y probabilidad | Texto en Markdown con análisis y plan de acción |
| Naturaleza | Estadística, determinista, auditable | Generativa, probabilística, creativa |
| Costo | Milisegundos, local, gratis | Petición de red a la nube, latencia y cuota |
| Riesgo | Aprende del dataset (si está sesgado, se sesga) | Puede alucinar; no conoce al alumno real |

El modelo predictivo es la **fuente de verdad numérica**: es el único que calcula la probabilidad (100 % de riesgo en el caso evaluado) y lo hace de forma reproducible: los mismos tres valores producen siempre el mismo diagnóstico. Pero un número no sirve pedagógicamente: decirle a un estudiante "0.98 de probabilidad de deserción" no lo ayuda a mejorar, y peor aún, puede ser estigmatizante si se presenta sin contexto.

Ahí entra Groq. El LLM **no decide** el riesgo ni lo recalcula: recibe el diagnóstico ya calculado como un dato del prompt (`- Diagnóstico del Modelo: ALTO RIESGO ACADÉMICO`) y lo traduce en un **dictamen accionable**: explica qué indicadores influyeron y propone un plan de tres pasos con acciones concretas (reorganizar el horario, subir la asistencia, pedir tutoría). Aporta empatía, contexto y redacción, cosas que un bosque aleatorio no puede producir.

Esta separación es además una decisión de diseño responsable: al mantener la **decisión** en un modelo pequeño y auditable, se evita que un LLM —que puede inventar cifras— determine el futuro académico de una persona. La API de Groq solo **redacta** sobre una decisión que ya está fundamentada en datos. El resultado es un sistema que aprovecha lo mejor de cada tecnología: rigor estadístico para medir y capacidad de comunicación para acompañar.

---

## ✅ Cierre

Los cuatro conceptos del laboratorio forman un flujo completo: **datos limpios** (`dropna`) → **modelo que aprende** (Random Forest) → **artefacto persistido** (`.pkl`) → **servicio híbrido** (Streamlit + Groq). Entender por qué cada pieza existe —y no solo cómo ejecutarla— es lo que permite reutilizar el patrón en cualquier proyecto real de analítica predictiva.
