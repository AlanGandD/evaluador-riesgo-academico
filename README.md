# ⚡ Evaluador Predictivo de Riesgo Académico

**Laboratorio 1 — ISOF CLASS**
Carga de Datos CSV + Machine Learning + Groq API + Streamlit

> Docente: Dra. Aiblis Susej Herrera Palafox
> Materia: Cómputo en la Nube

Sistema híbrido que combina **Machine Learning (scikit-learn)** para calcular la probabilidad de
riesgo académico de un estudiante y la **API de Groq** para redactar un plan de tutoría
personalizado a partir de ese diagnóstico.

> 🔗 **Repositorio:** <https://github.com/AlanGandD/evaluador-riesgo-academico>
>
> 🚀 **Desplegar en Streamlit Cloud:**
> <https://share.streamlit.io/deploy?repository=AlanGandD/evaluador-riesgo-academico&branch=main&mainModule=app.py>

---

## 📁 Estructura del proyecto

| Archivo | Para qué sirve |
|---|---|
| `alumnos.csv` | Dataset de entrenamiento (20 registros: asistencia, promedio, horas de estudio, riesgo). |
| `entrenar.py` | Lee el CSV, limpia datos, entrena el `RandomForestClassifier` y guarda `modelo_riesgo.pkl`. |
| `evaluar_modelo.py` | *(Complemento)* Mide exactitud, validación cruzada e importancia de variables del modelo. |
| `app.py` | Aplicación Streamlit: sliders, predicción del modelo y plan de tutoría con Groq. |
| `requirements.txt` | Dependencias para el despliegue en la nube. |
| `.gitignore` | Evita subir `secrets.toml`, `__pycache__/` y `.venv/` a GitHub. |
| `.streamlit/config.toml` | Tema visual oscuro de la app. |
| `.streamlit/secrets.toml` | **Tu API key real de Groq** (ignorado por git, no se sube). |
| `.streamlit/secrets.toml.example` | Plantilla de cómo debe verse el archivo de secretos. |
| `modelo_riesgo.pkl` | Modelo entrenado y serializado (se genera al ejecutar `entrenar.py`). |
| `RESPUESTAS_CUESTIONARIO.md` | Respuestas al cuestionario de reflexión pedagógica (Paso 7). |

---

## 🚀 Cómo ejecutarlo en VS Code (localhost)

### 1. Abrir la carpeta del proyecto

```bash
cd "proyecto_ml_isof"
```

### 2. Crear el entorno virtual e instalar dependencias

```bash
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
# .venv\Scripts\activate         # Windows

pip install -r requirements.txt
```

### 3. Configurar la API key de Groq

Obtén una llave gratuita en <https://console.groq.com/keys> y colócala en
`.streamlit/secrets.toml`:

```toml
GROQ_API_KEY = "gsk_TU_LLAVE_AQUI"
```

> Si ese archivo no existe, `app.py` usa el marcador `gsk_TU_API_KEY_LOCAL_AQUI` y la app
> cargará, pero el plan de tutoría mostrará un error de autenticación (401).

### 4. Entrenar el modelo (Paso 4 del laboratorio)

```bash
python entrenar.py
```

**Salida esperada:**

```
📂 1. Cargando conjunto de datos desde 'alumnos.csv'...
✅ Archivo cargado correctamente. Registros totales: 20
📋 Columnas detectadas: ['asistencia', 'promedio', 'horas_estudio', 'riesgo']
🧠 2. Entrenando el algoritmo RandomForestClassifier...
✅ 3. ¡Éxito! Modelo exportado como 'modelo_riesgo.pkl'.
```

### 5. (Opcional) Revisar la calidad del modelo

```bash
python evaluar_modelo.py
```

Resultado obtenido con este dataset:

- Exactitud en el conjunto de prueba (4 registros): **75 %**
- Validación cruzada de 5 pliegues: **90 % ± 12.2 %**
- Importancia de variables: `asistencia` 36 %, `promedio` 36 %, `horas_estudio` 28 %
- El modelo acierta los 3 casos de ejemplo (95 %/9.2/18 → bajo; 80 %/7.5/8 → bajo; 45 %/4.8/2 → alto)

### 6. Levantar la app web (Paso 5)

```bash
python -m streamlit run app.py
```

Se abre en `http://localhost:8501`. En la barra lateral ajusta **asistencia**, **promedio** y
**horas de estudio**, presiona **Evaluar Estudiante** y obtendrás:

1. El diagnóstico del modelo con su probabilidad.
2. El plan de tutoría en Markdown generado por Groq, indicando qué modelo respondió.

---

## ☁️ Despliegue en Streamlit Community Cloud (Paso 6)

1. El repositorio ya está publicado en
   <https://github.com/AlanGandD/evaluador-riesgo-academico> (rama `main`). `secrets.toml` y `.venv/`
   **no** se subieron gracias al `.gitignore`.
2. Entrar al enlace directo de despliegue
   <https://share.streamlit.io/deploy?repository=AlanGandD/evaluador-riesgo-academico&branch=main&mainModule=app.py>,
   autorizar la app de Streamlit en GitHub (solo la primera vez) y confirmar repositorio, rama `main`
   y *Main file path*: `app.py`. Si se prefiere a mano: <https://share.streamlit.io> → *Create app* →
   *Deploy a public app from GitHub*.
3. En **Advanced settings → Secrets** pegar:

   ```toml
   GROQ_API_KEY = "gsk_TU_LLAVE_AQUI"
   ```

4. Presionar **Deploy!** y esperar el enlace público.

### ⚠️ Recomendación de versiones

`requirements.txt` va sin fijar versiones, como indica el laboratorio. Si al desplegar aparece
un `InconsistentVersionWarning` al cargar el `.pkl` (el modelo se entrenó con
`scikit-learn 1.9.1`), fija la versión exacta para garantizar compatibilidad del binario:

```text
scikit-learn==1.9.1
pandas==3.1.1
numpy==2.5.5
```

---

## 🧪 Verificaciones realizadas

| Prueba | Resultado |
|---|---|
| `entrenar.py` con el CSV del laboratorio | 20 registros, 4 columnas, `.pkl` de 64 KB generado |
| `entrenar.py` con el CSV exportado **entrecomillado** | La validación automática dividió la fila y recuperó las 4 columnas correctamente |
| Predicción de un alumno en riesgo (45 % / 4.8 / 2 h) | ALTO RIESGO, 100 % de probabilidad |
| Predicción de un alumno sin riesgo (95 % / 9.2 / 18 h) | BAJO RIESGO, 100 % de probabilidad |
| Llamada real a Groq con la key del `.streamlit/secrets.toml` | 11 modelos detectados, respuesta en español y en Markdown en ~1.5 s con `openai/gpt-oss-120b` |

---

## 🔧 Notas de implementación

- **Selector automático de modelo mejorado.** El laboratorio elige el primer modelo cuyo nombre
  contenga `llama` y, si no hay, el primero de la lista. En la cuenta de Groq actual los únicos
  modelos con `llama` son `llama-prompt-guard-*` (modelos de seguridad, no de chat), por lo que
  el algoritmo terminaba seleccionando `allam-2-7b`, que redacta con calidad irregular en
  español (llegó a afirmar que un promedio de 4.8 era "bueno"). Se amplió la lista de exclusión
  (`orpheus`, `safeguard`, `rerank`, `moderation`, `realtime`) y se añadió un orden de
  preferencia (`gpt-oss-120b` → `llama-3.3-70b` → `llama-3.1-8b` → `qwen` → `gpt-oss-20b` →
  `llama` → `mixtral` → `allam`), conservando el respaldo al primer modelo disponible. La app
  sigue detectando el catálogo dinámicamente: si mañana desaparece `gpt-oss-120b`, elegirá otro
  sin tocar el código.
- **`max_tokens=1024` sigue siendo suficiente** para el dictamen de 3 pasos; si se pide un plan
  más extenso, conviene subirlo.
- **Caché del modelo.** `@st.cache_resource` evita leer el `.pkl` en cada interacción, así la
  predicción es instantánea.

---

## 🩺 Solución de problemas

| Síntoma | Causa y solución |
|---|---|
| `FileNotFoundError: alumnos.csv` | Ejecutar los scripts desde la carpeta del proyecto, no desde otra ruta. |
| `❌ Error al consultar Groq API: ... 401` | La key no es válida o no se cargó el `.streamlit/secrets.toml`. Verifica que el archivo esté en `.streamlit/` y con la clave `GROQ_API_KEY`. |
| `❌ Error al consultar Groq API: ... model_decommissioned` | El modelo elegido fue retirado por Groq; actualiza el orden de preferencia en `app.py`. |
| El CSV se cargó con una sola columna | El archivo tiene comas distintas o comillas mezcladas; el script lo corrige si toda la fila viene entre comillas, pero revisa el separador. |
| `InconsistentVersionWarning` en la nube | El `.pkl` se entrenó con otra versión de scikit-learn; fija la versión como se explica arriba o reentrena desde la nube. |

---

## 📚 Referencias

- Documentación de scikit-learn — RandomForestClassifier: <https://scikit-learn.org/stable/modules/ensemble.html#forest>
- Streamlit — `st.cache_resource` y secrets: <https://docs.streamlit.io>
- Groq API — modelos disponibles: <https://console.groq.com/docs/models>
