# ============================================================
# CABECERA
# ============================================================
# Alumno: Luis Fernando Neila Martínez
# URL Streamlit Cloud: https://...streamlit.app
# URL GitHub: https://github.com/...

# ============================================================
# IMPORTS
# ============================================================
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from openai import OpenAI
import json

# ============================================================
# CONSTANTES
# ============================================================
MODEL = "gpt-4.1-mini"

# -------------------------------------------------------
# >>> SYSTEM PROMPT — TU TRABAJO PRINCIPAL ESTÁ AQUÍ <<<
# -------------------------------------------------------
SYSTEM_PROMPT = """Eres un asistente analítico especializado en hábitos de escucha musical de Spotify.
Tu tarea es responder preguntas del usuario generando código Python con Plotly que analice el DataFrame `df`.

PERÍODO DE DATOS: desde {fecha_min} hasta {fecha_max}.

COLUMNAS DISPONIBLES EN `df`:
- ts: datetime con zona horaria (Europe/Madrid) — timestamp de fin de reproducción
- fecha: date — solo la fecha
- hora: int (0-23) — hora del día
- dia_semana: str — día en inglés (Monday, Tuesday, ...)
- mes: str — período mensual en formato "YYYY-MM"
- semestre: str — "S1" (enero-junio) o "S2" (julio-diciembre)
- estacion: str — "invierno", "primavera", "verano", "otoño"
- ms_played: int — milisegundos reproducidos
- minutos: float — minutos reproducidos (ya convertido)
- cancion: str — nombre de la canción
- artista: str — nombre del artista
- album: str — nombre del álbum
- spotify_track_uri: str — identificador único de la canción
- platform: str — plataforma usada. Valores posibles: {plataformas}
- reason_start: str — motivo de inicio. Valores posibles: {reason_start_values}
- reason_end: str — motivo de fin. Valores posibles: {reason_end_values}
- shuffle: bool — si el modo aleatorio estaba activado
- skipped: bool/null — True si se saltó, null si no

TIPOS DE PREGUNTAS QUE PUEDES RESPONDER:
A. Rankings y favoritos: artistas, canciones, álbumes más escuchados
B. Evolución temporal: tendencias por mes, semana, día
C. Patrones de uso: horas del día, días de la semana, plataformas
D. Comportamiento: porcentaje de skips, uso de shuffle, duración media
E. Comparación entre períodos: semestres, estaciones, meses concretos

INSTRUCCIONES DE CÓDIGO:
- El código debe crear una variable llamada `fig` usando plotly.express (px) o plotly.graph_objects (go)
- Usa `df` directamente — ya está cargado y preparado
- Para rankings usa gráficos de barras horizontales (px.bar con orientation='h')
- Para evolución temporal usa líneas (px.line)
- Para distribuciones horarias usa barras verticales (px.bar)
- Para proporciones usa gráficos de tarta (px.pie) solo si hay menos de 6 categorías
- Añade siempre title, labels legibles en español, y color donde aporte valor
- No uses plt (matplotlib). Solo plotly.

FORMATO DE RESPUESTA — OBLIGATORIO:
Debes responder SIEMPRE con un JSON válido y nada más. Sin texto antes ni después. Sin backticks.
El JSON debe tener exactamente estos tres campos:

{{"tipo": "grafico", "codigo": "fig = px.bar(...)", "interpretacion": "Texto explicativo en español."}}

Si la pregunta está fuera del alcance del dataset (no es sobre hábitos de escucha musical), responde:

{{"tipo": "fuera_de_alcance", "codigo": "", "interpretacion": "Solo puedo responder preguntas sobre tus hábitos de escucha en Spotify."}}

GUARDRAILS:
- No generes código que modifique `df`, elimine datos, o acceda a ficheros externos
- No respondas preguntas que no estén relacionadas con el historial de escucha
- Si una pregunta es ambigua, interpreta la opción más razonable y respóndela
- Nunca devuelvas texto fuera del JSON


"""


# ============================================================
# CARGA Y PREPARACIÓN DE DATOS
# ============================================================
@st.cache_data
def load_data():
    df = pd.read_json("streaming_history.json")

    # 1. Convertir timestamp a datetime con zona horaria de Madrid
    df["ts"] = pd.to_datetime(df["ts"], utc=True).dt.tz_convert("Europe/Madrid")

    # 2. Columnas temporales derivadas para facilitar el trabajo del LLM
    df["fecha"]      = df["ts"].dt.date
    df["hora"]       = df["ts"].dt.hour
    df["dia_semana"] = df["ts"].dt.day_name()
    df["mes"]        = df["ts"].dt.to_period("M").astype(str)
    df["semestre"]   = df["ts"].dt.month.apply(lambda m: "S1" if m <= 6 else "S2")
    df["estacion"]   = df["ts"].dt.month.apply(
                           lambda m: "invierno" if m in [12, 1, 2]
                           else "primavera" if m in [3, 4, 5]
                           else "verano" if m in [6, 7, 8]
                           else "otoño")

    # 3. Convertir milisegundos a minutos (más interpretable)
    df["minutos"] = (df["ms_played"] / 60000).round(2)

    # 4. Renombrar columnas largas para simplificar el código generado por el LLM
    df = df.rename(columns={
        "master_metadata_track_name":        "cancion",
        "master_metadata_album_artist_name": "artista",
        "master_metadata_album_album_name":  "album",
    })

    # 5. Filtrar reproducciones menores de 30 segundos (no son escuchas reales)
    df = df[df["ms_played"] >= 30000]

    return df


def build_prompt(df):
    fecha_min = df["ts"].min()
    fecha_max = df["ts"].max()
    plataformas = df["platform"].unique().tolist()
    reason_start_values = df["reason_start"].unique().tolist()
    reason_end_values = df["reason_end"].unique().tolist()

    return SYSTEM_PROMPT.format(
        fecha_min=fecha_min,
        fecha_max=fecha_max,
        plataformas=plataformas,
        reason_start_values=reason_start_values,
        reason_end_values=reason_end_values,
    )


# ============================================================
# FUNCIÓN DE LLAMADA A LA API
# ============================================================
def get_response(user_msg, system_prompt):
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.2,
    )
    return response.choices[0].message.content


# ============================================================
# PARSING DE LA RESPUESTA
# ============================================================
def parse_response(raw):
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()

    return json.loads(cleaned)


# ============================================================
# EJECUCIÓN DEL CÓDIGO GENERADO
# ============================================================
def execute_chart(code, df):
    local_vars = {"df": df, "pd": pd, "px": px, "go": go}
    exec(code, {}, local_vars)
    return local_vars.get("fig")


# ============================================================
# INTERFAZ STREAMLIT
# ============================================================
st.set_page_config(page_title="Spotify Analytics", layout="wide")

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    st.title("🔒 Acceso restringido")
    pwd = st.text_input("Contraseña:", type="password")
    if pwd:
        if pwd == st.secrets["PASSWORD"]:
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("Contraseña incorrecta.")
    st.stop()

st.title("🎵 Spotify Analytics Assistant")
st.caption("Pregunta lo que quieras sobre tus hábitos de escucha")

df = load_data()
system_prompt = build_prompt(df)

if prompt := st.chat_input("Ej: ¿Cuál es mi artista más escuchado?"):

    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Analizando..."):
            try:
                raw = get_response(prompt, system_prompt)
                parsed = parse_response(raw)

                if parsed["tipo"] == "fuera_de_alcance":
                    st.write(parsed["interpretacion"])
                else:
                    fig = execute_chart(parsed["codigo"], df)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                        st.write(parsed["interpretacion"])
                        st.code(parsed["codigo"], language="python")
                    else:
                        st.warning("El código no produjo ninguna visualización. Intenta reformular la pregunta.")
                        st.code(parsed["codigo"], language="python")

            except json.JSONDecodeError:
                st.error("No he podido interpretar la respuesta. Intenta reformular la pregunta.")
            except Exception as e:
                st.error("Ha ocurrido un error al generar la visualización. Intenta reformular la pregunta.")


# # ============================================================
# REFLEXIÓN TÉCNICA (máximo 30 líneas)
# ============================================================
#
# 1. ARQUITECTURA TEXT-TO-CODE
#    La aplicación sigue una arquitectura text-to-code: el LLM nunca
#    recibe los datos reales, sino solo la descripción de su estructura
#    (nombres de columnas, tipos, valores posibles). Con esa información,
#    genera código Python como texto. Ese código lo ejecuta la app en local
#    usando exec(), que busca la variable `fig` producida. El LLM no ve los
#    datos por dos razones: privacidad (los datos del usuario no salen del
#    dispositivo) y coste (enviar 15.000 filas en cada llamada consumiría
#    muchos tokens y haría la app lenta e inviable).
#
# 2. EL SYSTEM PROMPT COMO PIEZA CLAVE
#    El prompt le proporciona al LLM el esquema exacto del DataFrame tras
#    la preparación de datos: nombres de columnas simplificados (cancion,
#    artista, minutos), columnas derivadas (estacion, dia_semana, semestre)
#    y el formato de respuesta JSON obligatorio. Sin la columna `estacion`,
#    una pregunta como "compara verano vs invierno" fallaría porque el LLM
#    tendría que calcularla desde cero y probablemente lo haría mal. Sin el
#    formato JSON estricto, la función parse_response() lanzaría un error
#    al no poder convertir texto libre en diccionario Python.
#
# 3. EL FLUJO COMPLETO
#    El usuario escribe una pregunta en el chat. La app llama a get_response()
#    con esa pregunta y el system prompt ya construido (con fechas y valores
#    reales del dataset inyectados). El LLM devuelve un string JSON con tres
#    campos: tipo, codigo e interpretacion. parse_response() limpia y convierte
#    ese string en un diccionario Python. Si el tipo es "grafico", execute_chart()
#    ejecuta el código con exec() en un entorno local que tiene acceso a df, px
#    y go, y recupera la figura resultante. Streamlit renderiza esa figura
#    interactiva en pantalla junto con la interpretación en texto.