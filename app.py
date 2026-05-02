"""
Spotify Studio Albums + Lyrics Explorer
========================================
Web app interactiva construida con Streamlit que permite:
  1. Filtrar tracks de un CSV de Spotify para quedarse sólo con álbumes de estudio.
  2. Explorar variables de audio por álbum y artista.
  3. Analizar letras de canciones vía la API pública lyrics.ovh.
  4. Comparar artistas con métricas léxicas y acústicas.

Artistas incluidos: The Strokes, The National y Elliott Smith.

Estos artistas sustituyen a Radiohead y The Smiths porque tienen mejor cobertura
útil en el dataset y ofrecen contrastes claros entre indie/garage rock,
indie-rock melancólico y folk-rock acústico e introspectivo.
"""

import ast
import json
import os
import re
import time
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import streamlit as st

# ---------------------------------------------------------------------------
# Configuración de página
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Spotify Studio Albums + Lyrics Explorer",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Constantes y configuración de artistas
# ---------------------------------------------------------------------------
DATA_FILE = "tracks_features.csv"
LYRICS_CACHE_FILE = "lyrics_cache_streamlit.json"

# Cada artista lleva: Spotify artist_id + lista manual de álbumes de estudio.
# La lista se mantiene manualmente porque el CSV no incluye un `album_type` fiable.
ARTIST_CONFIG = {
    # --- The Strokes (6 álbumes de estudio, 2001-2020) ---
    # En el CSV usado para el trabajo suelen aparecer 5/6; normalmente falta
    # "Comedown Machine", pero se deja en la lista para que la cobertura lo indique.
    "The Strokes": {
        "artist_id": "0epOFNiUfyON9EYx7Tpr6V",
        "studio_albums": [
            "Is This It",
            "Room on Fire",
            "First Impressions of Earth",
            "Angles",
            "Comedown Machine",
            "The New Abnormal",
        ],
    },
    # --- The National (5 álbumes de estudio seleccionados, 2007-2019) ---
    # Se usa esta etapa central porque el dataset suele cubrirla bien y porque
    # sus letras melancólicas suelen funcionar mejor con lyrics.ovh que The Clash.
    "The National": {
        "artist_id": "2cCUtGK9sDU2EoElnk0GNB",
        "lyrics_aliases": ["The National", "National"],
        "studio_albums": [
            "Boxer",
            "High Violet",
            "Trouble Will Find Me",
            "Sleep Well Beast",
            "I Am Easy to Find",
        ],
    },
    # --- Elliott Smith (6 álbumes de estudio oficiales, 1994-2004) ---
    # Se mantiene sólo el criterio de álbumes de estudio. "New Moon" no se incluye
    # porque es una recopilación póstuma de material inédito, no un álbum de estudio.
    "Elliott Smith": {
        "artist_id": "2ApaG60P4r0yhBoDCGD8YG",
        "studio_albums": [
            "Roman Candle",
            "Elliott Smith",
            "Either/Or",
            "XO",
            "Figure 8",
            "From a Basement on the Hill",
        ],
    },
}

AUDIO_FEATURES = [
    "acousticness", "danceability", "duration_ms", "energy",
    "instrumentalness", "liveness", "loudness", "speechiness",
    "tempo", "valence",
]

# Stopwords básicas en inglés para el análisis léxico
BASIC_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "if", "then", "than", "so", "because",
    "to", "of", "in", "on", "for", "from", "with", "at", "by", "as", "is", "it",
    "its", "i", "im", "i'm", "me", "my", "mine", "you", "your", "yours", "we",
    "our", "ours", "they", "their", "them", "he", "she", "his", "her", "hers",
    "this", "that", "these", "those", "be", "been", "being", "am", "are", "was",
    "were", "do", "does", "did", "doing", "have", "has", "had", "having", "not",
    "no", "yes", "oh", "ah", "ooh", "la", "na", "yeah", "uh", "ha", "hey",
    "all", "any", "some", "can", "could", "would", "should", "will", "just",
    "dont", "don't", "cant", "can't", "wont", "won't", "youre", "you're",
    "ive", "i've", "ill", "i'll", "id", "i'd", "theres", "there's", "thats",
    "that's", "get", "got", "go", "know", "like", "want", "see", "say", "said",
    "now", "up", "out", "when", "what", "where", "who", "how", "there", "here",
    "more", "still", "never", "always", "every", "ever", "one", "two", "about",
    "over", "down", "back", "away", "into", "come", "came", "make", "made", "us",
}


# ---------------------------------------------------------------------------
# Carga y caché de datos
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_data():
    """Carga el CSV principal. Lanza FileNotFoundError si no existe."""
    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(
            f"No se encuentra {DATA_FILE}. "
            "Coloca el CSV en la misma carpeta que app.py."
        )
    return pd.read_csv(DATA_FILE)


# ---------------------------------------------------------------------------
# Funciones de normalización y parseo
# ---------------------------------------------------------------------------

def parse_python_list(text):
    """Convierte una cadena que representa una lista Python a lista real."""
    if isinstance(text, list):
        return text
    try:
        return ast.literal_eval(str(text))
    except Exception:
        return []


def normalize_album_name(album_name: str) -> str:
    """
    Elimina sufijos de reedición para normalizar nombres de álbum:
    'OK Computer (OKNOTOK 1997–2017)' → 'OK Computer'
    """
    album_name = str(album_name).strip()
    album_name = re.sub(r"\s*\([^\)]*\)", "", album_name)
    album_name = re.sub(r"\s*\[[^\]]*\]", "", album_name)
    album_name = re.sub(
        r"\s*-\s*(deluxe|expanded|special|legacy|collector's|remaster.*)$",
        "",
        album_name,
        flags=re.IGNORECASE,
    )
    return re.sub(r"\s+", " ", album_name).strip()


def normalize_track_name(track_name: str) -> str:
    """
    Elimina sufijos de remaster del nombre de la canción:
    'Karma Police - 2016 Remaster' → 'Karma Police'
    """
    track_name = str(track_name).strip()
    track_name = re.sub(r"\s*-\s*\d{4} remaster.*$", "", track_name, flags=re.IGNORECASE)
    track_name = re.sub(r"\s*-\s*remaster.*$", "", track_name, flags=re.IGNORECASE)
    track_name = re.sub(r"\s*-\s*remastered.*$", "", track_name, flags=re.IGNORECASE)
    track_name = re.sub(r"\s*\([^\)]*remaster[^\)]*\)", "", track_name, flags=re.IGNORECASE)
    track_name = re.sub(r"\s*\([^\)]*version[^\)]*\)", "", track_name, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", track_name).strip()


def contains_artist_id(artist_ids_text: str, target_id: str) -> bool:
    """Comprueba si el target_id aparece en la cadena de IDs del artista."""
    return target_id in str(artist_ids_text)


def extract_main_artist(artists_text: str) -> str:
    """Devuelve el primer artista de la lista."""
    artists = parse_python_list(artists_text)
    return artists[0] if artists else str(artists_text)


# ---------------------------------------------------------------------------
# Filtrado por artista y álbumes de estudio
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def get_artist_subset(df_in: pd.DataFrame, artist_name: str) -> pd.DataFrame:
    """
    Filtra el DataFrame global para quedarse sólo con los tracks del artista
    indicado que pertenecen a sus álbumes de estudio (según la lista manual).
    También desduplicа por (artista, álbum, canción) conservando la primera aparición.
    """
    config = ARTIST_CONFIG[artist_name]
    artist_id = config["artist_id"]
    studio_albums = set(config["studio_albums"])

    # Filtrar por artist_id
    artist_df = df_in[
        df_in["artist_ids"].astype(str).apply(lambda x: contains_artist_id(x, artist_id))
    ].copy()

    artist_df["artist_name"] = artist_df["artists"].apply(extract_main_artist)
    artist_df["album_clean"] = artist_df["album"].apply(normalize_album_name)
    artist_df["track_clean"] = artist_df["name"].apply(normalize_track_name)

    # Conservar sólo álbumes de estudio
    artist_df = artist_df[artist_df["album_clean"].isin(studio_albums)].copy()

    # Deduplicar: una fila por (artista, álbum, canción)
    artist_df = (
        artist_df
        .sort_values(["year", "album_clean", "disc_number", "track_number", "release_date"])
        .drop_duplicates(subset=["artist_name", "album_clean", "track_clean"], keep="first")
        .reset_index(drop=True)
    )
    return artist_df


@st.cache_data(show_spinner=False)
def build_coverage_table(df_in: pd.DataFrame) -> pd.DataFrame:
    """
    Genera una tabla de cobertura para cada artista: cuántos álbumes de estudio
    se encontraron en el dataset y cuáles faltan.
    """
    rows = []
    for artist_name, config in ARTIST_CONFIG.items():
        artist_id = config["artist_id"]
        studio_albums = config["studio_albums"]

        pre_df = df_in[
            df_in["artist_ids"].astype(str).apply(lambda x: contains_artist_id(x, artist_id))
        ].copy()
        before = len(pre_df)

        filtered_df = get_artist_subset(df_in, artist_name)
        albums_present = sorted(filtered_df["album_clean"].dropna().unique().tolist())
        missing = [a for a in studio_albums if a not in albums_present]

        rows.append({
            "Artista": artist_name,
            "Tracks (bruto)": before,
            "Tracks (filtrado)": len(filtered_df),
            "Álbumes esperados": len(studio_albums),
            "Álbumes en dataset": len(albums_present),
            "Álbumes presentes": ", ".join(albums_present) if albums_present else "(ninguno)",
            "Álbumes no encontrados": ", ".join(missing) if missing else "—",
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Lyrics: caché, fetch y métricas léxicas
# ---------------------------------------------------------------------------

def load_lyrics_cache() -> dict:
    """Carga la caché de letras desde disco (JSON). Devuelve {} si no existe."""
    if os.path.exists(LYRICS_CACHE_FILE):
        try:
            with open(LYRICS_CACHE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_lyrics_cache(cache_dict: dict):
    """Persiste la caché de letras en disco para evitar llamadas repetidas a la API."""
    try:
        with open(LYRICS_CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(cache_dict, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def clean_title_for_api(title: str) -> str:
    """
    Limpia el título de la canción antes de llamar a la API:
    elimina 'feat.', partes tras '/', etc.
    """
    title = normalize_track_name(title)
    title = re.sub(r"\s*/\s*.*$", "", title)
    title = re.sub(r"\s*feat\..*$", "", title, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", title).strip()


def build_title_candidates(title: str) -> list[str]:
    """Genera variantes del título para mejorar la cobertura de lyrics.ovh."""
    base = clean_title_for_api(title)
    variants = [base]

    # Algunas canciones aparecen en la API sin subtítulos entre paréntesis.
    no_parentheses = re.sub(r"\s*\([^\)]*\)", "", base).strip()
    if no_parentheses and no_parentheses != base:
        variants.append(no_parentheses)

    # Variante más agresiva: quitar comillas/apóstrofes/puntuación conflictiva.
    simplified = re.sub(r"[^a-zA-Z0-9\s]", "", no_parentheses or base)
    simplified = re.sub(r"\s+", " ", simplified).strip()
    if simplified and simplified not in variants:
        variants.append(simplified)

    return variants


def build_artist_candidates(artist_name: str) -> list[str]:
    """Genera variantes del artista para mejorar la cobertura de lyrics.ovh."""
    config = ARTIST_CONFIG.get(artist_name, {})
    aliases = config.get("lyrics_aliases", [artist_name])
    variants = []
    for alias in aliases:
        if alias not in variants:
            variants.append(alias)
        without_the = re.sub(r"^the\s+", "", alias, flags=re.IGNORECASE).strip()
        if without_the and without_the not in variants:
            variants.append(without_the)
    return variants


def fetch_lyrics(artist_name: str, song_title: str, sleep_seconds: float = 0.1):
    """
    Obtiene la letra de una canción usando la API pública https://lyrics.ovh/.
    Usa caché y prueba variantes de artista/título para evitar falsos negativos.

    Devuelve la letra como string o None si no se encuentra.
    """
    cache = st.session_state.setdefault("lyrics_cache", load_lyrics_cache())

    artist_candidates = build_artist_candidates(artist_name)
    title_candidates = build_title_candidates(song_title)
    cache_key = f"{artist_name}|||{title_candidates[0]}"

    if cache_key in cache:
        return cache[cache_key]

    lyrics = None
    for api_artist in artist_candidates:
        for api_title in title_candidates:
            url_artist = requests.utils.quote(api_artist)
            url_title = requests.utils.quote(api_title)
            url = f"https://api.lyrics.ovh/v1/{url_artist}/{url_title}"

            try:
                response = requests.get(url, timeout=15)
                if response.status_code == 200:
                    lyrics = response.json().get("lyrics", None)
                    if lyrics:
                        break
            except Exception:
                lyrics = None
        if lyrics:
            break

    cache[cache_key] = lyrics
    save_lyrics_cache(cache)
    time.sleep(sleep_seconds)
    return lyrics


def tokenize_lyrics(text: str) -> list:
    """
    Tokeniza la letra: minúsculas, sólo letras/apóstrofes, elimina stopwords
    y tokens de menos de 3 caracteres.
    """
    if not isinstance(text, str) or not text.strip():
        return []
    text = text.lower()
    tokens = re.findall(r"[a-zA-Z']+", text)
    tokens = [tok.strip("'") for tok in tokens if tok.strip("'")]
    tokens = [tok for tok in tokens if tok not in BASIC_STOPWORDS and len(tok) > 2]
    return tokens


def lyrics_metrics(text: str) -> dict:
    """
    Calcula métricas léxicas para una letra:
    - word_count: número de palabras (sin stopwords)
    - unique_words: número de palabras únicas
    - lexical_diversity: unique / total (índice Type-Token Ratio)
    """
    tokens = tokenize_lyrics(text)
    if not tokens:
        return {"word_count": np.nan, "unique_words": np.nan, "lexical_diversity": np.nan}
    unique_words = len(set(tokens))
    word_count = len(tokens)
    return {
        "word_count": word_count,
        "unique_words": unique_words,
        "lexical_diversity": unique_words / word_count if word_count else np.nan,
    }


def build_lyrics_dataset(
    artist_df: pd.DataFrame,
    artist_name: str,
    max_tracks: int = 20,
) -> pd.DataFrame:
    """
    Añade columnas de letras y métricas léxicas al DataFrame del artista.
    Se limita a max_tracks canciones (ordenadas cronológicamente) para
    no saturar la API ni tardar demasiado.
    """
    if artist_df.empty:
        return artist_df.assign(
            lyrics=None,
            word_count=np.nan,
            unique_words=np.nan,
            lexical_diversity=np.nan,
            lyrics_found=False,
        )

    subset = (
        artist_df
        .sort_values(["year", "album_clean", "disc_number", "track_number"])
        .head(max_tracks)
        .copy()
    )

    lyrics_list, word_counts, unique_counts, lexical_divs = [], [], [], []

    for _, row in subset.iterrows():
        lyrics = fetch_lyrics(artist_name, row["track_clean"])
        m = lyrics_metrics(lyrics if lyrics else "")
        lyrics_list.append(lyrics)
        word_counts.append(m["word_count"])
        unique_counts.append(m["unique_words"])
        lexical_divs.append(m["lexical_diversity"])

    subset["lyrics"] = lyrics_list
    subset["word_count"] = word_counts
    subset["unique_words"] = unique_counts
    subset["lexical_diversity"] = lexical_divs
    subset["lyrics_found"] = subset["lyrics"].notna()
    return subset


# ---------------------------------------------------------------------------
# Gráficos de audio
# ---------------------------------------------------------------------------

def plot_audio_summary(artist_df: pd.DataFrame, artist_name: str):
    """
    Muestra:
    1. Tabla resumen por álbum (tracks, año, valence media, energy media, acousticness media).
    2. Barras horizontales: número de canciones por álbum.
    3. Scatter: Acousticness vs Valence con tamaño = duración.
    """
    if artist_df.empty:
        st.warning(f"No hay canciones disponibles para {artist_name} después del filtrado.")
        return

    album_order = (
        artist_df
        .drop_duplicates("album_clean")
        .sort_values(["year", "album_clean"])["album_clean"]
        .tolist()
    )

    summary = (
        artist_df.groupby("album_clean")
        .agg(
            Canciones=("track_clean", "count"),
            Año=("year", "min"),
            Valence=("valence", "mean"),
            Energy=("energy", "mean"),
            Acousticness=("acousticness", "mean"),
        )
        .loc[album_order]
        .reset_index()
        .rename(columns={"album_clean": "Álbum"})
    )
    summary[["Valence", "Energy", "Acousticness"]] = summary[
        ["Valence", "Energy", "Acousticness"]
    ].round(3)

    st.dataframe(summary, use_container_width=True, hide_index=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    tracks_per_album = (
        artist_df.groupby("album_clean")["track_clean"].count().loc[album_order]
    )
    axes[0].barh(tracks_per_album.index, tracks_per_album.values, color="steelblue")
    axes[0].set_title(f"Canciones por álbum — {artist_name}")
    axes[0].set_xlabel("Número de canciones")
    axes[0].invert_yaxis()

    axes[1].scatter(
        artist_df["valence"],
        artist_df["acousticness"],
        s=np.clip(artist_df["duration_ms"] / 1000, 40, 300),
        alpha=0.7,
        color="steelblue",
    )
    axes[1].set_title(f"Acousticness vs Valence — {artist_name}")
    axes[1].set_xlabel("Valence")
    axes[1].set_ylabel("Acousticness")

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Gráficos de letras
# ---------------------------------------------------------------------------

def plot_top_words(lyrics_df: pd.DataFrame, top_n_words: int):
    """
    Gráfico de barras con las top_n_words palabras más frecuentes
    en el conjunto de letras encontradas.
    """
    all_tokens = []
    for text in lyrics_df.loc[lyrics_df["lyrics_found"], "lyrics"]:
        all_tokens.extend(tokenize_lyrics(text))

    if not all_tokens:
        st.info("No hay tokens suficientes para construir el ranking de palabras.")
        return

    top_words = Counter(all_tokens).most_common(top_n_words)
    words = [w for w, _ in top_words]
    counts = [c for _, c in top_words]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(words, counts, color="steelblue")
    ax.set_title(f"Top {top_n_words} palabras más frecuentes en letras")
    ax.set_ylabel("Frecuencia")
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def plot_metric_by_album(lyrics_df: pd.DataFrame, metric: str):
    """
    Barras con el promedio de la métrica léxica seleccionada agrupado por álbum.
    Útil para ver qué álbumes tienen letras más complejas o extensas.
    """
    valid = lyrics_df[lyrics_df["lyrics_found"]].copy()
    if valid.empty:
        st.info("No hay letras recuperadas para calcular métricas.")
        return

    summary = (
        valid.groupby("album_clean")
        .agg(avg_metric=(metric, "mean"), n_tracks=("track_clean", "count"))
        .sort_values("avg_metric", ascending=False)
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(summary["album_clean"], summary["avg_metric"], color="teal")
    ax.set_title(f"Promedio de '{metric}' por álbum")
    ax.set_ylabel(metric)
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def plot_lexical_scatter(lyrics_df: pd.DataFrame):
    """
    Scatter: Diversidad léxica vs número de palabras por canción.
    Permite ver si las canciones con más palabras son también más diversas.
    El tamaño del punto representa la duración de la canción.
    """
    valid = lyrics_df[lyrics_df["lyrics_found"]].dropna(
        subset=["lexical_diversity", "word_count", "duration_ms"]
    )
    if valid.empty:
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    sc = ax.scatter(
        valid["word_count"],
        valid["lexical_diversity"],
        s=np.clip(valid["duration_ms"] / 1000, 30, 250),
        alpha=0.7,
        c=valid["valence"],
        cmap="RdYlGn",
    )
    plt.colorbar(sc, ax=ax, label="Valence (Spotify)")
    ax.set_xlabel("Palabras (sin stopwords)")
    ax.set_ylabel("Diversidad léxica (TTR)")
    ax.set_title("Diversidad léxica vs. Longitud de letra")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Comparación entre artistas
# ---------------------------------------------------------------------------

def build_artist_comparison(df_in: pd.DataFrame, max_tracks: int = 20) -> pd.DataFrame:
    """
    Construye una tabla comparativa con métricas medias de letras y audio
    para todos los artistas configurados.
    """
    rows = []
    for artist_name in ARTIST_CONFIG:
        artist_df = get_artist_subset(df_in, artist_name)
        tmp = build_lyrics_dataset(artist_df, artist_name, max_tracks=max_tracks)
        tmp = tmp[tmp["lyrics_found"]].copy()
        rows.append({
            "Artista": artist_name,
            "Tracks con letra": len(tmp),
            "Palabras (media)": round(tmp["word_count"].mean(), 1) if not tmp.empty else np.nan,
            "Palabras únicas (media)": round(tmp["unique_words"].mean(), 1) if not tmp.empty else np.nan,
            "Diversidad léxica": round(tmp["lexical_diversity"].mean(), 3) if not tmp.empty else np.nan,
            "Valence (media)": round(tmp["valence"].mean(), 3) if not tmp.empty else np.nan,
            "Energy (media)": round(tmp["energy"].mean(), 3) if not tmp.empty else np.nan,
        })
    return pd.DataFrame(rows)


def plot_comparison(comparison_df: pd.DataFrame):
    """
    Dos gráficas de barras para comparar artistas:
    1. Longitud y vocabulario medio de las letras.
    2. Diversidad léxica (en gráfica separada para evitar aplastamiento de escala).
    """
    fig1, ax1 = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(comparison_df))
    width = 0.35
    ax1.bar(x - width / 2, comparison_df["Palabras (media)"], width, label="Palabras (media)")
    ax1.bar(x + width / 2, comparison_df["Palabras únicas (media)"], width, label="Palabras únicas (media)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(comparison_df["Artista"])
    ax1.set_title("Longitud y vocabulario de las letras")
    ax1.set_ylabel("Palabras")
    ax1.legend()
    plt.tight_layout()
    st.pyplot(fig1)
    plt.close(fig1)

    fig2, ax2 = plt.subplots(figsize=(9, 4.5))
    ax2.bar(comparison_df["Artista"], comparison_df["Diversidad léxica"], color="green")
    ax2.set_title("Diversidad léxica media (Type-Token Ratio)")
    ax2.set_ylabel("TTR")
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close(fig2)


# ---------------------------------------------------------------------------
# App principal
# ---------------------------------------------------------------------------

def main():
    st.title("Spotify Studio Albums + Lyrics Explorer")
    st.markdown(
        """
        App interactiva que combina datos de audio de Spotify con análisis de letras.
        - **Audio**: variables de Spotify (valence, energy, acousticness…) filtradas a álbumes de estudio.
        - **Letras**: obtenidas de [lyrics.ovh](https://lyrics.ovh/) con métricas de diversidad léxica.
        - **Artistas**: The Strokes, The National y Elliott Smith.
        """
    )

    # Sidebar de controles
    with st.sidebar:
        st.header("Controles")
        artist_name = st.selectbox("Artista", list(ARTIST_CONFIG.keys()), index=0)
        metric = st.selectbox(
            "Métrica léxica",
            ["word_count", "unique_words", "lexical_diversity"],
            index=0,
            help="Métrica a mostrar en el gráfico por álbum."
        )
        max_tracks = st.slider(
            "Máx. canciones para letras",
            min_value=5, max_value=30, step=5, value=20,
            help="Limitar el número de canciones evita saturar la API de letras."
        )
        top_n_words = st.slider(
            "Top palabras",
            min_value=5, max_value=20, step=1, value=12,
        )
        show_raw = st.checkbox("Mostrar tabla completa filtrada", value=False)

    # Carga de datos
    try:
        df = load_data()
    except FileNotFoundError as e:
        st.error(str(e))
        st.stop()

    # ------------------------------------------------------------------
    # Sección 1: Cobertura del filtrado
    # ------------------------------------------------------------------
    st.subheader("1) Cobertura del filtrado de álbumes de estudio")
    st.caption(
        "El filtrado es manual porque el CSV no incluye un campo `album_type` fiable. "
        "Los tres artistas elegidos tienen buena cobertura y contrastes claros de audio y letras."
    )
    coverage_df = build_coverage_table(df)
    st.dataframe(coverage_df, use_container_width=True, hide_index=True)

    # ------------------------------------------------------------------
    # Sección 2: Explorador de audio
    # ------------------------------------------------------------------
    selected_df = get_artist_subset(df, artist_name)

    st.subheader(f"2) Explorador de audio — {artist_name}")
    if selected_df.empty:
        st.warning("No hay datos disponibles para el artista seleccionado.")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Canciones", len(selected_df))
        c2.metric("Álbumes de estudio en dataset", selected_df["album_clean"].nunique())
        c3.metric("Año más antiguo", int(selected_df["year"].min()))
        plot_audio_summary(selected_df, artist_name)

    # ------------------------------------------------------------------
    # Sección 3: Explorador de letras
    # ------------------------------------------------------------------
    st.subheader(f"3) Explorador de letras — {artist_name}")
    st.caption(
        "Las letras se obtienen de lyrics.ovh. "
        "Se analiza el vocabulario (words, unique words) y la diversidad léxica (TTR = unique/total)."
    )

    if selected_df.empty:
        st.warning("Sin datos de audio, no se puede analizar letras.")
    else:
        with st.spinner("Recuperando letras y calculando métricas léxicas…"):
            lyrics_df = build_lyrics_dataset(selected_df, artist_name, max_tracks=max_tracks)

        total_tracks = len(lyrics_df)
        found_tracks = int(lyrics_df["lyrics_found"].sum()) if "lyrics_found" in lyrics_df else 0
        coverage_pct = 100 * found_tracks / total_tracks if total_tracks else 0.0

        c1, c2, c3 = st.columns(3)
        c1.metric("Canciones analizadas", total_tracks)
        c2.metric("Letras encontradas", found_tracks)
        c3.metric("Cobertura", f"{coverage_pct:.1f}%")

        if found_tracks > 0:
            # Gráficos principales de letras
            left, right = st.columns(2)
            with left:
                plot_metric_by_album(lyrics_df, metric)
            with right:
                plot_top_words(lyrics_df, top_n_words)

            # Scatter diversidad léxica vs longitud (gráfico extra)
            st.markdown("#### Diversidad léxica vs. longitud de letra")
            st.caption(
                "Cada punto es una canción. Color = Valence de Spotify (verde = más positivo). "
                "Tamaño = duración. Un TTR alto indica vocabulario más variado."
            )
            plot_lexical_scatter(lyrics_df)

            # Tabla detallada por canción
            st.markdown("#### Detalle por canción")
            st.dataframe(
                lyrics_df[[
                    "album_clean", "track_clean", "lyrics_found",
                    "word_count", "unique_words", "lexical_diversity",
                    "valence", "energy",
                ]].rename(columns={
                    "album_clean": "Álbum",
                    "track_clean": "Canción",
                    "lyrics_found": "Letra encontrada",
                    "word_count": "Palabras",
                    "unique_words": "Únicas",
                    "lexical_diversity": "TTR",
                })
                .sort_values(["Álbum", "Canción"]),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.warning("No se han recuperado letras con la configuración actual.")

    # ------------------------------------------------------------------
    # Sección 4: Comparación entre artistas
    # ------------------------------------------------------------------
    st.subheader("4) Comparación entre artistas")
    st.caption(
        "Se comparan métricas léxicas y acústicas medias para los tres artistas. "
        "La diversidad léxica se presenta en gráfica separada para evitar que quede "
        "aplastada por la escala de palabras totales."
    )

    with st.spinner("Construyendo comparación entre artistas…"):
        comparison_df = build_artist_comparison(df, max_tracks=max_tracks)

    st.dataframe(comparison_df, use_container_width=True, hide_index=True)
    plot_comparison(comparison_df)

    # ------------------------------------------------------------------
    # Tabla raw (opcional)
    # ------------------------------------------------------------------
    if show_raw and not selected_df.empty:
        st.subheader(f"Tabla completa filtrada — {artist_name}")
        cols = [
            "name", "track_clean", "album", "album_clean", "year", "release_date",
            "danceability", "energy", "acousticness", "valence", "tempo", "duration_ms",
        ]
        available_cols = [c for c in cols if c in selected_df.columns]
        sort_cols = [c for c in ["year", "album_clean", "track_number"] if c in selected_df.columns]
        raw_df = selected_df.sort_values(sort_cols) if sort_cols else selected_df.copy()
        st.dataframe(raw_df[available_cols], use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown(
        "**Fuentes:** Spotify 1.2M+ Songs (Kaggle) · "
        "[lyrics.ovh](https://lyrics.ovh/) · "
        "Discografías y páginas oficiales de Spotify."
    )


if __name__ == "__main__":
    main()