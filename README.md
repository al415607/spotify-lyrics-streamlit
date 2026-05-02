# Spotify Studio Albums + Lyrics Explorer

Web app interactiva desarrollada con **Streamlit** para analizar canciones de Spotify y letras de canciones de tres artistas: **The Strokes**, **The National** y **Elliott Smith**.

La aplicación combina variables de audio del dataset de Spotify con letras obtenidas mediante la API pública `lyrics.ovh`, permitiendo comparar rasgos acústicos y métricas léxicas entre artistas y álbumes.

## Archivos incluidos

- `app.py`: aplicación principal en Streamlit.
- `requirements.txt`: dependencias necesarias para ejecutar la app.
- `tracks_features.csv`: dataset filtrado con las canciones usadas en la aplicación.
- `lyrics_cache_streamlit.json`: caché local de letras para evitar llamadas repetidas a la API.

## Nota sobre el dataset

El dataset original de Spotify era demasiado grande para subirlo directamente al repositorio y para desplegarlo de forma cómoda en Streamlit Community Cloud.

Por ese motivo, se ha generado una versión reducida de `tracks_features.csv` que conserva únicamente las canciones de los tres artistas analizados en la aplicación:

- The Strokes
- The National
- Elliott Smith

Esta reducción no afecta al funcionamiento de la app, ya que el análisis se centra exclusivamente en esos artistas. Además, permite que el repositorio sea más ligero y que la aplicación cargue más rápido.

## Ejecución local

Instalar dependencias:

```bash
pip install -r requirements.txt
```

Ejecutar la aplicación:

```bash
py -m streamlit run app.py
```

También puede ejecutarse con:

```bash
streamlit run app.py
```

## Qué incluye la app

- Filtrado de álbumes de estudio mediante un catálogo manual.
- Tabla de cobertura para ver qué álbumes están presentes en el dataset.
- Explorador de variables de audio por artista y álbum.
- Visualización de métricas como `valence`, `energy`, `acousticness`, `tempo` y duración.
- Explorador de letras usando `lyrics.ovh`.
- Cálculo de métricas léxicas:
  - número de palabras,
  - palabras únicas,
  - diversidad léxica o Type-Token Ratio.
- Ranking de palabras más frecuentes.
- Comparación entre artistas mediante métricas acústicas y léxicas.

## Artistas analizados

Los artistas seleccionados son:

- **The Strokes**: grupo de indie/garage rock con evolución clara entre álbumes.
- **The National**: banda de indie rock melancólico con letras introspectivas.
- **Elliott Smith**: cantautor de folk-rock acústico con letras personales e introspectivas.

Se eligieron estos artistas porque tienen una cobertura razonable en el dataset y permiten comparar estilos musicales y líricos diferentes.

## Fuentes de datos

- Dataset de Spotify: `tracks_features.csv`.
- Letras: API pública de `lyrics.ovh`.
- Catálogo de álbumes de estudio: elaboración manual a partir de discografías oficiales y referencias públicas.

## Limitaciones

La recuperación de letras depende de la disponibilidad de cada canción en `lyrics.ovh`, por lo que algunas letras pueden no encontrarse.

Además, el dataset original no incluye un campo fiable de tipo de álbum, por eso el filtrado de álbumes de estudio se realiza manualmente.