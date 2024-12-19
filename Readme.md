Here's a README.md for your project:

```markdown
# Práctica 2: Limpieza y Análisis de Datos de Películas de Bollywood

## Descripción
Este proyecto consiste en un proceso ETL (Extract, Transform, Load) para datos de películas de Bollywood, combinando información de múltiples fuentes y generando visualizaciones relevantes. El sistema procesa datos de películas incluyendo calificaciones IMDb, información de elenco, géneros y más.

## Estructura del Proyecto
El proyecto está organizado en varias clases principales:

- `Logger`: Manejo de logs con timestamps
- `MovieDataExtractor`: Extracción de datos de archivos CSV/TSV
- `MovieDataTransformer`: Transformación y limpieza de datos
- `MovieDataLoader`: Guardado de datos procesados
- `DatasetVisualizer`: Generación de visualizaciones

## Requisitos
```python
pandas
numpy
matplotlib
seaborn
```

## Datasets Necesarios
El script requiere los siguientes archivos de entrada:
- `bollywood_movie_list.csv`: Lista principal de películas de Bollywood
- `title.akas.tsv`: Títulos alternativos de IMDb
- `title.basics.tsv`: Información básica de títulos de IMDb
- `title.crew.tsv`: Información de equipo técnico
- `title.ratings.tsv`: Calificaciones de IMDb
- `name.basics.tsv`: Información básica de personas

## Funcionalidades Principales

### Extracción de Datos
- Carga de múltiples datasets
- Normalización inicial de títulos

### Transformación de Datos
- Merge de datasets basado en títulos
- Procesamiento de títulos alternativos
- Limpieza de datos numéricos
- Procesamiento de información de géneros
- Manejo de información de directores y escritores

### Visualizaciones
El script genera las siguientes visualizaciones:
1. Distribución de puntuaciones IMDb
2. Frecuencia de géneros
3. Relación entre puntuaciones IMDb y número de votos
4. Películas por año
5. Distribución de duración de películas
6. Top 10 directores por número de películas

## Uso
```python
python imdb_etl.py
```

## Salidas
- `cleaned_movies_data.csv`: Dataset procesado y limpio
- Carpeta `image/`: Contiene todas las visualizaciones generadas

## Autores
Pablo Barranco López & Pablo Jiménez Cruz


## Licencia
CC BY-SA 4.0

Para instalar todas las dependencias ejecutando:
```bash
pip install -r requirements.txt
```

Para ejecutar el programa (Puede tardar varios minutos):
```bash
python -m source 
```