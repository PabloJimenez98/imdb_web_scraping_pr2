import pandas as pd
from datetime import datetime
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

class Logger:
    @staticmethod
    def log(message):
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f"[{timestamp}] {message}")

class MovieDataExtractor:
    def __init__(self):
        self.logger = Logger()

    def extract_main_dataset(self):
        self.logger.log("Loading main dataset from bollywood_movie_list.csv")
        df_main = pd.read_csv('bollywood_movie_list.csv')
        df_main['title_normalized'] = df_main['title'].str.lower().str.strip()
        return df_main

    def extract_imdb_datasets(self):
        datasets = {}
        files = {
            'akas': 'title.akas.tsv',
            'basics': 'title.basics.tsv',
            'crew': 'title.crew.tsv',
            'ratings': 'title.ratings.tsv',
            'names': 'name.basics.tsv'
        }

        for key, file in files.items():
            self.logger.log(f"Loading {file}")
            datasets[key] = pd.read_csv(file, sep='\t')
            self.logger.log(f"{key} dataset loaded with {len(datasets[key])} rows")

        return datasets

class MovieDataTransformer:
    def __init__(self):
        self.logger = Logger()

    def merge_initial_datasets(self, df_main, df_akas):
        df_akas_slim = df_akas[['primaryTitle', 'tconst']].copy()
        df_akas_slim['title_normalized'] = df_akas_slim['primaryTitle'].str.lower().str.strip()
        df_akas_slim = df_akas_slim.drop_duplicates(subset=['title_normalized'], keep='first')
        
        merged = pd.merge(df_main, df_akas_slim, on='title_normalized', how='left')
        self.logger.log(f"Initial merge completed with {len(merged)} rows")
        return merged

    def process_alternative_titles(self, merged_df, akas_df):
        akas_grouped = akas_df.groupby('titleId')['title'].apply(
            lambda x: ', '.join(str(val) for val in x if pd.notna(val))
        ).reset_index()
        akas_grouped.columns = ['titleId', 'alternative_titles']
        return pd.merge(merged_df, akas_grouped, left_on='tconst', right_on='titleId', how='left')

    def process_crew_data(self, merged_df, crew_df, names_df):
        names_map = names_df.set_index('nconst')['primaryName'].to_dict()

        def process_crew_column(df, column_name):
            df = df.merge(crew_df[['tconst', column_name]], on='tconst', how='left')
            df[column_name] = df[column_name].fillna('\\N')
            df = df.assign(temp_id=df[column_name].str.split(','))
            df = df.explode('temp_id')
            df[f'{column_name}_name'] = df['temp_id'].map(names_map).fillna('')
            
            names_grouped = df.groupby('tconst')[f'{column_name}_name'].agg(
                lambda x: ', '.join(name for name in x if name)
            ).reset_index(name=column_name)
            
            return df.drop([column_name, 'temp_id', f'{column_name}_name'], axis=1).drop_duplicates(), names_grouped

        merged_df, director_names = process_crew_column(merged_df, 'directors')
        merged_df, writer_names = process_crew_column(merged_df, 'writers')

        merged_df = merged_df.merge(director_names, on='tconst', how='left')
        merged_df = merged_df.merge(writer_names, on='tconst', how='left')
        
        return merged_df

    def clean_numerical_data(self, data):
        # Clean year and duration
        data['year'] = data['year'].replace(['No Year', '\\N'], np.nan).fillna(data['startYear']).fillna(0)
        data['year'] = pd.to_numeric(data['year'], errors='coerce').fillna(0).astype(int)
        
        data['duration'] = data['duration'].replace(['No Duration', '\\N'], np.nan).fillna(data['runtimeMinutes']).fillna(0)
        data['duration'] = pd.to_numeric(data['duration'], errors='coerce').fillna(0).astype(int)

        # Clean votes
        data['votes'] = data['votes'].astype(str).apply(self._clean_number)
        data['votes'] = pd.to_numeric(data['votes'], errors='coerce').fillna(0).astype(int)

        # Clean other numerical columns
        data['age_rating'] = data['age_rating'].replace(['No Age Rating', '\\N'], np.nan).fillna('Unknown')
        data['imdb_rating'] = pd.to_numeric(data['imdb_rating'], errors='coerce').fillna(0.0)
        data['metascore'] = pd.to_numeric(data['metascore'], errors='coerce').fillna(0).astype(int)

        return data

    @staticmethod
    def _clean_number(x):
        x = (x.replace('(', '')
             .replace(')', '')
             .replace(',', '')
             .replace(' ', '')
             .replace('\xa0', '')
             .strip())
        x = x.replace('mil', '000')
        return '0' if x in ('', '\\N', 'nan') else x

    def process_genres(self, data):
        # Create genre dummy columns
        genre_columns = data['genres'].str.get_dummies(sep=',')
        data = pd.concat([data, genre_columns], axis=1)

        # Combine genres
        data['combined_genres'] = data.apply(self._combine_genres, axis=1)
        
        return data

    @staticmethod
    def _combine_genres(row):
        genres = set()
        if pd.notna(row['genres']):
            genres.update(row['genres'].split(','))
        for col in ['genre1', 'genre2', 'genre3']:
            if pd.notna(row[col]):
                genres.add(row[col])
        return ','.join(genres)

class MovieDataLoader:
    def __init__(self):
        self.logger = Logger()

    def save_final_dataset(self, data, filename='cleaned_movies_data.csv'):
        self.logger.log(f"Saving cleaned dataset to {filename}")
        data.to_csv(filename, index=False)
        self.logger.log("Dataset saved successfully")

class DatasetVisualizer:
    def __init__(self, csv_path, output_dir="image"):
        self.csv_path = csv_path
        self.output_dir = output_dir
        self.data = None
        self._load_data()
        self._prepare_output_dir()

    def _load_data(self):
        """Load the dataset from the given CSV path."""
        self.data = pd.read_csv(self.csv_path)

    def _prepare_output_dir(self):
        """Create the output directory if it doesn't exist."""
        os.makedirs(self.output_dir, exist_ok=True)

    def plot_imdb_rating_distribution(self):
        """Create and save a histogram of IMDb ratings."""
        plt.figure(figsize=(10, 6))
        sns.histplot(self.data['imdb_rating'], bins=20, kde=True, color="blue")
        plt.title("Distribución de Puntuaciones IMDb", fontsize=16)
        plt.xlabel("Puntuación IMDb", fontsize=12)
        plt.ylabel("Número de Películas", fontsize=12)
        plt.grid(True)
        plt.savefig(os.path.join(self.output_dir, "imdb_rating_distribution.png"))
        plt.close()

    def plot_genre_frequency(self):
        """Create and save a bar plot of the most frequent genres."""
        genre_cols = [col for col in self.data.columns if col in ["Action", "Drama", "Comedy", "Adventure", "Horror"]] # Adjust genres as needed
        genre_counts = self.data[genre_cols].sum().sort_values(ascending=False)

        plt.figure(figsize=(12, 8))
        sns.barplot(x=genre_counts.index, y=genre_counts.values, palette="viridis")
        plt.title("Frecuencia de Géneros", fontsize=16)
        plt.xlabel("Género", fontsize=12)
        plt.ylabel("Cantidad de Películas", fontsize=12)
        plt.grid(True)
        plt.savefig(os.path.join(self.output_dir, "genre_frequency.png"))
        plt.close()

    def plot_imdb_rating_vs_votes(self):
        """Create and save a scatter plot of IMDb ratings vs. number of votes."""
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=self.data, x="votes", y="imdb_rating", hue="year", palette="coolwarm", size="duration", sizes=(20, 200), alpha=0.6)
        plt.title("Puntuaciones IMDb vs. Número de Votos", fontsize=16)
        plt.xlabel("Número de Votos", fontsize=12)
        plt.ylabel("Puntuación IMDb", fontsize=12)
        plt.legend(title="Año", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True)
        plt.savefig(os.path.join(self.output_dir, "imdb_rating_vs_votes.png"))
        plt.close()

    def plot_movies_per_year(self):
        """Create and save a line plot showing the number of movies released per year."""
        movies_per_year = self.data['year'].value_counts().sort_index()

        plt.figure(figsize=(12, 6))
        sns.lineplot(x=movies_per_year.index, y=movies_per_year.values, marker="o", color="green")
        plt.title("Número de Películas por Año", fontsize=16)
        plt.xlabel("Año", fontsize=12)
        plt.ylabel("Número de Películas", fontsize=12)
        plt.grid(True)
        plt.savefig(os.path.join(self.output_dir, "movies_per_year.png"))
        plt.close()

    def plot_duration_distribution(self):
        """Create and save a boxplot of movie durations."""
        plt.figure(figsize=(10, 6))
        sns.boxplot(x=self.data['duration'], color="orange")
        plt.title("Distribución de Duración de Películas", fontsize=16)
        plt.xlabel("Duración (minutos)", fontsize=12)
        plt.grid(True)
        plt.savefig(os.path.join(self.output_dir, "duration_distribution.png"))
        plt.close()

    def plot_top_directors(self):
        """Create and save a bar plot of the top directors by number of movies."""
        top_directors = self.data['directors'].value_counts().head(10)

        plt.figure(figsize=(12, 8))
        sns.barplot(x=top_directors.values, y=top_directors.index, palette="magma")
        plt.title("Top 10 Directores por Número de Películas", fontsize=16)
        plt.xlabel("Número de Películas", fontsize=12)
        plt.ylabel("Directores", fontsize=12)
        plt.grid(True)
        plt.savefig(os.path.join(self.output_dir, "top_directors.png"))
        plt.close()

    def generate_all_plots(self):
        """Generate all visualizations and save them to the output directory."""
        self.plot_imdb_rating_distribution()
        self.plot_genre_frequency()
        self.plot_imdb_rating_vs_votes()
        self.plot_movies_per_year()
        self.plot_duration_distribution()
        self.plot_top_directors()


def main():
    logger = Logger()
    logger.log("Starting ETL process")

    # Extract
    extractor = MovieDataExtractor()
    df_main = extractor.extract_main_dataset()
    imdb_data = extractor.extract_imdb_datasets()

    # Transform
    transformer = MovieDataTransformer()
    
    # Initial merge
    merged_df = transformer.merge_initial_datasets(df_main, imdb_data['akas'])
    
    # Process alternative titles and crew data
    merged_df = transformer.process_alternative_titles(merged_df, imdb_data['akas'])
    merged_df = transformer.process_crew_data(merged_df, imdb_data['crew'], imdb_data['names'])
    
    # Clean and process data
    merged_df = transformer.clean_numerical_data(merged_df)
    merged_df = transformer.process_genres(merged_df)

    # Load
    loader = MovieDataLoader()
    loader.save_final_dataset(merged_df)

    logger.log("ETL process completed")

    visualizer = DatasetVisualizer("cleaned_movies_data.csv")
    visualizer.generate_all_plots()

if __name__ == "__main__":
    main()