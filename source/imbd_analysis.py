import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.cluster import KMeans
from scipy.stats import shapiro, levene, ttest_ind, mannwhitneyu
from sklearn.decomposition import PCA


class MovieAnalysis:
    def __init__(self, input_path, output_folder, dataset_output_folder):
        self.input_path = input_path
        self.output_folder = output_folder
        self.dataset_output_folder = dataset_output_folder
        self.data = None
        self.processed_data = None
        self.clusters = None
        self._create_folders()

    def _create_folders(self):
        """
        Crea los directorios necesarios si no existen.
        """
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)
        if not os.path.exists(self.dataset_output_folder):
            os.makedirs(self.dataset_output_folder)

    def load_and_clean_data(self):
        """
        Carga y limpia los datos.
        """
        if not os.path.exists(self.input_path):
            raise FileNotFoundError(f"Archivo no encontrado en '{self.input_path}'.")

        data = pd.read_csv(self.input_path)
        data["duration"].replace(0, data["duration"].mean(), inplace=True)
        data["votes"].fillna(data["votes"].median(), inplace=True)
        data["metascore"].fillna(data["metascore"].median(), inplace=True)
        data["combined_genres"].fillna("Unknown", inplace=True)
        data["directors"].fillna("Unknown", inplace=True)
        self.data = data

    def supervised_analysis(self):
        """
        Realiza el análisis supervisado para predecir el rating de IMDb.
        """
        X = self.data.drop("imdb_rating", axis=1)
        y = self.data["imdb_rating"]

        # Preprocesamiento
        numerical_features = ["duration", "votes", "metascore"]
        categorical_features = ["combined_genres", "directors"]
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), numerical_features),
                ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
            ]
        )

        # Pipeline
        gb_model = GradientBoostingRegressor(random_state=42)
        param_grid = {
            "n_estimators": [50, 100, 200],
            "learning_rate": [0.01, 0.1, 0.2],
            "max_depth": [3, 5, 7],
        }
        grid_search = GridSearchCV(
            estimator=gb_model,
            param_grid=param_grid,
            scoring="neg_mean_squared_error",
            cv=3,
            verbose=2,
            n_jobs=-1,
        )
        pipeline = Pipeline(
            steps=[("preprocessor", preprocessor), ("regressor", grid_search)]
        )

        # Dividir los datos
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Entrenar el modelo
        pipeline.fit(X_train, y_train)

        # Evaluación
        test_predictions = pipeline.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, test_predictions))
        print(f"RMSE en conjunto de prueba: {rmse:.4f}")

        # Guardar gráfico
        plt.figure(figsize=(8, 6))
        plt.scatter(y_test, test_predictions, alpha=0.7)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
        plt.xlabel("Ratings Reales")
        plt.ylabel("Ratings Predichos")
        plt.title("Real vs Predicho")
        plt.savefig(f"{self.output_folder}/real_vs_predicted.png")
        plt.close()
        print("Gráfico de análisis supervisado guardado.")

    def unsupervised_analysis(self, n_clusters=5):
        """
        Realiza un análisis no supervisado utilizando K-Means y genera clusters.
        """
        numerical_features = ["duration", "votes", "metascore"]
        categorical_features = ["combined_genres", "directors"]
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), numerical_features),
                ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
            ]
        )

        self.processed_data = preprocessor.fit_transform(self.data)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        self.data["Cluster"] = kmeans.fit_predict(self.processed_data)

        # Guardar el dataset con clusters
        output_file = os.path.join(
            self.dataset_output_folder, "cleaned_dataset_with_clusters.csv"
        )
        self.data.to_csv(output_file, index=False)
        print(f"Dataset con clusters guardado en: {output_file}")

        # Visualización
        dense_data = self.processed_data.toarray()
        pca = PCA(n_components=2, random_state=42)
        reduced_data = pca.fit_transform(dense_data)

        plt.figure(figsize=(10, 8))
        sns.scatterplot(
            x=reduced_data[:, 0],
            y=reduced_data[:, 1],
            hue=self.data["Cluster"],
            palette="viridis",
        )
        plt.title("Clusters de Películas (PCA)")
        plt.xlabel("Componente Principal 1")
        plt.ylabel("Componente Principal 2")
        plt.legend(title="Cluster")
        plt.savefig(f"{self.output_folder}/clusters_pca.png")
        plt.close()
        print("Gráfico de clusters guardado.")

    def hypothesis_test(self, cluster1, cluster2):
        """
        Realiza un contraste de hipótesis sobre los votos entre dos clusters.
        """
        votes_cluster1 = self.data[self.data["Cluster"] == cluster1]["votes"]
        votes_cluster2 = self.data[self.data["Cluster"] == cluster2]["votes"]

        # Normalidad
        p_normal1 = shapiro(votes_cluster1)[1]
        p_normal2 = shapiro(votes_cluster2)[1]
        print(f"Normalidad (Cluster {cluster1}): p-value = {p_normal1:.4f}")
        print(f"Normalidad (Cluster {cluster2}): p-value = {p_normal2:.4f}")

        # Homocedasticidad
        p_levene = levene(votes_cluster1, votes_cluster2)[1]
        print(f"Homocedasticidad (Levene): p-value = {p_levene:.4f}")

        # Prueba de hipótesis
        if p_normal1 > 0.05 and p_normal2 > 0.05 and p_levene > 0.05:
            stat, p_value = ttest_ind(votes_cluster1, votes_cluster2)
            test_used = "t-test"
        else:
            stat, p_value = mannwhitneyu(
                votes_cluster1, votes_cluster2, alternative="two-sided"
            )
            test_used = "Mann-Whitney U"

        print(f"Prueba usada: {test_used}")
        print(f"Resultado: estadístico = {stat:.4f}, p-value = {p_value:.4f}")

        if p_value < 0.05:
            print(
                "Conclusión: Hay evidencia suficiente para rechazar la hipótesis nula."
            )
        else:
            print("Conclusión: No se puede rechazar la hipótesis nula.")


# Flujo Principal
if __name__ == "__main__":
    input_path = "../datasets/cleaned_movies_data_new.csv"  # Ruta al archivo dentro de 'datasets'
    output_folder = "../image"  # Carpeta para guardar imágenes
    dataset_output_folder = "../datasets"  # Carpeta para guardar dataset con clusters
    analysis = MovieAnalysis(
        input_path=input_path,
        output_folder=output_folder,
        dataset_output_folder=dataset_output_folder,
    )
    analysis.load_and_clean_data()
    analysis.supervised_analysis()
    analysis.unsupervised_analysis(n_clusters=5)
    analysis.hypothesis_test(cluster1=0, cluster2=1)
