
from config import  MODELS_DIR
import pandas as pd
from models import (
    train_logistic_regression,
    train_random_forest,
    train_xgboost,
    train_neural_network,
)

# ------------------------
# Clase para ejecutar y comparar varios modelos
# ------------------------
class ModelsTrainer:

    def __init__(self, metric="auc"):
        self.metric = metric
        self.results = []
        self.best_model = None
        self.best_model_name = None

    def run(self, X_train, y_train, X_test, y_test):
        models = [
            ("logistic_regression", train_logistic_regression),
            ("random_forest", train_random_forest),
            ("xgboost", train_xgboost),
            ("neural_network", train_neural_network),
        ]

        #Metricas rango de 0-1
        best_score = -1

        for name, train_fn in models:
            print(f"\nEntrenando modelo: {name}")

            result = train_fn(X_train, y_train, X_test, y_test)

            metrics = result["metrics"]
            score = metrics[self.metric]

            self.results.append({
                "model": name,
                # Pasa el Diccionario a clave/valor
                **metrics
            })

            if score > best_score:
                best_score = score
                self.best_model = result["model"]
                self.best_model_name = name

    def save_results(self, filename="model_results.csv"):
        df = pd.DataFrame(self.results)
        df.to_csv(MODELS_DIR / filename, index=False)
        return df