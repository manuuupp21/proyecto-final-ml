from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from config import RANDOM_STATE, MODELS_DIR
from models_evaluator import ModelEvaluator
import joblib

import numpy as np
import tensorflow as tf
import autokeras as ak

from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, roc_auc_score
import dill as pickle

def train_with_gridsearch(
    model,
    param_grid,
    X_train,
    y_train,
    X_test,
    y_test,
    model_filename,
    scoring="roc_auc",
    cv=5,
    n_jobs=-1,
):
    
    #Lógica común de entrenamiento + evaluación + guardado
    model_cv = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=cv,
        scoring=scoring,
        n_jobs=n_jobs,
    )

    model_cv.fit(X_train, y_train)

    print(f"Mejores hiperparámetros encontrados: {model_cv.best_params_}")
    print(f"Mejor score obtenido: {model_cv.best_score_:.2%}")

    best_model = model_cv.best_estimator_

    evaluator = ModelEvaluator()
    metrics = evaluator.evaluate(best_model, X_test, y_test, threshold=0.5)

    model_path = MODELS_DIR / model_filename
    joblib.dump(best_model, model_path)

    return {
        "model": best_model,
        "metrics": metrics
    }

# ------------------------
# Entrenamiento Regresión Logística
# ------------------------
def train_logistic_regression(
    X_train,
    y_train,
    X_test,
    y_test,
):
    param_grid = {
        "C": [0.01, 0.1, 1, 10, 100],
        "solver": ["liblinear"],
    }

    model = LogisticRegression(
        max_iter=1000,
        random_state=42
    )

    return train_with_gridsearch(
        model=model,
        param_grid=param_grid,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        model_filename="logistic_regression_best_model.pkl",
    )

# ------------------------
# Entrenamiento Random Forest
# ------------------------
def train_random_forest(
    X_train,
    y_train,
    X_test,
    y_test,
    random_state=RANDOM_STATE,
):
    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [None, 15],
        "min_samples_leaf": [1, 3],
        "max_features": ["sqrt"],
    }

    model = RandomForestClassifier(
        random_state=random_state
    )

    return train_with_gridsearch(
        model=model,
        param_grid=param_grid,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        model_filename="random_forest_best_model.pkl",
    )

# ------------------------
# Entrenamiento XGBoost
# ------------------------
def train_xgboost(
    X_train,
    y_train,
    X_test,
    y_test,
    random_state=RANDOM_STATE,
):
    param_grid = {
        "n_estimators": [100, 200],
        "learning_rate": [0.05, 0.1],
        "max_depth": [3, 5],
        "subsample": [0.8],
        "colsample_bytree": [0.8]
    }

    model = XGBClassifier(
        random_state=random_state,
    )

    return train_with_gridsearch(
        model=model,
        param_grid=param_grid,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        model_filename="xgboost_best_model.pkl",
    )


# ------------------------
# Entrenamiento Red Neuronal
# Usando Autokeras, con pesos de clase y ajuste de umbral.
# ------------------------
def train_neural_network(
    X_train,
    y_train,
    X_test,
    y_test,
    threshold: float = 0.35,
):
    # Cálculo de pesos de clase
    classes = np.unique(y_train)
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y_train
    )
    class_weight_dict = dict(zip(classes, class_weights))
    print(class_weight_dict)
    # Definición del modelo AutoKeras
    clf = ak.StructuredDataClassifier(
        max_trials=10,              
        objective="val_accuracy",
        overwrite=True,
        seed=RANDOM_STATE
    )
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=5,
        mode="max",
        restore_best_weights=True
    )
    # Entrenamiento
    clf.fit(
        X_train.toarray(),
        y_train.to_numpy(),
        epochs=40,
        validation_split=0.2,
        class_weight=class_weight_dict,
        callbacks=[early_stop],
        verbose=1
    )
  

    # Exportar mejor modelo
    best_model = clf.export_model()
    best_model.summary()

    # Evaluación UNIFICADA
    evaluator = ModelEvaluator()
    metrics = evaluator.evaluate(
        best_model,
        X_test.toarray(),
        y_test.to_numpy(),
        threshold=threshold
    )

    # Guardado
    with open(MODELS_DIR / "neural_network_best_model.pkl", "wb") as f:
        pickle.dump(best_model, f)

    return {
        "model": best_model,
        "metrics": metrics
    }