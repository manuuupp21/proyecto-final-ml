from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score
from .config import RANDOM_STATE, MODELS_DIR
from .model_evaluator import ModelEvaluator
import joblib




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
    metrics = evaluator.evaluate(best_model, X_test, y_test)

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
# ------------------------