from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


class ModelEvaluator:
    """
    Encapsula la evaluación de un modelo de clasificación binaria.
    """

    def evaluate(self, model, X_test, y_test):
        # Predicciones
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        # Métricas
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)

        # Prints de métricas
        print(f"Accuracy:  {acc:.2%}")
        print(f"Precisión: {prec:.2%}")
        print(f"Recall:    {rec:.2f}")
        print(f"F1-Score:  {f1:.2f}")
        print(f"AUC:       {auc:.2f}\n")

        metrics = {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "auc": auc
        }

        return metrics
