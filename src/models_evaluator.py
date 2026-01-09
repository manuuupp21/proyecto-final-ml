import numpy as np
import matplotlib.pyplot as plt
import os

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)


class ModelEvaluator:
    """
    Encapsula la evaluación de un modelo de clasificación binaria y guarda matrices de confusión.
    """
    _global_cm_counter = 1   # contador de clase

    def __init__(self):
        self.cm_files = []   # Lista de ficheros generados

    def evaluate(self, model, X_test, y_test, threshold=0.5):
        # Probabilidades
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)[:, 1]
        else:
            y_proba = model.predict(X_test).ravel()

        # Clases predichas
        y_pred = (y_proba > threshold).astype(int)

        # Métricas
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)

        print(f"Accuracy:  {acc:.2%}")
        print(f"Precisión: {prec:.2%}")
        print(f"Recall:    {rec:.2f}")
        print(f"F1-Score:  {f1:.2f}")
        print(f"AUC:       {auc:.2f}\n")

        # Guardar matriz de confusión
        self._save_confusion_matrix(y_test, y_pred)

        metrics = {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "auc": auc
        }

        return metrics

    def _save_confusion_matrix(self, y_test, y_pred):
        output_dir = os.path.join(".", "outputs")
        os.makedirs(output_dir, exist_ok=True)

        filename = f"cm_{ModelEvaluator._global_cm_counter}.png"
        save_path = os.path.join(output_dir, filename)

        cm = confusion_matrix(y_test, y_pred)

        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=["no cancelada", "cancelada"]
        )

        fig, ax = plt.subplots(figsize=(5, 5))
        disp.plot(ax=ax, cmap="Blues", colorbar=False)

        plt.title(f"Matriz de Confusión ({filename})")
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close(fig)

        # Guardar nombre y aumentar contador
        self.cm_files.append(filename)
        ModelEvaluator._global_cm_counter += 1

        print(f" Matriz de confusión guardada: {save_path}")


