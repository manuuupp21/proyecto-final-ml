from data_prep import prepare_data_for_train 
from models_trainer import ModelsTrainer

if __name__ == "__main__":
    # Preparar datos
    X_train, X_test, y_train, y_test, scaler, encoder = prepare_data_for_train()

    # Crear y entrenar los distintos modelos y guarda los resultados
    model_trainer = ModelsTrainer()
    model_trainer.run(X_train, y_train, X_test, y_test)
    results_df = model_trainer.save_results()
    print("\nResultados de los modelos:\n", results_df)

