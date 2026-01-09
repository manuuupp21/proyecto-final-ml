# Proyecto Final - Machine Learning

## Descripción del Proyecto

Este proyecto tiene como objetivo desarrollar y evaluar modelos de clasificación binaria para predecir la cancelación de reservas hoteleras. A partir de un conjunto de datos histórico que incluye información sobre el tipo de hotel, características del cliente, comportamiento de reserva y detalles de la estancia.

## Estructura del Proyecto

```plaintext
proyecto-final-ml/
├── .gitignore                       # Archivos que no se suben al repo (por ejemplo, modelos, datos temporales)
├── data/
│   ├── raw/                         # Datos originales sin procesar (por ejemplo, CSV descargado o Parquet)
│   └── processed/                   # Datos tras limpieza y transformación (listos para modelar)
├── docs/                            # Documentación adicional
├── models/                          # Modelos entrenados (guardados con joblib o pickle)
│   ├── tests/                       # Modelos intermedios de prueba
│   │   ├── logistic_regression.pkl  # Ejemplo: Modelo de Regresión Logística
│   │   ├── tree.pkl                 # Ejemplo: Modelo Árbol de Decisión
│   │   ├── random_forest.pkl        # Ejemplo: Modelo Random Forest
│   │   ├── xgboost.pkl              # Ejemplo: Modelo XGBoost
│   │   ├── lightgbm.pkl             # Ejemplo: Modelo LightGBM
│   │   ├── neural_network.pkl       # Ejemplo: Red Neuronal
│   │   └── best_model.pkl           # El mejor modelo seleccionado para producción
├── notebooks/                       # Todos los notebooks del proyecto
│   ├── exploracion/                 # Notebooks de pruebas, EDA inicial, prototipos
│   │   ├── eda_inicial.ipynb        # Ejemplo: análisis inicial del dataset
│   │   └── pruebas_modelos.ipynb    # Ejemplo: pruebas de diferentes modelos
│   ├── finales/                     # Notebooks finales con resultados o presentación
│   │   ├── eda_final.ipynb          # Ejemplo: EDA del dataset
│   │   └── comparativa_modelos.ipynb # Ejemplo: comparación final de modelo usada para los scripts
├── outputs/                         # Gráficos, reportes y resultados generados (PNGs, HTMLs si son interactivos, etc)
│   ├── confusion_matrix.png         # Ejemplo: Matriz de confusión
├── src/                             # Código fuente del proyecto
│   ├── __init__.py                  # Inicializador del paquete src
│   ├── config.py                    # Parámetros y configuración del proyecto
│   ├── data_prep.py                 # Funciones para cargar y preperar los datos
│   ├── main.py                      # Programa Principal con ejemplo de ejecución
│   ├── models_evaluator.py          # Funciones para la evaluación de los modelos
│   ├── models_trainer.py            # Clase para entrenar y seleccionar el mejor modelo
│   ├── models.py                    # Funciones para entrenar los distintos modelos
├── requirements.txt                 # Dependencias del proyecto
└── README.md                        # Documentación principal del proyecto con comandos de ejecución
