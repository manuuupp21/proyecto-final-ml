import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from scipy.sparse import csr_matrix, hstack
from config import DATASET_PATH, TEST_SIZE, RANDOM_STATE


SCALER = StandardScaler()
ENCODER = OneHotEncoder(handle_unknown="ignore")

# ------------------------
# Carga de datos
# ------------------------
def load_data(path=DATASET_PATH):
    return pd.read_csv(path)

# ------------------------
# Limpieza datos
# ------------------------
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    print("Ejecutando Cleaning de datos...")
    # Eliminar duplicados
    df = df.copy()

    df = df.drop_duplicates()

    # Convertir arrival_date_month a formato numérico
    df['arrival_date_month'] = df['arrival_date_month'].replace({
        'January': 1,'February': 2,'March': 3,'April': 4,
        'May': 5,'June': 6,'July': 7,'August': 8,'September': 9,
        'October': 10,'November': 11,'December': 12
    })

    # Rellenar valores nulos en 'children' con 0
    df['children'] = df['children'].fillna(0)

    #Para country se decidie que el valor NA será representado por 'Unknown'
    df['country'] = df['country'].fillna('Unknown')

    # Crear variable binaria has_company para 'company'
    df['has_company'] = df['company'].notna().astype(int)
    df = df.drop(columns=['company'])

    # Decido eliminar la variable Agent tiene una alta cardinalidad y bajo valor semantico
    df = df.drop(columns=['agent'])

    #Estas columnas no se van a tener en cuenta para el modelo
    df = df.drop(columns=['reservation_status', 'reservation_status_date'])

    # Eliminar valores negativos en adr
    df = df[df['adr'] >= 0]

    return df

# ------------------------
# Preprocesamiento para entrenamiento
# ------------------------
def preprocess_for_train(df, target_col="is_canceled", scaler=SCALER, encoder=ENCODER):
    print("Ejecutando Preprocesamiento de datos para entrenamiento...")
    X = df.drop(columns=target_col)
    y = df[target_col]

    num_cols = X.select_dtypes(include=["int64", "float64"]).columns
    cat_cols = X.select_dtypes(include="object").columns

    #Solo se hace fit en el momento de entrenar el modelo
    X_num = scaler.fit_transform(X[num_cols]) 
    X_cat = encoder.fit_transform(X[cat_cols]) 

    X = hstack([csr_matrix(X_num), X_cat])

    return X, y, scaler, encoder

# ------------------------
# Preprocesamiento para predicción
# ------------------------
def preprocess_for_predict(df, scaler=SCALER, encoder=ENCODER):
    print("Ejecutando Preprocesamiento de datos para predicción...")
    num_cols = df.select_dtypes(include=["int64", "float64"]).columns
    cat_cols = df.select_dtypes(include="object").columns

    X_num = scaler.transform(df[num_cols])
    X_cat = encoder.transform(df[cat_cols])

    X = hstack([csr_matrix(X_num), X_cat])

    return X

# ------------------------
# División train-test
# ------------------------
def split_ml(X, y, test_size=0.2, random_state=42):
    print("Ejecutando división train-test...")
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state
    )

# ------------------------
# Pipeline completo para entrenamiento
# ------------------------
def prepare_data_for_train(scaler=SCALER, encoder=ENCODER):
    print("Ejecutando Pipeline completo para entrenamiento...")
    df = load_data()
    df = clean_data(df)
    X, y, scaler, encoder = preprocess_for_train(df,scaler=scaler,encoder= encoder)
    X_train, X_test, y_train, y_test = split_ml(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    return X_train, X_test, y_train, y_test, scaler, encoder

# ------------------------
# Pipeline completo para predicción
# ------------------------
def prepare_data_for_predict(df, scaler=SCALER, encoder=ENCODER):
    print("Ejecutando Pipeline completo para predicción...")
    df = clean_data(df)
    X, _, _, _ = preprocess_for_predict(df, scaler=scaler, encoder=encoder)
    return X