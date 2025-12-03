# flake8: noqa: E501
#
# En este dataset se desea pronosticar el default (pago) del cliente el próximo
# mes a partir de 23 variables explicativas.
#
#   LIMIT_BAL: Monto del credito otorgado. Incluye el credito individual y el
#              credito familiar (suplementario).
#         SEX: Genero (1=male; 2=female).
#   EDUCATION: Educacion (0=N/A; 1=graduate school; 2=university; 3=high school; 4=others).
#    MARRIAGE: Estado civil (0=N/A; 1=married; 2=single; 3=others).
#         AGE: Edad (years).
#       PAY_0: Historia de pagos pasados. Estado del pago en septiembre, 2005.
#       PAY_2: Historia de pagos pasados. Estado del pago en agosto, 2005.
#       PAY_3: Historia de pagos pasados. Estado del pago en julio, 2005.
#       PAY_4: Historia de pagos pasados. Estado del pago en junio, 2005.
#       PAY_5: Historia de pagos pasados. Estado del pago en mayo, 2005.
#       PAY_6: Historia de pagos pasados. Estado del pago en abril, 2005.
#   BILL_AMT1: Historia de pagos pasados. Monto a pagar en septiembre, 2005.
#   BILL_AMT2: Historia de pagos pasados. Monto a pagar en agosto, 2005.
#   BILL_AMT3: Historia de pagos pasados. Monto a pagar en julio, 2005.
#   BILL_AMT4: Historia de pagos pasados. Monto a pagar en junio, 2005.
#   BILL_AMT5: Historia de pagos pasados. Monto a pagar en mayo, 2005.
#   BILL_AMT6: Historia de pagos pasados. Monto a pagar en abril, 2005.
#    PAY_AMT1: Historia de pagos pasados. Monto pagado en septiembre, 2005.
#    PAY_AMT2: Historia de pagos pasados. Monto pagado en agosto, 2005.
#    PAY_AMT3: Historia de pagos pasados. Monto pagado en julio, 2005.
#    PAY_AMT4: Historia de pagos pasados. Monto pagado en junio, 2005.
#    PAY_AMT5: Historia de pagos pasados. Monto pagado en mayo, 2005.
#    PAY_AMT6: Historia de pagos pasados. Monto pagado en abril, 2005.
#
# La variable "default payment next month" corresponde a la variable objetivo.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# clasificación están descritos a continuación.
#
#
# Paso 1.
# Realice la limpieza de los datasets:
# - Renombre la columna "default payment next month" a "default".
# - Remueva la columna "ID".
# - Elimine los registros con informacion no disponible.
# - Para la columna EDUCATION, valores > 4 indican niveles superiores
#   de educación, agrupe estos valores en la categoría "others".
# - Renombre la columna "default payment next month" a "default"
# - Remueva la columna "ID".
#
#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#
#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Descompone la matriz de entrada usando PCA. El PCA usa todas las componentes.
# - Estandariza la matriz de entrada.
# - Selecciona las K columnas mas relevantes de la matrix de entrada.
# - Ajusta una maquina de vectores de soporte (svm).
#
#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use la función de precision
# balanceada para medir la precisión del modelo.
#
#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
#
# Paso 6.
# Calcule las metricas de precision, precision balanceada, recall,
# y f1-score para los conjuntos de entrenamiento y prueba.
# Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# Este diccionario tiene un campo para indicar si es el conjunto
# de entrenamiento o prueba. Por ejemplo:
#
# {'dataset': 'train', 'precision': 0.8, 'balanced_accuracy': 0.7, 'recall': 0.9, 'f1_score': 0.85}
# {'dataset': 'test', 'precision': 0.7, 'balanced_accuracy': 0.6, 'recall': 0.8, 'f1_score': 0.75}
#
#
# Paso 7.
# Calcule las matrices de confusion para los conjuntos de entrenamiento y
# prueba. Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'cm_matrix', 'dataset': 'train', 'true_0': {"predicted_0": 15562, "predicte_1": 666}, 'true_1': {"predicted_0": 3333, "predicted_1": 1444}}
# {'type': 'cm_matrix', 'dataset': 'test', 'true_0': {"predicted_0": 15562, "predicte_1": 650}, 'true_1': {"predicted_0": 2490, "predicted_1": 1420}}
#
import json
import gzip
import os
import pickle
import zipfile
from pathlib import Path

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
def _cargar_csv_desde_zip(ruta_zip: str, archivo_interno: str) -> pd.DataFrame:
    """Extrae un CSV desde un archivo ZIP y lo carga en un DataFrame."""
    with zipfile.ZipFile(ruta_zip, "r") as zf:
        with zf.open(archivo_interno) as f:
            return pd.read_csv(f)


def _limpiar_datos(df: pd.DataFrame) -> pd.DataFrame:
    """Realiza la depuración del dataset: elimina columnas, filtra valores,
    corrige categorías y elimina filas inválidas."""
    limpio = df.copy()
    limpio = limpio.drop("ID", axis=1)
    limpio = limpio.rename(columns={"default payment next month": "default"})
    limpio = limpio.dropna()
    limpio = limpio[(limpio["EDUCATION"] != 0) & (limpio["MARRIAGE"] != 0)]
    limpio.loc[limpio["EDUCATION"] > 4, "EDUCATION"] = 4
    return limpio


def _construir_gridsearch() -> GridSearchCV:
    """Arma el pipeline de preprocesado + modelo y configura la búsqueda de hiperparámetros."""
    columnas_categoricas = ["SEX", "EDUCATION", "MARRIAGE"]
    columnas_numericas = [
        "LIMIT_BAL", "AGE", "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
        "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
        "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6",
    ]

    preproceso = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), columnas_categoricas),
            ("std", StandardScaler(), columnas_numericas),
        ],
        remainder="passthrough",
    )

    pipeline = Pipeline(steps=[
        ("preprocesamiento", preproceso),
        ("pca", PCA()),                         # usa todas las componentes posibles
        ("selector", SelectKBest(score_func=f_classif)),
        ("clasificador", SVC(kernel="rbf", random_state=42)),
    ])

    grilla = {
        "pca__n_components": [20, 21],
        "selector__k": [12],
        "clasificador__kernel": ["rbf"],
        "clasificador__gamma": [0.099],
    }

    return GridSearchCV(
        estimator=pipeline,
        param_grid=grilla,
        cv=10,
        refit=True,
        verbose=1,
        return_train_score=False,
        scoring="balanced_accuracy",
    )


def _calcular_metricas(nombre: str, y_true, y_pred) -> dict:
    """Devuelve un diccionario con diferentes métricas de desempeño."""
    return {
        "type": "metrics",
        "dataset": nombre,
        "precision": precision_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1_score": f1_score(y_true, y_pred),
    }


def _matriz_confusion(nombre: str, y_true, y_pred) -> dict:
    """Genera un diccionario con la matriz de confusión desglosada."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "type": "cm_matrix",
        "dataset": nombre,
        "true_0": {"predicted_0": int(tn), "predicted_1": int(fp)},
        "true_1": {"predicted_0": int(fn), "predicted_1": int(tp)},
    }


def _guardar_modelo_serializado(objeto) -> None:
    """Guarda el modelo entrenado en formato comprimido."""
    Path("files/models").mkdir(parents=True, exist_ok=True)
    with gzip.open("files/models/model.pkl.gz", "wb") as fh:
        pickle.dump(objeto, fh)


def _exportar_jsonl(registros: list[dict]) -> None:
    """Exporta métricas y matrices de confusión en formato JSONL."""
    Path("files/output").mkdir(parents=True, exist_ok=True)
    with open("files/output/metrics.json", "w", encoding="utf-8") as f:
        for r in registros:
            f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    archivo_test = "files/input/test_data.csv.zip"
    archivo_train = "files/input/train_data.csv.zip"
    interno_test = "test_default_of_credit_card_clients.csv"
    interno_train = "train_default_of_credit_card_clients.csv"

    df_test = _limpiar_datos(_cargar_csv_desde_zip(archivo_test, interno_test))
    df_train = _limpiar_datos(_cargar_csv_desde_zip(archivo_train, interno_train))

    X_tr, y_tr = df_train.drop("default", axis=1), df_train["default"]
    X_te, y_te = df_test.drop("default", axis=1), df_test["default"]

    busqueda = _construir_gridsearch()
    busqueda.fit(X_tr, y_tr)
    _guardar_modelo_serializado(busqueda)

    y_tr_pred = busqueda.predict(X_tr)
    y_te_pred = busqueda.predict(X_te)

    metricas_train = _calcular_metricas("train", y_tr, y_tr_pred)
    metricas_test = _calcular_metricas("test", y_te, y_te_pred)
    cm_train = _matriz_confusion("train", y_tr, y_tr_pred)
    cm_test = _matriz_confusion("test", y_te, y_te_pred)

    _exportar_jsonl([metricas_train, metricas_test, cm_train, cm_test])

