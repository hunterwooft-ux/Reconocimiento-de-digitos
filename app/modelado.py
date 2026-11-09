import cv2
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.datasets import load_digits
from sklearn.calibration import CalibratedClassifierCV
from sklearn.cluster import KMeans
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    silhouette_score,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from retroalimentacion import cargar_ejemplos

DATOS = load_digits()
CARACTERISTICAS = DATOS.data
NUMEROS = DATOS.target
caracteristicas_entrenamiento, caracteristicas_prueba, numeros_entrenamiento, numeros_prueba = train_test_split(
    CARACTERISTICAS, NUMEROS, test_size=0.20, random_state=42, stratify=NUMEROS
)


def aumentar_datos(caracteristicas, numeros, repeticiones=4):
    """Genera variaciones pequeñas del conjunto de entrenamiento para mejorar la robustez."""
    caracteristicas_aumentadas = []
    numeros_aumentados = []
    generador = np.random.default_rng(42)

    for muestra, numero in zip(caracteristicas, numeros):
        imagen = muestra.reshape(8, 8).astype(np.float32)
        for _ in range(repeticiones):
            variacion = imagen.copy()
            angulo = generador.uniform(-12, 12)
            transformacion = cv2.getRotationMatrix2D((3.5, 3.5), angulo, 1.0)
            variacion = cv2.warpAffine(variacion, transformacion, (8, 8), borderMode=cv2.BORDER_REPLICATE)

            desplazamiento_x = generador.integers(-1, 2)
            desplazamiento_y = generador.integers(-1, 2)
            transformacion = np.float32([[1, 0, desplazamiento_x], [0, 1, desplazamiento_y]])
            variacion = cv2.warpAffine(variacion, transformacion, (8, 8), borderMode=cv2.BORDER_REPLICATE)

            escala = generador.uniform(0.9, 1.1)
            redimensionada = cv2.resize(
                variacion,
                (int(8 * escala), int(8 * escala)),
                interpolation=cv2.INTER_LINEAR,
            )
            acolchada = np.full((8, 8), 0, dtype=np.float32)
            inicio_y = max(0, (8 - redimensionada.shape[0]) // 2)
            inicio_x = max(0, (8 - redimensionada.shape[1]) // 2)
            fin_y = inicio_y + min(8, redimensionada.shape[0])
            fin_x = inicio_x + min(8, redimensionada.shape[1])
            acolchada[inicio_y:fin_y, inicio_x:fin_x] = redimensionada[: fin_y - inicio_y, : fin_x - inicio_x]
            caracteristicas_aumentadas.append(acolchada.reshape(-1))
            numeros_aumentados.append(numero)

    return np.asarray(caracteristicas_aumentadas), np.asarray(numeros_aumentados)


@st.cache_resource
def obtener_modelos_entrenados():
    caracteristicas_usuario, numeros_usuario = cargar_ejemplos()
    caracteristicas_base = caracteristicas_entrenamiento.copy()
    numeros_base = numeros_entrenamiento.copy()
    if caracteristicas_usuario.size > 0:
        caracteristicas_base = np.vstack([caracteristicas_base, caracteristicas_usuario])
        numeros_base = np.concatenate([numeros_base, numeros_usuario])

    caracteristicas_aumentadas, numeros_aumentados = aumentar_datos(caracteristicas_base, numeros_base, repeticiones=4)

    configuraciones_modelos = {
        "KNN": (
            make_pipeline(StandardScaler(), KNeighborsClassifier()),
            {
                "kneighborsclassifier__n_neighbors": [3, 5, 7, 9],
                "kneighborsclassifier__weights": ["uniform", "distance"],
                "kneighborsclassifier__p": [1, 2],
            },
        ),
        "SVM": (
            make_pipeline(StandardScaler(), SVC(kernel="rbf", random_state=42)),
            {
                "svc__C": [1, 3, 7, 10, 20],
                "svc__gamma": ["scale", "auto", 0.001, 0.01],
            },
        ),
        "Random Forest": (
            RandomForestClassifier(random_state=42),
            {
                "n_estimators": [100, 200, 300],
                "max_depth": [None, 15, 25],
                "min_samples_leaf": [1, 2, 3],
            },
        ),
        "Árbol de decisión": (
            DecisionTreeClassifier(random_state=42),
            {
                "max_depth": [None, 10, 20, 30],
                "min_samples_leaf": [1, 2, 3],
                "criterion": ["gini", "entropy"],
            },
        ),
        "Extra Trees": (
            ExtraTreesClassifier(random_state=42),
            {
                "n_estimators": [100, 200, 300],
                "max_depth": [None, 15, 25],
                "min_samples_leaf": [1, 2, 3],
            },
        ),
        "Red neuronal MLP": (
            make_pipeline(
                StandardScaler(),
                MLPClassifier(
                    activation="relu",
                    solver="adam",
                    max_iter=500,
                    early_stopping=True,
                    random_state=42,
                ),
            ),
            {
                "mlpclassifier__hidden_layer_sizes": [(64,), (96, 64), (128, 64)],
                "mlpclassifier__alpha": [0.0001, 0.001, 0.01],
            },
        ),
        "Regresión logística": (
            make_pipeline(
                StandardScaler(),
                LogisticRegression(max_iter=1500, random_state=42),
            ),
            {
                "logisticregression__C": [0.1, 1, 10, 30],
                "logisticregression__class_weight": [None, "balanced"],
            },
        ),
    }

    validacion_cruzada = StratifiedKFold(
        n_splits=3,
        shuffle=True,
        random_state=42,
    )
    modelos_entrenados, metricas, predicciones, parametros = {}, [], {}, []
    for nombre, (modelo, espacio_parametros) in configuraciones_modelos.items():
        busqueda = RandomizedSearchCV(
            estimator=modelo,
            param_distributions=espacio_parametros,
            n_iter=4,
            scoring="accuracy",
            cv=validacion_cruzada,
            refit=False,
            random_state=42,
            n_jobs=-1,
        )
        busqueda.fit(caracteristicas_entrenamiento, numeros_entrenamiento)
        if nombre == "SVM":
            parametros_svm = {
                nombre_parametro.removeprefix("svc__"): valor
                for nombre_parametro, valor in busqueda.best_params_.items()
            }
            modelo = CalibratedClassifierCV(
                estimator=make_pipeline(
                    StandardScaler(),
                    SVC(kernel="rbf", random_state=42, **parametros_svm),
                ),
                cv=3,
                ensemble=False,
            )
        else:
            modelo.set_params(**busqueda.best_params_)
        modelo.fit(caracteristicas_aumentadas, numeros_aumentados)
        prediccion = modelo.predict(caracteristicas_prueba)
        modelos_entrenados[nombre] = modelo
        predicciones[nombre] = prediccion
        metricas.append({
            "Modelo": nombre,
            "Exactitud CV (promedio)": busqueda.best_score_,
            "Desv. estándar CV": busqueda.cv_results_["std_test_score"][
                busqueda.best_index_
            ],
            "Exactitud": accuracy_score(numeros_prueba, prediccion),
            "Precisión (macro)": precision_score(numeros_prueba, prediccion, average="macro", zero_division=0),
            "Recall (macro)": recall_score(numeros_prueba, prediccion, average="macro", zero_division=0),
            "F1 (macro)": f1_score(numeros_prueba, prediccion, average="macro", zero_division=0)
        })
        parametros.append({
            "Modelo": nombre,
            "Mejores hiperparámetros": busqueda.best_params_,
        })

    escalador_kmeans = StandardScaler()
    entrenamiento_escalado = escalador_kmeans.fit_transform(
        caracteristicas_entrenamiento
    )
    modelo_kmeans = KMeans(n_clusters=10, n_init=10, random_state=42)
    etiquetas_cluster = modelo_kmeans.fit_predict(entrenamiento_escalado)
    indices_muestra = np.random.default_rng(42).choice(
        len(entrenamiento_escalado),
        size=min(500, len(entrenamiento_escalado)),
        replace=False,
    )
    resultados_kmeans = {
        "Inercia": float(modelo_kmeans.inertia_),
        "Coeficiente de silueta": float(
            silhouette_score(
                entrenamiento_escalado[indices_muestra],
                etiquetas_cluster[indices_muestra],
            )
        ),
        "Distribución por dígito real": pd.crosstab(
            pd.Series(numeros_entrenamiento, name="Dígito real"),
            pd.Series(etiquetas_cluster, name="Cluster"),
        ),
    }
    return (
        modelos_entrenados,
        pd.DataFrame(metricas),
        predicciones,
        pd.DataFrame(parametros),
        resultados_kmeans,
    )


def predecir_todos(imagen8, modelos):
    muestra = imagen8.reshape(1, -1)
    filas = []
    for nombre, modelo in modelos.items():
        prediccion = int(modelo.predict(muestra)[0])
        confianza = None
        if hasattr(modelo, "predict_proba"):
            confianza = float(np.max(modelo.predict_proba(muestra)[0]))
        filas.append({
            "Modelo": nombre,
            "Predicción": prediccion,
            "Confianza estimada": confianza,
        })
    return pd.DataFrame(filas)


def predecir_secuencias(digitos, grupos, modelos):
    """Predice cada cifra y reúne las predicciones en números de varios dígitos."""
    predicciones_por_digito = [
        predecir_todos(digito["imagen"], modelos) for digito in digitos
    ]
    filas_resultado = []

    for indice_numero, grupo in enumerate(grupos, start=1):
        cifras_consenso = []
        for indice_digito in grupo:
            votos = predicciones_por_digito[indice_digito]["Predicción"].astype(int).value_counts()
            cifras_consenso.append(str(int(votos.index[0])))

        fila = {
            "N.º": indice_numero,
            "Consenso": "".join(cifras_consenso),
        }
        for nombre_modelo in modelos:
            fila[nombre_modelo] = "".join(
                str(int(predicciones_por_digito[indice]["Predicción"].iloc[
                    predicciones_por_digito[indice]["Modelo"].tolist().index(nombre_modelo)
                ]))
                for indice in grupo
            )
        filas_resultado.append(fila)

    return pd.DataFrame(filas_resultado)


def crear_fila_historial(tabla_predicciones):
    fila = {"Fecha y hora": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")}
    for _, fila_modelo in tabla_predicciones.iterrows():
        fila[fila_modelo["Modelo"]] = int(fila_modelo["Predicción"])
    return fila


def resumir_prediccion(tabla_predicciones):
    if tabla_predicciones.empty:
        return None, None
    votos = tabla_predicciones["Predicción"].astype(int).value_counts()
    numero_mas_votado = int(votos.index[0])
    cantidad_acuerdos = int(votos.iloc[0])
    return numero_mas_votado, cantidad_acuerdos
