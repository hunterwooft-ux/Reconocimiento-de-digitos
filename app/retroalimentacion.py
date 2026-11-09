import numpy as np
import pandas as pd

from configuracion import ARCHIVO_RETROALIMENTACION


def cargar_ejemplos():
    if not ARCHIVO_RETROALIMENTACION.exists():
        return np.empty((0, 64), dtype=float), np.empty((0,), dtype=int)

    tabla = pd.read_csv(ARCHIVO_RETROALIMENTACION)
    if tabla.empty:
        return np.empty((0, 64), dtype=float), np.empty((0,), dtype=int)

    columna_imagen = "imagen_plana" if "imagen_plana" in tabla.columns else "image_flat"
    columna_numero = "numero" if "numero" in tabla.columns else "label"
    if columna_imagen not in tabla.columns or columna_numero not in tabla.columns:
        return np.empty((0, 64), dtype=float), np.empty((0,), dtype=int)

    imagenes = []
    numeros = []
    for _, fila in tabla.iterrows():
        try:
            imagen_plana = np.array(
                [float(valor) for valor in str(fila[columna_imagen]).split(";") if valor.strip()],
                dtype=float,
            )
            if imagen_plana.size == 64:
                imagenes.append(imagen_plana)
                numeros.append(int(fila[columna_numero]))
        except Exception:
            continue

    if not imagenes:
        return np.empty((0, 64), dtype=float), np.empty((0,), dtype=int)

    return np.asarray(imagenes, dtype=float), np.asarray(numeros, dtype=int)


def guardar_ejemplo(imagen8, numero):
    imagen_plana = imagen8.reshape(-1)
    fila = {"imagen_plana": ";".join(str(float(valor)) for valor in imagen_plana), "numero": int(numero)}

    if ARCHIVO_RETROALIMENTACION.exists():
        tabla = pd.read_csv(ARCHIVO_RETROALIMENTACION)
        if "imagen_plana" not in tabla.columns or "numero" not in tabla.columns:
            columna_imagen = "image_flat" if "image_flat" in tabla.columns else None
            columna_numero = "label" if "label" in tabla.columns else None
            if columna_imagen and columna_numero:
                tabla = tabla.rename(columns={columna_imagen: "imagen_plana", columna_numero: "numero"})
            else:
                tabla = pd.DataFrame(columns=["imagen_plana", "numero"])
    else:
        tabla = pd.DataFrame(columns=["imagen_plana", "numero"])

    tabla = pd.concat([tabla, pd.DataFrame([fila])], ignore_index=True)
    tabla.to_csv(ARCHIVO_RETROALIMENTACION, index=False)


def contar_ejemplos_entrenamiento(cantidad_base):
    _, numeros = cargar_ejemplos()
    return int(cantidad_base + len(numeros))
