import cv2
import numpy as np
from PIL import Image, ImageOps


def preparar_imagen(imagen):
    """Normaliza una imagen real a la escala del conjunto Digits con mejor centrado."""
    imagen = ImageOps.grayscale(imagen)
    imagen = ImageOps.autocontrast(imagen)
    imagen = imagen.resize((420, 420), Image.Resampling.LANCZOS)

    arreglo = np.array(imagen, dtype=np.uint8)
    if arreglo.mean() > 127:
        arreglo = 255 - arreglo

    arreglo = cv2.GaussianBlur(arreglo, (3, 3), 0)
    arreglo = np.where(arreglo > 30, arreglo, 0).astype(np.uint8)

    coordenadas_y, coordenadas_x = np.where(arreglo > 10)
    if len(coordenadas_x) == 0:
        return np.zeros((8, 8), dtype=float)

    recorte = arreglo[coordenadas_y.min():coordenadas_y.max() + 1, coordenadas_x.min():coordenadas_x.max() + 1]
    alto, ancho = recorte.shape
    if alto == 0 or ancho == 0:
        return np.zeros((8, 8), dtype=float)

    tamano_objetivo = 28
    margen = 10
    imagen_acolchonada = np.full((tamano_objetivo + margen * 2, tamano_objetivo + margen * 2), 255, dtype=np.uint8)
    escala = min((tamano_objetivo + margen * 2) / max(alto, ancho), 1.0)
    nuevo_alto = max(1, int(alto * escala))
    nuevo_ancho = max(1, int(ancho * escala))
    imagen_redimensionada = Image.fromarray(recorte).resize((nuevo_ancho, nuevo_alto), Image.Resampling.LANCZOS)
    arreglo_redimensionado = np.array(imagen_redimensionada, dtype=np.uint8)
    inicio_y = (tamano_objetivo + margen * 2 - nuevo_alto) // 2
    inicio_x = (tamano_objetivo + margen * 2 - nuevo_ancho) // 2
    imagen_acolchonada[inicio_y:inicio_y + nuevo_alto, inicio_x:inicio_x + nuevo_ancho] = arreglo_redimensionado

    imagen_centrada = Image.fromarray(imagen_acolchonada).resize((8, 8), Image.Resampling.LANCZOS)
    salida = np.asarray(imagen_centrada, dtype=float)
    salida = np.clip(salida / 255.0 * 16.0, 0, 16)
    return salida


def preparar_hoja(imagen_pil, tipo_hoja, es_lienzo=False):
    """Prepara una hoja completa y elimina líneas de cuadrícula cuando se selecciona ese tipo."""
    arreglo_rgb = np.array(imagen_pil.convert("RGB"))
    gris = cv2.cvtColor(arreglo_rgb, cv2.COLOR_RGB2GRAY)
    gris = cv2.GaussianBlur(gris, (3, 3), 0)
    usar_mascara_color = False

    if es_lienzo:
        _, binaria = cv2.threshold(gris, 127, 255, cv2.THRESH_BINARY_INV)
    elif tipo_hoja == "Hoja cuadriculada":
        hsv = cv2.cvtColor(arreglo_rgb, cv2.COLOR_RGB2HSV)
        mascara_tinta_azul = cv2.inRange(
            hsv,
            np.array([85, 35, 25], dtype=np.uint8),
            np.array([145, 255, 245], dtype=np.uint8),
        )
        minimo_tinta = max(30, int(gris.size * 0.0004))
        if cv2.countNonZero(mascara_tinta_azul) >= minimo_tinta:
            binaria = cv2.morphologyEx(
                mascara_tinta_azul,
                cv2.MORPH_CLOSE,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
            )
            usar_mascara_color = True
        else:
            binaria = cv2.adaptiveThreshold(
                gris, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV, 31, 12
            )
    else:
        binaria = cv2.adaptiveThreshold(
            gris, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 31, 12
        )

    if not es_lienzo and not usar_mascara_color:
        alto, ancho = binaria.shape
        longitud_horizontal = max(18, ancho // 14)
        longitud_vertical = max(18, alto // 14)
        nucleo_linea_horizontal = cv2.getStructuringElement(cv2.MORPH_RECT, (longitud_horizontal, 1))
        nucleo_linea_vertical = cv2.getStructuringElement(cv2.MORPH_RECT, (1, longitud_vertical))
        lineas_horizontales = cv2.morphologyEx(binaria, cv2.MORPH_OPEN, nucleo_linea_horizontal)
        lineas_verticales = cv2.morphologyEx(binaria, cv2.MORPH_OPEN, nucleo_linea_vertical)

        filas_activas = np.sum(lineas_horizontales, axis=1) >= ancho * 255 * 0.2
        columnas_activas = np.sum(lineas_verticales, axis=0) >= alto * 255 * 0.2
        bandas_horizontales = np.count_nonzero(
            np.diff(np.r_[False, filas_activas, False].astype(np.int8)) == 1
        )
        bandas_verticales = np.count_nonzero(
            np.diff(np.r_[False, columnas_activas, False].astype(np.int8)) == 1
        )
        hay_cuadricula = bandas_horizontales >= 3 and bandas_verticales >= 3

        if tipo_hoja == "Hoja cuadriculada" or hay_cuadricula:
            cuadricula = cv2.bitwise_or(lineas_horizontales, lineas_verticales)
            cuadricula = cv2.morphologyEx(cuadricula, cv2.MORPH_DILATE, np.ones((9, 9), np.uint8))
            binaria = cv2.subtract(binaria, cuadricula)
            binaria = np.clip(binaria, 0, 255)
            binaria = cv2.morphologyEx(
                binaria, cv2.MORPH_CLOSE,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            )

    return binaria


def extraer_digitos(imagen_pil, tipo_hoja, es_lienzo=False):
    """Encuentra componentes que podrían ser dígitos y los ordena de arriba hacia abajo."""
    imagen_binaria = preparar_hoja(imagen_pil, tipo_hoja, es_lienzo=es_lienzo)
    contornos, _ = cv2.findContours(imagen_binaria, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    alto_hoja, ancho_hoja = imagen_binaria.shape
    candidatos = []
    area_minima = max(32, int(alto_hoja * ancho_hoja * 0.0001))
    for contorno in contornos:
        x, y, ancho, alto = cv2.boundingRect(contorno)
        area = cv2.contourArea(contorno)
        if area < area_minima or alto < 8 or ancho < 3:
            continue
        proporcion = max(ancho / alto, alto / ancho)
        if proporcion > 12:
            continue
        if alto > alto_hoja * 0.75 or ancho > ancho_hoja * 0.75:
            continue
        candidatos.append((x, y, ancho, alto))

    candidatos.sort(key=lambda caja: (caja[1] // max(18, int(alto_hoja * 0.035)), caja[0]))
    digitos = []
    for x, y, ancho, alto in candidatos:
        margen = max(4, int(max(ancho, alto) * 0.18))
        x1, y1 = max(0, x - margen), max(0, y - margen)
        x2, y2 = min(ancho_hoja, x + ancho + margen), min(alto_hoja, y + alto + margen)
        recorte = imagen_binaria[y1:y2, x1:x2]
        imagen_recortada = Image.fromarray(recorte).convert("L")
        imagen8 = preparar_imagen(imagen_recortada)
        digitos.append({"x": x, "y": y, "ancho": ancho, "alto": alto, "imagen": imagen8})
    return digitos, imagen_binaria


def agrupar_digitos(digitos):
    """Agrupa los recortes en números por fila y por espacios amplios entre cifras."""
    if not digitos:
        return []

    indices_por_y = sorted(
        range(len(digitos)),
        key=lambda indice: digitos[indice]["y"] + digitos[indice]["alto"] / 2,
    )
    filas = []
    for indice in indices_por_y:
        digito = digitos[indice]
        centro_y = digito["y"] + digito["alto"] / 2
        fila_cercana = None
        menor_distancia = float("inf")
        for fila in filas:
            centro_fila = float(np.median([
                digitos[i]["y"] + digitos[i]["alto"] / 2 for i in fila
            ]))
            tolerancia = max(14, float(np.median([digitos[i]["alto"] for i in fila])) * 0.55)
            distancia = abs(centro_y - centro_fila)
            if distancia <= tolerancia and distancia < menor_distancia:
                fila_cercana = fila
                menor_distancia = distancia
        if fila_cercana is None:
            filas.append([indice])
        else:
            fila_cercana.append(indice)

    grupos = []
    for fila in filas:
        fila.sort(key=lambda indice: digitos[indice]["x"])
        anchos = [digitos[indice]["ancho"] for indice in fila]
        separacion = max(10, float(np.median(anchos)) * 0.9)
        grupo_actual = [fila[0]]
        for indice_anterior, indice_actual in zip(fila, fila[1:]):
            borde_anterior = digitos[indice_anterior]["x"] + digitos[indice_anterior]["ancho"]
            espacio = digitos[indice_actual]["x"] - borde_anterior
            if espacio > separacion:
                grupos.append(grupo_actual)
                grupo_actual = []
            grupo_actual.append(indice_actual)
        grupos.append(grupo_actual)

    return grupos
