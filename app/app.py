import base64
import importlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
from streamlit_drawable_canvas import st_canvas

import configuracion as modulo_configuracion
import modelado as modulo_modelado
import procesamiento as modulo_procesamiento
from retroalimentacion import contar_ejemplos_entrenamiento, guardar_ejemplo

MODO_UN_NUMERO = "Un número"
MODO_VARIOS_NUMEROS = "Varios números"

modulo_configuracion = importlib.reload(modulo_configuracion)
if not all(
    hasattr(modulo_modelado, nombre)
    for nombre in ("obtener_modelos_entrenados", "predecir_secuencias")
):
    modulo_modelado = importlib.reload(modulo_modelado)
modulo_procesamiento = importlib.reload(modulo_procesamiento)

CARPETA_INTRO = modulo_configuracion.CARPETA_INTRO
CARPETA_MUSICA = modulo_configuracion.CARPETA_MUSICA
RUTA_ESTILOS = modulo_configuracion.RUTA_ESTILOS

caracteristicas_prueba = modulo_modelado.caracteristicas_prueba
numeros_entrenamiento = modulo_modelado.numeros_entrenamiento
numeros_prueba = modulo_modelado.numeros_prueba
obtener_modelos_entrenados = modulo_modelado.obtener_modelos_entrenados
predecir_todos = modulo_modelado.predecir_todos
predecir_secuencias = modulo_modelado.predecir_secuencias
resumir_prediccion = modulo_modelado.resumir_prediccion
crear_fila_historial = modulo_modelado.crear_fila_historial
agrupar_digitos = modulo_procesamiento.agrupar_digitos
extraer_digitos = modulo_procesamiento.extraer_digitos
preparar_imagen = modulo_procesamiento.preparar_imagen


def obtener_imagen_lienzo(lienzo):
    if lienzo is None:
        return None

    try:
        datos_imagen = lienzo.image_data
    except RuntimeError:
        return None

    return datos_imagen


# Configuración y estilos globales
st.set_page_config(page_title="TecnoForms | Dígitos", page_icon="🔢", layout="wide")
with open(RUTA_ESTILOS, "r", encoding="utf-8") as archivo_estilos:
    st.markdown(
        f"<style>{archivo_estilos.read()}</style>",
        unsafe_allow_html=True,
    )

# Introducción animada
intro_activa = False
gifs_intro = list(CARPETA_INTRO.glob("*.gif"))
if not st.session_state.get("intro_mostrada", False):
    if len(gifs_intro) == 1:
        gif_base64 = base64.b64encode(gifs_intro[0].read_bytes()).decode("ascii")
        st.markdown(
            f"""
            <div class="tv-intro">
                <img src="data:image/gif;base64,{gif_base64}" alt="Introducción de TecnoForms">
                <div class="tv-intro-caption">Preparando TecnoForms...</div>
            </div>
            <div class="tv-transition" aria-hidden="true"></div>
            <div class="tv-power-light" aria-hidden="true"></div>
            """,
            unsafe_allow_html=True,
        )
        st.session_state["intro_mostrada"] = True
        intro_activa = True
    elif not gifs_intro:
        st.warning(f"No se encontró un GIF de entrada en {CARPETA_INTRO}.")
    else:
        st.error(f"Deja un solo GIF de entrada en {CARPETA_INTRO}.")

# Música de fondo y controles personalizados
archivos_musica = list(CARPETA_MUSICA.glob("*.mp3"))
if len(archivos_musica) == 1:
    with st.sidebar:
        st.caption("Música de fondo · volumen bajo")
        st.audio(archivos_musica[0].read_bytes(), format="audio/mpeg", loop=True, autoplay=True)
        st.markdown(
            """
            <div class="music-control" role="group" aria-label="Controles de música">
                <button
                    class="music-toggle"
                    type="button"
                    aria-label="Reproducir música"
                    title="Reproducir música"
                >
                    <svg class="music-icon-play" viewBox="0 0 24 24" aria-hidden="true">
                        <path d="M8 5.5v13l10-6.5z" fill="currentColor"></path>
                    </svg>
                    <svg class="music-icon-pause" viewBox="0 0 24 24" aria-hidden="true">
                        <path d="M7 5h4v14H7zm6 0h4v14h-4z" fill="currentColor"></path>
                    </svg>
                </button>
                <input
                    class="music-progress"
                    type="range"
                    min="0"
                    max="1000"
                    value="0"
                    aria-label="Progreso de la música"
                >
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.html(
            """
            <script>
            (() => {
                const obtenerAudio = () => document.querySelector('audio[data-testid="stAudio"]');
                const actualizarControl = (audio) => {
                    document.querySelectorAll(".music-control").forEach((control) => {
                        const boton = control.querySelector(".music-toggle");
                        const progreso = control.querySelector(".music-progress");
                        if (!boton || !progreso) return;
                        const reproduciendo = !audio.paused && !audio.ended;
                        boton.dataset.playing = String(reproduciendo);
                        boton.setAttribute(
                            "aria-label",
                            reproduciendo ? "Pausar música" : "Reproducir música"
                        );
                        boton.title = reproduciendo ? "Pausar música" : "Reproducir música";
                        const porcentaje = Number.isFinite(audio.duration) && audio.duration > 0
                            ? (audio.currentTime / audio.duration) * 100
                            : 0;
                        progreso.value = String(Math.round(porcentaje * 10));
                        progreso.style.setProperty("--music-progress", `${porcentaje}%`);
                    });
                };

                const sincronizarAudio = () => {
                    const audio = obtenerAudio();
                    if (!audio) return;
                    audio.volume = 0.12;
                    audio.loop = true;
                    if (!audio.dataset.musicEventsBound) {
                        [
                            "play",
                            "pause",
                            "ended",
                            "timeupdate",
                            "loadedmetadata",
                            "durationchange",
                        ].forEach((evento) => {
                            audio.addEventListener(
                                evento, () => actualizarControl(audio)
                            );
                        });
                        audio.dataset.musicEventsBound = "true";
                    }
                    actualizarControl(audio);
                };

                if (!document.documentElement.dataset.musicControlBound) {
                    document.addEventListener("click", (evento) => {
                        const boton = evento.target.closest(".music-toggle");
                        if (!boton) return;
                        const audio = obtenerAudio();
                        if (!audio) return;
                        if (audio.paused) {
                            audio.play().catch(() => {
                                boton.title =
                                    "El navegador bloqueó la reproducción; " +
                                    "vuelve a intentarlo.";
                            });
                        } else {
                            audio.pause();
                        }
                    });
                    document.addEventListener("input", (evento) => {
                        if (!evento.target.matches(".music-progress")) return;
                        const audio = obtenerAudio();
                        if (audio && Number.isFinite(audio.duration)) {
                            audio.currentTime = (
                                Number(evento.target.value) / 1000
                            ) * audio.duration;
                        }
                    });
                    document.documentElement.dataset.musicControlBound = "true";
                }

                sincronizarAudio();
                if (!document.documentElement.dataset.musicObserverBound) {
                    new MutationObserver(sincronizarAudio).observe(document.body, {
                        childList: true,
                        subtree: true
                    });
                    document.documentElement.dataset.musicObserverBound = "true";
                }
            })();
            </script>
            """,
            unsafe_allow_javascript=True,
        )
elif not archivos_musica:
    st.warning(f"No se encontró un archivo MP3 en {CARPETA_MUSICA}.")
else:
    st.error(f"Deja un solo archivo MP3 de música en {CARPETA_MUSICA}.")

mostrar_carga_entrenamiento = st.session_state.get("mostrar_carga_entrenamiento", False)
mostrar_carga_secuencias = st.session_state.get("mostrar_carga_secuencias", False)
carga_entrenamiento = st.empty()
if mostrar_carga_entrenamiento or mostrar_carga_secuencias:
    if len(gifs_intro) == 1:
        gif_carga_base64 = base64.b64encode(gifs_intro[0].read_bytes()).decode("ascii")
        imagen_carga = (
            f'<img src="data:image/gif;base64,{gif_carga_base64}" '
            'alt="Animación de carga">'
        )
    else:
        imagen_carga = ""

    mensaje_carga = (
        "Guardando la corrección y actualizando los modelos..."
        if mostrar_carga_entrenamiento
        else "Reconociendo los números de la hoja..."
    )
    carga_entrenamiento.markdown(
        f"""
        <div class="training-overlay" role="status" aria-live="polite">
            <div class="training-card">
                {imagen_carga}
                <p>{mensaje_carga}</p>
                <div class="training-progress" aria-hidden="true"><span></span></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Carga de modelos y estado de las acciones prolongadas
with st.spinner("Validando y ajustando los modelos; esto puede tardar un momento..."):
    (
        modelos,
        resultados,
        predicciones_prueba,
        parametros_modelos,
        resultados_kmeans,
    ) = obtener_modelos_entrenados()
if mostrar_carga_entrenamiento:
    carga_entrenamiento.empty()
    st.session_state.pop("mostrar_carga_entrenamiento", None)
    numero_guardado = st.session_state.pop("numero_correccion_guardado", None)
    cantidad_correcciones = st.session_state.pop(
        "cantidad_correcciones_guardadas", None
    )
    if cantidad_correcciones is not None:
        st.success(
            "Se guardaron "
            f"{cantidad_correcciones} correcciones y los modelos ya se actualizaron."
        )
    elif numero_guardado is not None:
        st.success(
            f"Corrección guardada para el número {numero_guardado}; "
            "los modelos ya se actualizaron."
        )
if intro_activa:
    st.markdown(
        "<style>.tv-intro { animation-play-state: running !important; }</style>",
        unsafe_allow_html=True,
    )

# Encabezado y navegación
st.markdown(
    """
    <div class="hero-card">
        <div class="hero-badge">⋆.ೃ࿔🌸*:･</div>
        <div class="hero-title">TecnoForms<br>Digit Bloom</div>
        <div class="hero-sub">Reconocimiento de dígitos.</div>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown(
        "<div class='sidebar-brand'>"
        "<span class='brand-mark'>✦</span>TecnoForms</div>",
        unsafe_allow_html=True,
    )
    st.markdown("<div class='nav-panel'>", unsafe_allow_html=True)
    seccion = st.radio(
        "Ir a:",
        [
            "Reconocer un número",
            "Comparar modelos",
            "Evaluación y gráficos",
            "Acerca del proyecto",
        ],
    )
    st.markdown("</div>", unsafe_allow_html=True)
    st.divider()

    if st.button("Reentrenar modelos"):
        st.cache_resource.clear()
        st.session_state.pop("ultima_prediccion", None)
        st.session_state.pop("resultados_hoja", None)
        st.session_state.pop("historial", None)
        st.rerun()

    st.metric("Modelos supervisados", str(len(modelos)))
    st.metric(
        "Imágenes de prueba",
        contar_ejemplos_entrenamiento(len(numeros_entrenamiento)),
    )
    st.markdown(
        "<div class='sidebar-tech-note'>Datos: sklearn.datasets.load_digits (imágenes 8×8).</div>",
        unsafe_allow_html=True,
    )

# Sección de reconocimiento de dígitos
if seccion == "Reconocer un número":
    with st.container(key="reconocimiento_panel"):
        st.subheader("Prueba con tu propio número")
        st.write("Puedes dibujar o subir una fotografía.")

        modo_reconocimiento = st.radio(
            "¿Qué deseas reconocer?",
            [MODO_UN_NUMERO, MODO_VARIOS_NUMEROS],
            horizontal=True,
            index=0,
            key="modo_reconocimiento",
        )

    if "contador_lienzo" not in st.session_state:
        st.session_state.contador_lienzo = 0

    columna_izquierda, columna_derecha = st.columns(2, gap="large")
    with columna_izquierda:
        with st.container(key="opcion_dibujo"):
            st.markdown("**Opción A. Dibujar en el recuadro**")
            lienzo = st_canvas(
                fill_color="rgba(255,255,255,0)",
                stroke_width=18,
                stroke_color="#FFFFFF",
                background_color="#000000",
                height=220,
                width=220,
                drawing_mode="freedraw",
                return_image_data=True,
                update_streamlit=True,
                key=f"lienzo_digito_{st.session_state.contador_lienzo}",
            )
            if st.button("Limpiar dibujo"):
                st.session_state.contador_lienzo += 1
                st.rerun()

    with columna_derecha:
        with st.container(key="opcion_subir"):
            st.markdown("**Opción B. Subir una fotografía**")
            if modo_reconocimiento == MODO_VARIOS_NUMEROS:
                tipo_hoja = st.selectbox(
                    "Tipo de hoja",
                    ["Hoja blanca", "Hoja cuadriculada"],
                    index=1,
                    key="tipo_hoja_v2",
                )
            else:
                tipo_hoja = "Hoja blanca"
            archivo_subido = st.file_uploader(
                "Selecciona una imagen JPG o PNG",
                type=["png", "jpg", "jpeg"],
            )
            st.caption("Arrastra una imagen aquí o selecciónala desde tu dispositivo.")

    imagen8 = None
    digitos_hoja = None
    hay_imagen_hoja = False
    imagen_lienzo = obtener_imagen_lienzo(lienzo)

    if archivo_subido is not None:
        imagen_pil = Image.open(archivo_subido).convert("RGB")
        st.image(imagen_pil, caption="Imagen cargada", width=300)
        if modo_reconocimiento == MODO_VARIOS_NUMEROS:
            hay_imagen_hoja = True
            digitos_hoja, imagen_hoja_binaria = extraer_digitos(imagen_pil, tipo_hoja)
            st.markdown("**Vista previa de la hoja procesada**")
            st.image(imagen_hoja_binaria, clamp=True, width=350)
            st.caption(
                f"Se detectaron {len(digitos_hoja)} posibles dígitos. "
                "Las líneas, sombras o números muy juntos pueden afectar la detección."
            )
        else:
            imagen8 = preparar_imagen(imagen_pil)
    elif imagen_lienzo is not None and np.max(imagen_lienzo[:, :, :3]) > 30:
        imagen_rgba = imagen_lienzo.astype(np.uint8)
        imagen_pil = Image.fromarray(imagen_rgba, mode="RGBA").convert("RGB")
        if modo_reconocimiento == MODO_VARIOS_NUMEROS:
            hay_imagen_hoja = True
            imagen_hoja = Image.fromarray(255 - np.asarray(imagen_pil), mode="RGB")
            digitos_hoja, imagen_hoja_binaria = extraer_digitos(
                imagen_hoja, "Hoja blanca", es_lienzo=True
            )
            st.markdown("**Vista previa del dibujo procesado**")
            st.image(imagen_hoja_binaria, clamp=True, width=350)
            st.caption(f"Se detectaron {len(digitos_hoja)} posibles dígitos dibujados.")
        else:
            imagen8 = preparar_imagen(imagen_pil)

    if imagen8 is not None:
        st.markdown(
            "<div class='soft-panel'>"
            "<strong>Vista previa procesada (8 × 8 píxeles)</strong></div>",
            unsafe_allow_html=True,
        )
        st.image((imagen8 / 16 * 255).astype(np.uint8), width=150, clamp=True)
        if st.button(f"Reconocer con los {len(modelos)} modelos", type="primary"):
            tabla_predicciones = predecir_todos(imagen8, modelos)
            st.session_state["ultima_prediccion"] = tabla_predicciones
            st.session_state["ultima_imagen"] = imagen8
            historial = st.session_state.get("historial", [])
            historial.append(crear_fila_historial(tabla_predicciones))
            st.session_state["historial"] = historial

    if modo_reconocimiento == MODO_UN_NUMERO and "ultima_prediccion" in st.session_state:
        st.subheader("Resultado de los modelos")
        tabla_predicciones = st.session_state["ultima_prediccion"]
        numero_consenso, acuerdos = resumir_prediccion(tabla_predicciones)
        if numero_consenso is not None:
            st.success(
                f"Número detectado en la imagen: {numero_consenso} "
                f"(acuerdo de {acuerdos} de {len(modelos)} modelos)"
            )

        columnas = st.columns(3)
        for indice, (_, fila) in enumerate(tabla_predicciones.iterrows()):
            with columnas[indice % 3]:
                st.metric(
                    fila["Modelo"],
                    f"Número {fila['Predicción']}",
                    help="Confianza estimada del modelo",
                )
                if fila["Confianza estimada"] is not None:
                    st.caption(f"Confianza: {fila['Confianza estimada'] * 100:.1f}%")

        st.dataframe(tabla_predicciones, use_container_width=True, hide_index=True)
        st.caption(
            "La confianza es una estimación del modelo, "
            "no una garantía de que la predicción sea correcta."
        )

        numero_correccion = st.number_input(
            "¿Cuál es el número correcto?",
            min_value=0,
            max_value=9,
            value=int(numero_consenso) if numero_consenso is not None else 0,
            step=1,
            help=(
                "Si alguna predicción fue correcta, deja el valor correcto aquí "
                "y guarda el ejemplo para aprender."
            ),
        )
        if st.button("Guardar corrección y entrenar con este ejemplo", type="primary"):
            ultima_imagen = st.session_state.get("ultima_imagen")
            if ultima_imagen is not None:
                guardar_ejemplo(ultima_imagen, numero_correccion)
                st.cache_resource.clear()
                st.session_state["numero_correccion_guardado"] = numero_correccion
                st.session_state["mostrar_carga_entrenamiento"] = True
                st.rerun()
            else:
                st.warning("No hay imagen disponible para guardar como ejemplo.")

        if st.session_state.get("historial"):
            tabla_historial = pd.DataFrame(st.session_state["historial"])
            st.markdown("**Historial de pruebas de esta sesión**")
            st.dataframe(tabla_historial, use_container_width=True, hide_index=True)
            st.download_button(
                "Descargar historial en CSV",
                tabla_historial.to_csv(index=False).encode("utf-8"),
                file_name="historial_tecnoforms.csv",
                mime="text/csv",
            )

    # Reconocimiento y corrección de varios números
    if modo_reconocimiento == MODO_VARIOS_NUMEROS and digitos_hoja:
        st.subheader("Resultados de la hoja")
        if st.button("Reconocer todos los números detectados", type="primary"):
            st.session_state["mostrar_carga_secuencias"] = True
            st.rerun()

        if st.session_state.get("mostrar_carga_secuencias", False):
            try:
                grupos_digitos = agrupar_digitos(digitos_hoja)
                st.session_state["resultados_hoja"] = predecir_secuencias(
                    digitos_hoja, grupos_digitos, modelos
                )
                indices_ordenados = [
                    indice for grupo in grupos_digitos for indice in grupo
                ]
                st.session_state["imagenes_correccion_hoja"] = [
                    digitos_hoja[indice]["imagen"] for indice in indices_ordenados
                ]
            finally:
                st.session_state.pop("mostrar_carga_secuencias", None)
                carga_entrenamiento.empty()

        if "resultados_hoja" in st.session_state:
            st.dataframe(
                st.session_state["resultados_hoja"],
                use_container_width=True,
                hide_index=True,
            )
            st.download_button(
                "Descargar números reconocidos en CSV",
                st.session_state["resultados_hoja"].to_csv(index=False).encode("utf-8"),
                file_name="numeros_reconocidos_tecnoforms.csv",
                mime="text/csv",
            )
            st.caption(
                "Escribe los números correctos en el mismo orden. "
                "Sepáralos con comas, espacios o saltos de línea."
            )
            correccion_secuencias = st.text_input(
                "¿Cuáles son los números correctos?",
                placeholder="13, 25, 67, 100",
                key="correccion_secuencias_hoja",
            )
            if st.button("Guardar corrección y aprender", type="primary"):
                separadores_permitidos = {",", ";", " ", "\t", "\n", "\r"}
                entrada_valida = all(
                    caracter.isdigit() or caracter in separadores_permitidos
                    for caracter in correccion_secuencias
                )
                cifras_correctas = [
                    caracter for caracter in correccion_secuencias if caracter.isdigit()
                ]
                imagenes_correccion = st.session_state.get(
                    "imagenes_correccion_hoja", []
                )

                if not entrada_valida or not cifras_correctas:
                    st.warning(
                        "Escribe números separados por comas, espacios o saltos de línea."
                    )
                elif len(cifras_correctas) != len(imagenes_correccion):
                    st.warning(
                        f"Escribiste {len(cifras_correctas)} cifras, pero se detectaron "
                        f"{len(imagenes_correccion)}. "
                        "Revisa los números o la detección de la hoja."
                    )
                else:
                    for imagen_digito, cifra in zip(imagenes_correccion, cifras_correctas):
                        guardar_ejemplo(imagen_digito, int(cifra))
                    st.cache_resource.clear()
                    st.session_state.pop("resultados_hoja", None)
                    st.session_state.pop("imagenes_correccion_hoja", None)
                    st.session_state.pop("correccion_secuencias_hoja", None)
                    st.session_state["cantidad_correcciones_guardadas"] = len(
                        imagenes_correccion
                    )
                    st.session_state["mostrar_carga_entrenamiento"] = True
                    st.rerun()
    elif hay_imagen_hoja and modo_reconocimiento == MODO_VARIOS_NUMEROS and not digitos_hoja:
        st.warning(
            "No se encontraron números con claridad. "
            "Prueba con una foto más cercana, enfocada y con mejor iluminación."
        )

# Comparación del rendimiento de los modelos
elif seccion == "Comparar modelos":
    st.subheader("Comparación general")
    st.write(
        "Los hiperparámetros se seleccionan con validación cruzada estratificada "
        "de 3 pliegues dentro del conjunto de entrenamiento. La exactitud de prueba "
        "se calcula aparte, con datos que no participan en el ajuste."
    )
    columnas_metricas = {
        columna: "{:.3f}"
        for columna in resultados.columns
        if columna != "Modelo"
    }
    st.dataframe(
        resultados.style.format(columnas_metricas),
        use_container_width=True,
        hide_index=True,
    )

    with st.expander("Ver mejores hiperparámetros por modelo"):
        tabla_parametros = parametros_modelos.copy()
        tabla_parametros["Mejores hiperparámetros"] = tabla_parametros[
            "Mejores hiperparámetros"
        ].map(
            lambda parametros: ", ".join(
                f"{nombre}: {valor}" for nombre, valor in parametros.items()
            )
        )
        st.dataframe(tabla_parametros, use_container_width=True, hide_index=True)

    parametros_mlp = parametros_modelos.loc[
        parametros_modelos["Modelo"] == "Red neuronal MLP",
        "Mejores hiperparámetros",
    ].iloc[0]
    capas_ocultas = parametros_mlp["mlpclassifier__hidden_layer_sizes"]
    nodos_por_capa = " → ".join(str(nodos) for nodos in capas_ocultas)
    st.info(
        f"MLP seleccionado: {len(capas_ocultas)} capa(s) oculta(s), "
        f"{nodos_por_capa} nodo(s) por capa; salida de 10 nodos. "
        "Usa activación ReLU y optimizador Adam."
    )

    with st.expander("Ejemplo no supervisado: K-Means"):
        st.write(
            "K-Means agrupa las imágenes sin usar sus etiquetas durante el "
            "entrenamiento. Los números de cluster no equivalen directamente "
            "a los dígitos; la tabla permite comparar después los grupos con "
            "las etiquetas conocidas."
        )
        columna_silueta, columna_inercia = st.columns(2)
        columna_silueta.metric(
            "Silueta (muestra de entrenamiento)",
            f"{resultados_kmeans['Coeficiente de silueta']:.3f}",
        )
        columna_inercia.metric(
            "Inercia de K-Means",
            f"{resultados_kmeans['Inercia']:.1f}",
        )
        st.caption("Distribución de los dígitos reales dentro de cada cluster:")
        st.dataframe(
            resultados_kmeans["Distribución por dígito real"],
            use_container_width=True,
        )

    figura, eje = plt.subplots(figsize=(9, 4))
    eje.bar(resultados["Modelo"], resultados["Exactitud"])
    eje.set_ylim(0, 1.05)
    eje.set_ylabel("Exactitud")
    eje.set_title("Exactitud de cada modelo en datos de prueba")
    eje.tick_params(axis="x", rotation=15)
    eje.grid(axis="y", alpha=.25)
    st.pyplot(figura)

    mejor_modelo = resultados.loc[resultados["Exactitud"].idxmax()]
    st.success(
        "En esta ejecución, el mejor resultado por exactitud fue "
        f"{mejor_modelo['Modelo']} "
        f"({mejor_modelo['Exactitud'] * 100:.2f}%)."
    )

# Evaluación, matriz de confusión y ejemplos
elif seccion == "Evaluación y gráficos":
    st.subheader("Matriz de confusión")

    modelo_elegido = st.selectbox("Selecciona un modelo", list(modelos.keys()))
    predicciones = predicciones_prueba[modelo_elegido]

    matriz = confusion_matrix(numeros_prueba, predicciones, labels=list(range(10)))
    figura, eje = plt.subplots(figsize=(7, 5))
    mapa_colores = eje.imshow(matriz, cmap="Blues")
    eje.set_title(f"Matriz de confusión — {modelo_elegido}")
    eje.set_xlabel("Predicción")
    eje.set_ylabel("Valor real")
    eje.set_xticks(range(10))
    eje.set_yticks(range(10))
    for fila in range(10):
        for columna in range(10):
            eje.text(
                columna,
                fila,
                matriz[fila, columna],
                ha="center",
                va="center",
                fontsize=8,
            )
    figura.colorbar(mapa_colores, ax=eje)
    st.pyplot(figura)

    st.markdown("**Ejemplos: valor real frente a predicción**")
    cantidad_ejemplos = 12
    figura_ejemplos, ejes = plt.subplots(3, 4, figsize=(9, 6))
    for indice, eje_ejemplo in enumerate(ejes.flat):
        if indice >= cantidad_ejemplos:
            eje_ejemplo.axis("off")
            continue
        eje_ejemplo.imshow(caracteristicas_prueba[indice].reshape(8, 8), cmap="gray_r")
        eje_ejemplo.set_title(
            f"Real: {numeros_prueba[indice]} | Pred: {predicciones[indice]}"
        )
        eje_ejemplo.axis("off")
    figura_ejemplos.tight_layout()
    st.pyplot(figura_ejemplos)
    st.caption(
        "Estos ejemplos pertenecen al conjunto de prueba, "
        "por eso sí conocemos su valor real."
    )

    st.markdown("**Reporte de clasificación**")
    reporte = classification_report(
        numeros_prueba,
        predicciones,
        output_dict=True,
        zero_division=0,
    )
    st.dataframe(pd.DataFrame(reporte).transpose().round(3), use_container_width=True)

# Descripción técnica del proyecto
else:
    st.subheader("Descripción del proyecto")
    st.write("""
    **Problema:** TecnoForms necesita reconocer automáticamente dígitos escritos a mano
    en formularios. La escritura cambia de una persona a otra y las fotografías pueden
    tener sombras, tamaños o fondos diferentes.

    **Preprocesamiento:** convertir la imagen a escala de grises, aplicar contraste
    y umbral para separar tinta y fondo, segmentar los dígitos de una hoja, recortar
    y centrar cada cifra, redimensionarla a 8 × 8 y normalizar sus intensidades al
    rango 0–16. KNN, SVM, regresión logística y MLP estandarizan además las
    características durante el entrenamiento.

    **Modelos supervisados:** KNN compara ejemplos cercanos; SVM y regresión
    logística clasifican con fronteras aprendidas; Random Forest y Extra Trees
    combinan árboles; Árbol de decisión usa reglas; MLP es una red neuronal.
    La arquitectura MLP seleccionada, cantidad de capas y nodos, activación y
    optimizador se muestran en «Comparar modelos».

    **K-Means:** se incluye como ejemplo de agrupamiento no supervisado con 10
    clusters. No predice directamente las etiquetas 0–9; se informa la silueta y
    se muestra la distribución de etiquetas reales por cluster para interpretarlo.

    **Alcance:** esta versión es un prototipo académico. Para formularios reales de una
    empresa sería necesario entrenar y probar con fotografías de formularios reales,
    además de mejorar la segmentación de cada casilla antes de reconocer sus dígitos.
    """)
    st.markdown("**Flujo de trabajo**")
    st.code(
        "Imagen → grises/contraste → segmentación → recorte y centrado → "
        "8×8 y normalización 0–16 → 7 clasificadores → predicción y evaluación",
        language="text",
    )
    st.markdown(
        "**Tecnologías utilizadas:** Python, Streamlit, scikit-learn, "
        "NumPy, Pandas, Matplotlib y Pillow."
    )
