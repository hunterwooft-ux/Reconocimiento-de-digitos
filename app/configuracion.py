from pathlib import Path

CARPETA_PROYECTO = Path(__file__).resolve().parent.parent
ARCHIVO_RETROALIMENTACION = CARPETA_PROYECTO / "correcciones_modelos.csv"
CARPETA_INTRO = CARPETA_PROYECTO / "extras" / "intro"
CARPETA_MUSICA = CARPETA_PROYECTO / "extras" / "musica"
RUTA_ESTILOS = Path(__file__).with_name("styles.css")
