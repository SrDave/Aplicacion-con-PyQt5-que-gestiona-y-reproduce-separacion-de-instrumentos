# 🎵 Audio APP

## Introducción
**Audio APP** es una aplicación de escritorio desarrollada con **PyQt5** que permite la **reproducción, mezcla y manipulación de pistas de audio separadas** (voces, batería, bajo y otros).  
Además, integra un sistema de **separación de fuentes musicales mediante un modelo de Deep Learning**, permitiendo el control de volumen y *pitch shifting* en tiempo real.

---

## Estructura general

### Clases principales
- **EqualizerWidget:** Ecualizador gráfico en tiempo real.  
- **Dialog:** Ventana emergente personalizada.  
- **MainWindow:** Clase principal que gestiona la interfaz y la lógica del sistema.

### Librerías clave
`PyQt5`, `librosa`, `numpy`, `torch`, `pyaudio`, `scipy.io.wavfile`, `pydub`.

---

## Características principales
- **Reproducción y mezcla** de pistas en tiempo real.  
- **Control individual de volumen y pitch.**  
- **Visualización FFT** mediante ecualizador dinámico.  
- **Separación de fuentes musicales** mediante red neuronal convolucional (`MusicSeparationModel`).  
- **Empaquetado como ejecutable (.exe)** con PyInstaller para distribución sencilla.  

---

## Flujo de trabajo
1. **Lectura de audio** con `scipy.io.wavfile`.  
2. **Normalización y segmentación** del audio con `librosa`.  
3. **Inferencia del modelo neuronal** para separar pistas.  
4. **Procesamiento en tiempo real** con `PyAudio`.  
5. **Visualización de magnitudes FFT** con `pyqtgraph`.  
6. **Exportación de pistas** separadas en formato `.wav`.  

---

## Procesamiento de audio
El modelo analiza el espectro de frecuencias del audio (STFT) y estima las magnitudes logarítmicas desplazadas, reconstruyendo cada pista (bajo, batería, voz, otros) mediante **ISTFT**.  
El *pitch shifting* se logra con **líneas de retardo interpoladas**, manteniendo la duración original sin distorsión.

---

## Distribución
El proyecto se empaqueta como un ejecutable con **PyInstaller**, utilizando rutas relativas y recursos incrustados para permitir su ejecución sin instalar Python.  
Esto facilita la distribución a usuarios no técnicos y garantiza compatibilidad multiplataforma.


`pyinstaller --onefile --windowed RamDomMusic.py`


---

## Autor
**David Ramos Domingo**  
📚 Proyecto desarrollado como parte de estudios en **Tecnología Digital y Multimedia – UPV**  
🔗 [GitHub: SrDave](https://github.com/SrDave)
