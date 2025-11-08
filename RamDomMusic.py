import scipy.io.wavfile as wav
import os
import sys
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import QApplication, QMainWindow, QGraphicsDropShadowEffect, QFileDialog, QDialog
from PyQt5.QtCore import QPropertyAnimation, QEasingCurve
from PyQt5 import QtCore, QtWidgets
import torch
import librosa
import numpy as np
from scipy.io.wavfile import write as wav_write
from ModeloDENSE import MusicSeparationModel

import pyaudio
import pyqtgraph as pg
from pydub import AudioSegment

n = 0  # 'n' es la variable que nos moverá por el tiempo




class EqualizerWidget(pg.PlotWidget):
    def __init__(self, parent=None):
        super(EqualizerWidget, self).__init__(parent)
        self.init_ui()

    def init_ui(self):
        #self.setBackground('w')
        self.setYRange(0, 1)
        self.setXRange(0, 31)
        self.bars = pg.BarGraphItem(x=np.arange(32), height=np.zeros(32), width=0.5, brush='g')
        self.addItem(self.bars)

    def update_bars(self, data):
        self.bars.setOpts(height=data)
        
from dialog import Ui_Dialog  # dialog.py generado con pyuic5
from Design import Ui_MainWindow  # dialog.py generado con pyuic5
class Dialog(QDialog, Ui_Dialog):
    def __init__(self):
        super().__init__()
        self.setupUi(self)  # Inicializa la interfaz desde el archivo .py

        self.setWindowFlag(QtCore.Qt.FramelessWindowHint)
        self.setWindowOpacity(1)

class MainWindow(QMainWindow,Ui_MainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setupUi(self)  # Asegúrate de que esta ruta sea correcta
        self.bt_menu_iz.clicked.connect(self.mover_menu)
        self.bt_menu_dr.clicked.connect(self.mover_menu)
        self.bt_menu_iz_2.clicked.connect(self.mover_menu)
        self.bt_menu_dr_2.clicked.connect(self.mover_menu)
        
        self.cancelado = False
        self.aceptado = False

        # Ocultar botones
        self.bt_pausa.hide()
        self.bt_menu_dr.hide()
        self.bt_menu_dr_2.hide()
        self.bt_minimizar.hide()
        self.progressBar.hide()
        self.progresBarAceptar.hide()
        self.progresBarCancelar.hide()

        # Sombra widgets
        self.sombra_frame(self.stackedWidget)
        self.sombra_frame(self.frame_superior)
        self.sombra_frame(self.toolBox)
        self.sombra_frame(self.bt_1)
        self.sombra_frame(self.bt_2)
        self.sombra_frame(self.bt_3)
        self.sombra_frame(self.bt_play)
        self.sombra_frame(self.bt_rewind)
        self.sombra_frame(self.bt_stop)
        self.sombra_frame(self.bt_forward)
        self.sombra_frame(self.horizontalSlider)
        self.sombra_frame(self.bass_slider)
        self.sombra_frame(self.drum_slider)
        self.sombra_frame(self.other_slider)
        self.sombra_frame(self.vocals_slider)
        self.sombra_frame(self.label_Volumen)
        self.sombra_frame(self.label_Pitch)
        self.sombra_frame(self.bt_separar)
        self.sombra_frame(self.label_2)

        # Control barra superior
        self.bt_cerrar.clicked.connect(self.control_bt_cerrar)
        self.bt_minimizar.clicked.connect(self.control_bt_normal)
        self.bt_mini.clicked.connect(self.control_bt_minimizar)
        self.bt_maximizar.clicked.connect(self.control_bt_maximizar)

        # Eliminar borde de la ventana
        self.setWindowFlag(QtCore.Qt.FramelessWindowHint)
        self.setWindowOpacity(1)

        # SizeGrip
        self.gripSize = 10
        self.grip = QtWidgets.QSizeGrip(self)
        self.grip.resize(self.gripSize, self.gripSize)

        # Mover ventana
        self.frame_superior.mouseMoveEvent = self.mover_ventana

        # Páginas
        self.bt_1.clicked.connect(self.pagina_1)
        self.bt_2.clicked.connect(self.pagina_2)
        self.bt_3.clicked.connect(self.pagina_3)

        # Conectar los botones a sus funciones correspondientes
        self.bt_separar.clicked.connect(self.separate_audio)
        self.bt_play.clicked.connect(self.start_callback)
        self.bt_stop.clicked.connect(self.stop_callback)
        self.bt_pausa.clicked.connect(self.pausa_callback)
        self.bass_slider.valueChanged.connect(self.update_bass)
        self.drum_slider.valueChanged.connect(self.update_drums)
        self.other_slider.valueChanged.connect(self.update_other)
        self.vocals_slider.valueChanged.connect(self.update_vocals)
        self.bt_busqueda.clicked.connect(self.browse_folder_sparator)
        self.bt_busqueda_2.clicked.connect(self.browse_folder)
        self.path_edit = self.lineEdit_2
        self.path_edit_separator = self.lineEdit
        self.bt_rewind.clicked.connect(self.retraso)
        self.bt_forward.clicked.connect(self.avance)
        self.horizontalSlider.valueChanged.connect(self.update_pitch)
        self.progresBarCancelar.clicked.connect(self.cancelar_proceso)

        # Conectar la señal valueChanged de barra_tiempo al método update_n
        self.barra_tiempo.valueChanged.connect(self.update_n_from_barra_tiempo)
        self.barra_tiempo.setMinimum(0)
        self.statusLabel
        
        # Timer para actualizar la barra de tiempo
        self.update_timer = QtCore.QTimer()
        self.update_timer.timeout.connect(self.update_barra_tiempo)
        self.update_timer.start(1000)  # Actualiza cada segundo

        # Variables de clase
        self.bass = 1.0
        self.drums = 1.0
        self.other = 1.0
        self.vocals = 1.0
        self.pitch = 0.0
        self.stream = None
        self.p = pyaudio.PyAudio()

        # Añadir ecualizador al layout verticalLayout_3
        self.equalizer_widget = EqualizerWidget(self)
        self.verticalLayout_3.addWidget(self.equalizer_widget)
        
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_equalizer)
        self.timer.start(50)
        
        if self.aceptado:
            self.statusLabel.setText("Separación completada.")
            self.progresBarAceptar.hide()
            self.progressBar.hide()
            self.progressBar.setValue(0)
            self.aceptado = False
            

    # Mover el menú
    def mover_menu(self):
        width = self.frame_2.width()
        normal = 0
        if width == 0:
            extender = 300
            self.bt_menu_dr.hide()
            self.bt_menu_iz.show()
            self.bt_menu_dr_2.hide()
            self.bt_menu_iz_2.show()
        else:
            self.bt_menu_dr.show()
            self.bt_menu_iz.hide()
            self.bt_menu_dr_2.show()
            self.bt_menu_iz_2.hide()
            extender = normal
        self.animacion = QPropertyAnimation(self.frame_2, b"maximumWidth")
        self.animacion.setStartValue(width)
        self.animacion.setEndValue(extender)
        self.animacion.setDuration(500)
        self.animacion.setEasingCurve(QEasingCurve().OutInBack)
        self.animacion.start()

    def sombra_frame(self, frame):
        sombra = QGraphicsDropShadowEffect(self)
        sombra.setBlurRadius(30)
        sombra.setXOffset(8)
        sombra.setYOffset(8)
        sombra.setColor(QColor(20, 200, 220, 255))
        frame.setGraphicsEffect(sombra)
        
            
    def control_bt_minimizar(self):
        self.showMinimized()

    def control_bt_normal(self):
        self.showNormal()
        self.bt_minimizar.hide()
        self.bt_maximizar.show()

    def control_bt_maximizar(self):
        self.showMaximized()
        self.bt_minimizar.show()
        self.bt_maximizar.hide()

    def resizeEvent(self, event):
        rect = self.rect()
        self.grip.move(rect.right() - self.gripSize, rect.bottom() - self.gripSize)

    # Mover Ventana
    def mousePressEvent(self, event):
        self.clickPosition = event.globalPos()

    def mover_ventana(self, event):
        if not self.isMaximized():
            if event.buttons() == QtCore.Qt.LeftButton:
                self.move(self.pos() + event.globalPos() - self.clickPosition)
                self.clickPosition = event.globalPos()
                event.accept()
        if event.globalPos().y() <= 10:
            self.showMaximized()
            self.bt_maximizar.hide()
            self.bt_mini.show()
        else:
            self.showNormal()
            self.bt_maximizar.show()

    def pagina_1(self):
        self.stackedWidget.setCurrentWidget(self.page_1)

    def pagina_2(self):
        self.stackedWidget.setCurrentWidget(self.page_2)

    def pagina_3(self):
        self.stackedWidget.setCurrentWidget(self.page_3)
        
        
    def cancelar_proceso(self):
        self.cancelado = True
        
        
    def resource_path(self,relative_path):
        if hasattr(sys, '_MEIPASS'):
            return os.path.join(sys._MEIPASS, relative_path)
        return os.path.join(os.path.abspath("."), relative_path)
            
        
    def separate_audio(self):

        self.progressBar.hide()
        self.progressBar.setValue(0)
        self.progressBar.show()
        
        input_file =  self.lineEdit.text()
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Rutas a checkpoints por instrumento
        checkpoint_paths = {
            "bass": self.resource_path("RamDOM_weight_bass.pth"),
            "drums": self.resource_path("RamDOM_weight_drums.pth"),
            "other": self.resource_path("RamDOM_weight_other.pth"),
            "vocals": self.resource_path("RamDOM_weight_vocals.pth")
        }
        
        chunk_duration = 4.0
        sample_rate = 16000
        n_fft = 4096
        hop_length = 1024
        
        hann_window = torch.hann_window(n_fft).to(device)
        
        # Modelo base (debes definir esta clase en tu código)
        model = MusicSeparationModel().to(device)
        
        # Función principal de inferencia
        def infer_song(song_path):
            
            # Cargar audio estéreo
            mix_audio, sr = librosa.load(song_path, sr=sample_rate, mono=False)
            if mix_audio.ndim == 1:
                mix_audio = np.stack([mix_audio, mix_audio], axis=0)
        
            max_val = np.max(np.abs(mix_audio))
            if max_val > 0:
                mix_audio = mix_audio / max_val
        
            chunk_samples = int(chunk_duration * sample_rate)
            total_chunks = int(np.ceil(mix_audio.shape[1] / chunk_samples))
        
            # Nombre y carpeta de la canción
            song_name = os.path.splitext(os.path.basename(song_path))[0]
            song_dir = os.path.dirname(song_path)
            output_folder = os.path.join(song_dir, song_name)
            os.makedirs(output_folder, exist_ok=True)
        
            for stem_name, ckpt_path in checkpoint_paths.items():
                checkpoint = torch.load(ckpt_path, map_location=device)
                model.load_state_dict(checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint)
                model.eval()
                
                self.progresBarCancelar.show()
                
                # Reiniciar barra para este stem
                self.progressBar.setMinimum(0)
                self.progressBar.setMaximum(total_chunks)
                self.progressBar.setValue(0)
                QApplication.processEvents()
                
                pred_audio = []
                progress = 0
        
                with torch.no_grad():
                    for i in range(total_chunks):
                        if self.cancelado:
                            self.statusLabel.setText("Proceso cancelado.")
                            self.progresBarCancelar.hide()
                            self.progressBar.hide()
                            self.cancelado = False
                            break  # O usa break si quieres continuar con otra parte
                            
                            
                        start = i * chunk_samples
                        end = start + chunk_samples
                        chunk = mix_audio[:, start:end]
                        
        
                        if chunk.shape[1] < chunk_samples:
                            pad_width = chunk_samples - chunk.shape[1]
                            chunk = np.pad(chunk, ((0, 0), (0, pad_width)), mode='constant')
        
                        chunk_tensor = torch.from_numpy(chunk).float().to(device)
        
                        stft_mix = torch.stack(
                            [torch.stft(chunk_tensor[c], n_fft=n_fft, hop_length=hop_length, window=hann_window, return_complex=True)
                             for c in range(chunk_tensor.shape[0])],
                            dim=0
                        ).unsqueeze(0).to(device)
        
                        log_mag_mix = torch.log1p(torch.clamp(torch.abs(stft_mix), min=1e-8))
                        phase_mix = torch.angle(stft_mix)
        
                        pred_mag = model(log_mag_mix)
                        mag_pred = torch.expm1(pred_mag).clamp(min=0.0)
                        pred_complex = mag_pred * torch.exp(1j * phase_mix)
        
                        pred_chunk = []
                        for c in range(pred_complex.shape[1]):
                            istft_audio = torch.istft(pred_complex[0, c].cpu(), n_fft=n_fft, hop_length=hop_length, length=chunk_samples, window=hann_window.cpu())
                            pred_chunk.append(istft_audio.numpy())
        
                        pred_chunk = np.stack(pred_chunk, axis=0)
                        pred_audio.append(pred_chunk)
                        
                        # Actualizar barra para este stem
                        self.statusLabel.setText(f"Procesando: {stem_name}")
                        self.progressBar.setValue(progress)
                        QApplication.processEvents()
                        progress += 1
                        
                
                full_audio = np.concatenate(pred_audio, axis=1)
                full_audio = full_audio * max_val
        
                out_path = os.path.join(output_folder, f"{stem_name}.wav")
                wav_write(out_path, sample_rate, (full_audio.T * 32767).astype(np.int16))
                
            self.progresBarCancelar.hide()
            self.statusLabel.setText("Separación completada.")            
            self.progressBar.hide()

        return infer_song(input_file)
    
    def lee_audio(self, fichero):
        """Lee fichero de audio wav (devuelve frecuencia de muestreo y array)
        Argumentos de Entrada:
            fichero (String): Cadena con la rura y el archivo de audio en formato wav
            
        Salida:
            fs (float): frecuencia de muestreo del audio
            x (np.ndarray): variable con las muestras de la señal de audio
        """    
        fs,x = wav.read(fichero)
        # Convertir a float32 y normalizar
        if x.dtype == np.int16:
            x = x.astype(np.float32) / 32768.0
        elif x.dtype == np.int32:
            x = x.astype(np.float32) / 2147483648.0
        elif x.dtype == np.uint8:
            x = (x.astype(np.float32) - 128) / 128.0
        else:
            x = x.astype(np.float32)  # para float WAVs
        return fs,x

        
    def start_callback(self):

        global Tb, Td, In, Nb, sd, Nbuf, B

        # Define la ruta de las pistas de audio
        ruta = self.path_edit.text()
        
        if ruta.endswith("mp3") or ruta.endswith(".wav"):
           self.play_audio()
           
        else:

            # Carga las pistas de audio
            fs1, x1 = self.lee_audio(ruta + '/bass.wav')
            fs2, x2 = self.lee_audio(ruta + '/drums.wav')
            fs3, x3 = self.lee_audio(ruta + '/other.wav')
            fs4, x4 = self.lee_audio(ruta + '/vocals.wav')
            
            self.barra_tiempo.setMaximum(len(x1) // fs1)
            
            Tb= 0.5 # 0.5
            Tdg = 0.025 # 0.025
            self.Ret1=0
            self.Ret2=0
            SAMPLE_RATE = fs1

            self.Td=1* Tdg # Tdg = 0.025
            self.Overlap=0.25 # Porcentaje de solapamiemto entre las dos líneas de retardo (10%) 0.1
            self.fgain = 1 / self.Overlap

            # fases normalizadas de las funciones de las líneas de retardo            
            self.ph1, self.ph2 = 0, (1 - self.Overlap)
            

            Nb=int(np.ceil(SAMPLE_RATE *Tb))  # Muestras del bloque
            

            sd=int(np.ceil(SAMPLE_RATE *self.Td))  # Muestras del retardo

            B=int(np.ceil(SAMPLE_RATE *(2*Tb)))

            In=False
                     
            # Ponemos el número de canales en función de las pistas si son mono o estéreo
            if (x1.shape[1], x2.shape[1], x3.shape[1], x4.shape[1]) == (1, 1, 1, 1):
                CHANNELS = 1
            elif (x1.shape[1], x2.shape[1], x3.shape[1], x4.shape[1]) == (2, 2, 2, 2):
                CHANNELS = 2
            else:
                print("Las pistas deben ser todas mono o estéreo")
    
            RATE = fs1   # Usamos la frecuencia de muestreo de la primera pista
            CHUNK = 1*B # Tamaño del chunk
            #print("CHUNK: ",CHUNK)
            x = []
            x.append(x1);x.append(x2);x.append(x3);x.append(x4)
            x = sum(x)
            
            if RATE == 44100:
                CHANNELS = 1
            
            CHANNELS = CHANNELS
    
            
            if all(x is None for x in [x1, x2, x3, x4]):
                In=True
            else:
                numtramas=np.ceil(len(x)/CHUNK)
                if CHANNELS==1:
                    xss1=np.zeros(int(numtramas)*CHUNK)
                    x = np.mean(x, axis=1)  # Convertir estéreo a mono
                    xss1[0:len(x)]=x
                    self.Nbuf=np.zeros(int(2+Nb+sd))  # Buffer de datos incluyendo líneas de retardo
                else:
                    xss1=np.zeros([int(numtramas)*CHUNK,CHANNELS])
                    xss1[0:len(x),:]=x
                    self.Nbuf1=np.zeros([int(2+Nb+sd),CHANNELS])  # Buffer de datos incluyendo líneas de retardo
            
            fss=1*fs1
            self.x = x
            self.barraTiempo = 0.0
            self.fs1 = fs1
            self.x1 = x1
            self.x2 = x2
            self.x3 = x3
            self.x4 = x4
            self.CHANNELS = CHANNELS
            self.RATE = RATE
            self.CHUNK = CHUNK
            self.numtramas = numtramas

            self.Nb = Nb
            self.fss = fss
            self.In = In
            self.xss1 = xss1
            self.sd = sd
            
    
            # Llamamos a la función para iniciar la reproducción
            self.start_stream()

    def start_stream(self):
        global n
        
        self.bt_play.hide()
        self.bt_pausa.show()
        self.yb=np.zeros(self.CHUNK)
        self.yb1=np.zeros([self.CHUNK,self.CHANNELS])
    
        # Función callback donde se mezclan las pistas
        def callback(in_data, frame_count, time_info, status):
            
            global n, Ret1, Ret2, ph1, ph2, Overlap, fgain, In, Tb, Td, In, Nb, Nbuf, B, Nbuf1, Nbuf2
            
            flag = pyaudio.paContinue
            # Si hemos alcanzado el final de las pistas, reiniciamos la posición
            
            # Si las pistas son mono
            if self.CHANNELS == 1:
                self.xss1 = self.x1[n:n + self.CHUNK, 0] * self.bass + self.x2[n:n + self.CHUNK, 0] * self.drums + self.x3[n:n + self.CHUNK, 0] * self.other + self.x4[n:n + self.CHUNK, 0] * self.vocals
                # self.xss1 = self.xss1.astype(np.int16)
                if n >= len(self.x1):
                    n = 0
                if in_data is None:
                    #print("Advertencia: in_data es None")
                    in_data = np.zeros(frame_count, dtype=np.int16).tobytes()
                
                audio_data = np.frombuffer(in_data, dtype=np.int16)
                
                m = (1-2**(self.pitch/12)) # Variación del retardo (en muestras) de una muesra a la siguiente
                #print(self.pitch,m)
                pRate = m /self.Td        # Variación del retardo en muestras por segundo
                pstep = pRate / self.fss     # Paso de las fases (adimensional)
                
                if self.In==False:
                    if n==self.numtramas*self.CHUNK:
                        self.stop_callback()
                 
                    in_data=self.xss1
                    
                                        
                else:
                    in_data=audio_data
                    
                self.Nbuf=np.concatenate((in_data[-1::-1],self.Nbuf[:-self.Nb]))
                self.update_barra_tiempo()
                if m==0:
                    out_data=1*in_data
                    #print(f"Salida - out_data: {out_data[:5]}")
                    
                else:
                    #t0 = time.time()
                    out_data = self.pitching(in_data,pstep,self.yb,self.Nbuf, self.ph1, self.ph2, self.Overlap, self.fgain, self.sd)
                    #print("pitching duró", time.time() - t0, "segundos")
                    #print(f"Salida - out_data: {out_data[:5]}")
                if isinstance(out_data, tuple):
                    out_data = out_data[0]  # Convierte a array NumPy si no lo es 
                    
                n=n+self.CHUNK
                    
                # Para convertir de vuelta a int16 antes del .tobytes()
                out_data = np.clip(out_data * 32767, -32768, 32767).astype(np.int16)
                return (out_data.tobytes(), flag)
            
            # Si las pistas son estéreo
            elif self.CHANNELS == 2:
                # Multiplica cada pista del lado izquierdo por su volumen respectivo
                self.xss1 = self.x1[n:n + self.CHUNK,:] * self.bass + self.x2[n:n + self.CHUNK,:] * self.drums + self.x3[n:n + self.CHUNK,:] * self.other + self.x4[n:n + self.CHUNK,:] * self.vocals               
                
                if n >= len(self.x1):
                     n = 0
                     
                if in_data is None:
                    in_data = np.zeros(frame_count, dtype=np.int16).tobytes()
           
                # Convertir los datos de bytes a un array de NumPy
                audio_data = np.frombuffer(in_data, dtype=np.int16).reshape(-1, self.CHANNELS)
                #print(f"Forma in_data: {in_data.shape}, Tipo: {in_data.dtype}")
                
                     
                m = (1-2**(self.pitch/12)) # Variación del retardo (en muestras) de una muesra a la siguiente
                #print(self.pitch,m)
                pRate = m /self.Td        # Variación del retardo en muestras por segundo
                pstep1 = pRate / self.fss 

                #print("pRate: ",pRate,"pstep: ", pstep1, "fs: ", self.fss)
                if self.In==False:
                    if n==self.numtramas*self.CHUNK:
                        self.stop_callback()

                    in_data= self.xss1
  
                else:
                    in_data=audio_data
                    
                self.Nbuf1=np.concatenate((in_data[-1::-1,:],self.Nbuf1[:-self.Nb,:]))
                #print("Nbuf1: ", self.Nbuf.shape, "Nbuf2: ",self.Nbuf2.shape)
                self.update_barra_tiempo()
                if m==0:
                    out_data1 = 1*in_data
                   
                else:
                    #t0 = time.time()                    
                    out_data1 = self.pitching(
                        in_data, pstep1, self.yb1, self.Nbuf1, self.ph1, self.ph2, self.Overlap, self.fgain, self.sd
                    )
                    #print("pitching duró", time.time() - t0, "segundos")

                n=n+self.CHUNK 
            
  
                mixed_data = np.clip(out_data1 * 32767, -32768, 32767).astype(np.int16)
                return (mixed_data.tobytes(), flag)


        # Abre un nuevo stream solo si no está abierto actualmente
        if self.stream is None or not self.stream.is_active():
            # Abre un stream de audio
            self.stream = self.p.open(format=pyaudio.paInt16,
                                      channels=self.CHANNELS,
                                      rate=self.RATE,
                                      frames_per_buffer=self.CHUNK,
                                      output=True,
                                      stream_callback=callback)
            
    def pitching(self, in_data, pstep, yb, Nbuf, ph1, ph2, overlap, fgain, sd):
        if np.all(Nbuf == 0):
            return in_data, ph1, ph2
        
        Nb = len(yb)
        Ret1 = 0.0
        Ret2 = 0.0

        for muestra in range(Nb):
            ph1 = (ph1 + pstep) % 1.0
            ph2 = (ph2 + pstep) % 1.0

            # Transiciones entre fases
            if (ph1 < overlap) and (ph2 >= (1.0 - overlap)):
                Ret1 = sd * ph1
                Ret2 = sd * ph2
                Gan1 = np.cos((1.0 - (ph1 * fgain)) * np.pi / 2)
                Gan2 = np.cos(((ph2 - (1.0 - overlap)) * fgain) * np.pi / 2)
            elif (ph1 > overlap) and (ph1 < (1.0 - overlap)):
                ph2 = 0.0
                Ret1 = sd * ph1
                Gan1 = 1.0
                Gan2 = 0.0
            elif (ph1 >= (1.0 - overlap)) and (ph2 < overlap):
                Ret1 = sd * ph1
                Ret2 = sd * ph2
                Gan1 = np.cos(((ph1 - (1.0 - overlap)) * fgain) * np.pi / 2)
                Gan2 = np.cos((1.0 - (ph2 * fgain)) * np.pi / 2)
            elif (ph2 > overlap) and (ph2 < (1.0 - overlap)):
                ph1 = 0.0
                Ret2 = sd * ph2
                Gan1 = 0.0
                Gan2 = 1.0
            else:
                Gan1 = 0.0
                Gan2 = 0.0

            # Línea 1
            retardo1 = Nb - muestra + Ret1
            Nretardo1 = int(np.floor(retardo1))
            if Nretardo1 + 1 >= len(Nbuf):
                continue
            frac1 = retardo1 - Nretardo1
            y1=Nbuf[int(Nretardo1)]*(1-frac1)+Nbuf[int(Nretardo1)+1]*(frac1) 

            # Línea 2
            retardo2 = Nb - muestra + Ret2
            Nretardo2 = int(np.floor(retardo2))
            if Nretardo2 + 1 >= len(Nbuf):
                continue
            frac2 = retardo2 - Nretardo2
            y2=Nbuf[int(Nretardo2)]*(1-frac2)+Nbuf[int(Nretardo2)+1]*(frac2) 

            # Mezcla de líneas
            yb[muestra] = Gan1 * y1 + Gan2 * y2

        return yb

    def stop_callback(self):
        global n
        self.bt_pausa.hide()
        self.bt_play.show()

        self.stream.stop_stream()
        self.stream.close()
        print("Reproducción detenida.")
        self.stream = None
        n = 0

            
    def pausa_callback(self):
        global n
        self.bt_pausa.hide()
        self.bt_play.show()
 
        if self.stream is not None:
            self.stream.stop_stream()
            self.stream.close()
            print("Reproducción detenida.")
            self.stream = None
        else:
            print("No hay reproducción activa.")

    def control_bt_cerrar(self):
        global n

        if self.stream is not None:
            self.stream.stop_stream()
            self.stream.close()
            print("Reproducción detenida.")
            self.stream = None
            n = 0
            self.close() 
        else:
            self.close()
            
               

    def update_bass(self, value):
        self.bass = value / 100.0

    def update_drums(self, value):
        self.drums = value / 100.0

    def update_other(self, value):
        self.other = value / 100.0

    def update_vocals(self, value):
        self.vocals = value / 100.0
        
    def update_pitch(self, value):
        self.pitch = value / 3
        
    def update_barra_tiempo(self):
        if self.stream is not None and self.stream.is_active():
            global n
            self.barra_tiempo.setValue(n // self.fs1 )
            seconds = n // self.fs1 
            minutes = seconds // 60
            seconds = seconds % 60
            time_str = f"{minutes:02}:{seconds:02}"
            self.label_3.setText(time_str)
        
    def update_n_from_barra_tiempo(self, value):
        global n
        n = value * self.fs1   # Actualizar n según el valor de barra_tiempo
        # Actualizar el label_3 para mostrar el nuevo tiempo
        seconds = n // self.fs1 
        minutes = seconds // 60
        seconds = seconds % 60
        time_str = f"{minutes:02}:{seconds:02}"
        self.label_3.setText(time_str)

    def browse_folder(self):
        dialog = Dialog()
        if dialog.exec_() == QDialog.Accepted:
            if dialog.radioButton.isChecked():
                file_dialog = QFileDialog()
                file_path, _ = file_dialog.getOpenFileName(self, "Select Audio File", "", "Audio Files (*.wav *.mp3 *.ogg *.flac)")
                if file_path:
                    self.lineEdit_2.setText(file_path)
            elif dialog.radioButton_3.isChecked():
                file_dialog = QFileDialog()
                folder_path = file_dialog.getExistingDirectory(self, "Select Folder")
                if folder_path:
                    self.lineEdit_2.setText(folder_path)
                    
    def browse_folder_sparator(self):
        folder_path,_ = QFileDialog.getOpenFileName(self, "Select Audio File", "", "Audio Files (*.wav *.mp3)")
        if folder_path:
            self.lineEdit.setText(folder_path)
                    
    def play_audio(self):

        
        self.bt_play.hide()
        self.bt_pausa.show()

        # Define la ruta de las pistas de audio
        ruta = self.path_edit.text()
        # Reproducir un archivo MP3
        if ruta.endswith(".mp3"):
            sound = AudioSegment.from_mp3(ruta)
            wav_path = ruta.replace('.mp3', '.wav')
            sound.export(wav_path, format="wav")
            # Carga las pistas de audio
            fs, xs = self.lee_audio(wav_path)
        # Reproducir un archivo WAV
        elif ruta.endswith(".wav"): 
            
            fs, xs = self.lee_audio(ruta)
        else:
            print("Solo archivos .wav o .mp3")
            
        self.barra_tiempo.setMaximum(len(xs) // self.fs1)
        x1 = None

        # Ponemos el número de canales en función de las pistas si son mono o estéreo
        if (xs.shape[1]) == 1:
            CHANNELS = 1
        elif (xs.shape[1]) == 2:
            CHANNELS = 2
        else:
            print("Las pistas deben ser todas mono o estéreo")

        RATE = fs  # Usamos la frecuencia de muestreo de la primera pista
        CHUNK = 1024  # Tamaño del chunk
        # CHANNELS = 1
        self.x1 = x1
        self.xs = xs
        self.CHUNK = CHUNK
        self.CHANNELS = CHANNELS
            
        def callback(in_data, frame_count, time_info, status):
            global n
            
            if self.CHANNELS == 1:
                mixed_data = self.xs[n:n + self.CHUNK, 0]
                mixed_data = mixed_data.astype(np.int16)
            # Si las pistas son estéreo
            elif self.CHANNELS == 2:
                # Multiplica cada pista del lado izquierdo por su volumen respectivo
                mixed_data_left = self.xs[n:n + self.CHUNK, 0] 
                # Convierte los datos a formato adecuado para PyAudio (int16)
                mixed_data_left = mixed_data_left.astype(np.int16)
                # Multiplica cada pista del lado derecho por su volumen respectivo
                mixed_data_right = self.xs[n:n + self.CHUNK, 1] 
                mixed_data_right = mixed_data_right.astype(np.int16)
                mixed_data = np.column_stack((mixed_data_left, mixed_data_right))  # Juntamos los dos canales
            n += self.CHUNK
            
            self.update_barra_tiempo()
            
            # Retorna los datos mezclados para su reproducción
            return mixed_data.tobytes(), pyaudio.paContinue
            
            
        if self.stream is None or not self.stream.is_active():
            # Abre un stream de audio
            self.stream = self.p.open(format=pyaudio.paInt16,
                                      channels=CHANNELS,
                                      rate=RATE,
                                      frames_per_buffer=CHUNK,
                                      output=True,
                                      stream_callback=callback)
            print("Reproducción iniciada.")

        os.remove(wav_path)

    def avance(self):
        global n
        n += self.fs1

    def retraso(self):
        global n
        n -= self.fs1

    def update_equalizer(self):
        if self.stream is not None and self.stream.is_active():
            # Obtener los datos de audio actuales para el ecualizador
            mixed_data = self.get_current_mixed_data()
            if mixed_data is not None:
                frequencias = np.fft.rfft(mixed_data)
                magnitudes = np.abs(frequencias)
                magnitudes = magnitudes[:32]  # Tomamos las primeras 32 frecuencias
                magnitudes = magnitudes / np.max(magnitudes)  # Normalizamos los datos
                self.equalizer_widget.update_bars(magnitudes)

    def get_current_mixed_data(self):
        global n
        
        if self.x1 is not None:
            if n >= (len(self.x1)):
                return None
            # Si las pistas son mono
            if self.CHANNELS == 1:
                mixed_data = self.x1[n:n + self.CHUNK, 0] * self.bass + self.x2[n:n + self.CHUNK, 0] * self.drums + self.x3[n:n + self.CHUNK, 0] * self.other + self.x4[n:n + self.CHUNK, 0] * self.vocals
            # Si las pistas son estéreo
            elif self.CHANNELS == 2:
                mixed_data_left = self.x1[n:n + self.CHUNK, 0] * self.bass + self.x2[n:n + self.CHUNK, 0] * self.drums + self.x3[n:n + self.CHUNK, 0] * self.other + self.x4[n:n + self.CHUNK, 0] * self.vocals
                mixed_data_right = self.x1[n:n + self.CHUNK, 1] * self.bass + self.x2[n:n + self.CHUNK, 1] * self.drums + self.x3[n:n + self.CHUNK, 1] * self.other + self.x4[n:n + self.CHUNK, 1] * self.vocals
                mixed_data = np.column_stack((mixed_data_left, mixed_data_right))
                mixed_data = np.mean(mixed_data, axis=1)  # Convertimos a mono para la FFT
            return mixed_data.astype(np.float32)
        
        if self.xs is not None:
            if n >= (len(self.xs)):
                return None
            # Si las pistas son mono
            if self.CHANNELS == 1:
                mixed_data = self.xs[n:n + self.CHUNK, 0]
            # Si las pistas son estéreo
            elif self.CHANNELS == 2:
                mixed_data_left = self.xs[n:n + self.CHUNK, 0]
                mixed_data_right = self.xs[n:n + self.CHUNK, 1] 
                mixed_data = np.column_stack((mixed_data_left, mixed_data_right))
                mixed_data = np.mean(mixed_data, axis=1)  # Convertimos a mono para la FFT
            return mixed_data.astype(np.float32) 

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
