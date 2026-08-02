"""Conexão e leitura do stream de vídeo.

Tenta conectar à URL MJPEG do celular (ex.: http://192.168.x.x:8080/video) e
cai para a webcam local (`cv2.VideoCapture(indice_fallback)`) se a URL não
for fornecida ou falhar. A leitura roda em thread própria, escrevendo o
frame mais recente num buffer protegido por lock — quem consome nunca
bloqueia esperando o próximo frame de rede.
"""
import threading
import time
from enum import Enum
from typing import Callable, Optional

import cv2
import numpy as np


class EstadoConexao(Enum):
    CONECTADO = "conectado"
    RECONECTANDO = "reconectando"
    DESCONECTADO = "desconectado"


class CameraSource:
    """Fonte de frames de vídeo com reconexão automática em thread própria.

    O construtor não abre nada (sem efeito colateral) — quem conecta de
    fato é `iniciar()`/`__enter__`, o que permite instanciar em testes sem
    hardware. `fabrica_captura` é injetável para testes com um
    `cv2.VideoCapture` falso.
    """

    def __init__(
        self,
        url_mjpeg: Optional[str] = None,
        indice_fallback: int = 0,
        timeout_conexao_s: float = 3.0,
        intervalo_reconexao_s: float = 2.0,
        falhas_consecutivas_para_reconectar: int = 5,
        fabrica_captura: Callable[[object], "cv2.VideoCapture"] = cv2.VideoCapture,
    ) -> None:
        self.url_mjpeg = url_mjpeg
        self.indice_fallback = indice_fallback
        self.timeout_conexao_s = timeout_conexao_s
        self.intervalo_reconexao_s = intervalo_reconexao_s
        self.falhas_consecutivas_para_reconectar = falhas_consecutivas_para_reconectar
        self._fabrica_captura = fabrica_captura

        self._captura: Optional["cv2.VideoCapture"] = None
        self._fonte_ativa: Optional[object] = None
        self._thread: Optional[threading.Thread] = None
        self._deve_parar = threading.Event()

        self._lock = threading.Lock()
        self._frame_mais_recente: Optional[np.ndarray] = None
        self._estado = EstadoConexao.DESCONECTADO

    def _abrir_fonte(self, fonte: object) -> Optional["cv2.VideoCapture"]:
        captura = self._fabrica_captura(fonte)
        try:
            captura.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, self.timeout_conexao_s * 1000)
            captura.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, self.timeout_conexao_s * 1000)
        except Exception:
            pass  # best-effort: nem toda build/backend do OpenCV suporta essas props
        if not captura.isOpened():
            captura.release()
            return None
        ok, _ = captura.read()
        if not ok:
            captura.release()
            return None
        return captura

    def iniciar(self) -> "CameraSource":
        if self._thread is not None:
            return self  # idempotente

        captura = None
        fonte_escolhida: object = self.indice_fallback
        if self.url_mjpeg:
            captura = self._abrir_fonte(self.url_mjpeg)
            fonte_escolhida = self.url_mjpeg
        if captura is None:
            captura = self._abrir_fonte(self.indice_fallback)
            fonte_escolhida = self.indice_fallback

        self._captura = captura
        self._fonte_ativa = fonte_escolhida
        self._estado = EstadoConexao.CONECTADO if captura is not None else EstadoConexao.DESCONECTADO

        self._deve_parar.clear()
        self._thread = threading.Thread(target=self._loop_leitura, daemon=True)
        self._thread.start()
        return self

    def __enter__(self) -> "CameraSource":
        return self.iniciar()

    def __exit__(self, *_exc) -> None:
        self.close()

    def _loop_leitura(self) -> None:
        falhas_seguidas = 0
        proxima_tentativa_reconexao = 0.0

        while not self._deve_parar.is_set():
            if self._captura is None:
                if time.monotonic() < proxima_tentativa_reconexao:
                    time.sleep(0.05)
                    continue
                nova_captura = self._abrir_fonte(self._fonte_ativa)
                if nova_captura is None:
                    proxima_tentativa_reconexao = time.monotonic() + self.intervalo_reconexao_s
                    continue
                self._captura = nova_captura
                falhas_seguidas = 0
                self._estado = EstadoConexao.CONECTADO
                continue

            try:
                ok, frame = self._captura.read()
            except Exception:
                ok, frame = False, None

            if ok and frame is not None:
                falhas_seguidas = 0
                self._estado = EstadoConexao.CONECTADO
                with self._lock:
                    self._frame_mais_recente = frame
                continue

            falhas_seguidas += 1
            if falhas_seguidas >= self.falhas_consecutivas_para_reconectar:
                self._estado = EstadoConexao.RECONECTANDO
                try:
                    self._captura.release()
                except Exception:
                    pass
                self._captura = None
                proxima_tentativa_reconexao = time.monotonic() + self.intervalo_reconexao_s

    def obter_frame_mais_recente(self) -> Optional[np.ndarray]:
        with self._lock:
            return self._frame_mais_recente

    def obter_estado(self) -> EstadoConexao:
        return self._estado

    def close(self) -> None:
        self._deve_parar.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._captura is not None:
            try:
                self._captura.release()
            except Exception:
                pass
            self._captura = None
        self._estado = EstadoConexao.DESCONECTADO
