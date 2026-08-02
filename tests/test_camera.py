import time

import numpy as np
import pytest

from vision.camera import CameraSource, EstadoConexao


class _CapturaFalsa:
    """Substitui cv2.VideoCapture em teste: `leituras` é a sequência de
    sucesso/falha que `.read()` devolve; após esgotar, repete a última."""

    def __init__(self, leituras: list[bool], abre: bool = True) -> None:
        self._leituras = leituras
        self._posicao = 0
        self._abre = abre
        self.liberada = False

    def isOpened(self) -> bool:
        return self._abre

    def read(self):
        if not self._leituras:
            return False, None
        indice = min(self._posicao, len(self._leituras) - 1)
        ok = self._leituras[indice]
        self._posicao += 1
        if ok:
            return True, np.zeros((2, 2, 3), dtype=np.uint8)
        return False, None

    def set(self, *_args, **_kwargs) -> None:
        pass

    def release(self) -> None:
        self.liberada = True


class _FabricaSequencial:
    """Devolve uma `_CapturaFalsa` diferente a cada chamada, na ordem dada
    (repete a última depois de esgotar) — simula reconexão trocando de
    instância de captura como o `cv2.VideoCapture` real faria."""

    def __init__(self, capturas: list[_CapturaFalsa]) -> None:
        self._capturas = capturas
        self._indice = 0

    def __call__(self, _fonte) -> _CapturaFalsa:
        captura = self._capturas[min(self._indice, len(self._capturas) - 1)]
        self._indice += 1
        return captura


def _esperar_ate(condicao, timeout: float = 2.0, intervalo: float = 0.02) -> bool:
    inicio = time.monotonic()
    while time.monotonic() - inicio < timeout:
        if condicao():
            return True
        time.sleep(intervalo)
    return False


class TestCameraSourceConexaoInicial:
    def test_conecta_com_sucesso(self):
        fabrica = _FabricaSequencial([_CapturaFalsa(leituras=[True] * 50)])
        camera = CameraSource(fabrica_captura=fabrica, indice_fallback=0)
        camera.iniciar()
        try:
            assert camera.obter_estado() == EstadoConexao.CONECTADO
            assert _esperar_ate(lambda: camera.obter_frame_mais_recente() is not None)
        finally:
            camera.close()

    def test_falha_ao_abrir_nao_lanca_excecao(self):
        fabrica = _FabricaSequencial([_CapturaFalsa(leituras=[], abre=False)])
        camera = CameraSource(fabrica_captura=fabrica, indice_fallback=0, intervalo_reconexao_s=0.05)
        camera.iniciar()
        try:
            assert camera.obter_estado() == EstadoConexao.DESCONECTADO
            assert camera.obter_frame_mais_recente() is None
        finally:
            camera.close()  # não deve travar (thread precisa parar mesmo sem conexão)

    def test_url_mjpeg_com_falha_cai_para_fallback(self):
        captura_mjpeg_falha = _CapturaFalsa(leituras=[], abre=False)
        captura_fallback_ok = _CapturaFalsa(leituras=[True] * 20)
        fabrica = _FabricaSequencial([captura_mjpeg_falha, captura_fallback_ok])
        camera = CameraSource(url_mjpeg="http://invalido/video", fabrica_captura=fabrica, indice_fallback=0)
        camera.iniciar()
        try:
            assert camera.obter_estado() == EstadoConexao.CONECTADO
        finally:
            camera.close()


class TestCameraSourceReconexao:
    def test_reconecta_apos_falhas_consecutivas(self):
        captura_inicial = _CapturaFalsa(leituras=[True] * 3 + [False] * 20)
        captura_recuperada = _CapturaFalsa(leituras=[True] * 50)
        fabrica = _FabricaSequencial([captura_inicial, captura_recuperada])
        camera = CameraSource(
            fabrica_captura=fabrica,
            indice_fallback=0,
            intervalo_reconexao_s=0.02,
            falhas_consecutivas_para_reconectar=3,
        )
        camera.iniciar()
        try:
            assert _esperar_ate(lambda: camera.obter_estado() == EstadoConexao.CONECTADO and captura_inicial.liberada)
        finally:
            camera.close()

    def test_nao_crasha_com_falha_permanente(self):
        captura_que_sempre_falha = _CapturaFalsa(leituras=[True] * 2 + [False] * 100)
        fabrica = _FabricaSequencial([captura_que_sempre_falha])
        camera = CameraSource(
            fabrica_captura=fabrica,
            indice_fallback=0,
            intervalo_reconexao_s=0.02,
            falhas_consecutivas_para_reconectar=2,
        )
        camera.iniciar()
        try:
            assert _esperar_ate(lambda: camera.obter_estado() in (EstadoConexao.RECONECTANDO, EstadoConexao.DESCONECTADO))
        finally:
            camera.close()  # thread precisa parar de forma limpa mesmo em falha permanente


class TestCameraSourceCloseIdempotente:
    def test_close_pode_ser_chamado_duas_vezes(self):
        fabrica = _FabricaSequencial([_CapturaFalsa(leituras=[True] * 10)])
        camera = CameraSource(fabrica_captura=fabrica, indice_fallback=0)
        camera.iniciar()
        camera.close()
        camera.close()  # não deve lançar
        assert camera.obter_estado() == EstadoConexao.DESCONECTADO

    def test_funciona_como_context_manager(self):
        fabrica = _FabricaSequencial([_CapturaFalsa(leituras=[True] * 10)])
        with CameraSource(fabrica_captura=fabrica, indice_fallback=0) as camera:
            assert camera.obter_estado() == EstadoConexao.CONECTADO
        assert camera.obter_estado() == EstadoConexao.DESCONECTADO
