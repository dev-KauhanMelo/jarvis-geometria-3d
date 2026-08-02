"""Testes da thread de detecção e da associação temporal de mãos.

Rodam sem MediaPipe, sem câmera e sem GPU: o detector e a câmera são falsos,
injetados pelo construtor.
"""
import threading
import time

import pytest

from vision.rastreador import (
    MaoDetectada,
    RastreadorMaos,
    SnapshotMaos,
    associar_maos,
    centroide_palma,
)


def _landmarks(cx: float = 0.5, cy: float = 0.5):
    """21 pontos com a palma centrada em (cx, cy)."""
    return [(cx, cy, 0.0) for _ in range(21)]


class _Categoria:
    def __init__(self, nome, score):
        self.category_name = nome
        self.score = score


class _Ponto:
    def __init__(self, x, y, z):
        self.x, self.y, self.z = x, y, z


class _ResultadoFalso:
    """Imita HandLandmarkerResult."""

    def __init__(self, maos):
        self.hand_landmarks = [[_Ponto(*p) for p in lm] for lm, _ in maos]
        self.hand_world_landmarks = [[_Ponto(*p) for p in lm] for lm, _ in maos]
        self.handedness = [[_Categoria("Right" if destra else "Left", 0.99)] for _, destra in maos]


class _DetectorFalso:
    def __init__(self, resultados):
        self._resultados = list(resultados)
        self.chamadas = 0
        self.fechado = False
        self.timestamps = []

    def detect_for_video(self, imagem, timestamp_ms):
        self.timestamps.append(timestamp_ms)
        resultado = self._resultados[min(self.chamadas, len(self._resultados) - 1)]
        self.chamadas += 1
        return resultado

    def close(self):
        self.fechado = True


class _DetectorQueExplode:
    def __init__(self):
        self.chamadas = 0

    def detect_for_video(self, imagem, timestamp_ms):
        self.chamadas += 1
        raise RuntimeError("falha simulada de detecção")

    def close(self):
        pass


class _CameraFalsa:
    """Devolve frames novos (objetos distintos) a cada chamada."""

    def __init__(self, frames=None, repetir_mesmo_objeto=False):
        import numpy as np

        self._np = np
        self._repetir = repetir_mesmo_objeto
        self._frame_fixo = np.zeros((48, 64, 3), dtype=np.uint8)
        self.chamadas = 0

    def obter_frame_mais_recente(self):
        import numpy as np

        self.chamadas += 1
        if self._repetir:
            return self._frame_fixo
        return np.zeros((48, 64, 3), dtype=np.uint8)


class _ConfigFalso:
    HAND_LANDMARKER_MODEL_PATH = "models/hand_landmarker.task"
    HAND_MAX_NUM_HANDS = 2
    HAND_MIN_DETECTION_CONFIDENCE = 0.5
    HAND_MIN_TRACKING_CONFIDENCE = 0.5
    INVERTER_ESPELHO_CAMERA = True  # evita depender do cv2.flip nos testes


def _esperar(condicao, timeout=3.0):
    limite = time.monotonic() + timeout
    while time.monotonic() < limite:
        if condicao():
            return True
        time.sleep(0.01)
    return False


class TestAssociarMaos:
    def test_sem_maos_anteriores_atribui_ids_novos(self):
        atribuicoes, proximo = associar_maos([(0.2, 0.5), (0.8, 0.5)], {}, 0.25, 0)
        assert atribuicoes == {0: 0, 1: 1}
        assert proximo == 2

    def test_mao_proxima_mantem_o_id(self):
        atribuicoes, _ = associar_maos([(0.21, 0.51)], {7: (0.2, 0.5)}, 0.25, 8)
        assert atribuicoes == {0: 7}

    def test_mao_longe_demais_ganha_id_novo(self):
        atribuicoes, proximo = associar_maos([(0.9, 0.9)], {7: (0.1, 0.1)}, 0.25, 8)
        assert atribuicoes == {0: 8}
        assert proximo == 9

    def test_troca_de_ordem_nao_troca_as_identidades(self):
        """O ponto do módulo: se os índices trocam entre frames, o id tem de
        seguir a mão, senão um arrasto salta para a outra mão."""
        anteriores = {0: (0.2, 0.5), 1: (0.8, 0.5)}
        # a mão que era índice 1 agora vem primeiro
        atribuicoes, _ = associar_maos([(0.81, 0.5), (0.19, 0.5)], anteriores, 0.25, 2)
        assert atribuicoes == {0: 1, 1: 0}

    def test_duas_maos_nao_recebem_o_mesmo_id(self):
        atribuicoes, _ = associar_maos([(0.5, 0.5), (0.52, 0.5)], {3: (0.51, 0.5)}, 0.25, 4)
        assert len(set(atribuicoes.values())) == 2

    def test_casamento_e_guloso_pelo_mais_proximo(self):
        anteriores = {10: (0.10, 0.5), 20: (0.90, 0.5)}
        atribuicoes, _ = associar_maos([(0.88, 0.5), (0.12, 0.5)], anteriores, 0.5, 30)
        assert atribuicoes == {0: 20, 1: 10}

    def test_sem_maos_devolve_vazio(self):
        atribuicoes, proximo = associar_maos([], {1: (0.5, 0.5)}, 0.25, 5)
        assert atribuicoes == {}
        assert proximo == 5


class TestCentroidePalma:
    def test_usa_pulso_e_mcps(self):
        lm = [(0.0, 0.0, 0.0)] * 21
        for i in (0, 5, 9, 13, 17):
            lm[i] = (0.4, 0.6, 0.0)
        assert centroide_palma(lm) == pytest.approx((0.4, 0.6))

    def test_ignora_as_pontas_dos_dedos(self):
        """Fechar a mão move as pontas mas não a palma — o centroide precisa
        ficar parado, senão a associação temporal perde a mão no gesto."""
        base = _landmarks(0.5, 0.5)
        com_dedos_movidos = list(base)
        for i in (4, 8, 12, 16, 20):
            com_dedos_movidos[i] = (0.9, 0.9, 0.0)
        assert centroide_palma(base) == pytest.approx(centroide_palma(com_dedos_movidos))


class TestSnapshot:
    def test_snapshot_inicial_e_vazio_e_nao_bloqueia(self):
        r = RastreadorMaos(_CameraFalsa(), _ConfigFalso(), fabrica_detector=lambda c: _DetectorFalso([]))
        snapshot = r.obter_snapshot()
        assert isinstance(snapshot, SnapshotMaos)
        assert snapshot.maos == ()
        r.close()

    def test_snapshot_e_imutavel(self):
        mao = MaoDetectada(landmarks=(), world_landmarks=(), destra=True, confianca_lado=1.0)
        with pytest.raises(Exception):
            mao.destra = False  # frozen dataclass
        snapshot = SnapshotMaos()
        with pytest.raises(Exception):
            snapshot.frame_id = 5


class TestLoopDeDeteccao:
    def test_publica_maos_detectadas(self):
        resultado = _ResultadoFalso([(_landmarks(0.3, 0.4), True)])
        detector = _DetectorFalso([resultado])
        r = RastreadorMaos(_CameraFalsa(), _ConfigFalso(), fabrica_detector=lambda c: detector).iniciar()
        try:
            assert _esperar(lambda: len(r.obter_snapshot().maos) == 1)
            mao = r.obter_snapshot().maos[0]
            assert mao.destra is True
            assert mao.confianca_lado == pytest.approx(0.99)
            assert len(mao.landmarks) == 21
            assert len(mao.world_landmarks) == 21
        finally:
            r.close()

    def test_ids_das_maos_sao_estaveis_entre_deteccoes(self):
        resultado = _ResultadoFalso([(_landmarks(0.3, 0.4), True)])
        detector = _DetectorFalso([resultado])
        r = RastreadorMaos(_CameraFalsa(), _ConfigFalso(), fabrica_detector=lambda c: detector).iniciar()
        try:
            assert _esperar(lambda: r.obter_snapshot().frame_id >= 1)
            primeiro = r.obter_snapshot().maos[0].id_mao
            assert _esperar(lambda: r.obter_snapshot().frame_id >= 5)
            assert r.obter_snapshot().maos[0].id_mao == primeiro
        finally:
            r.close()

    def test_frame_id_avanca_a_cada_deteccao(self):
        detector = _DetectorFalso([_ResultadoFalso([])])
        r = RastreadorMaos(_CameraFalsa(), _ConfigFalso(), fabrica_detector=lambda c: detector).iniciar()
        try:
            assert _esperar(lambda: r.obter_snapshot().frame_id >= 3)
        finally:
            r.close()

    def test_timestamps_sao_estritamente_crescentes(self):
        """detect_for_video rejeita timestamp que não avança."""
        detector = _DetectorFalso([_ResultadoFalso([])])
        r = RastreadorMaos(_CameraFalsa(), _ConfigFalso(), fabrica_detector=lambda c: detector).iniciar()
        try:
            assert _esperar(lambda: len(detector.timestamps) >= 5)
            r.close()
            ts = detector.timestamps
            assert all(b > a for a, b in zip(ts, ts[1:]))
        finally:
            r.close()

    def test_frame_repetido_nao_e_redetectado(self):
        """Com a câmera devolvendo o mesmo objeto, detectar de novo gastaria
        51 ms para reprocessar pixels idênticos."""
        detector = _DetectorFalso([_ResultadoFalso([])])
        camera = _CameraFalsa(repetir_mesmo_objeto=True)
        r = RastreadorMaos(camera, _ConfigFalso(), fabrica_detector=lambda c: detector).iniciar()
        try:
            assert _esperar(lambda: r.obter_metricas().frames_repetidos_ignorados > 3)
            assert detector.chamadas == 1  # só o primeiro frame foi detectado
        finally:
            r.close()

    def test_erro_na_deteccao_nao_mata_a_thread(self):
        detector = _DetectorQueExplode()
        r = RastreadorMaos(_CameraFalsa(), _ConfigFalso(), fabrica_detector=lambda c: detector).iniciar()
        try:
            assert _esperar(lambda: r.obter_metricas().erros >= 2)
            assert r._thread.is_alive()
        finally:
            r.close()

    def test_metricas_sao_preenchidas(self):
        detector = _DetectorFalso([_ResultadoFalso([])])
        r = RastreadorMaos(_CameraFalsa(), _ConfigFalso(), fabrica_detector=lambda c: detector).iniciar()
        try:
            assert _esperar(lambda: r.obter_metricas().deteccoes >= 2)
            m = r.obter_metricas()
            assert m.ms_por_deteccao >= 0.0
            assert m.fps_deteccao > 0.0
        finally:
            r.close()


class TestCicloDeVida:
    def test_iniciar_e_idempotente(self):
        r = RastreadorMaos(_CameraFalsa(), _ConfigFalso(), fabrica_detector=lambda c: _DetectorFalso([_ResultadoFalso([])]))
        try:
            r.iniciar()
            thread = r._thread
            r.iniciar()
            assert r._thread is thread
        finally:
            r.close()

    def test_close_encerra_a_thread(self):
        detector = _DetectorFalso([_ResultadoFalso([])])
        r = RastreadorMaos(_CameraFalsa(), _ConfigFalso(), fabrica_detector=lambda c: detector).iniciar()
        thread = r._thread
        r.close()
        assert not thread.is_alive()

    def test_close_fecha_o_detector_depois_de_parar_a_thread(self):
        """Fechar o detector com a thread ainda usando-o seria uso após
        liberação dentro do C++ do MediaPipe."""
        detector = _DetectorFalso([_ResultadoFalso([])])
        r = RastreadorMaos(_CameraFalsa(), _ConfigFalso(), fabrica_detector=lambda c: detector).iniciar()
        _esperar(lambda: detector.chamadas > 0)
        r.close()
        assert detector.fechado is True

    def test_close_e_idempotente(self):
        detector = _DetectorFalso([_ResultadoFalso([])])
        r = RastreadorMaos(_CameraFalsa(), _ConfigFalso(), fabrica_detector=lambda c: detector).iniciar()
        r.close()
        r.close()  # não pode levantar

    def test_funciona_como_gerenciador_de_contexto(self):
        detector = _DetectorFalso([_ResultadoFalso([])])
        with RastreadorMaos(_CameraFalsa(), _ConfigFalso(), fabrica_detector=lambda c: detector) as r:
            assert _esperar(lambda: r.obter_snapshot().frame_id >= 1)
            thread = r._thread
        assert not thread.is_alive()

    def test_nao_deixa_threads_penduradas(self):
        antes = threading.active_count()
        detector = _DetectorFalso([_ResultadoFalso([])])
        r = RastreadorMaos(_CameraFalsa(), _ConfigFalso(), fabrica_detector=lambda c: detector).iniciar()
        _esperar(lambda: r.obter_snapshot().frame_id >= 1)
        r.close()
        assert threading.active_count() == antes
