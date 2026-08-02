"""Wrapper do MediaPipe Hands: converte landmarks da mão em `EstadoEntrada`.

As funções de cálculo puro (posição, pinça, abertura, suavização, dead zone,
detecção de punho sustentado) não dependem do MediaPipe — recebem apenas
listas de 21 tuplas (x, y, z) normalizadas, o que as torna testáveis com
landmarks sintéticos, sem câmera nem modelo carregado.
"""
import time
from dataclasses import dataclass
from math import sqrt
from pathlib import Path
from typing import Callable, Optional, Sequence

from render.viewer import EstadoEntrada

Landmarks = Sequence[tuple[float, float, float]]

# Índices dos landmarks do MediaPipe Hands (ordem oficial, 21 pontos).
PULSO = 0
POLEGAR_PONTA = 4
INDICADOR_MCP, INDICADOR_PONTA = 5, 8
MEDIO_MCP, MEDIO_PONTA = 9, 12
ANELAR_MCP, ANELAR_PONTA = 13, 16
MINDINHO_MCP, MINDINHO_PONTA = 17, 20

PONTOS_PALMA = (PULSO, INDICADOR_MCP, MEDIO_MCP, ANELAR_MCP, MINDINHO_MCP)
PONTAS_DEDOS = (POLEGAR_PONTA, INDICADOR_PONTA, MEDIO_PONTA, ANELAR_PONTA, MINDINHO_PONTA)


def _distancia_xy(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def tamanho_mao(landmarks: Landmarks) -> float:
    """Distância pulso -> base do dedo médio; referência para normalizar as
    demais medidas e torná-las invariantes à distância da mão até a câmera."""
    return max(_distancia_xy(landmarks[PULSO], landmarks[MEDIO_MCP]), 1e-6)


def posicao_mao(landmarks: Landmarks) -> tuple[float, float]:
    """Centroide de pulso + 4 bases dos dedos — mais estável que só o pulso,
    que oscila bastante com a rotação do punho."""
    xs = sum(landmarks[i][0] for i in PONTOS_PALMA)
    ys = sum(landmarks[i][1] for i in PONTOS_PALMA)
    n = len(PONTOS_PALMA)
    return (xs / n, ys / n)


def distancia_pinca(landmarks: Landmarks) -> float:
    """Distância polegar-ponta <-> indicador-ponta, normalizada por `tamanho_mao`."""
    return _distancia_xy(landmarks[POLEGAR_PONTA], landmarks[INDICADOR_PONTA]) / tamanho_mao(landmarks)


def abertura_mao(landmarks: Landmarks) -> float:
    """Média das distâncias do centroide da palma a cada ponta de dedo,
    normalizada por `tamanho_mao`."""
    cx, cy = posicao_mao(landmarks)
    tamanho = tamanho_mao(landmarks)
    distancias = [sqrt((landmarks[i][0] - cx) ** 2 + (landmarks[i][1] - cy) ** 2) for i in PONTAS_DEDOS]
    return (sum(distancias) / len(distancias)) / tamanho


def mao_fechada(landmarks: Landmarks, limiar: float) -> bool:
    return abertura_mao(landmarks) < limiar


def suavizar_ema(bruto: float, suavizado_anterior: Optional[float], alfa: float) -> float:
    """Média móvel exponencial. Retorna o valor bruto na primeira amostra
    (bootstrap), evitando um salto inicial artificial a partir de 0."""
    if suavizado_anterior is None:
        return bruto
    return alfa * bruto + (1 - alfa) * suavizado_anterior


def aplicar_dead_zone(delta: float, limiar: float) -> float:
    return 0.0 if abs(delta) < limiar else delta


class DetectorPunhoSustentado:
    """Detecta punho fechado sustentado por `duracao_hold` segundos e
    dispara exatamente uma vez por hold — é preciso reabrir a mão antes de
    disparar de novo. Relógio injetável para permitir teste determinístico
    sem `time.sleep` real."""

    def __init__(self, duracao_hold: float = 1.0, relogio: Callable[[], float] = time.monotonic) -> None:
        self.duracao_hold = duracao_hold
        self._relogio = relogio
        self._inicio_fechado: Optional[float] = None
        self._disparado = False

    def atualizar(self, mao_esta_fechada: bool) -> bool:
        if not mao_esta_fechada:
            self._inicio_fechado = None
            self._disparado = False
            return False

        agora = self._relogio()
        if self._inicio_fechado is None:
            self._inicio_fechado = agora

        if not self._disparado and (agora - self._inicio_fechado) >= self.duracao_hold:
            self._disparado = True
            return True
        return False


@dataclass
class _EstadoMaoSuavizado:
    posicao_x: Optional[float] = None
    posicao_y: Optional[float] = None
    pinca: Optional[float] = None
    abertura: Optional[float] = None


class FonteEntradaGestos:
    """Implementa `render.viewer.FonteEntrada` traduzindo gestos de mão
    (capturados via MediaPipe HandLandmarker) em `EstadoEntrada`. Mesma
    interface de `FonteEntradaMouseTeclado` — o `Viewer` não muda."""

    def __init__(self, camera=None, config=None, mostrar_janela_debug: bool = True) -> None:
        # Import tardio: mediapipe/cv2/pygame só são necessários para uso real
        # (não para importar/testar as funções puras acima).
        import pygame

        from config import Config
        from vision.camera import CameraSource

        self._pygame = pygame
        self.config = config or Config()
        self.mostrar_janela_debug = mostrar_janela_debug

        modelo = Path(self.config.HAND_LANDMARKER_MODEL_PATH)
        if not modelo.exists():
            raise RuntimeError(
                f"Modelo do MediaPipe não encontrado em '{modelo}'. Baixe o "
                "hand_landmarker.task em https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker "
                f"e salve nesse caminho (ver README)."
            )

        self.camera = camera or CameraSource(
            url_mjpeg=self.config.CAMERA_URL,
            indice_fallback=self.config.CAMERA_FALLBACK_INDICE,
            timeout_conexao_s=self.config.CAMERA_TIMEOUT_CONEXAO_S,
            intervalo_reconexao_s=self.config.CAMERA_INTERVALO_RECONEXAO_S,
            falhas_consecutivas_para_reconectar=self.config.CAMERA_FALHAS_CONSECUTIVAS_PARA_RECONECTAR,
        )
        self.camera.iniciar()

        self._detector_punho = DetectorPunhoSustentado(duracao_hold=self.config.DURACAO_HOLD_RESET_S)
        self._estado_suavizado = _EstadoMaoSuavizado()
        self._ultimo_timestamp_ms = 0

        self._landmarker = self._criar_hand_landmarker()
        self._janela_debug_disponivel = mostrar_janela_debug

    def _criar_hand_landmarker(self):
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision

        options = mp_vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=self.config.HAND_LANDMARKER_MODEL_PATH),
            running_mode=mp_vision.RunningMode.VIDEO,
            num_hands=self.config.HAND_MAX_NUM_HANDS,
            min_hand_detection_confidence=self.config.HAND_MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=self.config.HAND_MIN_TRACKING_CONFIDENCE,
        )
        return mp_vision.HandLandmarker.create_from_options(options)

    def _detectar(self, frame_bgr):
        import cv2
        import mediapipe as mp

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        imagem_mp = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        timestamp_ms = int(time.monotonic() * 1000)
        if timestamp_ms <= self._ultimo_timestamp_ms:
            timestamp_ms = self._ultimo_timestamp_ms + 1
        self._ultimo_timestamp_ms = timestamp_ms

        return self._landmarker.detect_for_video(imagem_mp, timestamp_ms)

    def _processar_landmarks(self, landmarks: Landmarks) -> EstadoEntrada:
        cfg = self.config
        estado = EstadoEntrada()

        pos_x, pos_y = posicao_mao(landmarks)
        pinca_bruta = distancia_pinca(landmarks)
        abertura_bruta = abertura_mao(landmarks)

        es = self._estado_suavizado
        pos_x_suave = suavizar_ema(pos_x, es.posicao_x, cfg.ALFA_SUAVIZACAO)
        pos_y_suave = suavizar_ema(pos_y, es.posicao_y, cfg.ALFA_SUAVIZACAO)
        pinca_suave = suavizar_ema(pinca_bruta, es.pinca, cfg.ALFA_SUAVIZACAO)
        abertura_suave = suavizar_ema(abertura_bruta, es.abertura, cfg.ALFA_SUAVIZACAO)

        if es.posicao_x is not None:
            delta_x = (pos_x_suave - es.posicao_x) * cfg.SENSIBILIDADE_ROTACAO_GESTO
            delta_y = (pos_y_suave - es.posicao_y) * cfg.SENSIBILIDADE_ROTACAO_GESTO
            delta_zoom = (pinca_suave - es.pinca) * cfg.SENSIBILIDADE_ZOOM_GESTO
            delta_aresta = (abertura_suave - es.abertura) * cfg.SENSIBILIDADE_ARESTA_GESTO

            estado.delta_rotacao_y = aplicar_dead_zone(delta_x, cfg.DEAD_ZONE_ROTACAO)
            estado.delta_rotacao_x = aplicar_dead_zone(delta_y, cfg.DEAD_ZONE_ROTACAO)
            estado.delta_zoom = aplicar_dead_zone(delta_zoom, cfg.DEAD_ZONE_ZOOM)
            estado.delta_aresta = aplicar_dead_zone(delta_aresta, cfg.DEAD_ZONE_ARESTA)

        self._estado_suavizado = _EstadoMaoSuavizado(pos_x_suave, pos_y_suave, pinca_suave, abertura_suave)

        fechada = mao_fechada(landmarks, cfg.LIMIAR_MAO_FECHADA)
        estado.resetar = self._detector_punho.atualizar(fechada)

        return estado

    def _desenhar_debug(self, frame, landmarks: Optional[Landmarks]) -> None:
        if not self._janela_debug_disponivel or frame is None:
            return
        try:
            import cv2

            imagem = frame.copy()
            if landmarks is not None:
                self._desenhar_landmarks_na_imagem(imagem, landmarks)
            cv2.putText(
                imagem,
                f"camera: {self.camera.obter_estado().value}",
                (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )
            cv2.imshow("Jarvis Tetraedro - debug da mao", imagem)
            cv2.waitKey(1)
        except Exception as erro:
            print(f"Janela de debug desativada ({erro}).")
            self._janela_debug_disponivel = False

    @staticmethod
    def _desenhar_landmarks_na_imagem(imagem_bgr, landmarks: Landmarks) -> None:
        # mediapipe 1.0.0 não expõe mais `mediapipe.solutions` (API legada removida);
        # drawing_utils/drawing_styles/HandLandmarksConnections agora vivem em
        # mediapipe.tasks.python.vision.
        from mediapipe.tasks.python import vision as mp_vision
        from mediapipe.tasks.python.components.containers.landmark import NormalizedLandmark

        landmark_list = [NormalizedLandmark(x=x, y=y, z=z) for x, y, z in landmarks]
        mp_vision.drawing_utils.draw_landmarks(
            imagem_bgr,
            landmark_list,
            mp_vision.HandLandmarksConnections.HAND_CONNECTIONS,
            mp_vision.drawing_styles.get_default_hand_landmarks_style(),
            mp_vision.drawing_styles.get_default_hand_connections_style(),
        )

    def capturar(self) -> EstadoEntrada:
        import cv2

        estado = EstadoEntrada()

        for evento in self._pygame.event.get():
            if evento.type == self._pygame.QUIT:
                estado.sair = True
            elif evento.type == self._pygame.KEYDOWN and evento.key == self._pygame.K_ESCAPE:
                estado.sair = True
            elif evento.type == self._pygame.VIDEORESIZE:
                estado.redimensionado = (evento.w, evento.h)

        frame = self.camera.obter_frame_mais_recente()
        landmarks: Optional[Landmarks] = None

        if frame is not None:
            if not self.config.INVERTER_ESPELHO_CAMERA:
                frame = cv2.flip(frame, 1)
            resultado = self._detectar(frame)
            if resultado.hand_landmarks:
                landmarks = [(lm.x, lm.y, lm.z) for lm in resultado.hand_landmarks[0]]
                gestos = self._processar_landmarks(landmarks)
                estado.delta_rotacao_x = gestos.delta_rotacao_x
                estado.delta_rotacao_y = gestos.delta_rotacao_y
                estado.delta_zoom = gestos.delta_zoom
                estado.delta_aresta = gestos.delta_aresta
                estado.resetar = gestos.resetar
            else:
                self._estado_suavizado = _EstadoMaoSuavizado()

        # Entrega o frame ao viewer para a composição AR (o sólido é desenhado
        # por cima dele). Já vem espelhado, para o usuário se ver como num
        # espelho e o gesto ir para o mesmo lado que a mão.
        estado.frame_camera = frame
        estado.estado_camera = self.camera.obter_estado().value

        self._desenhar_debug(frame, landmarks)
        return estado

    def close(self) -> None:
        if self._landmarker is not None:
            self._landmarker.close()
        self.camera.close()
        try:
            import cv2

            cv2.destroyAllWindows()
        except Exception:
            pass
