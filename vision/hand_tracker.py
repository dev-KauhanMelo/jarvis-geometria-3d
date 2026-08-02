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
from typing import TYPE_CHECKING, Callable, Optional, Sequence

if TYPE_CHECKING:  # pragma: no cover
    from render.viewer import EstadoEntrada

# `EstadoEntrada` é importado tardiamente (dentro dos métodos que o usam) para
# quebrar o ciclo render.viewer -> interaction.gestos -> vision.hand_tracker.
# As funções puras deste módulo não dependem dele, e é justamente elas que a
# camada de interação consome.

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


# ---------------------------------------------------------------------------
# Curvatura dos dedos e orientação da palma (Etapa 3)
# ---------------------------------------------------------------------------
# Cada dedo como cadeia MCP -> PIP -> DIP -> ponta.
CADEIAS_DEDOS: dict[str, tuple[int, int, int, int]] = {
    "indicador": (5, 6, 7, 8),
    "medio": (9, 10, 11, 12),
    "anelar": (13, 14, 15, 16),
    "mindinho": (17, 18, 19, 20),
}


def _distancia_3d(a, b) -> float:
    return sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)


def extensao_dedo(landmarks: Landmarks, cadeia: tuple[int, int, int, int]) -> float:
    """Razão corda/arco do dedo: |ponta - MCP| dividido pelo comprimento total.

    Vale ~1,0 com o dedo reto e cai para ~0,25-0,45 totalmente dobrado. É
    adimensional e não depende de normalizar pelo tamanho da mão.

    Usar de preferência com `hand_world_landmarks` (3D métrico), onde fica
    imune ao encurtamento por perspectiva. Medi nas fotos reais: punho
    [0,31 0,25 0,33 0,46], apontando [0,96 0,35 0,41 0,42], vitória
    [0,94 0,99 0,62 0,55] — em 2D os mesmos gestos separam bem menos.
    """
    mcp, pip, dip, ponta = (landmarks[i] for i in cadeia)
    arco = _distancia_3d(mcp, pip) + _distancia_3d(pip, dip) + _distancia_3d(dip, ponta)
    if arco < 1e-9:
        return 0.0
    return _distancia_3d(mcp, ponta) / arco


def extensoes_dedos(landmarks: Landmarks) -> tuple[float, float, float, float]:
    """(indicador, médio, anelar, mindinho), na ordem de CADEIAS_DEDOS."""
    return tuple(extensao_dedo(landmarks, cadeia) for cadeia in CADEIAS_DEDOS.values())


def mao_em_concha(
    extensoes: Sequence[float],
    pinca: float,
    minimo: float = 0.50,
    maximo: float = 0.88,
    pinca_minima: float = 0.50,
) -> bool:
    """Mão em concha/garra — "carregando um poder na mão".

    Exige que os QUATRO dedos estejam semi-dobrados: nenhum esticado
    (`max <= maximo`) e nenhum totalmente fechado (`min >= minimo`), com a
    pinça aberta (senão o gesto de agarrar dispararia junto).

    Recebe as extensões já calculadas em vez dos landmarks de propósito: fica
    trivial de testar e permite alimentar com os valores já suavizados do
    render, sem recalcular.

    `abertura_mao` não serve para isso: uma concha vista de topo tem as pontas
    quase tão próximas do centroide quanto um punho.
    """
    if pinca < pinca_minima:
        return False
    return min(extensoes) >= minimo and max(extensoes) <= maximo


def matriz_orientacao_palma(world_landmarks: Landmarks, destra: bool = True):
    """Base ortonormal da palma a partir dos world landmarks.

    Usa pulso e as juntas MCP — os cinco pontos mais estáveis da mão — e nunca
    as pontas dos dedos, cujo ruído viraria tremor na rotação do objeto.

    Devolve a matriz já convertida para os eixos do OpenGL. Levanta ValueError
    se a mão estiver degenerada (pontos colineares).
    """
    from geometry.transformacoes import base_destra, subtrair

    pulso = world_landmarks[PULSO]
    indicador = world_landmarks[INDICADOR_MCP]
    mindinho = world_landmarks[MINDINHO_MCP]

    ao_longo = subtrair(indicador, pulso)   # através da palma
    para_cima = subtrair(mindinho, pulso)

    # A ordem do produto vetorial depende da mão, senão a normal da palma
    # aponta para dentro numa delas e o giro sai invertido.
    normal = (
        _produto_vetorial(ao_longo, para_cima) if destra
        else _produto_vetorial(para_cima, ao_longo)
    )
    matriz = base_destra(normal, subtrair(_media(indicador, mindinho), pulso))
    return converter_orientacao_mp_para_gl(matriz)


def _produto_vetorial(a, b):
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _media(a, b):
    return ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2, (a[2] + b[2]) / 2)


def converter_orientacao_mp_para_gl(matriz):
    """Troca de eixos de imagem (y para baixo, z afastando) para eixos do
    OpenGL (y para cima, z em direção ao observador), via S·M·S com
    S = diag(1,-1,-1).

    Como det(S) = +1 e S é sua própria inversa, a similaridade preserva a
    rotação. Sem isso o giro sai invertido em dois eixos — o sintoma é
    "mexo a mão para cima e ele gira para baixo".
    """
    s = ((1.0, 0.0, 0.0), (0.0, -1.0, 0.0), (0.0, 0.0, -1.0))
    from geometry.transformacoes import multiplicar_matrizes

    return multiplicar_matrizes(multiplicar_matrizes(s, matriz), s)


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

    def __init__(
        self,
        camera=None,
        config=None,
        mostrar_janela_debug: bool = True,
        mostrar_diagnostico: bool = False,
    ) -> None:
        # Import tardio: mediapipe/cv2/pygame só são necessários para uso real
        # (não para importar/testar as funções puras acima).
        import pygame

        from config import Config
        from vision.camera import CameraSource

        self._pygame = pygame
        self.config = config or Config()
        self.mostrar_janela_debug = mostrar_janela_debug
        self.mostrar_diagnostico = mostrar_diagnostico

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

        # A detecção roda em thread própria: o loop de render não espera por
        # ela. Construir aqui (e não dentro da thread) mantém o contrato de
        # main.py, que conta com a exceção de "modelo não encontrado" para
        # cair no fallback de mouse/teclado.
        from vision.rastreador import RastreadorMaos

        self._rastreador = RastreadorMaos(self.camera, self.config).iniciar()
        self._ultimo_frame_id = 0
        self._janela_debug_disponivel = mostrar_janela_debug

    # A criação do detector e a chamada de detecção migraram para
    # `vision/rastreador.py`, que as executa na thread dedicada.

    def _processar_landmarks(self, landmarks: Landmarks) -> "EstadoEntrada":
        cfg = self.config
        from render.viewer import EstadoEntrada

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

    def capturar(self) -> "EstadoEntrada":
        import cv2

        from render.viewer import EstadoEntrada

        estado = EstadoEntrada()

        for evento in self._pygame.event.get():
            if evento.type == self._pygame.QUIT:
                estado.sair = True
            elif evento.type == self._pygame.KEYDOWN and evento.key == self._pygame.K_ESCAPE:
                estado.sair = True
            elif evento.type == self._pygame.VIDEORESIZE:
                estado.redimensionado = (evento.w, evento.h)

        # Frame para o fundo AR: vem direto da câmera (mais fresco) e não do
        # snapshot da detecção (mais velho). O flip é o mesmo que a thread de
        # detecção aplica, para que os landmarks e o vídeo concordem.
        frame = self.camera.obter_frame_mais_recente()
        if frame is not None and not self.config.INVERTER_ESPELHO_CAMERA:
            frame = cv2.flip(frame, 1)

        snapshot = self._rastreador.obter_snapshot()
        landmarks: Optional[Landmarks] = None

        if snapshot.maos:
            landmarks = list(snapshot.maos[0].landmarks)
            # Os gestos da Etapa 2 acumulam deltas, então só podem ser
            # processados uma vez por DETECÇÃO — a 60 fps de render sobre 19 fps
            # de detecção, reprocessar o mesmo snapshot triplicaria a rotação.
            if snapshot.frame_id != self._ultimo_frame_id:
                self._ultimo_frame_id = snapshot.frame_id
                gestos = self._processar_landmarks(landmarks)
                estado.delta_rotacao_x = gestos.delta_rotacao_x
                estado.delta_rotacao_y = gestos.delta_rotacao_y
                estado.delta_zoom = gestos.delta_zoom
                estado.delta_aresta = gestos.delta_aresta
                estado.resetar = gestos.resetar
        else:
            self._estado_suavizado = _EstadoMaoSuavizado()

        estado.maos = snapshot.maos

        # Entrega o frame ao viewer para a composição AR (o sólido é desenhado
        # por cima dele). Já vem espelhado, para o usuário se ver como num
        # espelho e o gesto ir para o mesmo lado que a mão.
        estado.frame_camera = frame
        estado.estado_camera = self.camera.obter_estado().value

        if self.mostrar_diagnostico:
            m = self._rastreador.obter_metricas()
            estado.diagnostico = (
                f"detecção: {m.fps_deteccao:.0f} fps ({m.ms_por_deteccao:.0f} ms)"
                f" | mãos: {len(snapshot.maos)}",
            )

        self._desenhar_debug(frame, landmarks)
        return estado

    def close(self) -> None:
        # O rastreador para a thread antes de fechar o detector; a câmera vem
        # por último, porque a thread de detecção lê frames dela.
        self._rastreador.close()
        self.camera.close()
        try:
            import cv2

            cv2.destroyAllWindows()
        except Exception:
            pass
