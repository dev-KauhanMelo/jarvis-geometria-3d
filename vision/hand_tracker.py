"""Wrapper do MediaPipe Hands: mede o que a mão está fazendo.

As funções de cálculo puro (posição, ponto da pinça, abertura, curvatura dos
dedos, orientação da palma) não dependem do MediaPipe — recebem apenas listas
de 21 tuplas (x, y, z) normalizadas, o que as torna testáveis com landmarks
sintéticos, sem câmera nem modelo carregado.

Este módulo só produz FATOS sobre a mão. Traduzi-los em movimento do sólido é
tarefa de `interaction/gestos.py`. A Etapa 2 misturava as duas coisas aqui e
convertia qualquer deslocamento da mão em rotação; quando a máquina de gestos
da Etapa 3 entrou, as duas passaram a disputar o mesmo objeto. O controle
contínuo foi removido — ver o comentário em `FonteEntradaGestos.capturar`.
"""
from math import sqrt
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Sequence

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


def ponto_pinca(landmarks: Landmarks) -> tuple[float, float]:
    """Onde os dedos se encontram: meio do caminho entre as pontas do polegar
    e do indicador. É AQUI que o cursor deve ficar.

    Antes o cursor usava `posicao_mao` (centroide da palma) e o erro era
    grosseiro: medindo a foto "apontando", a palma cai em y=0,55 e a ponta do
    indicador em y=0,20 — uns 270 px de distância numa janela de 768. O
    usuário mira com o dedo e o programa lia a palma, então nada do que ele
    apontava era o que ele pegava.
    """
    a, b = landmarks[POLEGAR_PONTA], landmarks[INDICADOR_PONTA]
    return ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)


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


def mao_em_garra(abertura: float, limiar: float) -> bool:
    """Mão inteira fechando sobre algo — o gesto de apertar um balão.

    Substituiu a antiga `mao_em_concha`, que exigia uma POSE estática com os
    quatro dedos semi-dobrados dentro de uma faixa estreita. Duas coisas
    estavam erradas nela: o usuário descreveu um movimento de apertar
    ("poing, poing"), não uma pose; e a faixa que aceitava a concha também
    aceitava a mão relaxada, então o gesto disparava sozinho.

    `abertura_mao` separa bem os casos que medi com o MediaPipe: punho 0,243,
    dedo apontando 0,434, "vitória" 0,582. Um limiar em ~0,38 pega a mão
    fechando e rejeita qualquer mão aberta.
    """
    return abertura < limiar


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
            elif evento.type == self._pygame.KEYDOWN:
                if evento.key == self._pygame.K_ESCAPE:
                    estado.sair = True
                elif evento.key == self._pygame.K_r:
                    # Único jeito de resetar. Antes o reset era um gesto
                    # (punho fechado por 1 s) e isso ficou insustentável: no
                    # modelo novo o punho fechado é justamente como se SEGURA
                    # o sólido, então segurá-lo por um segundo o mandava de
                    # volta para a origem sozinho.
                    estado.resetar = True
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
        self._ultimo_frame_id = snapshot.frame_id

        # Só fatos sobre as mãos daqui para baixo. O controle contínuo da
        # Etapa 2 (que traduzia qualquer deslocamento da mão em rotação, e
        # qualquer variação de abertura em tamanho de aresta) foi desligado:
        # ele rodava EM PARALELO com a máquina de gestos da Etapa 3 e as duas
        # brigavam pelo mesmo sólido. Era a maior parte do "ele fica se
        # movendo loucamente" — o objeto reagia à mão mesmo sem ninguém tê-lo
        # agarrado, e não havia como descansar a mão no quadro.
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
