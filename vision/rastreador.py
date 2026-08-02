"""Detecção de mãos numa thread própria, fora do caminho crítico do render.

Na Etapa 2 a detecção rodava dentro do loop de render, então o programa
inteiro andava na velocidade do MediaPipe. Medi neste ambiente: 25,6 ms por
frame com uma mão e 51,2 ms com duas — ou seja, o app ficava preso a ~20-39
fps, e foi isso que o usuário sentiu como "delay extremamente atrasado".

Aqui a detecção vira produtora: consome frames da câmera no ritmo dela e
publica um `SnapshotMaos` imutável sob lock. O render lê o snapshot mais
recente sem nunca bloquear e roda a 60 fps independentemente.

Não há fila entre as threads, de propósito: um snapshot atrasado é lixo, não
trabalho acumulado a processar. Vale sempre o último.
"""
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Mapping, Optional, Sequence

Vec3 = tuple[float, float, float]


@dataclass(frozen=True)
class MaoDetectada:
    """Uma mão num único frame, no espaço da imagem já espelhada."""

    landmarks: tuple[Vec3, ...]          # 21 pontos normalizados em [0,1]
    world_landmarks: tuple[Vec3, ...]    # 21 pontos métricos, origem no centro da mão
    destra: bool                          # handedness == "Right"
    confianca_lado: float
    id_mao: int = -1                      # identidade estável entre frames (ver associar_maos)


@dataclass(frozen=True)
class SnapshotMaos:
    """Resultado de UMA rodada de detecção, publicado atomicamente.

    Imutável por construção (frozen + tuplas): atravessa fronteira de thread,
    então o consumidor não pode mutar algo que a thread produtora ainda
    referencia.
    """

    maos: tuple[MaoDetectada, ...] = ()
    timestamp_s: float = 0.0
    frame_id: int = 0
    largura_frame: int = 0
    altura_frame: int = 0


@dataclass
class MetricasRastreador:
    ms_por_deteccao: float = 0.0
    fps_deteccao: float = 0.0
    frames_repetidos_ignorados: int = 0
    deteccoes: int = 0
    erros: int = 0


def associar_maos(
    centroides_atuais: Sequence[tuple[float, float]],
    centroides_anteriores: Mapping[int, tuple[float, float]],
    distancia_max: float,
    proximo_id: int,
) -> tuple[dict[int, int], int]:
    """Casa as mãos deste frame com as do frame anterior, por proximidade.

    Retorna ({índice_atual: id_mao}, próximo_id_livre).

    Por que não usar `handedness` como identidade: o rótulo é confiável (medi
    score 0.99), mas a ORDEM em que as mãos aparecem em `hand_landmarks`
    troca entre frames. Como um gesto fica ancorado numa mão específica, se a
    identidade trocar no meio de um arrasto o vértice salta para a outra mão.
    Casamento guloso por menor distância resolve, e é função pura.
    """
    atribuicoes: dict[int, int] = {}
    disponiveis = dict(centroides_anteriores)

    candidatos = []
    for indice, atual in enumerate(centroides_atuais):
        for id_mao, anterior in disponiveis.items():
            distancia = ((atual[0] - anterior[0]) ** 2 + (atual[1] - anterior[1]) ** 2) ** 0.5
            if distancia <= distancia_max:
                candidatos.append((distancia, indice, id_mao))
    candidatos.sort()

    usados_indice: set[int] = set()
    usados_id: set[int] = set()
    for _, indice, id_mao in candidatos:
        if indice in usados_indice or id_mao in usados_id:
            continue
        atribuicoes[indice] = id_mao
        usados_indice.add(indice)
        usados_id.add(id_mao)

    for indice in range(len(centroides_atuais)):
        if indice not in atribuicoes:
            atribuicoes[indice] = proximo_id
            proximo_id += 1

    return atribuicoes, proximo_id


def centroide_palma(landmarks: Sequence[Vec3]) -> tuple[float, float]:
    """Centro dos pontos da palma — âncora estável para a associação temporal.

    Usa pulso e as juntas MCP (não as pontas dos dedos, que se mexem muito
    quando o gesto muda sem a mão sair do lugar).
    """
    indices = (0, 5, 9, 13, 17)
    return (
        sum(landmarks[i][0] for i in indices) / len(indices),
        sum(landmarks[i][1] for i in indices) / len(indices),
    )


class RastreadorMaos:
    """Roda a detecção numa thread daemon e publica o último resultado.

    O detector é criado no construtor (e não na thread) porque `main.py`
    depende de a exceção "modelo não encontrado" subir aqui para cair no
    fallback de mouse/teclado. Depois de criado, ele é usado exclusivamente
    pela thread de detecção.
    """

    def __init__(
        self,
        camera,
        config,
        fabrica_detector: Optional[Callable] = None,
        relogio: Callable[[], float] = time.monotonic,
        distancia_max_associacao: float = 0.25,
    ) -> None:
        self.camera = camera
        self.config = config
        self.relogio = relogio
        self.distancia_max_associacao = distancia_max_associacao

        self._detector = (fabrica_detector or self._criar_detector_mediapipe)(config)
        self._lock = threading.Lock()
        self._snapshot = SnapshotMaos()
        self._metricas = MetricasRastreador()
        self._parar = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._fechado = False

        self._ultimo_frame = None
        self._ultimo_timestamp_ms = 0
        self._frame_id = 0
        self._proximo_id_mao = 0
        self._centroides_anteriores: dict[int, tuple[float, float]] = {}

    # ------------------------------------------------------------------ API
    def iniciar(self) -> "RastreadorMaos":
        """Sobe a thread de detecção. Idempotente."""
        if self._thread is None:
            self._thread = threading.Thread(
                target=self._loop_deteccao, name="deteccao-maos", daemon=True
            )
            self._thread.start()
        return self

    def obter_snapshot(self) -> SnapshotMaos:
        """Último resultado disponível. Nunca bloqueia, nunca devolve None."""
        with self._lock:
            return self._snapshot

    def obter_metricas(self) -> MetricasRastreador:
        with self._lock:
            return MetricasRastreador(**vars(self._metricas))

    def close(self) -> None:
        """Para a thread e fecha o detector, nessa ordem. Idempotente."""
        if self._fechado:
            return
        self._fechado = True
        self._parar.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        # Só depois do join: fechar o detector com a thread ainda usando-o
        # seria uso após liberação dentro do C++ do MediaPipe.
        fechar = getattr(self._detector, "close", None)
        if callable(fechar):
            try:
                fechar()
            except Exception:
                pass

    def __enter__(self) -> "RastreadorMaos":
        return self.iniciar()

    def __exit__(self, *_) -> None:
        self.close()

    # -------------------------------------------------------------- interno
    def _criar_detector_mediapipe(self, config):
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision

        opcoes = mp_vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=config.HAND_LANDMARKER_MODEL_PATH),
            running_mode=mp_vision.RunningMode.VIDEO,
            num_hands=config.HAND_MAX_NUM_HANDS,
            min_hand_detection_confidence=config.HAND_MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=config.HAND_MIN_TRACKING_CONFIDENCE,
        )
        return mp_vision.HandLandmarker.create_from_options(opcoes)

    def _loop_deteccao(self) -> None:
        while not self._parar.is_set():
            try:
                frame = self.camera.obter_frame_mais_recente()
                # Comparação por identidade: a CameraSource só troca a
                # referência quando lê um frame novo. Com a câmera a 15 fps e a
                # detecção a 19, isso evita gastar 51 ms redetectando pixels
                # idênticos.
                if frame is None or frame is self._ultimo_frame:
                    if frame is not None:
                        with self._lock:
                            self._metricas.frames_repetidos_ignorados += 1
                    self._parar.wait(0.005)
                    continue
                self._ultimo_frame = frame
                self._processar(frame)
            except Exception:
                # Nenhuma exceção pode escapar e matar a thread — mesma
                # disciplina de CameraSource._loop_leitura.
                with self._lock:
                    self._metricas.erros += 1
                self._parar.wait(0.05)

    def _processar(self, frame) -> None:
        import cv2
        import mediapipe as mp

        inicio = self.relogio()
        if not self.config.INVERTER_ESPELHO_CAMERA:
            frame = cv2.flip(frame, 1)

        imagem = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
        )
        self._ultimo_timestamp_ms = max(self._ultimo_timestamp_ms + 1, int(inicio * 1000))
        resultado = self._detector.detect_for_video(imagem, self._ultimo_timestamp_ms)

        altura, largura = frame.shape[:2]
        snapshot = self._montar_snapshot(resultado, largura, altura, inicio)

        decorrido = (self.relogio() - inicio) * 1000.0
        with self._lock:
            self._snapshot = snapshot
            self._metricas.ms_por_deteccao = decorrido
            self._metricas.fps_deteccao = 1000.0 / decorrido if decorrido > 0 else 0.0
            self._metricas.deteccoes += 1

    def _montar_snapshot(self, resultado, largura: int, altura: int, agora: float) -> SnapshotMaos:
        landmarks_por_mao = getattr(resultado, "hand_landmarks", None) or []
        world_por_mao = getattr(resultado, "hand_world_landmarks", None) or []
        handedness = getattr(resultado, "handedness", None) or []

        brutas = []
        for i, pontos in enumerate(landmarks_por_mao):
            lm = tuple((p.x, p.y, p.z) for p in pontos)
            world = (
                tuple((p.x, p.y, p.z) for p in world_por_mao[i])
                if i < len(world_por_mao)
                else ()
            )
            destra, confianca = True, 0.0
            if i < len(handedness) and handedness[i]:
                categoria = handedness[i][0]
                destra = categoria.category_name == "Right"
                confianca = float(categoria.score)
            brutas.append((lm, world, destra, confianca))

        centroides = [centroide_palma(lm) for lm, _, _, _ in brutas]
        atribuicoes, self._proximo_id_mao = associar_maos(
            centroides, self._centroides_anteriores,
            self.distancia_max_associacao, self._proximo_id_mao,
        )
        self._centroides_anteriores = {
            atribuicoes[i]: centroides[i] for i in range(len(centroides))
        }

        maos = tuple(
            MaoDetectada(
                landmarks=lm, world_landmarks=world, destra=destra,
                confianca_lado=confianca, id_mao=atribuicoes[i],
            )
            for i, (lm, world, destra, confianca) in enumerate(brutas)
        )

        self._frame_id += 1
        return SnapshotMaos(
            maos=maos, timestamp_s=agora, frame_id=self._frame_id,
            largura_frame=largura, altura_frame=altura,
        )
