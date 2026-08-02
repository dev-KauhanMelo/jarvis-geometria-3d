"""Configurações centrais do Jarvis Tetraedro."""
from dataclasses import dataclass
from typing import Optional


@dataclass
class Config:
    # --- Etapa 1: geometria e render ---
    ARESTA_INICIAL: float = 1.5
    ARESTA_MIN: float = 0.3
    ARESTA_MAX: float = 4.0
    LARGURA_JANELA: int = 1024
    ALTURA_JANELA: int = 768
    FPS_ALVO: int = 60
    TITULO_JANELA: str = "Jarvis Tetraedro"

    # --- Etapa 2 (visão): rastreamento de mão ---
    CAMERA_URL: Optional[str] = None  # ex.: "http://192.168.0.10:8080/video"; None = usa direto o fallback local
    CAMERA_FALLBACK_INDICE: int = 0
    CAMERA_TIMEOUT_CONEXAO_S: float = 3.0
    CAMERA_INTERVALO_RECONEXAO_S: float = 2.0
    CAMERA_FALHAS_CONSECUTIVAS_PARA_RECONECTAR: int = 5

    HAND_LANDMARKER_MODEL_PATH: str = "models/hand_landmarker.task"
    HAND_MAX_NUM_HANDS: int = 1
    HAND_MIN_DETECTION_CONFIDENCE: float = 0.5
    HAND_MIN_TRACKING_CONFIDENCE: float = 0.5

    SENSIBILIDADE_ROTACAO_GESTO: float = 200.0
    SENSIBILIDADE_ZOOM_GESTO: float = 15.0
    SENSIBILIDADE_ARESTA_GESTO: float = 3.0
    DEAD_ZONE_ROTACAO: float = 0.015
    DEAD_ZONE_ZOOM: float = 0.02
    DEAD_ZONE_ARESTA: float = 0.01
    ALFA_SUAVIZACAO: float = 0.4
    # Calibrado com fotos reais processadas pelo MediaPipe: punho fechado ~0.24,
    # um dedo esticado já sobe pra ~0.43 — 0.32 separa bem os dois casos. Ainda
    # assim, é ponto de partida: pode precisar de ajuste fino por usuário/mão
    # (ver valores brutos na janela de debug).
    LIMIAR_MAO_FECHADA: float = 0.32
    DURACAO_HOLD_RESET_S: float = 1.0
    INVERTER_ESPELHO_CAMERA: bool = False

    # --- Etapa 3 (voz): reservado, ainda não usado ---
    # VOSK_MODEL_PATH: str = "models/vosk-model-small-pt"
    # PALAVRAS_ATIVACAO: tuple[str, ...] = ("jarvis", "tetraedro")
