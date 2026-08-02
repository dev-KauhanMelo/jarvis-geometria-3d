"""Configurações centrais do Jarvis Tetraedro."""
from dataclasses import dataclass


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

    # --- Etapa 2 (visão): reservado, ainda não usado ---
    # CAMERA_URL: str = "http://192.168.0.10:8080/video"
    # CAMERA_FALLBACK_INDICE: int = 0

    # --- Etapa 3 (voz): reservado, ainda não usado ---
    # VOSK_MODEL_PATH: str = "models/vosk-model-small-pt"
    # PALAVRAS_ATIVACAO: tuple[str, ...] = ("jarvis", "tetraedro")
