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
    # Duas mãos, para a manipulação a duas mãos da Etapa 3. Medi o custo: 51 ms
    # por detecção contra 26 ms com uma só. Como a detecção roda em thread
    # própria (vision/rastreador.py), isso não segura o render — mas com
    # hardware fraco `--uma-mao` devolve a diferença.
    HAND_MAX_NUM_HANDS: int = 2
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

    # --- Etapa 3: realidade aumentada ---
    # O sólido é composto sobre a imagem da câmera, na mesma janela.
    AR_ATIVO: bool = True
    # "preencher" cobre a janela cortando as bordas do frame; "caber" mostra o
    # frame inteiro com barras pretas. Preencher é o padrão porque barra preta
    # quebra a ilusão de que o objeto está no ambiente.
    AR_MODO_AJUSTE: str = "preencher"
    # Véu escuro sobre o vídeo: o sólido é semi-transparente e some sobre um
    # ambiente claro ou movimentado. Suba se estiver difícil de enxergar.
    AR_ESCURECIMENTO_FUNDO: float = 0.35

    # --- Etapa 3: manipulação direta ---
    # Pinça: fecha em PINCA_FECHA, mas só solta em PINCA_ABRE. A folga entre os
    # dois é histerese — sem ela o objeto solta sozinho quando a mão treme em
    # cima do limiar. Medi nas fotos: punho 0,13; mão aberta 0,99.
    PINCA_FECHA: float = 0.35
    PINCA_ABRE: float = 0.50

    # Concha ("carregando um poder na mão"): todos os quatro dedos
    # semi-dobrados. Medi as extensões (corda/arco) em fotos reais: punho
    # [0,31 0,25 0,33 0,46], apontando [0,96 0,35 ...], vitória [0,94 0,99 ...] —
    # os três são rejeitados. O positivo depende da SUA mão: rode
    # `--debug-gestos`, faça a concha, leia os quatro números e ajuste.
    CONCHA_EXTENSAO_MIN: float = 0.50
    CONCHA_EXTENSAO_MAX: float = 0.88
    CONCHA_EXTENSAO_MIN_SAI: float = 0.42
    CONCHA_EXTENSAO_MAX_SAI: float = 0.94

    # Quão perto do vértice a mão precisa estar para pegá-lo (~6% da altura).
    RAIO_PICK_VERTICE_PX: float = 45.0
    # Frames de DETECÇÃO consecutivos para confirmar um gesto. Filtra o
    # falso-positivo de um landmark ruim isolado, ao custo de ~80 ms de engate.
    QUADROS_CONFIRMACAO_GESTO: int = 2
    # Carência para a mão que acabou de entrar no quadro, cujos landmarks
    # ainda estão convergindo.
    TEMPO_MINIMO_MAO_VISIVEL_S: float = 0.15

    # Constantes de tempo da suavização (segundos). Maior = mais suave e mais
    # lento. A orientação usa a maior porque o z dos world landmarks é ruidoso.
    TAU_POSICAO_MAO: float = 0.06
    TAU_ESCALARES_MAO: float = 0.08
    TAU_ORIENTACAO_MAO: float = 0.12
    # A pinça é um GATILHO ("agarrei agora"), não uma classificação de pose:
    # suavizá-la tanto quanto as demais medidas adiava o engate em ~180 ms e
    # o gesto parecia não responder. A histerese e o debounce já cuidam do
    # ruído, então aqui vale privilegiar a latência.
    TAU_PINCA: float = 0.03

    # Conversão de pixels de tela para as unidades da cena.
    SENSIBILIDADE_ORBITA_PX: float = 0.35   # graus de rotação por pixel
    SENSIBILIDADE_PAN_PX: float = 0.006     # unidades de mundo por pixel

    # --- Etapa 3 (voz): reservado, ainda não usado ---
    # VOSK_MODEL_PATH: str = "models/vosk-model-small-pt"
    # PALAVRAS_ATIVACAO: tuple[str, ...] = ("jarvis", "tetraedro")
