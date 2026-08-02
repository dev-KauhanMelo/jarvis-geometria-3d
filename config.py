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

    INVERTER_ESPELHO_CAMERA: bool = False
    # O controle contínuo da Etapa 2 (sensibilidades, dead zones, limiar de
    # punho e hold de reset) saiu junto com o modo em que qualquer movimento da
    # mão já mexia no sólido. Quem manda agora são os limiares de agarre logo
    # abaixo, na seção da Etapa 3.

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
    # Dois agarres distintos. Cada um fecha num limiar e só solta noutro mais
    # frouxo: a folga é histerese, sem ela o objeto se solta sozinho quando a
    # mão treme em cima do limiar.
    #
    # GARRA = a mão inteira fechando (apertar um balão). Pega o sólido todo.
    # Medi `abertura_mao` com o MediaPipe: punho 0,243, dedo apontando 0,434,
    # "vitória" 0,582. 0,38 fica no meio do vão entre fechar e apontar.
    GARRA_FECHA: float = 0.38
    GARRA_ABRE: float = 0.48
    # PINÇA = só polegar e indicador, com a mão aberta. Pega um vértice.
    # Medi `distancia_pinca`: punho 0,199; mão aberta ~1,05.
    PINCA_FECHA: float = 0.35
    PINCA_ABRE: float = 0.50

    # Quanto a mão pode chegar/afastar da câmera dentro de um mesmo agarre,
    # como razão do tamanho aparente da mão. Serve de trava: uma leitura ruim
    # do tamanho jogaria o sólido para o infinito ou para dentro da câmera.
    RAZAO_PROFUNDIDADE_MIN: float = 0.6
    RAZAO_PROFUNDIDADE_MAX: float = 1.7

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
    # O tamanho aparente da mão vira PROFUNDIDADE, que é onde tremer mais
    # incomoda, e é a medida mais ruidosa (muda com o ângulo do punho, não só
    # com a distância). Por isso a constante é muito maior que as demais.
    TAU_TAMANHO_MAO: float = 0.25

    # Ligado por --debug-gestos: mostra os valores crus no HUD para calibração.
    DEBUG_GESTOS: bool = False

    # A translação não tem constante de sensibilidade: o Viewer converte pixel
    # em unidade de mundo pela geometria da câmera, então o sólido acompanha a
    # mão exatamente 1:1 em qualquer nível de zoom.

    # --- Etapa 3 (voz): reservado, ainda não usado ---
    # VOSK_MODEL_PATH: str = "models/vosk-model-small-pt"
    # PALAVRAS_ATIVACAO: tuple[str, ...] = ("jarvis", "tetraedro")
