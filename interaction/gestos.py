"""Máquina de estados da manipulação direta.

Modelo de controle: o objeto flutua num espaço e só se mexe quando uma mão o
AGARRA de propósito. Mão aberta = objeto parado, exatamente onde foi deixado.

São dois agarres, fisicamente distintos e por isso nunca ambíguos:

- **Garra** (a mão inteira fechando, como apertar um balão): pega o SÓLIDO
  INTEIRO. Enquanto durar, o objeto fica travado na mão — anda com ela,
  gira com a palma e chega mais perto quando a mão se aproxima da câmera.
- **Pinça** (só polegar e indicador, mão aberta no resto): pega UM VÉRTICE e
  o deforma. É o gesto de precisão.

O que havia antes e foi removido, com o motivo:

- *Órbita por deslocamento na tela* — convertia pixels percorridos em graus a
  0,35°/px, então atravessar a janela girava o sólido 358° e um tremor de
  20 px já rodava 7°. Era o "ele fica se movendo loucamente". A garra gira
  1:1 com a palma: onde a mão aponta, o objeto aponta.
- *Concha como pose estática* — exigia os quatro dedos numa faixa estreita de
  curvatura, e essa faixa também aceitava a mão relaxada, então o gesto
  disparava sozinho.

Tudo aqui é função pura: sem OpenGL, sem relógio implícito, sem I/O. A cena
entra pelo protocolo `ContextoCena`, que nos testes é uma projeção
ortográfica escrita à mão.
"""
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Protocol, Sequence

from geometry.transformacoes import Quat, compor_rotacao_relativa
from vision.hand_tracker import mao_em_garra

Vec3 = tuple[float, float, float]


class Fase(Enum):
    OCIOSO = "ocioso"
    SEGURANDO_OBJETO = "segurando_objeto"
    ARRASTANDO_VERTICE = "arrastando_vertice"
    ESCALANDO = "escalando"


MENSAGENS = {
    Fase.OCIOSO: "",
    Fase.SEGURANDO_OBJETO: "segurando o sólido",
    Fase.ARRASTANDO_VERTICE: "deformando vértice {vertice}",
    Fase.ESCALANDO: "redimensionando",
}


@dataclass(frozen=True)
class ParametrosGesto:
    """Limiares com histerese: o valor de ENTRAR difere do de SAIR, senão o
    gesto fica oscilando quando a mão para bem em cima do limiar."""

    # Pinça (precisão): medi punho 0,199 e mão aberta ~1,05.
    pinca_fecha: float = 0.35
    pinca_abre: float = 0.50
    # Garra (mão inteira): medi punho 0,243, apontando 0,434, vitória 0,582.
    garra_fecha: float = 0.38
    garra_abre: float = 0.48
    raio_pick_px: float = 45.0
    quadros_confirmacao: int = 2
    tempo_minimo_visivel_s: float = 0.15
    # Quanto a mão pode chegar/afastar da câmera dentro de um mesmo agarre.
    # Sem trava, um erro de leitura do tamanho da mão jogaria o sólido para o
    # infinito ou para dentro da câmera de uma vez.
    razao_profundidade_min: float = 0.6
    razao_profundidade_max: float = 1.7


@dataclass(frozen=True)
class EstadoInteracao:
    fase: Fase = Fase.OCIOSO
    id_mao_ancora: Optional[int] = None
    id_mao_secundaria: Optional[int] = None
    vertice_ativo: Optional[int] = None

    # âncoras capturadas no INÍCIO do gesto
    cursor_anterior: Optional[tuple[float, float]] = None
    offset_cursor_vertice: tuple[float, float] = (0.0, 0.0)
    z_janela_vertice: float = 0.0
    distancia_inicial_maos: float = 0.0
    escala_inicial: float = 0.0
    ponto_medio_anterior: Optional[tuple[float, float]] = None
    orientacao_mao_inicial: Optional[Quat] = None
    orientacao_objeto_inicial: Optional[Quat] = None
    tamanho_mao_inicial: float = 0.0
    distancia_camera_inicial: float = 0.0

    # debounce de entrada
    candidato: Fase = Fase.OCIOSO
    quadros_candidato: int = 0
    ultimo_frame_id: int = -1


@dataclass(frozen=True)
class ComandoInteracao:
    """O que o Viewer deve aplicar neste frame.

    Declarativo de propósito: o modelo de "deltas" da Etapa 2 não sabe
    expressar "leve o vértice 2 exatamente para esta posição". Aqui os deltas
    continuam existindo, mas como um campo entre outros.
    """

    fase: Fase = Fase.OCIOSO
    orientacao_absoluta: Optional[Quat] = None
    delta_pan_tela: tuple[float, float] = (0.0, 0.0)
    distancia_camera_absoluta: Optional[float] = None
    escala_absoluta: Optional[float] = None
    vertice_sob_mira: Optional[int] = None
    vertice_movido: Optional[int] = None
    posicao_vertice_objeto: Optional[Vec3] = None
    mensagem: str = ""


class ContextoCena(Protocol):
    def vertices_tela(self) -> Sequence[Optional[tuple[float, float, float]]]:
        """(x_px, y_px, z_janela) de cada vértice; None se não projetável."""

    def desprojetar(self, x_px: float, y_px: float, z_janela: float) -> Optional[Vec3]: ...

    def orientacao_objeto(self) -> Quat: ...

    def escala_objeto(self) -> float: ...

    def distancia_camera(self) -> float: ...


# ---------------------------------------------------------------------------
# Funções auxiliares puras
# ---------------------------------------------------------------------------
def vertice_sob_cursor(
    cursor: tuple[float, float],
    vertices_tela: Sequence[Optional[tuple[float, float, float]]],
    raio_px: float,
) -> Optional[int]:
    """Índice do vértice mais próximo do cursor dentro do raio, ou None.

    Havendo empate visual, ganha o mais próximo na tela; a profundidade não
    entra no critério porque o usuário mira pelo que vê.
    """
    melhor_indice, melhor_distancia = None, raio_px
    for indice, projetado in enumerate(vertices_tela):
        if projetado is None:
            continue
        distancia = ((cursor[0] - projetado[0]) ** 2 + (cursor[1] - projetado[1]) ** 2) ** 0.5
        if distancia <= melhor_distancia:
            melhor_indice, melhor_distancia = indice, distancia
    return melhor_indice


def esta_em_garra(mao, params: ParametrosGesto, ja_em_garra: bool) -> bool:
    """Mão inteira fechada, com histerese."""
    limiar = params.garra_abre if ja_em_garra else params.garra_fecha
    return mao_em_garra(mao.abertura, limiar)


def esta_agarrando(mao, params: ParametrosGesto, ja_agarrando: bool) -> bool:
    """Pinça de precisão: polegar e indicador juntos com a mão ABERTA.

    A exigência de mão aberta é o que separa a pinça da garra. Sem ela o
    punho fechado dispararia as duas (medi pinça 0,199 num punho), e o
    usuário nunca saberia se ia deformar um vértice ou mover o sólido.
    """
    # A garra é sempre testada pelo limiar ESTRITO, mesmo com a pinça já em
    # curso. `ja_agarrando` vale para a pinça, não para a garra: reaproveitá-lo
    # aqui abriria o limiar da garra para 0,48 e a mão que segura um vértice
    # (abertura ~0,43, dedos esticados) passaria a ser lida como punho,
    # derrubando o arrasto no meio.
    if esta_em_garra(mao, params, False):
        return False
    limiar = params.pinca_abre if ja_agarrando else params.pinca_fecha
    return mao.pinca < limiar


def _distancia(a: tuple[float, float], b: tuple[float, float]) -> float:
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def _ponto_medio(a: tuple[float, float], b: tuple[float, float]) -> tuple[float, float]:
    return ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)


def _por_id(maos: Sequence) -> dict[int, object]:
    return {m.id_mao: m for m in maos}


def _profundidade(estado: EstadoInteracao, mao, params: ParametrosGesto) -> Optional[float]:
    """Distância da câmera a partir de quanto a mão cresceu na imagem.

    A mão chegando perto da câmera aparece maior, então a razão sobe e a
    distância cai — o sólido vem para frente. É o mapeamento que o usuário
    esperava de cara ("se eu aproximo a mão, ele fica mais perto").
    """
    if estado.tamanho_mao_inicial <= 0 or mao.tamanho <= 0:
        return None
    razao = mao.tamanho / estado.tamanho_mao_inicial
    razao = max(params.razao_profundidade_min, min(params.razao_profundidade_max, razao))
    return estado.distancia_camera_inicial / razao


# ---------------------------------------------------------------------------
# Máquina de estados
# ---------------------------------------------------------------------------
def avaliar_gestos(
    estado: EstadoInteracao,
    maos: Sequence,
    cena: ContextoCena,
    params: ParametrosGesto,
    frame_id: int,
) -> tuple[EstadoInteracao, ComandoInteracao]:
    """Um passo da máquina de estados. Pura: mesma entrada, mesma saída."""
    indexadas = _por_id(maos)

    # 1) Gesto em curso vence tudo — com uma exceção deliberada.
    #
    #    Um gesto ancorado num ALVO (o arrasto de um vértice) é protegido:
    #    encostar a outra mão no campo de visão não pode sequestrá-lo.
    #    Já segurar o sólido é genérico e PODE virar escala quando a segunda
    #    mão fecha a garra — as duas mãos praticamente nunca entram no quadro
    #    no mesmo frame, então sem essa promoção o gesto de duas mãos seria
    #    impossível de iniciar na prática.
    if estado.fase is Fase.SEGURANDO_OBJETO:
        promovido = _promover_para_escala(estado, indexadas, maos, cena, params)
        if promovido is not None:
            return promovido

    if estado.fase is not Fase.OCIOSO:
        continuado = _continuar(estado, indexadas, cena, params)
        if continuado is not None:
            return continuado
        estado = EstadoInteracao(ultimo_frame_id=estado.ultimo_frame_id)

    # 2) Ocioso: escolhe o candidato e aplica o debounce.
    elegiveis = [m for m in maos if m.visivel_ha_s >= params.tempo_minimo_visivel_s]
    candidato, dados = _escolher_candidato(elegiveis, cena, params)

    novo_frame = frame_id != estado.ultimo_frame_id
    if candidato is estado.candidato:
        # Conta frames de SNAPSHOT, não de render: a 60 fps sobre 25 de
        # detecção, contar frames de render filtraria só 33 ms e não pegaria
        # o falso-positivo de um landmark ruim.
        quadros = estado.quadros_candidato + (1 if novo_frame else 0)
    else:
        quadros = 1
    estado_base = EstadoInteracao(
        candidato=candidato, quadros_candidato=quadros, ultimo_frame_id=frame_id
    )

    mira = _vertice_sob_mira(elegiveis, cena, params)
    if candidato is Fase.OCIOSO or quadros < params.quadros_confirmacao:
        return estado_base, ComandoInteracao(fase=Fase.OCIOSO, vertice_sob_mira=mira)

    return _iniciar(candidato, dados, cena, frame_id), ComandoInteracao(
        fase=candidato, vertice_sob_mira=mira, mensagem=_mensagem(candidato, dados)
    )


def _promover_para_escala(estado, indexadas, maos, cena, params):
    """Uma segunda mão fechando a garra enquanto a primeira segura o sólido
    inicia a escala.

    Exige que a mão âncora ainda esteja segurando, senão o gesto já acabou e
    o caminho normal cuida disso.
    """
    ancora = indexadas.get(estado.id_mao_ancora)
    if ancora is None or not esta_em_garra(ancora, params, True):
        return None

    outras = [
        m for m in maos
        if m.id_mao != estado.id_mao_ancora
        and m.visivel_ha_s >= params.tempo_minimo_visivel_s
        and esta_em_garra(m, params, False)
    ]
    if not outras:
        return None

    secundaria = min(outras, key=lambda m: m.id_mao)
    dados = {"ancora": ancora, "secundaria": secundaria}
    return _iniciar(Fase.ESCALANDO, dados, cena, estado.ultimo_frame_id), ComandoInteracao(
        fase=Fase.ESCALANDO, mensagem=MENSAGENS[Fase.ESCALANDO]
    )


def _vertice_sob_mira(maos, cena, params) -> Optional[int]:
    """Realce de proximidade: mostra qual vértice seria pego, antes de agarrar."""
    if not maos:
        return None
    vertices = cena.vertices_tela()
    for mao in maos:
        encontrado = vertice_sob_cursor(mao.cursor_tela, vertices, params.raio_pick_px)
        if encontrado is not None:
            return encontrado
    return None


def _escolher_candidato(maos, cena, params) -> tuple[Fase, dict]:
    em_garra = [m for m in maos if esta_em_garra(m, params, False)]

    if len(em_garra) >= 2:
        # Ordena por id para a escolha ser determinística entre frames.
        a, b = sorted(em_garra, key=lambda m: m.id_mao)[:2]
        return Fase.ESCALANDO, {"ancora": a, "secundaria": b}

    # A pinça de precisão é avaliada ANTES da garra de uma mão só: mirar um
    # vértice é o gesto mais específico e por isso o mais intencional.
    for mao in sorted(maos, key=lambda m: m.id_mao):
        if not esta_agarrando(mao, params, False):
            continue
        indice = vertice_sob_cursor(mao.cursor_tela, cena.vertices_tela(), params.raio_pick_px)
        if indice is not None:
            return Fase.ARRASTANDO_VERTICE, {"ancora": mao, "vertice": indice}

    if em_garra:
        return Fase.SEGURANDO_OBJETO, {"ancora": em_garra[0]}

    return Fase.OCIOSO, {}


def _mensagem(fase: Fase, dados: dict) -> str:
    return MENSAGENS[fase].format(vertice=dados.get("vertice"))


def _iniciar(fase: Fase, dados: dict, cena: ContextoCena, frame_id: int) -> EstadoInteracao:
    """Captura as âncoras do início do gesto."""
    ancora = dados.get("ancora")
    comum = dict(fase=fase, id_mao_ancora=ancora.id_mao, ultimo_frame_id=frame_id,
                 candidato=fase, quadros_candidato=0, cursor_anterior=ancora.cursor_tela)

    if fase is Fase.ESCALANDO:
        secundaria = dados["secundaria"]
        return EstadoInteracao(
            **comum,
            id_mao_secundaria=secundaria.id_mao,
            distancia_inicial_maos=max(_distancia(ancora.cursor_tela, secundaria.cursor_tela), 1e-6),
            escala_inicial=cena.escala_objeto(),
            ponto_medio_anterior=_ponto_medio(ancora.palma_tela, secundaria.palma_tela),
        )

    if fase is Fase.ARRASTANDO_VERTICE:
        indice = dados["vertice"]
        projetado = cena.vertices_tela()[indice]
        # Guarda o deslocamento entre o cursor e o vértice para o sólido não
        # "pular" no instante em que é agarrado.
        offset = (projetado[0] - ancora.cursor_tela[0], projetado[1] - ancora.cursor_tela[1])
        return EstadoInteracao(
            **comum, vertice_ativo=indice,
            offset_cursor_vertice=offset, z_janela_vertice=projetado[2],
        )

    # SEGURANDO_OBJETO: o sólido trava na mão. Guarda a pose da mão e a do
    # objeto no instante do agarre, e move tudo em relação a elas — assim o
    # objeto nunca salta ao ser pego e nunca acumula deriva enquanto é levado.
    return EstadoInteracao(
        **comum,
        orientacao_mao_inicial=ancora.orientacao,
        orientacao_objeto_inicial=cena.orientacao_objeto(),
        tamanho_mao_inicial=ancora.tamanho,
        distancia_camera_inicial=cena.distancia_camera(),
        ponto_medio_anterior=ancora.palma_tela,
    )


def _continuar(estado, indexadas, cena, params):
    """Mantém o gesto em curso, ou devolve None quando ele acabou."""
    ancora = indexadas.get(estado.id_mao_ancora)
    if ancora is None:
        return None  # a mão sumiu (saiu do quadro ou foi ocluída)

    if estado.fase is Fase.ESCALANDO:
        secundaria = indexadas.get(estado.id_mao_secundaria)
        if secundaria is None:
            return None
        if not (esta_em_garra(ancora, params, True) and esta_em_garra(secundaria, params, True)):
            return None
        distancia = max(_distancia(ancora.cursor_tela, secundaria.cursor_tela), 1e-6)
        medio = _ponto_medio(ancora.palma_tela, secundaria.palma_tela)
        anterior = estado.ponto_medio_anterior or medio
        # Escala absoluta em relação ao início do gesto: acumular razões
        # quadro a quadro derivaria ao longo do tempo.
        escala = estado.escala_inicial * (distancia / estado.distancia_inicial_maos)
        return (
            _substituir(estado, ponto_medio_anterior=medio),
            ComandoInteracao(
                fase=estado.fase,
                escala_absoluta=escala,
                delta_pan_tela=(medio[0] - anterior[0], medio[1] - anterior[1]),
                mensagem=MENSAGENS[estado.fase],
            ),
        )

    if estado.fase is Fase.SEGURANDO_OBJETO:
        if not esta_em_garra(ancora, params, True):
            return None
        anterior = estado.ponto_medio_anterior or ancora.palma_tela
        orientacao = compor_rotacao_relativa(
            estado.orientacao_objeto_inicial,
            estado.orientacao_mao_inicial,
            ancora.orientacao,
        )
        return (
            _substituir(estado, ponto_medio_anterior=ancora.palma_tela,
                        cursor_anterior=ancora.cursor_tela),
            ComandoInteracao(
                fase=estado.fase,
                orientacao_absoluta=orientacao,
                delta_pan_tela=(
                    ancora.palma_tela[0] - anterior[0],
                    ancora.palma_tela[1] - anterior[1],
                ),
                distancia_camera_absoluta=_profundidade(estado, ancora, params),
                mensagem=MENSAGENS[estado.fase],
            ),
        )

    # ARRASTANDO_VERTICE
    if not esta_agarrando(ancora, params, True):
        return None

    alvo_x = ancora.cursor_tela[0] + estado.offset_cursor_vertice[0]
    alvo_y = ancora.cursor_tela[1] + estado.offset_cursor_vertice[1]
    # Move no plano paralelo à câmera, preservando a profundidade que o
    # vértice tinha quando foi agarrado.
    posicao = cena.desprojetar(alvo_x, alvo_y, estado.z_janela_vertice)
    return (
        _substituir(estado, cursor_anterior=ancora.cursor_tela),
        ComandoInteracao(
            fase=estado.fase,
            vertice_movido=estado.vertice_ativo,
            posicao_vertice_objeto=posicao,
            vertice_sob_mira=estado.vertice_ativo,
            mensagem=MENSAGENS[estado.fase].format(vertice=estado.vertice_ativo),
        ),
    )


def _substituir(estado: EstadoInteracao, **campos) -> EstadoInteracao:
    from dataclasses import replace

    return replace(estado, **campos)
