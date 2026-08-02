"""Geometria do tetraedro regular: vértices, topologia e fórmulas.

Os 4 vértices são construídos a partir de cantos alternados de um cubo
centrado na origem — uma forma simples e numericamente estável de obter
um tetraedro regular sem trigonometria, já centrado no baricentro.
"""
from dataclasses import dataclass, field
from math import sqrt
from typing import ClassVar, Optional, Sequence

VERTICES_UNITARIOS: tuple[tuple[float, float, float], ...] = (
    (1.0, 1.0, 1.0),
    (1.0, -1.0, -1.0),
    (-1.0, 1.0, -1.0),
    (-1.0, -1.0, 1.0),
)

# Distância entre quaisquer 2 vértices unitários acima é sempre 2*sqrt(2).
_ARESTA_UNITARIA = 2 * sqrt(2)

FACES: tuple[tuple[int, int, int], ...] = (
    (0, 1, 2),
    (0, 1, 3),
    (0, 2, 3),
    (1, 2, 3),
)

ARESTAS_INDICES: tuple[tuple[int, int], ...] = (
    (0, 1),
    (0, 2),
    (0, 3),
    (1, 2),
    (1, 3),
    (2, 3),
)


def _validar_aresta(aresta: float) -> None:
    if aresta <= 0:
        raise ValueError(f"aresta deve ser positiva, recebido: {aresta!r}")


def calcular_vertices(aresta: float) -> list[tuple[float, float, float]]:
    """Retorna os 4 vértices do tetraedro regular de aresta `aresta`, centrado na origem."""
    _validar_aresta(aresta)
    escala = aresta / _ARESTA_UNITARIA
    return [(x * escala, y * escala, z * escala) for x, y, z in VERTICES_UNITARIOS]


def _subtrair(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _produto_vetorial(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _produto_escalar(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _centroide(pontos: list[tuple[float, float, float]]) -> tuple[float, float, float]:
    n = len(pontos)
    soma = (sum(p[0] for p in pontos), sum(p[1] for p in pontos), sum(p[2] for p in pontos))
    return (soma[0] / n, soma[1] / n, soma[2] / n)


def calcular_normal_face(
    vertices: list[tuple[float, float, float]], face: tuple[int, int, int]
) -> tuple[float, float, float]:
    """Normal unitária da face, apontando para fora do sólido.

    Calcula a normal via produto vetorial e, se ela apontar para dentro
    (produto escalar negativo contra o vetor centroide_face -> centroide_solido),
    inverte o sinal — evita bugs de iluminação por winding incorreto.
    """
    v0, v1, v2 = vertices[face[0]], vertices[face[1]], vertices[face[2]]
    normal = _produto_vetorial(_subtrair(v1, v0), _subtrair(v2, v0))
    modulo = sqrt(_produto_escalar(normal, normal))
    if modulo == 0:
        raise ValueError("vértices da face são colineares, normal indefinida")
    normal = (normal[0] / modulo, normal[1] / modulo, normal[2] / modulo)

    centroide_solido = _centroide(vertices)
    centroide_face = _centroide([v0, v1, v2])
    direcao_externa = _subtrair(centroide_face, centroide_solido)
    if _produto_escalar(normal, direcao_externa) < 0:
        normal = (-normal[0], -normal[1], -normal[2])
    return normal


def area_face(aresta: float) -> float:
    """Área de uma face triangular equilátera."""
    _validar_aresta(aresta)
    return (sqrt(3) / 4) * aresta**2


def area_total(aresta: float) -> float:
    """Área total das 4 faces: a²√3."""
    _validar_aresta(aresta)
    return aresta**2 * sqrt(3)


def volume(aresta: float) -> float:
    """Volume: a³√2/12."""
    _validar_aresta(aresta)
    return aresta**3 * sqrt(2) / 12


def altura(aresta: float) -> float:
    """Altura (vértice ao baricentro da face oposta): a√6/3."""
    _validar_aresta(aresta)
    return aresta * sqrt(6) / 3


def apotema_face(aresta: float) -> float:
    """Apótema da face triangular equilátera: a√3/2."""
    _validar_aresta(aresta)
    return aresta * sqrt(3) / 2


# ---------------------------------------------------------------------------
# Fórmulas genéricas — valem para um tetraedro QUALQUER (deformado)
#
# As fórmulas fechadas acima (a²√3, a³√2/12, ...) só valem enquanto o sólido
# é regular. Quando o usuário arrasta um vértice, o sólido deixa de ser
# regular e estes cálculos numéricos assumem o lugar delas. No caso regular
# os dois caminhos coincidem — propriedade verificada nos testes.
# ---------------------------------------------------------------------------
Vec3 = tuple[float, float, float]


def area_face_generica(vertices: Sequence[Vec3], face: tuple[int, int, int]) -> float:
    """Área do triângulo: ½·|(v1-v0) × (v2-v0)|."""
    v0, v1, v2 = vertices[face[0]], vertices[face[1]], vertices[face[2]]
    normal = _produto_vetorial(_subtrair(v1, v0), _subtrair(v2, v0))
    return 0.5 * sqrt(_produto_escalar(normal, normal))


def area_total_generica(vertices: Sequence[Vec3], faces=FACES) -> float:
    """Soma das áreas das 4 faces. No caso regular converge para a²√3."""
    return sum(area_face_generica(vertices, face) for face in faces)


def volume_generico(vertices: Sequence[Vec3]) -> float:
    """|det(B-A, C-A, D-A)| / 6. Não-negativo, independe do winding."""
    a, b, c, d = vertices[0], vertices[1], vertices[2], vertices[3]
    return abs(_produto_escalar(_subtrair(b, a), _produto_vetorial(_subtrair(c, a), _subtrair(d, a)))) / 6.0


def comprimentos_arestas(vertices: Sequence[Vec3], arestas=ARESTAS_INDICES) -> list[float]:
    """As 6 distâncias, na ordem de ARESTAS_INDICES."""
    return [
        sqrt(_produto_escalar(_subtrair(vertices[i], vertices[j]), _subtrair(vertices[i], vertices[j])))
        for i, j in arestas
    ]


def altura_generica(vertices: Sequence[Vec3], indice_apice: int = 0) -> float:
    """Distância do vértice `indice_apice` ao plano da face oposta = 3V/A_base.

    No caso regular reduz a a√6/3.
    """
    face_oposta = tuple(i for i in range(4) if i != indice_apice)
    area_base = area_face_generica(vertices, face_oposta)  # type: ignore[arg-type]
    if area_base < 1e-12:
        return 0.0
    return 3.0 * volume_generico(vertices) / area_base


def apotema_face_generica(vertices: Sequence[Vec3], face: tuple[int, int, int]) -> float:
    """Média das três alturas do triângulo (2·Área/lado).

    Definida assim de propósito para reduzir exatamente a `apotema_face(a)`
    = a√3/2 no caso regular, mantendo a coerência com o que o app já exibe.
    """
    area = area_face_generica(vertices, face)
    lados = []
    for i in range(3):
        p, q = vertices[face[i]], vertices[face[(i + 1) % 3]]
        lados.append(sqrt(_produto_escalar(_subtrair(p, q), _subtrair(p, q))))
    alturas = [2.0 * area / lado for lado in lados if lado > 1e-12]
    return sum(alturas) / len(alturas) if alturas else 0.0


def e_regular(vertices: Sequence[Vec3], tolerancia_relativa: float = 1e-6) -> bool:
    """True se as 6 arestas forem iguais dentro da tolerância relativa."""
    medidas = comprimentos_arestas(vertices)
    media = sum(medidas) / len(medidas)
    if media < 1e-12:
        return False
    return all(abs(m - media) / media <= tolerancia_relativa for m in medidas)


def centroide_solido(vertices: Sequence[Vec3]) -> Vec3:
    return _centroide(list(vertices))


def escalar_em_torno_do_centroide(vertices: Sequence[Vec3], fator: float) -> list[Vec3]:
    """Escala uniforme preservando o centro do sólido."""
    c = centroide_solido(vertices)
    return [
        (
            c[0] + (v[0] - c[0]) * fator,
            c[1] + (v[1] - c[1]) * fator,
            c[2] + (v[2] - c[2]) * fator,
        )
        for v in vertices
    ]


@dataclass
class Tetraedro:
    """Estado do tetraedro exibido — única fonte de verdade da geometria.

    Começa REGULAR, descrito só pelo escalar `aresta`. Quando um vértice é
    arrastado (`mover_vertice`), passa a guardar as posições explícitas em
    `_vertices_deformados` e todos os cálculos migram para as fórmulas
    genéricas. `resetar()` volta ao estado regular.
    """

    aresta: float = 1.5
    aresta_min: float = 0.3
    aresta_max: float = 4.0
    _aresta_padrao: float = field(init=False, repr=False, default=0.0)
    _vertices_deformados: Optional[list[Vec3]] = field(init=False, repr=False, default=None)

    # Fração do volume regular equivalente abaixo da qual um arrasto de vértice
    # é recusado — impede que o sólido colapse num plano (volume ~0), o que
    # deixaria a normal da face indefinida no meio do render.
    VOLUME_MINIMO_RELATIVO: ClassVar[float] = 0.02

    def __post_init__(self) -> None:
        _validar_aresta(self.aresta)
        self._aresta_padrao = self.aresta

    # --- consulta -----------------------------------------------------------
    def esta_regular(self) -> bool:
        return self._vertices_deformados is None

    def vertices(self) -> list[Vec3]:
        if self._vertices_deformados is not None:
            return list(self._vertices_deformados)
        return calcular_vertices(self.aresta)

    def faces(self) -> tuple[tuple[int, int, int], ...]:
        return FACES

    def arestas(self) -> tuple[tuple[int, int], ...]:
        return ARESTAS_INDICES

    def area_total(self) -> float:
        if self._vertices_deformados is not None:
            return area_total_generica(self._vertices_deformados)
        return area_total(self.aresta)

    def volume(self) -> float:
        if self._vertices_deformados is not None:
            return volume_generico(self._vertices_deformados)
        return volume(self.aresta)

    def altura(self) -> float:
        if self._vertices_deformados is not None:
            return altura_generica(self._vertices_deformados)
        return altura(self.aresta)

    def apotema_face(self) -> float:
        if self._vertices_deformados is not None:
            return apotema_face_generica(self._vertices_deformados, FACES[0])
        return apotema_face(self.aresta)

    def comprimentos_arestas(self) -> list[float]:
        return comprimentos_arestas(self.vertices())

    def aresta_media(self) -> float:
        medidas = self.comprimentos_arestas()
        return sum(medidas) / len(medidas)

    def centroide(self) -> Vec3:
        return centroide_solido(self.vertices())

    # --- modificação --------------------------------------------------------
    def ajustar_aresta(self, delta: float) -> None:
        """Soma `delta` à aresta, com clamp em [aresta_min, aresta_max].

        Deformado, `aresta` funciona como alça de escala: a razão entre o valor
        novo e o antigo vira um fator aplicado em torno do centroide, mantendo
        a forma deformada e apenas mudando o tamanho.
        """
        anterior = self.aresta
        self.aresta = min(max(anterior + delta, self.aresta_min), self.aresta_max)
        if self._vertices_deformados is not None and anterior > 0:
            fator = self.aresta / anterior
            if fator != 1.0:
                self._vertices_deformados = escalar_em_torno_do_centroide(
                    self._vertices_deformados, fator
                )

    def escalar(self, fator: float) -> None:
        """Escala uniforme em torno do centroide (gesto de duas mãos).

        Respeita [aresta_min, aresta_max] através da aresta média, para que o
        gesto nunca faça o sólido sumir nem estourar a tela.
        """
        if fator <= 0:
            raise ValueError(f"fator de escala deve ser positivo, recebido: {fator!r}")
        alvo = min(max(self.aresta * fator, self.aresta_min), self.aresta_max)
        fator_efetivo = alvo / self.aresta if self.aresta > 0 else 1.0
        self.aresta = alvo
        if self._vertices_deformados is not None and fator_efetivo != 1.0:
            self._vertices_deformados = escalar_em_torno_do_centroide(
                self._vertices_deformados, fator_efetivo
            )

    def mover_vertice(self, indice: int, nova_posicao: Vec3) -> bool:
        """Move um vértice e entra em estado DEFORMADO.

        Retorna False sem alterar nada se o movimento achataria o sólido a
        ponto de deixá-lo degenerado (volume abaixo de `VOLUME_MINIMO_RELATIVO`
        do volume regular equivalente) — evita normais indefinidas no render.
        """
        if not 0 <= indice < 4:
            raise IndexError(f"índice de vértice fora do intervalo 0..3: {indice}")

        candidatos = self.vertices()
        candidatos[indice] = tuple(float(c) for c in nova_posicao)  # type: ignore[assignment]

        if volume_generico(candidatos) < volume(self.aresta) * self.VOLUME_MINIMO_RELATIVO:
            return False

        self._vertices_deformados = candidatos
        return True

    def resetar(self, aresta_padrao: float | None = None) -> None:
        """Volta ao tetraedro regular, descartando qualquer deformação."""
        self.aresta = aresta_padrao if aresta_padrao is not None else self._aresta_padrao
        self._vertices_deformados = None
