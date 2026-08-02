"""Geometria do tetraedro regular: vértices, topologia e fórmulas.

Os 4 vértices são construídos a partir de cantos alternados de um cubo
centrado na origem — uma forma simples e numericamente estável de obter
um tetraedro regular sem trigonometria, já centrado no baricentro.
"""
from dataclasses import dataclass, field
from math import sqrt

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


@dataclass
class Tetraedro:
    """Estado do tetraedro exibido — única fonte de verdade da aresta atual.

    O viewer (e, futuramente, o hand tracker) sempre consulta/ajusta a
    aresta através deste objeto, nunca guardando o valor em variável solta.
    """

    aresta: float = 1.5
    aresta_min: float = 0.3
    aresta_max: float = 4.0
    _aresta_padrao: float = field(init=False, repr=False, default=0.0)

    def __post_init__(self) -> None:
        _validar_aresta(self.aresta)
        self._aresta_padrao = self.aresta

    def vertices(self) -> list[tuple[float, float, float]]:
        return calcular_vertices(self.aresta)

    def faces(self) -> tuple[tuple[int, int, int], ...]:
        return FACES

    def arestas(self) -> tuple[tuple[int, int], ...]:
        return ARESTAS_INDICES

    def area_total(self) -> float:
        return area_total(self.aresta)

    def volume(self) -> float:
        return volume(self.aresta)

    def altura(self) -> float:
        return altura(self.aresta)

    def apotema_face(self) -> float:
        return apotema_face(self.aresta)

    def ajustar_aresta(self, delta: float) -> None:
        """Soma `delta` à aresta, com clamp em [aresta_min, aresta_max].

        Mesma assinatura que será chamada tanto por teclado (etapa 1)
        quanto pela abertura da mão (etapa 2).
        """
        nova_aresta = self.aresta + delta
        self.aresta = min(max(nova_aresta, self.aresta_min), self.aresta_max)

    def resetar(self, aresta_padrao: float | None = None) -> None:
        self.aresta = aresta_padrao if aresta_padrao is not None else self._aresta_padrao
