"""Álgebra de vetores, matrizes 3x3 e quaternions — 100% puro, sem OpenGL.

Convenções adotadas em todo o projeto:

- `Vec3` é uma tupla (x, y, z).
- `Mat3` é uma tupla de 3 linhas; `m[i][j]` é linha i, coluna j.
- `Quat` é (w, x, y, z), sempre unitário quando representa rotação.
- Uma matriz de rotação tem os vetores da base nas COLUNAS: ela leva
  coordenadas locais para coordenadas do mundo.

O estado de orientação do objeto é quaternion (e não ângulos de Euler)
por três motivos: renormalização barata (a rotação é multiplicada a cada
frame e uma matriz acumularia deriva de ortonormalidade), interpolação
correta via `slerp`, e ausência de gimbal lock.
"""
from math import acos, cos, exp, isclose, radians, sin, sqrt
from typing import Sequence

Vec3 = tuple[float, float, float]
Mat3 = tuple[Vec3, Vec3, Vec3]
Quat = tuple[float, float, float, float]

QUAT_IDENTIDADE: Quat = (1.0, 0.0, 0.0, 0.0)
MAT3_IDENTIDADE: Mat3 = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))

EPSILON = 1e-12


# --------------------------------------------------------------------------
# Vetores
# --------------------------------------------------------------------------
def somar(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def subtrair(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def escalar_vetor(v: Vec3, k: float) -> Vec3:
    return (v[0] * k, v[1] * k, v[2] * k)


def produto_escalar(a: Vec3, b: Vec3) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def produto_vetorial(a: Vec3, b: Vec3) -> Vec3:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def norma(v: Vec3) -> float:
    return sqrt(produto_escalar(v, v))


def normalizar(v: Vec3, epsilon: float = EPSILON) -> Vec3:
    n = norma(v)
    if n < epsilon:
        raise ValueError(f"vetor de norma quase nula não pode ser normalizado: {v!r}")
    return (v[0] / n, v[1] / n, v[2] / n)


def centroide(pontos: Sequence[Vec3]) -> Vec3:
    if not pontos:
        raise ValueError("centroide de sequência vazia")
    n = len(pontos)
    return (
        sum(p[0] for p in pontos) / n,
        sum(p[1] for p in pontos) / n,
        sum(p[2] for p in pontos) / n,
    )


def distancia(a: Vec3, b: Vec3) -> float:
    return norma(subtrair(a, b))


# --------------------------------------------------------------------------
# Matrizes 3x3
# --------------------------------------------------------------------------
def transpor(m: Mat3) -> Mat3:
    return (
        (m[0][0], m[1][0], m[2][0]),
        (m[0][1], m[1][1], m[2][1]),
        (m[0][2], m[1][2], m[2][2]),
    )


def multiplicar_matrizes(a: Mat3, b: Mat3) -> Mat3:
    return tuple(  # type: ignore[return-value]
        tuple(sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)) for i in range(3)
    )


def aplicar_matriz(m: Mat3, v: Vec3) -> Vec3:
    return (
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    )


def determinante(m: Mat3) -> float:
    return (
        m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
    )


def matriz_de_colunas(ex: Vec3, ey: Vec3, ez: Vec3) -> Mat3:
    """Monta a matriz cujas COLUNAS são ex, ey, ez (local -> mundo)."""
    return (
        (ex[0], ey[0], ez[0]),
        (ex[1], ey[1], ez[1]),
        (ex[2], ey[2], ez[2]),
    )


def base_destra(ez_bruto: Vec3, referencia_ey: Vec3) -> Mat3:
    """Base ortonormal destra (det = +1) a partir de dois vetores não paralelos.

    `ez_bruto` define o terceiro eixo; `referencia_ey` só orienta o plano —
    ele é ortogonalizado por Gram-Schmidt, não precisa ser perpendicular.
    Levanta ValueError se os vetores forem paralelos (base indefinida).
    """
    ez = normalizar(ez_bruto)
    ex = produto_vetorial(referencia_ey, ez)
    if norma(ex) < 1e-9:
        raise ValueError("vetores paralelos: base ortonormal indefinida")
    ex = normalizar(ex)
    ey = produto_vetorial(ez, ex)  # já unitário: produto de dois unitários ortogonais
    return matriz_de_colunas(ex, ey, ez)


# --------------------------------------------------------------------------
# Quaternions
# --------------------------------------------------------------------------
def normalizar_quaternion(q: Quat) -> Quat:
    n = sqrt(q[0] ** 2 + q[1] ** 2 + q[2] ** 2 + q[3] ** 2)
    if n < EPSILON:
        raise ValueError(f"quaternion de norma quase nula: {q!r}")
    return (q[0] / n, q[1] / n, q[2] / n, q[3] / n)


def conjugado(q: Quat) -> Quat:
    return (q[0], -q[1], -q[2], -q[3])


def multiplicar_quaternions(a: Quat, b: Quat) -> Quat:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    )


def aplicar_quaternion(q: Quat, v: Vec3) -> Vec3:
    """Rotaciona `v` por `q` (forma de Rodrigues, sem montar a matriz)."""
    w, x, y, z = q
    u: Vec3 = (x, y, z)
    t = escalar_vetor(produto_vetorial(u, v), 2.0)
    return somar(somar(v, escalar_vetor(t, w)), produto_vetorial(u, t))


def quaternion_de_eixo_angulo(eixo: Vec3, angulo_rad: float) -> Quat:
    ex, ey, ez = normalizar(eixo)
    meio = angulo_rad / 2.0
    s = sin(meio)
    return (cos(meio), ex * s, ey * s, ez * s)


def quaternion_de_euler_graus(angulo_x: float, angulo_y: float) -> Quat:
    """Equivalente a `glRotatef(angulo_x,1,0,0)` seguido de `glRotatef(angulo_y,0,1,0)`.

    No OpenGL as chamadas consecutivas multiplicam à direita (M = M·Rx·Ry),
    então o quaternion composto é qx·qy — usado para reproduzir exatamente
    a orientação inicial das Etapas 1 e 2 (20°, -30°).
    """
    qx = quaternion_de_eixo_angulo((1.0, 0.0, 0.0), radians(angulo_x))
    qy = quaternion_de_eixo_angulo((0.0, 1.0, 0.0), radians(angulo_y))
    return multiplicar_quaternions(qx, qy)


def matriz_de_quaternion(q: Quat) -> Mat3:
    w, x, y, z = normalizar_quaternion(q)
    return (
        (1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)),
        (2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)),
        (2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)),
    )


def quaternion_de_matriz(m: Mat3) -> Quat:
    """Converte matriz de rotação em quaternion pelo método de Shepperd.

    Escolhe o maior entre traço/m00/m11/m22 como divisor — a fórmula ingênua
    com sqrt(1+traço) perde precisão (e chega a estourar) quando a rotação se
    aproxima de 180°, o que acontece de verdade quando o usuário vira a palma
    para trás.
    """
    m00, m01, m02 = m[0]
    m10, m11, m12 = m[1]
    m20, m21, m22 = m[2]
    traco = m00 + m11 + m22

    if traco > 0.0:
        s = sqrt(traco + 1.0) * 2.0
        q = (0.25 * s, (m21 - m12) / s, (m02 - m20) / s, (m10 - m01) / s)
    elif m00 > m11 and m00 > m22:
        s = sqrt(1.0 + m00 - m11 - m22) * 2.0
        q = ((m21 - m12) / s, 0.25 * s, (m01 + m10) / s, (m02 + m20) / s)
    elif m11 > m22:
        s = sqrt(1.0 + m11 - m00 - m22) * 2.0
        q = ((m02 - m20) / s, (m01 + m10) / s, 0.25 * s, (m12 + m21) / s)
    else:
        s = sqrt(1.0 + m22 - m00 - m11) * 2.0
        q = ((m10 - m01) / s, (m02 + m20) / s, (m12 + m21) / s, 0.25 * s)
    return normalizar_quaternion(q)


def angulo_entre_quaternions(a: Quat, b: Quat) -> float:
    """Menor ângulo de rotação (radianos) que leva `a` em `b`."""
    a = normalizar_quaternion(a)
    b = normalizar_quaternion(b)
    produto = abs(sum(x * y for x, y in zip(a, b)))  # abs: q e -q são a mesma rotação
    return 2.0 * acos(min(1.0, max(-1.0, produto)))


def slerp(a: Quat, b: Quat, t: float) -> Quat:
    """Interpolação esférica de `a` para `b`, tomando sempre o caminho curto.

    Usada para interpolar a orientação entre atualizações da detecção (~19 Hz)
    no ritmo do render (60 Hz). Nunca interpolar quaternion componente a
    componente: desnormaliza e gera velocidade angular não uniforme.
    """
    a = normalizar_quaternion(a)
    b = normalizar_quaternion(b)
    t = min(1.0, max(0.0, t))

    produto = sum(x * y for x, y in zip(a, b))
    if produto < 0.0:  # hemisfério oposto: inverte b para pegar o caminho curto
        b = (-b[0], -b[1], -b[2], -b[3])
        produto = -produto

    if produto > 0.9995:  # quase paralelos: lerp evita divisão por sin(~0)
        interpolado = tuple(x + t * (y - x) for x, y in zip(a, b))
        return normalizar_quaternion(interpolado)  # type: ignore[arg-type]

    theta_0 = acos(min(1.0, max(-1.0, produto)))
    theta = theta_0 * t
    sin_theta_0 = sin(theta_0)
    peso_a = sin(theta_0 - theta) / sin_theta_0
    peso_b = sin(theta) / sin_theta_0
    return normalizar_quaternion(
        tuple(peso_a * x + peso_b * y for x, y in zip(a, b))  # type: ignore[arg-type]
    )


def matriz4_coluna_maior_de_quaternion(q: Quat) -> tuple[float, ...]:
    """16 floats prontos para `glMultMatrixf`/`glLoadMatrixf`.

    OpenGL espera COLUMN-MAJOR: os elementos 0,1,2 são a PRIMEIRA COLUNA,
    não a primeira linha. Como `Mat3` é armazenada por linhas, esta função
    já emite transposto — trocar isso é o erro clássico do pipeline fixo.
    """
    m = matriz_de_quaternion(q)
    return (
        m[0][0], m[1][0], m[2][0], 0.0,
        m[0][1], m[1][1], m[2][1], 0.0,
        m[0][2], m[1][2], m[2][2], 0.0,
        0.0, 0.0, 0.0, 1.0,
    )


# --------------------------------------------------------------------------
# Suavização
# --------------------------------------------------------------------------
def alfa_temporal(dt: float, tau: float) -> float:
    """Alfa de média móvel exponencial equivalente a uma constante de tempo `tau`.

    `alfa = 1 - exp(-dt/tau)`. Ao contrário de um alfa fixo, é independente da
    taxa de amostragem: o mesmo `tau` suaviza igual a 19 fps e a 60 fps — o que
    importa aqui, porque a detecção e o render rodam em ritmos diferentes.
    """
    if tau <= 0.0:
        return 1.0
    return 1.0 - exp(-max(dt, 0.0) / tau)


def compor_rotacao_relativa(
    orientacao_inicial_objeto: Quat,
    orientacao_inicial_mao: Quat,
    orientacao_atual_mao: Quat,
) -> Quat:
    """Aplica ao objeto a rotação que a mão sofreu desde o início do gesto.

    A multiplicação é à ESQUERDA:

        ΔR = R_mão_atual · R_mão_inicial⁻¹
        R_objeto = ΔR · R_objeto_inicial

    Mão e objeto vivem no mesmo referencial (o da câmera), então o delta tem
    de ser aplicado nesse referencial. Compondo à direita, o objeto giraria em
    torno dos próprios eixos — que apontam para qualquer lado dependendo de
    como ele já estava — e o usuário lê isso como "não obedece".
    """
    delta = multiplicar_quaternions(orientacao_atual_mao, conjugado(orientacao_inicial_mao))
    return normalizar_quaternion(multiplicar_quaternions(delta, orientacao_inicial_objeto))


def eh_ortonormal(m: Mat3, tolerancia: float = 1e-6) -> bool:
    """True se `m` for ortonormal destra (linhas unitárias, mutuamente
    ortogonais, det = +1). Usado em testes e em guardas de sanidade."""
    produto = multiplicar_matrizes(m, transpor(m))
    for i in range(3):
        for j in range(3):
            esperado = 1.0 if i == j else 0.0
            if not isclose(produto[i][j], esperado, abs_tol=tolerancia):
                return False
    return isclose(determinante(m), 1.0, abs_tol=tolerancia)
