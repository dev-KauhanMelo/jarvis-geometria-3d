"""Composição de realidade aumentada: o frame da câmera como fundo da cena 3D.

O sólido deixa de flutuar numa tela preta e passa a aparecer sobre a imagem
ao vivo, no mesmo espaço que o usuário.

A matemática de encaixe (como o frame da câmera cabe na janela) mora em
funções puras testáveis sem OpenGL. Ela é usada em dois lugares que precisam
concordar exatamente, sob pena de o cursor da mão não coincidir com a mão no
vídeo: o desenho do fundo e a conversão das coordenadas dos landmarks para
pixels de tela.
"""
from dataclasses import dataclass
from typing import Optional

import numpy as np
from OpenGL.GL import (
    GL_BGR,
    GL_BLEND,
    GL_CLAMP_TO_EDGE,
    GL_COLOR_BUFFER_BIT,
    GL_DEPTH_BUFFER_BIT,
    GL_DEPTH_TEST,
    GL_LIGHTING,
    GL_LINEAR,
    GL_MODELVIEW,
    GL_ONE_MINUS_SRC_ALPHA,
    GL_PROJECTION,
    GL_RGB,
    GL_SRC_ALPHA,
    GL_TEXTURE_2D,
    GL_TEXTURE_MAG_FILTER,
    GL_TEXTURE_MIN_FILTER,
    GL_TEXTURE_WRAP_S,
    GL_TEXTURE_WRAP_T,
    GL_TRIANGLES,
    GL_UNPACK_ALIGNMENT,
    GL_UNSIGNED_BYTE,
    glBegin,
    glBindTexture,
    glBlendFunc,
    glClear,
    glClearColor,
    glColor4f,
    glDeleteTextures,
    glDisable,
    glEnable,
    glEnd,
    glGenTextures,
    glLoadIdentity,
    glMatrixMode,
    glOrtho,
    glPixelStorei,
    glPopMatrix,
    glPushMatrix,
    glTexCoord2f,
    glTexImage2D,
    glTexParameteri,
    glTexSubImage2D,
    glVertex2f,
)

MODO_PREENCHER = "preencher"  # cobre a janela inteira, corta o excesso do frame
MODO_CABER = "caber"          # mostra o frame inteiro, com barras pretas


@dataclass(frozen=True)
class AjusteFundo:
    """Como o frame da câmera se encaixa na janela.

    - (u0, v0)-(u1, v1): sub-retângulo da TEXTURA que será exibido, em [0,1].
    - (x0, y0)-(x1, y1): retângulo em PIXELS da janela onde ele é desenhado,
      na convenção do pygame (origem no canto superior esquerdo, y cresce
      para baixo) — a mesma usada pelo HUD e pelo projetor.
    """

    u0: float
    v0: float
    u1: float
    v1: float
    x0: float
    y0: float
    x1: float
    y1: float


def calcular_ajuste(
    largura_frame: int,
    altura_frame: int,
    largura_janela: int,
    altura_janela: int,
    modo: str = MODO_PREENCHER,
) -> AjusteFundo:
    """Encaixa um frame de câmera numa janela sem distorcer a imagem.

    `MODO_PREENCHER` mantém a janela toda coberta cortando as bordas do frame
    (padrão: em AR, barra preta quebra a ilusão de que o objeto está no
    ambiente). `MODO_CABER` mostra o frame inteiro com barras.
    """
    if largura_frame <= 0 or altura_frame <= 0:
        raise ValueError(f"dimensões de frame inválidas: {largura_frame}x{altura_frame}")
    if largura_janela <= 0 or altura_janela <= 0:
        raise ValueError(f"dimensões de janela inválidas: {largura_janela}x{altura_janela}")

    aspecto_frame = largura_frame / altura_frame
    aspecto_janela = largura_janela / altura_janela

    if modo == MODO_PREENCHER:
        if aspecto_frame > aspecto_janela:
            # frame mais largo que a janela: corta laterais
            fracao = aspecto_janela / aspecto_frame
            margem = (1.0 - fracao) / 2.0
            u0, u1, v0, v1 = margem, 1.0 - margem, 0.0, 1.0
        else:
            fracao = aspecto_frame / aspecto_janela
            margem = (1.0 - fracao) / 2.0
            u0, u1, v0, v1 = 0.0, 1.0, margem, 1.0 - margem
        return AjusteFundo(u0, v0, u1, v1, 0.0, 0.0, float(largura_janela), float(altura_janela))

    if modo == MODO_CABER:
        if aspecto_frame > aspecto_janela:
            largura = float(largura_janela)
            altura = largura / aspecto_frame
        else:
            altura = float(altura_janela)
            largura = altura * aspecto_frame
        x0 = (largura_janela - largura) / 2.0
        y0 = (altura_janela - altura) / 2.0
        return AjusteFundo(0.0, 0.0, 1.0, 1.0, x0, y0, x0 + largura, y0 + altura)

    raise ValueError(f"modo de ajuste desconhecido: {modo!r}")


def mapear_uv_para_tela(u: float, v: float, ajuste: AjusteFundo) -> tuple[float, float]:
    """Converte coordenada normalizada do frame da câmera em pixel da janela.

    É o elo que faz o cursor da mão cair exatamente sobre a mão no vídeo: os
    landmarks do MediaPipe vêm em [0,1] no espaço do frame, e precisam sofrer
    o mesmo recorte/escala que o fundo sofreu. Pontos cortados pelo modo
    "preencher" caem fora do retângulo da janela, o que é o correto.
    """
    largura_uv = ajuste.u1 - ajuste.u0
    altura_uv = ajuste.v1 - ajuste.v0
    if largura_uv <= 0 or altura_uv <= 0:
        raise ValueError("ajuste com sub-retângulo de textura degenerado")
    x = ajuste.x0 + (u - ajuste.u0) / largura_uv * (ajuste.x1 - ajuste.x0)
    y = ajuste.y0 + (v - ajuste.v0) / altura_uv * (ajuste.y1 - ajuste.y0)
    return (x, y)


class FundoCamera:
    """Desenha o frame da câmera como fundo, em projeção ortográfica.

    A textura é alocada uma vez e atualizada com `glTexSubImage2D` (medi
    1,18 ms por frame a 640x480 — irrelevante perto dos 16 ms de um frame a
    60fps). O upload usa `GL_BGR` diretamente, dispensando a conversão de cor
    que o OpenCV exigiria.
    """

    def __init__(self, modo_ajuste: str = MODO_PREENCHER, escurecimento: float = 0.35) -> None:
        self.modo_ajuste = modo_ajuste
        self.escurecimento = escurecimento
        self._id_textura: Optional[int] = None
        self._largura_textura = 0
        self._altura_textura = 0
        self._ajuste: Optional[AjusteFundo] = None
        self.tem_frame = False

    def _garantir_textura(self, largura: int, altura: int) -> None:
        if self._id_textura is not None and (largura, altura) == (
            self._largura_textura,
            self._altura_textura,
        ):
            return
        if self._id_textura is not None:
            glDeleteTextures([self._id_textura])
        self._id_textura = int(glGenTextures(1))
        glBindTexture(GL_TEXTURE_2D, self._id_textura)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        # CLAMP_TO_EDGE evita que o filtro linear puxe a borda oposta da imagem
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, largura, altura, 0, GL_BGR, GL_UNSIGNED_BYTE, None)
        self._largura_textura, self._altura_textura = largura, altura

    def atualizar(self, frame: Optional[np.ndarray]) -> None:
        """Sobe o frame mais recente para a GPU. `None` mantém o último."""
        if frame is None or frame.size == 0:
            return
        altura, largura = frame.shape[:2]
        self._garantir_textura(largura, altura)
        glBindTexture(GL_TEXTURE_2D, self._id_textura)
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1)  # linhas não são múltiplas de 4
        glTexSubImage2D(
            GL_TEXTURE_2D, 0, 0, 0, largura, altura,
            GL_BGR, GL_UNSIGNED_BYTE, np.ascontiguousarray(frame),
        )
        self.tem_frame = True

    def ajuste_atual(self, largura_janela: int, altura_janela: int) -> Optional[AjusteFundo]:
        """Encaixe vigente, ou None se nenhum frame chegou ainda."""
        if not self.tem_frame:
            return None
        return calcular_ajuste(
            self._largura_textura, self._altura_textura,
            largura_janela, altura_janela, self.modo_ajuste,
        )

    def desenhar(self, largura_janela: int, altura_janela: int) -> None:
        """Limpa a tela e desenha o vídeo ao fundo.

        Sem frame de câmera, apenas limpa com o fundo escuro da Etapa 1 — o
        sólido continua utilizável, só que sem AR.
        """
        glClearColor(0.05, 0.05, 0.08, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        if not self.tem_frame:
            self._ajuste = None
            return

        self._ajuste = calcular_ajuste(
            self._largura_textura, self._altura_textura,
            largura_janela, altura_janela, self.modo_ajuste,
        )
        a = self._ajuste

        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        glOrtho(0, largura_janela, altura_janela, 0, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()

        glDisable(GL_DEPTH_TEST)
        glDisable(GL_LIGHTING)
        glEnable(GL_TEXTURE_2D)
        glBindTexture(GL_TEXTURE_2D, self._id_textura)
        glColor4f(1.0, 1.0, 1.0, 1.0)
        _quad_texturizado(a.x0, a.y0, a.x1, a.y1, a.u0, a.v0, a.u1, a.v1)
        glDisable(GL_TEXTURE_2D)

        # Véu escuro: o sólido é semi-transparente e some sobre um ambiente
        # claro ou movimentado. Escurecer o vídeo devolve o contraste.
        if self.escurecimento > 0.0:
            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            glColor4f(0.0, 0.0, 0.0, self.escurecimento)
            _quad_simples(0, 0, largura_janela, altura_janela)
            glDisable(GL_BLEND)

        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)
        glPopMatrix()

    def close(self) -> None:
        """Libera a textura. Idempotente."""
        if self._id_textura is not None:
            try:
                glDeleteTextures([self._id_textura])
            except Exception:
                pass  # contexto GL já pode ter sido destruído
            self._id_textura = None
        self.tem_frame = False


def _quad_texturizado(x0, y0, x1, y1, u0, v0, u1, v1) -> None:
    """Dois triângulos (e não GL_QUADS, removido do core profile)."""
    glBegin(GL_TRIANGLES)
    for u, v, x, y in (
        (u0, v0, x0, y0), (u1, v0, x1, y0), (u1, v1, x1, y1),
        (u0, v0, x0, y0), (u1, v1, x1, y1), (u0, v1, x0, y1),
    ):
        glTexCoord2f(u, v)
        glVertex2f(x, y)
    glEnd()


def _quad_simples(x0, y0, x1, y1) -> None:
    glBegin(GL_TRIANGLES)
    for x, y in ((x0, y0), (x1, y0), (x1, y1), (x0, y0), (x1, y1), (x0, y1)):
        glVertex2f(x, y)
    glEnd()
