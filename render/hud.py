"""Sobreposição 2D: valores geométricos, estado do gesto e cursores das mãos.

O HUD da Etapa 1 criava e destruía uma textura de OpenGL por linha de texto
por frame. Aqui as linhas são compostas numa única superfície, enviada como
uma única textura, e só reenviada quando o texto realmente muda — o que
importa porque sobre o vídeo o painel precisa de um fundo opaco, e portanto
ficou maior.
"""
from typing import Optional, Sequence

import pygame
from OpenGL.GL import (
    GL_BLEND,
    GL_CLAMP_TO_EDGE,
    GL_DEPTH_TEST,
    GL_LIGHTING,
    GL_LINEAR,
    GL_MODELVIEW,
    GL_ONE_MINUS_SRC_ALPHA,
    GL_PROJECTION,
    GL_RGBA,
    GL_SRC_ALPHA,
    GL_TEXTURE_2D,
    GL_TEXTURE_MAG_FILTER,
    GL_TEXTURE_MIN_FILTER,
    GL_TEXTURE_WRAP_S,
    GL_TEXTURE_WRAP_T,
    GL_TRIANGLE_FAN,
    GL_TRIANGLES,
    GL_UNPACK_ALIGNMENT,
    GL_UNSIGNED_BYTE,
    glBegin,
    glBindTexture,
    glBlendFunc,
    glColor4f,
    glDeleteTextures,
    glDisable,
    glEnable,
    glEnd,
    glGenTextures,
    glLineWidth,
    glLoadIdentity,
    glMatrixMode,
    glOrtho,
    glPixelStorei,
    glPopMatrix,
    glPushMatrix,
    glTexCoord2f,
    glTexImage2D,
    glTexParameteri,
    glVertex2f,
)

BRANCO = (235, 238, 245)
VERDE = (90, 230, 140)
AMBAR = (255, 200, 60)


class ContextoOrtografico:
    """Entra e sai da projeção 2D (origem no topo-esquerdo, y para baixo).

    Usado como gerenciador de contexto para que nenhum caminho de erro deixe a
    matriz de projeção empilhada — o que corromperia o frame seguinte.
    """

    def __init__(self, largura: int, altura: int) -> None:
        self.largura = largura
        self.altura = altura

    def __enter__(self) -> "ContextoOrtografico":
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        glOrtho(0, self.largura, self.altura, 0, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()
        glDisable(GL_DEPTH_TEST)
        glDisable(GL_LIGHTING)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        return self

    def __exit__(self, *_) -> None:
        glDisable(GL_BLEND)
        glEnable(GL_LIGHTING)
        glEnable(GL_DEPTH_TEST)
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)
        glPopMatrix()


class PainelHUD:
    """Painel de texto com fundo, desenhado como uma textura só."""

    def __init__(
        self,
        fonte: pygame.font.Font,
        margem: int = 10,
        respiro: int = 8,
        altura_linha: int = 24,
        opacidade_fundo: int = 150,
    ) -> None:
        self.fonte = fonte
        self.margem = margem
        self.respiro = respiro
        self.altura_linha = altura_linha
        self.opacidade_fundo = opacidade_fundo
        self._id_textura: Optional[int] = None
        self._tamanho = (0, 0)
        self._chave_cache: Optional[tuple] = None

    def _reconstruir(self, linhas: Sequence[tuple[str, tuple[int, int, int]]]) -> None:
        larguras = [self.fonte.size(texto)[0] for texto, _ in linhas] or [0]
        largura = max(larguras) + self.respiro * 2
        altura = self.altura_linha * len(linhas) + self.respiro * 2

        superficie = pygame.Surface((largura, altura), pygame.SRCALPHA)
        superficie.fill((10, 12, 18, self.opacidade_fundo))
        for i, (texto, cor) in enumerate(linhas):
            superficie.blit(
                self.fonte.render(texto, True, cor),
                (self.respiro, self.respiro + i * self.altura_linha),
            )

        # flipped=False: v=0 fica no topo, combinando com a projeção ortográfica
        dados = pygame.image.tostring(superficie, "RGBA", False)
        if self._id_textura is None:
            self._id_textura = int(glGenTextures(1))
        glBindTexture(GL_TEXTURE_2D, self._id_textura)
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, largura, altura, 0, GL_RGBA, GL_UNSIGNED_BYTE, dados)
        self._tamanho = (largura, altura)

    def desenhar(self, linhas: Sequence[tuple[str, tuple[int, int, int]]]) -> None:
        """Desenha o painel. Deve ser chamado dentro de um ContextoOrtografico."""
        if not linhas:
            return
        chave = tuple(linhas)
        if chave != self._chave_cache:
            self._reconstruir(linhas)
            self._chave_cache = chave

        largura, altura = self._tamanho
        x, y = self.margem, self.margem
        glEnable(GL_TEXTURE_2D)
        glBindTexture(GL_TEXTURE_2D, self._id_textura)
        glColor4f(1.0, 1.0, 1.0, 1.0)
        glBegin(GL_TRIANGLES)
        for u, v, vx, vy in (
            (0, 0, x, y), (1, 0, x + largura, y), (1, 1, x + largura, y + altura),
            (0, 0, x, y), (1, 1, x + largura, y + altura), (0, 1, x, y + altura),
        ):
            glTexCoord2f(u, v)
            glVertex2f(vx, vy)
        glEnd()
        glDisable(GL_TEXTURE_2D)

    def close(self) -> None:
        if self._id_textura is not None:
            try:
                glDeleteTextures([self._id_textura])
            except Exception:
                pass  # contexto GL já pode ter sido destruído
            self._id_textura = None
        self._chave_cache = None


def desenhar_circulo(x: float, y: float, raio: float, cor: tuple[float, float, float], alfa: float = 1.0,
                     segmentos: int = 24, preenchido: bool = True) -> None:
    """Marcador circular em coordenadas de tela. Dentro de ContextoOrtografico."""
    import math

    glColor4f(cor[0], cor[1], cor[2], alfa)
    if preenchido:
        glBegin(GL_TRIANGLE_FAN)
        glVertex2f(x, y)
    else:
        glBegin(GL_TRIANGLE_FAN)
    for i in range(segmentos + 1):
        angulo = 2.0 * math.pi * i / segmentos
        glVertex2f(x + raio * math.cos(angulo), y + raio * math.sin(angulo))
    glEnd()


def desenhar_anel(x: float, y: float, raio: float, cor: tuple[float, float, float],
                  alfa: float = 1.0, espessura: float = 2.0, segmentos: int = 24) -> None:
    """Contorno circular — usado para o cursor da mão quando NÃO está agarrando."""
    import math

    from OpenGL.GL import GL_LINE_LOOP

    glColor4f(cor[0], cor[1], cor[2], alfa)
    glLineWidth(espessura)
    glBegin(GL_LINE_LOOP)
    for i in range(segmentos):
        angulo = 2.0 * math.pi * i / segmentos
        glVertex2f(x + raio * math.cos(angulo), y + raio * math.sin(angulo))
    glEnd()
