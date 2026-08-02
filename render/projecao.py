"""Ponte entre o espaço 3D do sólido e os pixels da janela.

Serve à seleção de vértice: para saber se a mão está "em cima" de um vértice,
é preciso projetar os 4 vértices para a tela; para arrastar, é preciso o
caminho inverso.

Toda a convenção do OpenGL fica presa aqui dentro. Em particular, `gluProject`
devolve y com origem no canto INFERIOR esquerdo, enquanto pygame, o HUD e o
fundo AR usam origem no canto SUPERIOR esquerdo. A conversão acontece nesta
classe e em nenhum outro lugar.
"""
from typing import Optional, Sequence

from OpenGL.GL import (
    GL_MODELVIEW_MATRIX,
    GL_PROJECTION_MATRIX,
    GL_VIEWPORT,
    glGetDoublev,
    glGetIntegerv,
)
from OpenGL.GLU import gluProject, gluUnProject

Vec3 = tuple[float, float, float]


class ProjetorOpenGL:
    """Captura as matrizes do pipeline fixo e projeta/desprojeta pontos.

    `capturar()` precisa ser chamado com a MESMA modelview usada para desenhar
    o sólido — na prática, logo depois de montá-la e antes de desenhar. Se for
    chamado com outra matriz ativa (a do HUD, por exemplo), a projeção aponta
    para o lugar errado e a seleção de vértice erra o alvo.
    """

    def __init__(self) -> None:
        self._modelview = None
        self._projecao = None
        self._viewport = None

    def capturar(self) -> None:
        self._modelview = glGetDoublev(GL_MODELVIEW_MATRIX)
        self._projecao = glGetDoublev(GL_PROJECTION_MATRIX)
        self._viewport = glGetIntegerv(GL_VIEWPORT)

    @property
    def pronto(self) -> bool:
        return self._viewport is not None

    @property
    def altura_viewport(self) -> int:
        return int(self._viewport[3]) if self._viewport is not None else 0

    def projetar(self, ponto: Vec3) -> Optional[tuple[float, float, float]]:
        """Ponto do espaço do objeto -> (x_px, y_px, z_janela).

        `y_px` já vem na convenção do pygame (origem em cima). `z_janela` está
        em [0,1] e é guardado no início de um arrasto para que o vértice seja
        movido num plano paralelo à câmera, preservando a profundidade.
        Retorna None se as matrizes ainda não foram capturadas.
        """
        if not self.pronto:
            return None
        x, y, z = gluProject(ponto[0], ponto[1], ponto[2], self._modelview, self._projecao, self._viewport)
        return (float(x), float(self.altura_viewport - y), float(z))

    def desprojetar(self, x_px: float, y_px: float, z_janela: float) -> Optional[Vec3]:
        """(x_px, y_px, z_janela) -> ponto no espaço do objeto.

        Inverso exato de `projetar` — `y_px` é interpretado na convenção do
        pygame, como o que `projetar` devolve.
        """
        if not self.pronto:
            return None
        x, y, z = gluUnProject(
            x_px, self.altura_viewport - y_px, z_janela,
            self._modelview, self._projecao, self._viewport,
        )
        return (float(x), float(y), float(z))

    def projetar_varios(self, pontos: Sequence[Vec3]) -> list[Optional[tuple[float, float, float]]]:
        return [self.projetar(p) for p in pontos]
