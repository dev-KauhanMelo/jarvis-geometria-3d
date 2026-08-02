"""Janela OpenGL e loop de renderização do tetraedro.

A entrada (mouse/teclado nesta etapa, gestos de mão na Etapa 2) é
abstraída pelo protocolo `FonteEntrada`, para que o `Viewer` nunca
precise mudar quando a fonte de controle trocar.
"""
from dataclasses import dataclass
from math import radians
from typing import Optional, Protocol

import pygame
from OpenGL.GL import (
    GL_AMBIENT_AND_DIFFUSE,
    GL_BLEND,
    GL_COLOR_MATERIAL,
    GL_DEPTH_TEST,
    GL_FRONT_AND_BACK,
    GL_LIGHT0,
    GL_LIGHTING,
    GL_LINES,
    GL_MODELVIEW,
    GL_ONE_MINUS_SRC_ALPHA,
    GL_PROJECTION,
    GL_RGBA,
    GL_SMOOTH,
    GL_SRC_ALPHA,
    GL_TEXTURE_2D,
    GL_TEXTURE_MAG_FILTER,
    GL_TEXTURE_MIN_FILTER,
    GL_LINEAR,
    GL_TRIANGLES,
    GL_UNSIGNED_BYTE,
    GL_AMBIENT,
    GL_DIFFUSE,
    GL_POSITION,
    glBegin,
    glBindTexture,
    glBlendFunc,
    glClear,
    glClearColor,
    glColor3f,
    glColor4f,
    glColorMaterial,
    glDeleteTextures,
    glDepthMask,
    glDisable,
    glEnable,
    glEnd,
    glGenTextures,
    glLightfv,
    glLineWidth,
    glLoadIdentity,
    glMatrixMode,
    glNormal3f,
    glOrtho,
    glPopMatrix,
    glPushMatrix,
    glMultMatrixf,
    glShadeModel,
    glTexCoord2f,
    glTexImage2D,
    glTexParameteri,
    glTranslatef,
    glVertex2f,
    glVertex3f,
    glViewport,
)
from OpenGL.GL import GL_COLOR_BUFFER_BIT, GL_DEPTH_BUFFER_BIT
from OpenGL.GLU import gluDeleteQuadric, gluNewQuadric, gluPerspective, gluSphere

from config import Config
from geometry.tetraedro import Tetraedro, calcular_normal_face
from geometry.transformacoes import (
    QUAT_IDENTIDADE,
    Quat,
    aplicar_quaternion,
    matriz4_coluna_maior_de_quaternion,
    multiplicar_quaternions,
    normalizar_quaternion,
    quaternion_de_eixo_angulo,
    quaternion_de_euler_graus,
)


@dataclass
class EstadoEntrada:
    """Delta de controle produzido por qualquer fonte de entrada
    (mouse/teclado hoje, gestos de mão na Etapa 2)."""

    delta_rotacao_x: float = 0.0
    delta_rotacao_y: float = 0.0
    delta_zoom: float = 0.0
    delta_aresta: float = 0.0
    resetar: bool = False
    sair: bool = False
    redimensionado: Optional[tuple[int, int]] = None


class FonteEntrada(Protocol):
    def capturar(self) -> EstadoEntrada: ...


class FonteEntradaMouseTeclado:
    """Etapa 1: arrastar com o botão esquerdo do mouse gira o sólido,
    a roda do mouse controla o zoom, +/- (ou setas) mudam a aresta,
    R reseta e ESC/fechar a janela encerra o viewer.
    """

    def __init__(
        self,
        sensibilidade_rotacao: float = 0.3,
        sensibilidade_zoom_scroll: float = 0.5,
        velocidade_aresta: float = 0.02,
    ) -> None:
        self.sensibilidade_rotacao = sensibilidade_rotacao
        self.sensibilidade_zoom_scroll = sensibilidade_zoom_scroll
        self.velocidade_aresta = velocidade_aresta

    def capturar(self) -> EstadoEntrada:
        estado = EstadoEntrada()

        for evento in pygame.event.get():
            if evento.type == pygame.QUIT:
                estado.sair = True
            elif evento.type == pygame.KEYDOWN:
                if evento.key == pygame.K_ESCAPE:
                    estado.sair = True
                elif evento.key == pygame.K_r:
                    estado.resetar = True
            elif evento.type == pygame.VIDEORESIZE:
                estado.redimensionado = (evento.w, evento.h)
            elif evento.type == pygame.MOUSEWHEEL:
                estado.delta_zoom -= evento.y * self.sensibilidade_zoom_scroll

        deslocamento_x, deslocamento_y = pygame.mouse.get_rel()
        if pygame.mouse.get_pressed()[0]:
            estado.delta_rotacao_y += deslocamento_x * self.sensibilidade_rotacao
            estado.delta_rotacao_x += deslocamento_y * self.sensibilidade_rotacao

        teclas = pygame.key.get_pressed()
        if teclas[pygame.K_PLUS] or teclas[pygame.K_EQUALS] or teclas[pygame.K_KP_PLUS] or teclas[pygame.K_UP]:
            estado.delta_aresta += self.velocidade_aresta
        if teclas[pygame.K_MINUS] or teclas[pygame.K_KP_MINUS] or teclas[pygame.K_DOWN]:
            estado.delta_aresta -= self.velocidade_aresta

        return estado


def _quaternion_de_arrasto(delta_x: float, delta_y: float) -> Quat:
    """Converte um arrasto na tela (graus) numa rotação de trackball.

    Arrastar na horizontal gira em torno do eixo Y da TELA e na vertical em
    torno do X da TELA — por isso o quaternion resultante é composto à
    esquerda da orientação atual (mesma matemática do gesto da mão).
    """
    q = QUAT_IDENTIDADE
    if delta_y:
        q = multiplicar_quaternions(quaternion_de_eixo_angulo((1.0, 0.0, 0.0), radians(delta_y)), q)
    if delta_x:
        q = multiplicar_quaternions(quaternion_de_eixo_angulo((0.0, 1.0, 0.0), radians(delta_x)), q)
    return q


class Viewer:
    """Janela 3D do tetraedro. `executar()` roda o ciclo completo
    (init -> loop -> cleanup) e retorna normalmente ao chamador quando
    o usuário sai (ESC/fechar janela) — nunca chama sys.exit(), para
    permitir reabrir a janela depois (Etapa 3: ciclo ouvir <-> ativo).
    """

    def __init__(
        self,
        tetraedro: Tetraedro,
        fonte_entrada: Optional[FonteEntrada] = None,
        config: Optional[Config] = None,
    ) -> None:
        self.tetraedro = tetraedro
        self.config = config or Config()
        self.fonte_entrada = fonte_entrada or FonteEntradaMouseTeclado()

        # Orientação como quaternion (e não ângulos de Euler): renormalização
        # barata a cada frame, interpolação por slerp e sem gimbal lock.
        self.orientacao_inicial: Quat = quaternion_de_euler_graus(20.0, -30.0)
        self.orientacao: Quat = self.orientacao_inicial
        self.distancia_inicial = 6.0
        self.distancia_camera = self.distancia_inicial
        self.distancia_min = 2.0
        self.distancia_max = 20.0

        self.largura = self.config.LARGURA_JANELA
        self.altura = self.config.ALTURA_JANELA
        self._fonte_texto = None
        self._deve_sair = False

    def executar(self) -> None:
        self._configurar_janela()
        self._configurar_opengl()
        self._deve_sair = False
        relogio = pygame.time.Clock()
        try:
            while not self._deve_sair:
                estado = self.fonte_entrada.capturar()
                self._processar_entrada(estado)
                self._renderizar_frame()
                pygame.display.flip()
                relogio.tick(self.config.FPS_ALVO)
        finally:
            pygame.quit()

    def _configurar_janela(self) -> None:
        pygame.init()
        pygame.display.set_mode(
            (self.largura, self.altura),
            pygame.DOUBLEBUF | pygame.OPENGL | pygame.RESIZABLE,
        )
        pygame.display.set_caption(self.config.TITULO_JANELA)
        self._fonte_texto = pygame.font.Font(None, 22)
        pygame.mouse.get_rel()  # descarta deslocamento acumulado antes do loop

    def _configurar_opengl(self) -> None:
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glEnable(GL_COLOR_MATERIAL)
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
        glLightfv(GL_LIGHT0, GL_POSITION, (2.0, 3.0, 4.0, 0.0))
        glLightfv(GL_LIGHT0, GL_DIFFUSE, (1.0, 1.0, 1.0, 1.0))
        glLightfv(GL_LIGHT0, GL_AMBIENT, (0.25, 0.25, 0.25, 1.0))
        glShadeModel(GL_SMOOTH)
        glClearColor(0.05, 0.05, 0.08, 1.0)
        self._tratar_redimensionamento(self.largura, self.altura)

    def _tratar_redimensionamento(self, largura: int, altura: int) -> None:
        self.largura, self.altura = largura, max(altura, 1)
        glViewport(0, 0, self.largura, self.altura)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45.0, self.largura / self.altura, 0.1, 100.0)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

    def _processar_entrada(self, estado: EstadoEntrada) -> None:
        if estado.redimensionado is not None:
            self._tratar_redimensionamento(*estado.redimensionado)

        if estado.delta_rotacao_x or estado.delta_rotacao_y:
            arrasto = _quaternion_de_arrasto(estado.delta_rotacao_y, estado.delta_rotacao_x)
            self.orientacao = normalizar_quaternion(
                multiplicar_quaternions(arrasto, self.orientacao)
            )

        self.distancia_camera = max(
            self.distancia_min, min(self.distancia_max, self.distancia_camera + estado.delta_zoom)
        )
        if estado.delta_aresta:
            self.tetraedro.ajustar_aresta(estado.delta_aresta)

        if estado.resetar:
            self.tetraedro.resetar()
            self.orientacao = self.orientacao_inicial
            self.distancia_camera = self.distancia_inicial

        if estado.sair:
            self._deve_sair = True

    def _renderizar_frame(self) -> None:
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        self._desenhar_solido()
        self._desenhar_overlay()

    def _desenhar_solido(self) -> None:
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        glTranslatef(0.0, 0.0, -self.distancia_camera)
        glMultMatrixf(matriz4_coluna_maior_de_quaternion(self.orientacao))

        vertices = self.tetraedro.vertices()
        faces = self.tetraedro.faces()
        arestas = self.tetraedro.arestas()

        def profundidade_face(face: tuple[int, int, int]) -> float:
            # A translação da câmera é uniforme, então não altera a ordem
            # relativa em z — basta rotacionar o centroide da face.
            cx = sum(vertices[i][0] for i in face) / 3
            cy = sum(vertices[i][1] for i in face) / 3
            cz = sum(vertices[i][2] for i in face) / 3
            return aplicar_quaternion(self.orientacao, (cx, cy, cz))[2]

        faces_ordenadas = sorted(faces, key=profundidade_face)

        # Faces semi-transparentes: algoritmo do pintor (mais distantes primeiro),
        # com depth write desligado para evitar artefatos de blending.
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glDepthMask(False)
        glColor4f(0.25, 0.55, 1.0, 0.45)
        for face in faces_ordenadas:
            normal = calcular_normal_face(vertices, face)
            glNormal3f(*normal)
            glBegin(GL_TRIANGLES)
            for indice in face:
                glVertex3f(*vertices[indice])
            glEnd()
        glDepthMask(True)
        glDisable(GL_BLEND)

        # Arestas destacadas por cima das faces.
        glDisable(GL_LIGHTING)
        glColor3f(1.0, 1.0, 1.0)
        glLineWidth(3.0)
        glBegin(GL_LINES)
        for i, j in arestas:
            glVertex3f(*vertices[i])
            glVertex3f(*vertices[j])
        glEnd()

        # Vértices marcados como pequenas esferas.
        glColor3f(1.0, 0.85, 0.2)
        quadrica = gluNewQuadric()
        raio_vertice = max(0.03, self.tetraedro.aresta * 0.02)
        for vertice in vertices:
            glPushMatrix()
            glTranslatef(*vertice)
            gluSphere(quadrica, raio_vertice, 12, 12)
            glPopMatrix()
        gluDeleteQuadric(quadrica)
        glEnable(GL_LIGHTING)

    def _desenhar_overlay(self) -> None:
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

        linhas = [
            f"Aresta (a): {self.tetraedro.aresta:.3f}",
            f"Área total: {self.tetraedro.area_total():.3f}",
            f"Volume: {self.tetraedro.volume():.3f}",
            f"Altura: {self.tetraedro.altura():.3f}",
            f"Apótema da face: {self.tetraedro.apotema_face():.3f}",
        ]
        y = 10
        for linha in linhas:
            self._desenhar_texto(linha, 10, y)
            y += 24

        glDisable(GL_BLEND)
        glEnable(GL_LIGHTING)
        glEnable(GL_DEPTH_TEST)

        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)
        glPopMatrix()

    def _desenhar_texto(self, texto: str, x: float, y: float) -> None:
        superficie = self._fonte_texto.render(texto, True, (255, 255, 255))
        largura, altura = superficie.get_size()
        # flipped=False: a textura fica com origem em cima (v=0 = topo), o que
        # já combina com os glTexCoord2f/glVertex2f usados abaixo (y cresce para baixo).
        dados = pygame.image.tostring(superficie, "RGBA", False)

        id_textura = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, id_textura)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, largura, altura, 0, GL_RGBA, GL_UNSIGNED_BYTE, dados)

        glEnable(GL_TEXTURE_2D)
        glColor4f(1.0, 1.0, 1.0, 1.0)
        glBegin(GL_TRIANGLES)
        # dois triângulos formando o quad do texto (evita depender de GL_QUADS, removido no core profile)
        for tx, ty, vx, vy in (
            (0, 0, x, y),
            (1, 0, x + largura, y),
            (1, 1, x + largura, y + altura),
            (0, 0, x, y),
            (1, 1, x + largura, y + altura),
            (0, 1, x, y + altura),
        ):
            glTexCoord2f(tx, ty)
            glVertex2f(vx, vy)
        glEnd()
        glDisable(GL_TEXTURE_2D)
        glDeleteTextures([id_textura])
