"""Janela OpenGL e loop de renderização do tetraedro.

A entrada (mouse/teclado nesta etapa, gestos de mão na Etapa 2) é
abstraída pelo protocolo `FonteEntrada`, para que o `Viewer` nunca
precise mudar quando a fonte de controle trocar.
"""
from dataclasses import dataclass
from math import radians
from typing import Any, Optional, Protocol

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
    GL_SMOOTH,
    GL_SRC_ALPHA,
    GL_TRIANGLES,
    GL_AMBIENT,
    GL_DIFFUSE,
    GL_POSITION,
    glBegin,
    glBlendFunc,
    glClear,
    glClearColor,
    glColor3f,
    glColor4f,
    glColorMaterial,
    glDepthMask,
    glDisable,
    glEnable,
    glEnd,
    glLightfv,
    glLineWidth,
    glLoadIdentity,
    glMatrixMode,
    glNormal3f,
    glPopMatrix,
    glPushMatrix,
    glMultMatrixf,
    glShadeModel,
    glTranslatef,
    glVertex3f,
    glViewport,
)
from OpenGL.GL import GL_COLOR_BUFFER_BIT, GL_DEPTH_BUFFER_BIT
from OpenGL.GLU import gluDeleteQuadric, gluNewQuadric, gluPerspective, gluSphere

from config import Config
from geometry.tetraedro import Tetraedro, calcular_normal_face
from render.ar import FundoCamera
from render.hud import AMBAR, BRANCO, VERDE, ContextoOrtografico, PainelHUD
from render.projecao import ProjetorOpenGL
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
    """O que a fonte de entrada observou neste frame.

    Campos das Etapas 1 e 2 são deltas de controle. Os da Etapa 3 são
    *sensoriais*: descrevem o que as mãos e a câmera estão fazendo, sem
    interpretar. A interpretação (qual vértice está sob a mão, o que o gesto
    significa) depende da cena e por isso mora no Viewer, não aqui.

    Todos os campos novos têm default inerte, então `FonteEntradaMouseTeclado`
    continua válida sem alteração.
    """

    delta_rotacao_x: float = 0.0
    delta_rotacao_y: float = 0.0
    delta_zoom: float = 0.0
    delta_aresta: float = 0.0
    resetar: bool = False
    sair: bool = False
    redimensionado: Optional[tuple[int, int]] = None

    # --- Etapa 3 ---
    frame_camera: Any = None                 # np.ndarray BGR já espelhado, ou None
    estado_camera: Optional[str] = None      # "conectado" | "reconectando" | "desconectado"
    diagnostico: tuple[str, ...] = ()        # linhas extras para o HUD de depuração


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

        self._projetor = ProjetorOpenGL()
        self._fundo = FundoCamera(
            modo_ajuste=self.config.AR_MODO_AJUSTE,
            escurecimento=self.config.AR_ESCURECIMENTO_FUNDO,
        )
        self._hud: Optional[PainelHUD] = None
        self._estado_camera: Optional[str] = None
        self._diagnostico: tuple[str, ...] = ()

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
            # Libera as texturas ANTES de destruir o contexto GL.
            self._fundo.close()
            if self._hud is not None:
                self._hud.close()
            pygame.quit()

    def _configurar_janela(self) -> None:
        pygame.init()
        pygame.display.set_mode(
            (self.largura, self.altura),
            pygame.DOUBLEBUF | pygame.OPENGL | pygame.RESIZABLE,
        )
        pygame.display.set_caption(self.config.TITULO_JANELA)
        self._fonte_texto = pygame.font.Font(None, 22)
        self._hud = PainelHUD(self._fonte_texto)
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

        if self.config.AR_ATIVO:
            self._fundo.atualizar(estado.frame_camera)
        self._estado_camera = estado.estado_camera
        self._diagnostico = estado.diagnostico

    def _renderizar_frame(self) -> None:
        # O fundo limpa a tela (com o vídeo ou com a cor sólida) e o sólido
        # vem por cima, em projeção perspectiva.
        if self.config.AR_ATIVO:
            self._fundo.desenhar(self.largura, self.altura)
        else:
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        self._desenhar_solido()
        self._desenhar_overlay()

    def _desenhar_solido(self) -> None:
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        glTranslatef(0.0, 0.0, -self.distancia_camera)
        glMultMatrixf(matriz4_coluna_maior_de_quaternion(self.orientacao))
        # Captura as matrizes com a modelview do SÓLIDO ativa: é o que torna a
        # seleção de vértice possível (fatia 3E).
        self._projetor.capturar()

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

    def _linhas_hud(self) -> list[tuple[str, tuple[int, int, int]]]:
        """Texto do painel. Quando o sólido está deformado não existe uma
        "aresta a" única, então mostra a média com o intervalo — escrever
        "Aresta: 1.500" num sólido de arestas 1.5 a 2.4 seria mentir."""
        t = self.tetraedro
        if t.esta_regular():
            linhas = [(f"Aresta (a): {t.aresta:.3f}", BRANCO)]
        else:
            medidas = t.comprimentos_arestas()
            linhas = [
                (f"Aresta média: {t.aresta_media():.3f}"
                 f"  (min {min(medidas):.3f} / max {max(medidas):.3f})", AMBAR),
                ("DEFORMADO — fórmulas do tetraedro regular não valem", AMBAR),
            ]
        linhas += [
            (f"Área total: {t.area_total():.3f}", BRANCO),
            (f"Volume: {t.volume():.3f}", BRANCO),
            (f"Altura: {t.altura():.3f}", BRANCO),
            (f"Apótema da face: {t.apotema_face():.3f}", BRANCO),
        ]
        if self._estado_camera is not None:
            cor = VERDE if self._estado_camera == "conectado" else AMBAR
            linhas.append((f"Câmera: {self._estado_camera}", cor))
        linhas += [(texto, BRANCO) for texto in self._diagnostico]
        return linhas

    def _desenhar_overlay(self) -> None:
        if self._hud is None:
            return
        with ContextoOrtografico(self.largura, self.altura):
            self._hud.desenhar(self._linhas_hud())
