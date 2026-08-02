"""Janela OpenGL e loop de renderização do tetraedro.

A entrada (mouse/teclado nesta etapa, gestos de mão na Etapa 2) é
abstraída pelo protocolo `FonteEntrada`, para que o `Viewer` nunca
precise mudar quando a fonte de controle trocar.
"""
from dataclasses import dataclass
from math import radians, tan
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
from interaction.controlador import ControladorInteracao
from interaction.gestos import ComandoInteracao, Fase, esta_agarrando, esta_em_garra
from render.ar import FundoCamera, mapear_uv_para_tela
from render.hud import (
    AMBAR,
    BRANCO,
    VERDE,
    ContextoOrtografico,
    PainelHUD,
    desenhar_anel,
    desenhar_circulo,
)

from render.projecao import ProjetorOpenGL

# Cores do feedback de manipulação (0-1, para o OpenGL).
COR_LIVRE = (0.85, 0.88, 0.95)      # mão presente, nada agarrado
COR_AGARRANDO = (0.35, 0.95, 0.55)  # gesto ativo
COR_MIRA = (1.0, 0.78, 0.20)        # vértice que seria pego


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
    maos: tuple = ()                         # MaoDetectada do último snapshot
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

        # Mão sintética nos botões direito e do meio: exercita a máquina de
        # gestos (mira, seleção, arrasto de vértice e segurar o sólido) sem
        # câmera nenhuma. É o único caminho para testar a manipulação num
        # ambiente sem webcam.
        mao = self._mao_sintetica()
        if mao is not None:
            estado.maos = (mao,)

        return estado

    def _mao_sintetica(self):
        """Uma `MaoDetectada` plausível, posicionada no cursor do mouse.

        Botão direito = pinça (pega um vértice e deforma).
        Botão do meio = garra (segura o sólido inteiro e o arrasta).

        Os landmarks não são degenerados: montá-los com uma palma e cinco
        pontas de verdade é o que faz `abertura_mao` e `distancia_pinca`
        renderem valores realistas, e portanto o que faz este caminho
        exercitar a MESMA máquina de estados que a câmera alimenta. Só a
        orientação da palma fica de fora (o mouse não tem como expressá-la),
        então pelo mouse o sólido translada e aproxima, mas não gira.
        """
        from vision.rastreador import MaoDetectada

        botoes = pygame.mouse.get_pressed(num_buttons=3)
        garra, pinca = botoes[1], botoes[2]
        if not (garra or pinca):
            return None

        largura, altura = pygame.display.get_surface().get_size()
        x, y = pygame.mouse.get_pos()
        u, v = x / max(largura, 1), y / max(altura, 1)

        landmarks = [(u, v, 0.0)] * 21
        landmarks[0] = (u, v + 0.10, 0.0)  # pulso, abaixo das juntas
        for k, indice in enumerate((5, 9, 13, 17)):
            landmarks[indice] = (u - 0.045 + k * 0.03, v, 0.0)

        # Polegar e indicador simétricos em torno do mouse: o ponto médio
        # deles é o cursor, então ele cai exatamente onde o mouse está.
        meia_pinca = 0.0 if pinca else 0.05
        landmarks[4] = (u - meia_pinca, v, 0.0)
        landmarks[8] = (u + meia_pinca, v, 0.0)
        # Na garra os outros dedos recolhem para a palma; na pinça ficam
        # esticados, que é o que separa um gesto do outro.
        ponta_livre = (u, v + 0.02, 0.0) if garra else (u, v - 0.09, 0.0)
        for indice in (12, 16, 20):
            landmarks[indice] = ponta_livre

        return MaoDetectada(
            landmarks=tuple(landmarks), world_landmarks=(),
            destra=True, confianca_lado=1.0, id_mao=9000,
        )


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

        self.pan_x = 0.0
        self.pan_y = 0.0
        self._controlador = ControladorInteracao(self.config)
        self._comando = ComandoInteracao()
        self._frame_id_gesto = 0

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
            self.pan_x = self.pan_y = 0.0
            self._controlador.resetar()

        if estado.sair:
            self._deve_sair = True

        if self.config.AR_ATIVO:
            self._fundo.atualizar(estado.frame_camera)
        self._estado_camera = estado.estado_camera
        self._diagnostico = estado.diagnostico

        self._processar_gestos(estado)

    def _mapear_para_tela(self, u: float, v: float) -> tuple[float, float]:
        """Landmark normalizado -> pixel da janela.

        Passa pelo mesmo ajuste que o fundo AR sofreu; é o que faz o cursor
        cair exatamente sobre a mão no vídeo. Sem AR, mapeia direto na janela.
        """
        ajuste = self._fundo.ajuste_atual(self.largura, self.altura)
        if ajuste is None:
            return (u * self.largura, v * self.altura)
        return mapear_uv_para_tela(u, v, ajuste)

    def _processar_gestos(self, estado: EstadoEntrada) -> None:
        """Traduz as mãos em comandos e aplica na cena."""
        if not estado.maos:
            self._comando = ComandoInteracao()
            if self._controlador.estado.fase is not Fase.OCIOSO:
                self._controlador.resetar()
            self._controlador.maos_suaves = ()
            return

        self._frame_id_gesto += 1
        comando = self._controlador.avaliar(
            estado.maos, self._projetor, self.tetraedro, self.orientacao,
            self._mapear_para_tela, self._frame_id_gesto,
            distancia_camera=self.distancia_camera,
        )
        self._comando = comando
        self._aplicar_comando(comando)

    def _unidades_por_pixel(self) -> float:
        """Quanto vale um pixel da janela em unidades de mundo, no plano onde
        o sólido está.

        Existe para a translação ser 1:1 de verdade: o sólido acompanha a mão
        na mesma velocidade, e continua acompanhando depois de aproximar ou
        afastar. Uma constante fixa (o antigo SENSIBILIDADE_PAN_PX) só
        acertava numa distância de câmera e errava em todas as outras.
        """
        return 2.0 * self.distancia_camera * tan(radians(45.0) / 2.0) / max(self.altura, 1)

    def _aplicar_comando(self, comando: ComandoInteracao) -> None:
        if comando.orientacao_absoluta is not None:
            self.orientacao = normalizar_quaternion(comando.orientacao_absoluta)

        if comando.escala_absoluta is not None:
            atual = self.tetraedro.aresta
            if atual > 0 and comando.escala_absoluta > 0:
                self.tetraedro.escalar(comando.escala_absoluta / atual)

        px, py = comando.delta_pan_tela
        if px or py:
            escala = self._unidades_por_pixel()
            self.pan_x += px * escala
            self.pan_y -= py * escala  # y da tela cresce para baixo

        if comando.distancia_camera_absoluta is not None:
            self.distancia_camera = max(
                self.distancia_min,
                min(self.distancia_max, comando.distancia_camera_absoluta),
            )

        if comando.vertice_movido is not None and comando.posicao_vertice_objeto is not None:
            self.tetraedro.mover_vertice(comando.vertice_movido, comando.posicao_vertice_objeto)

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
        glTranslatef(self.pan_x, self.pan_y, -self.distancia_camera)
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
        if self._comando.mensagem:
            linhas.append((f"» {self._comando.mensagem}", VERDE))
        elif self._controlador.maos_suaves:
            n = len(self._controlador.maos_suaves)
            linhas.append((f"{n} mão(s) — feche a mão para pegar o sólido", BRANCO))

        if self._estado_camera is not None:
            cor = VERDE if self._estado_camera == "conectado" else AMBAR
            linhas.append((f"Câmera: {self._estado_camera}", cor))
        linhas += [(texto, BRANCO) for texto in self._diagnostico]
        linhas += self._linhas_calibracao()
        return linhas

    def _linhas_calibracao(self) -> list[tuple[str, tuple[int, int, int]]]:
        """Valores crus por mão, para calibrar os limiares (`--debug-gestos`).

        Os limiares vêm de fotos que passei pelo MediaPipe (punho, dedo
        apontando, "vitória"), mas quanto cada mão fecha varia por pessoa. É
        isto aqui que permite ajustar: feche a mão, leia `abert`, e mexa em
        GARRA_FECHA/GARRA_ABRE no config.py.
        """
        if not getattr(self.config, "DEBUG_GESTOS", False):
            return []

        params = self._controlador.params
        linhas = [(
            f"garra fecha < {params.garra_fecha:.2f}"
            f" | pinça fecha < {params.pinca_fecha:.2f}", BRANCO,
        )]
        for mao in self._controlador.maos_suaves:
            garra = esta_em_garra(mao, params, False)
            pinca = esta_agarrando(mao, params, False)
            marca = "GARRA" if garra else ("PINÇA" if pinca else "—")
            cor = VERDE if (garra or pinca) else BRANCO
            linhas.append((
                f"mão {mao.id_mao} abert: {mao.abertura:.2f} | pinça: {mao.pinca:.2f}"
                f" | tam: {mao.tamanho:.3f} | {marca}", cor,
            ))
        return linhas

    def _desenhar_overlay(self) -> None:
        if self._hud is None:
            return
        with ContextoOrtografico(self.largura, self.altura):
            self._desenhar_cursores()
            self._desenhar_realce_vertice()
            self._hud.desenhar(self._linhas_hud())

    def _desenhar_cursores(self) -> None:
        """Um cursor por mão: anel vazado quando solta, disco cheio quando
        agarrando. É o que torna o modelo de controle legível — sem isso o
        usuário não tem como saber que precisa fechar a pinça."""
        agarrando_agora = self._comando.fase is not Fase.OCIOSO
        ancora = self._controlador.estado.id_mao_ancora
        secundaria = self._controlador.estado.id_mao_secundaria

        for mao in self._controlador.maos_suaves:
            x, y = mao.cursor_tela
            ativa = agarrando_agora and mao.id_mao in (ancora, secundaria)
            if ativa:
                desenhar_circulo(x, y, 13.0, COR_AGARRANDO, alfa=0.85)
                desenhar_anel(x, y, 20.0, COR_AGARRANDO, alfa=0.9, espessura=2.0)
            else:
                desenhar_anel(x, y, 15.0, COR_LIVRE, alfa=0.75, espessura=2.0)

    def _desenhar_realce_vertice(self) -> None:
        """Anel em volta do vértice sob a mira, para o usuário saber o que vai
        pegar ANTES de fechar a pinça."""
        indice = self._comando.vertice_sob_mira
        if indice is None or not self._projetor.pronto:
            return
        projetado = self._projetor.projetar(self.tetraedro.vertices()[indice])
        if projetado is None:
            return
        arrastando = self._comando.vertice_movido is not None
        cor = COR_AGARRANDO if arrastando else COR_MIRA
        desenhar_anel(projetado[0], projetado[1], 18.0, cor, alfa=0.95, espessura=3.0)
