"""Testes da máquina de estados de gestos.

Zero OpenGL: `ContextoCena` é uma projeção ortográfica escrita à mão. As mãos
são `MaoSuave` sintéticas, então dá para exercitar toda a lógica de
travamento, histerese e debounce sem câmera.
"""
import pytest

from geometry.transformacoes import (
    QUAT_IDENTIDADE,
    aplicar_quaternion,
    quaternion_de_eixo_angulo,
)
from interaction.gestos import (
    ComandoInteracao,
    EstadoInteracao,
    Fase,
    ParametrosGesto,
    avaliar_gestos,
    esta_agarrando,
    esta_em_garra,
    vertice_sob_cursor,
)
from interaction.suavizacao import MaoSuave

PARAMS = ParametrosGesto()
PINCA_FECHADA = 0.10
PINCA_ABERTA = 0.95
# Valores de `abertura_mao` medidos com o MediaPipe em fotos reais.
ABERTURA_PUNHO = 0.24
ABERTURA_APONTANDO = 0.43
ABERTURA_ABERTA = 0.58
EXT_ABERTA = (0.98, 0.97, 0.96, 0.95)
TAMANHO_PADRAO = 0.30


def mao(
    id_mao=0, x=500.0, y=400.0, pinca=PINCA_ABERTA, abertura=ABERTURA_ABERTA,
    orientacao=QUAT_IDENTIDADE, visivel_ha_s=1.0, destra=True,
    tamanho=TAMANHO_PADRAO, palma=None,
) -> MaoSuave:
    return MaoSuave(
        id_mao=id_mao, destra=destra, cursor_tela=(x, y), pinca=pinca,
        extensoes=EXT_ABERTA, orientacao=orientacao, visivel_ha_s=visivel_ha_s,
        palma_tela=palma if palma is not None else (x, y),
        abertura=abertura, tamanho=tamanho,
    )


def garra(id_mao=0, **kw) -> MaoSuave:
    """Mão inteira fechada: o gesto que pega o sólido."""
    kw.setdefault("abertura", ABERTURA_PUNHO)
    kw.setdefault("pinca", PINCA_FECHADA)  # fechar a mão junta polegar e indicador
    return mao(id_mao=id_mao, **kw)


def pinca(id_mao=0, **kw) -> MaoSuave:
    """Pinça de precisão: polegar e indicador juntos, resto da mão aberto."""
    kw.setdefault("abertura", ABERTURA_APONTANDO)
    kw.setdefault("pinca", PINCA_FECHADA)
    return mao(id_mao=id_mao, **kw)


class CenaFalsa:
    """Projeção ortográfica trivial: (x,y,z) do objeto -> pixels."""

    def __init__(self, vertices=None, orientacao=QUAT_IDENTIDADE, escala=1.5, distancia=6.0):
        self._vertices = vertices or [
            (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)
        ]
        self._orientacao = orientacao
        self._escala = escala
        self._distancia = distancia
        self.centro = (512.0, 384.0)
        self.zoom = 100.0

    def vertices_tela(self):
        return [
            (self.centro[0] + v[0] * self.zoom, self.centro[1] - v[1] * self.zoom, 0.5)
            for v in self._vertices
        ]

    def desprojetar(self, x_px, y_px, z_janela):
        return ((x_px - self.centro[0]) / self.zoom, (self.centro[1] - y_px) / self.zoom, 0.0)

    def orientacao_objeto(self):
        return self._orientacao

    def escala_objeto(self):
        return self._escala

    def distancia_camera(self):
        return self._distancia


def avancar(estado, maos, cena, frame_id, params=PARAMS):
    return avaliar_gestos(estado, maos, cena, params, frame_id)


def confirmar(maos, cena, params=PARAMS, estado=None, frame_inicial=1):
    """Roda frames suficientes para vencer o debounce de entrada."""
    estado = estado or EstadoInteracao()
    comando = ComandoInteracao()
    for i in range(params.quadros_confirmacao):
        estado, comando = avancar(estado, maos, cena, frame_inicial + i, params)
    return estado, comando


class TestVerticeSobCursor:
    def test_acha_o_vertice_dentro_do_raio(self):
        vertices = [(100.0, 100.0, 0.5), (500.0, 500.0, 0.5)]
        assert vertice_sob_cursor((110.0, 105.0), vertices, 45.0) == 0

    def test_devolve_none_quando_todos_estao_longe(self):
        vertices = [(100.0, 100.0, 0.5), (500.0, 500.0, 0.5)]
        assert vertice_sob_cursor((300.0, 300.0), vertices, 45.0) is None

    def test_escolhe_o_mais_proximo(self):
        vertices = [(100.0, 100.0, 0.5), (120.0, 100.0, 0.5)]
        assert vertice_sob_cursor((118.0, 100.0), vertices, 45.0) == 1

    def test_ignora_vertices_nao_projetaveis(self):
        vertices = [None, (100.0, 100.0, 0.5)]
        assert vertice_sob_cursor((100.0, 100.0), vertices, 45.0) == 1

    def test_borda_do_raio_conta_como_acerto(self):
        assert vertice_sob_cursor((145.0, 100.0), [(100.0, 100.0, 0.5)], 45.0) == 0
        assert vertice_sob_cursor((146.0, 100.0), [(100.0, 100.0, 0.5)], 45.0) is None


class TestHisterese:
    def test_garra_fecha_no_limiar_de_entrada(self):
        assert esta_em_garra(mao(abertura=0.30), PARAMS, False) is True
        assert esta_em_garra(mao(abertura=0.43), PARAMS, False) is False

    def test_garra_ja_fechada_so_solta_no_limiar_maior(self):
        """Entre 0,38 e 0,48 o gesto continua se já estava segurando — é o que
        impede o sólido de escapar da mão quando ela treme no limiar."""
        assert esta_em_garra(mao(abertura=0.43), PARAMS, True) is True
        assert esta_em_garra(mao(abertura=0.43), PARAMS, False) is False
        assert esta_em_garra(mao(abertura=0.55), PARAMS, True) is False

    def test_pinca_fecha_no_limiar_de_entrada(self):
        assert esta_agarrando(mao(pinca=0.30, abertura=ABERTURA_APONTANDO), PARAMS, False) is True
        assert esta_agarrando(mao(pinca=0.40, abertura=ABERTURA_APONTANDO), PARAMS, False) is False

    def test_pinca_ja_agarrando_so_solta_no_limiar_maior(self):
        aberta = dict(abertura=ABERTURA_ABERTA)
        assert esta_agarrando(mao(pinca=0.42, **aberta), PARAMS, True) is True
        assert esta_agarrando(mao(pinca=0.42, **aberta), PARAMS, False) is False
        assert esta_agarrando(mao(pinca=0.55, **aberta), PARAMS, True) is False

    def test_punho_fechado_nao_conta_como_pinca(self):
        """Fechar a mão inteira também aproxima polegar e indicador (medi 0,199
        num punho). Sem exigir a mão aberta, a garra e a pinça disparariam
        juntas e o usuário nunca saberia qual das duas ia acontecer."""
        assert esta_agarrando(garra(), PARAMS, False) is False
        assert esta_em_garra(garra(), PARAMS, False) is True


class TestEntradaNosGestos:
    def test_mao_aberta_nao_faz_nada(self):
        estado, comando = confirmar([mao()], CenaFalsa())
        assert estado.fase is Fase.OCIOSO
        assert comando.fase is Fase.OCIOSO

    def test_fechar_a_mao_pega_o_solido(self):
        estado, _ = confirmar([garra(x=200.0, y=200.0)], CenaFalsa())
        assert estado.fase is Fase.SEGURANDO_OBJETO

    def test_garra_pega_o_solido_mesmo_longe_dele(self):
        """Não exige mirar: procurar o sólido com a mão fechada antes de ele
        reagir seria o oposto de "agarrar e travar"."""
        estado, _ = confirmar([garra(x=50.0, y=50.0)], CenaFalsa())
        assert estado.fase is Fase.SEGURANDO_OBJETO

    def test_pinca_sobre_vertice_deforma(self):
        cena = CenaFalsa()
        x, y, _ = cena.vertices_tela()[1]
        estado, _ = confirmar([pinca(x=x, y=y)], cena)
        assert estado.fase is Fase.ARRASTANDO_VERTICE
        assert estado.vertice_ativo == 1

    def test_pinca_longe_de_qualquer_vertice_nao_faz_nada(self):
        """A pinça é o gesto de precisão e só serve para vértice. Antes ela
        virava órbita, e converter pixels em graus fazia o sólido girar 358°
        ao atravessar a janela — era o "movendo loucamente"."""
        estado, comando = confirmar([pinca(x=50.0, y=50.0)], CenaFalsa())
        assert estado.fase is Fase.OCIOSO
        assert comando.fase is Fase.OCIOSO

    def test_duas_garras_escalam(self):
        maos = [garra(id_mao=0, x=300.0), garra(id_mao=1, x=700.0)]
        estado, _ = confirmar(maos, CenaFalsa())
        assert estado.fase is Fase.ESCALANDO
        assert {estado.id_mao_ancora, estado.id_mao_secundaria} == {0, 1}

    def test_vertice_vence_garra_quando_ambos_sao_possiveis(self):
        """Mirar um vértice com a pinça é mais específico, logo mais
        intencional, que fechar a mão em qualquer lugar."""
        cena = CenaFalsa()
        x, y, _ = cena.vertices_tela()[1]
        maos = [pinca(id_mao=0, x=x, y=y), garra(id_mao=1, x=900.0)]
        estado, _ = confirmar(maos, cena)
        assert estado.fase is Fase.ARRASTANDO_VERTICE

    def test_mao_recem_aparecida_nao_dispara_gesto(self):
        """Nos primeiros instantes os landmarks ainda estão convergindo."""
        estado, _ = confirmar([garra(x=200.0, visivel_ha_s=0.05)], CenaFalsa())
        assert estado.fase is Fase.OCIOSO

    def test_sem_maos_fica_ocioso(self):
        estado, _ = confirmar([], CenaFalsa())
        assert estado.fase is Fase.OCIOSO


class TestDebounce:
    def test_um_unico_frame_nao_dispara(self):
        estado, _ = avancar(EstadoInteracao(), [garra(x=200.0)], CenaFalsa(), 1)
        assert estado.fase is Fase.OCIOSO

    def test_dois_frames_consecutivos_disparam(self):
        m = [garra(x=200.0)]
        estado, _ = avancar(EstadoInteracao(), m, CenaFalsa(), 1)
        estado, _ = avancar(estado, m, CenaFalsa(), 2)
        assert estado.fase is Fase.SEGURANDO_OBJETO

    def test_frame_id_repetido_nao_conta(self):
        """A 60fps de render sobre 25 de detecção, contar frames de render
        faria o debounce filtrar só 33 ms e não pegar landmark ruim."""
        m = [garra(x=200.0)]
        estado = EstadoInteracao()
        for _ in range(5):
            estado, _ = avancar(estado, m, CenaFalsa(), 1)  # sempre o mesmo frame_id
        assert estado.fase is Fase.OCIOSO

    def test_candidato_intermitente_reinicia_a_contagem(self):
        cena = CenaFalsa()
        fechada = [garra(x=200.0)]
        solta = [mao(x=200.0)]
        estado, _ = avancar(EstadoInteracao(), fechada, cena, 1)
        estado, _ = avancar(estado, solta, cena, 2)
        estado, _ = avancar(estado, fechada, cena, 3)
        assert estado.fase is Fase.OCIOSO


class TestTravamentoDoGesto:
    def test_segunda_mao_nao_sequestra_um_arrasto_em_curso(self):
        """Sem travamento, encostar a outra mão no campo de visão trocaria o
        gesto no meio — falha clássica de usabilidade."""
        cena = CenaFalsa()
        x, y, _ = cena.vertices_tela()[1]
        estado, _ = confirmar([pinca(id_mao=0, x=x, y=y)], cena)
        assert estado.fase is Fase.ARRASTANDO_VERTICE

        duas = [pinca(id_mao=0, x=x, y=y), garra(id_mao=1, x=800.0, y=300.0)]
        for i in range(5):
            estado, comando = avancar(estado, duas, cena, 10 + i)
        assert estado.fase is Fase.ARRASTANDO_VERTICE
        assert comando.escala_absoluta is None

    def test_segunda_mao_promove_o_agarre_para_escala(self):
        """As duas mãos quase nunca entram no quadro no mesmo frame: a
        primeira engata o agarre e, sem promoção, o gesto de duas mãos seria
        impossível de iniciar na prática."""
        cena = CenaFalsa()
        estado, _ = confirmar([garra(id_mao=0, x=300.0, y=400.0)], cena)
        assert estado.fase is Fase.SEGURANDO_OBJETO

        duas = [garra(id_mao=0, x=300.0, y=400.0), garra(id_mao=1, x=700.0, y=400.0)]
        estado, _ = avancar(estado, duas, cena, 20)
        assert estado.fase is Fase.ESCALANDO
        assert {estado.id_mao_ancora, estado.id_mao_secundaria} == {0, 1}

    def test_segunda_mao_recem_chegada_nao_promove(self):
        cena = CenaFalsa()
        estado, _ = confirmar([garra(id_mao=0, x=300.0, y=400.0)], cena)
        duas = [garra(id_mao=0, x=300.0, y=400.0),
                garra(id_mao=1, x=700.0, y=400.0, visivel_ha_s=0.02)]
        estado, _ = avancar(estado, duas, cena, 20)
        assert estado.fase is Fase.SEGURANDO_OBJETO

    def test_promocao_nao_vale_para_arrasto_de_vertice(self):
        """O arrasto é ancorado num alvo, então continua protegido — este é o
        contraste que justifica a promoção ser só do agarre genérico."""
        cena = CenaFalsa()
        x, y, _ = cena.vertices_tela()[1]
        estado, _ = confirmar([pinca(id_mao=0, x=x, y=y)], cena)
        duas = [pinca(id_mao=0, x=x, y=y), garra(id_mao=1, x=800.0, y=300.0)]
        estado, _ = avancar(estado, duas, cena, 20)
        assert estado.fase is Fase.ARRASTANDO_VERTICE

    def test_agarre_nao_vira_deformacao_no_meio_do_gesto(self):
        """Começou segurando o sólido: passar a mão sobre um vértice não pode
        trocar o gesto."""
        cena = CenaFalsa()
        estado, _ = confirmar([garra(x=200.0, y=200.0)], cena)
        x, y, _ = cena.vertices_tela()[1]
        estado, comando = avancar(estado, [garra(x=x, y=y)], cena, 20)
        assert estado.fase is Fase.SEGURANDO_OBJETO
        assert comando.vertice_movido is None

    def test_gesto_termina_quando_a_mao_abre(self):
        cena = CenaFalsa()
        estado, _ = confirmar([garra(x=200.0, y=200.0)], cena)
        estado, comando = avancar(estado, [mao(x=200.0, y=200.0)], cena, 20)
        assert estado.fase is Fase.OCIOSO
        assert comando.fase is Fase.OCIOSO

    def test_gesto_termina_quando_a_mao_some(self):
        cena = CenaFalsa()
        estado, _ = confirmar([garra(x=200.0, y=200.0)], cena)
        estado, _ = avancar(estado, [], cena, 20)
        assert estado.fase is Fase.OCIOSO

    def test_escala_termina_se_uma_das_maos_some(self):
        cena = CenaFalsa()
        maos = [garra(id_mao=0, x=300.0), garra(id_mao=1, x=700.0)]
        estado, _ = confirmar(maos, cena)
        assert estado.fase is Fase.ESCALANDO
        estado, _ = avancar(estado, [maos[0]], cena, 20)
        assert estado.fase is Fase.OCIOSO


class TestSegurarOSolido:
    def test_mover_a_mao_arrasta_o_solido(self):
        cena = CenaFalsa()
        estado, _ = confirmar([garra(x=200.0, y=200.0)], cena)
        _, comando = avancar(estado, [garra(x=250.0, y=230.0)], cena, 20)
        assert comando.delta_pan_tela == pytest.approx((50.0, 30.0))

    def test_mao_parada_nao_arrasta(self):
        cena = CenaFalsa()
        estado, _ = confirmar([garra(x=200.0, y=200.0)], cena)
        _, comando = avancar(estado, [garra(x=200.0, y=200.0)], cena, 20)
        assert comando.delta_pan_tela == pytest.approx((0.0, 0.0))

    def test_girar_a_mao_gira_o_solido_no_referencial_da_tela(self):
        cena = CenaFalsa(orientacao=quaternion_de_eixo_angulo((1.0, 0.0, 0.0), 1.2))
        estado, _ = confirmar([garra(x=200.0, y=200.0, orientacao=QUAT_IDENTIDADE)], cena)
        assert estado.fase is Fase.SEGURANDO_OBJETO

        giro = quaternion_de_eixo_angulo((0.0, 1.0, 0.0), 0.7)
        _, comando = avancar(estado, [garra(x=200.0, y=200.0, orientacao=giro)], cena, 20)
        for v in [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.3, -0.6, 0.8)]:
            esperado = aplicar_quaternion(giro, aplicar_quaternion(cena.orientacao_objeto(), v))
            assert aplicar_quaternion(comando.orientacao_absoluta, v) == pytest.approx(
                esperado, abs=1e-9)

    def test_mao_parada_nao_gira_o_solido(self):
        cena = CenaFalsa(orientacao=quaternion_de_eixo_angulo((1.0, 0.0, 0.0), 1.2))
        parada = garra(x=200.0, y=200.0)
        estado, _ = confirmar([parada], cena)
        _, comando = avancar(estado, [parada], cena, 20)
        for v in [(1.0, 0.0, 0.0), (0.0, 0.0, 1.0)]:
            assert aplicar_quaternion(comando.orientacao_absoluta, v) == pytest.approx(
                aplicar_quaternion(cena.orientacao_objeto(), v), abs=1e-9
            )

    def test_aproximar_a_mao_da_camera_traz_o_solido(self):
        """A mão perto da câmera aparece maior. É o mapeamento que o usuário
        esperava sem que ninguém explicasse."""
        cena = CenaFalsa(distancia=6.0)
        estado, _ = confirmar([garra(x=400.0, tamanho=0.30)], cena)
        _, comando = avancar(estado, [garra(x=400.0, tamanho=0.36)], cena, 20)
        assert comando.distancia_camera_absoluta == pytest.approx(6.0 / 1.2)

    def test_afastar_a_mao_afasta_o_solido(self):
        cena = CenaFalsa(distancia=6.0)
        estado, _ = confirmar([garra(x=400.0, tamanho=0.30)], cena)
        _, comando = avancar(estado, [garra(x=400.0, tamanho=0.24)], cena, 20)
        assert comando.distancia_camera_absoluta == pytest.approx(6.0 / 0.8)

    def test_profundidade_e_absoluta_e_nao_acumula_deriva(self):
        """Voltar a mão à distância inicial devolve a distância inicial, por
        mais que se mexa no meio do caminho."""
        cena = CenaFalsa(distancia=6.0)
        estado, _ = confirmar([garra(x=400.0, tamanho=0.30)], cena)
        for t in (0.36, 0.25, 0.40, 0.28):
            estado, _ = avancar(estado, [garra(x=400.0, tamanho=t)], cena, 30)
        _, comando = avancar(estado, [garra(x=400.0, tamanho=0.30)], cena, 40)
        assert comando.distancia_camera_absoluta == pytest.approx(6.0)

    def test_razao_de_profundidade_e_limitada(self):
        """Uma leitura ruim do tamanho da mão não pode jogar o sólido para o
        infinito nem para dentro da câmera."""
        cena = CenaFalsa(distancia=6.0)
        estado, _ = confirmar([garra(x=400.0, tamanho=0.30)], cena)
        _, comando = avancar(estado, [garra(x=400.0, tamanho=3.0)], cena, 20)
        assert comando.distancia_camera_absoluta == pytest.approx(
            6.0 / PARAMS.razao_profundidade_max)
        _, comando = avancar(estado, [garra(x=400.0, tamanho=0.001)], cena, 21)
        assert comando.distancia_camera_absoluta == pytest.approx(
            6.0 / PARAMS.razao_profundidade_min)

    def test_o_solido_nao_salta_no_instante_em_que_e_pego(self):
        """Primeiro frame do agarre: sem deslocamento, sem giro, sem mudança
        de profundidade — só trava."""
        cena = CenaFalsa(orientacao=quaternion_de_eixo_angulo((0.0, 1.0, 0.0), 0.4))
        m = garra(x=200.0, y=300.0, tamanho=0.33,
                  orientacao=quaternion_de_eixo_angulo((1.0, 0.0, 0.0), 0.9))
        estado, _ = confirmar([m], cena)
        _, comando = avancar(estado, [m], cena, 20)
        assert comando.delta_pan_tela == pytest.approx((0.0, 0.0))
        assert comando.distancia_camera_absoluta == pytest.approx(6.0)
        for v in [(1.0, 0.0, 0.0), (0.0, 0.0, 1.0)]:
            assert aplicar_quaternion(comando.orientacao_absoluta, v) == pytest.approx(
                aplicar_quaternion(cena.orientacao_objeto(), v), abs=1e-9)


class TestComandoDeEscala:
    def test_afastar_as_maos_aumenta(self):
        cena = CenaFalsa(escala=1.5)
        maos = [garra(id_mao=0, x=400.0), garra(id_mao=1, x=600.0)]
        estado, _ = confirmar(maos, cena)
        afastadas = [garra(id_mao=0, x=300.0), garra(id_mao=1, x=700.0)]
        _, comando = avancar(estado, afastadas, cena, 20)
        assert comando.escala_absoluta == pytest.approx(1.5 * 2.0)

    def test_juntar_as_maos_diminui(self):
        cena = CenaFalsa(escala=2.0)
        maos = [garra(id_mao=0, x=300.0), garra(id_mao=1, x=700.0)]
        estado, _ = confirmar(maos, cena)
        juntas = [garra(id_mao=0, x=400.0), garra(id_mao=1, x=600.0)]
        _, comando = avancar(estado, juntas, cena, 20)
        assert comando.escala_absoluta == pytest.approx(1.0)

    def test_escala_e_absoluta_e_nao_acumula_deriva(self):
        """Voltar as mãos à distância inicial tem de devolver a escala inicial,
        por mais que se mexa no meio do caminho."""
        cena = CenaFalsa(escala=1.5)
        inicio = [garra(id_mao=0, x=400.0), garra(id_mao=1, x=600.0)]
        estado, _ = confirmar(inicio, cena)
        for larg in (250.0, 350.0, 150.0, 450.0):
            estado, _ = avancar(
                estado,
                [garra(id_mao=0, x=500.0 - larg), garra(id_mao=1, x=500.0 + larg)],
                cena, 30,
            )
        _, comando = avancar(estado, inicio, cena, 40)
        assert comando.escala_absoluta == pytest.approx(1.5)

    def test_mover_as_duas_maos_juntas_faz_pan(self):
        cena = CenaFalsa()
        maos = [garra(id_mao=0, x=400.0, y=300.0), garra(id_mao=1, x=600.0, y=300.0)]
        estado, _ = confirmar(maos, cena)
        deslocadas = [garra(id_mao=0, x=450.0, y=320.0), garra(id_mao=1, x=650.0, y=320.0)]
        _, comando = avancar(estado, deslocadas, cena, 20)
        assert comando.delta_pan_tela == pytest.approx((50.0, 20.0))
        assert comando.escala_absoluta == pytest.approx(cena.escala_objeto())


class TestComandoDeVertice:
    def test_arrastar_leva_o_vertice_para_a_posicao_do_cursor(self):
        cena = CenaFalsa()
        x, y, _ = cena.vertices_tela()[1]
        estado, _ = confirmar([pinca(x=x, y=y)], cena)
        _, comando = avancar(estado, [pinca(x=x + 100.0, y=y)], cena, 20)
        assert comando.vertice_movido == 1
        # vértice 1 estava em (1,0,0); cursor andou 100px = 1 unidade
        assert comando.posicao_vertice_objeto == pytest.approx((2.0, 0.0, 0.0))

    def test_offset_evita_que_o_vertice_salte_ao_ser_agarrado(self):
        """Agarrar com o cursor um pouco ao lado do vértice não pode
        teletransportá-lo."""
        cena = CenaFalsa()
        x, y, _ = cena.vertices_tela()[1]
        estado, _ = confirmar([pinca(x=x - 20.0, y=y + 10.0)], cena)
        _, comando = avancar(estado, [pinca(x=x - 20.0, y=y + 10.0)], cena, 20)
        assert comando.posicao_vertice_objeto == pytest.approx((1.0, 0.0, 0.0))


class TestRealceDeProximidade:
    def test_indica_o_vertice_sob_a_mira_sem_agarrar(self):
        cena = CenaFalsa()
        x, y, _ = cena.vertices_tela()[2]
        _, comando = confirmar([mao(x=x, y=y)], cena)
        assert comando.fase is Fase.OCIOSO
        assert comando.vertice_sob_mira == 2

    def test_sem_vertice_por_perto_nao_indica_nada(self):
        _, comando = confirmar([mao(x=50.0, y=50.0)], CenaFalsa())
        assert comando.vertice_sob_mira is None


class TestPureza:
    def test_avaliar_nao_muda_o_estado_recebido(self):
        estado = EstadoInteracao()
        avancar(estado, [garra(x=200.0)], CenaFalsa(), 1)
        assert estado == EstadoInteracao()

    def test_mesma_entrada_produz_mesma_saida(self):
        cena = CenaFalsa()
        m = [garra(x=200.0)]
        assert avancar(EstadoInteracao(), m, cena, 1) == avancar(EstadoInteracao(), m, cena, 1)

    def test_ordem_das_maos_nao_altera_o_resultado(self):
        cena = CenaFalsa()
        maos = [garra(id_mao=0, x=300.0), garra(id_mao=1, x=700.0)]
        estado_a, _ = confirmar(maos, cena)
        estado_b, _ = confirmar(list(reversed(maos)), cena)
        assert estado_a.id_mao_ancora == estado_b.id_mao_ancora
        assert estado_a.id_mao_secundaria == estado_b.id_mao_secundaria
