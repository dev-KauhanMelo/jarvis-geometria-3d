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
    esta_em_concha,
    vertice_sob_cursor,
)
from interaction.suavizacao import MaoSuave

PARAMS = ParametrosGesto()
PINCA_FECHADA = 0.10
PINCA_ABERTA = 0.95
EXT_CONCHA = (0.65, 0.68, 0.66, 0.62)
EXT_ABERTA = (0.98, 0.97, 0.96, 0.95)
EXT_PUNHO = (0.31, 0.25, 0.33, 0.46)


def mao(
    id_mao=0, x=500.0, y=400.0, pinca=PINCA_ABERTA, extensoes=EXT_ABERTA,
    orientacao=QUAT_IDENTIDADE, visivel_ha_s=1.0, destra=True,
) -> MaoSuave:
    return MaoSuave(
        id_mao=id_mao, destra=destra, cursor_tela=(x, y), pinca=pinca,
        extensoes=extensoes, orientacao=orientacao, visivel_ha_s=visivel_ha_s,
    )


class CenaFalsa:
    """Projeção ortográfica trivial: (x,y,z) do objeto -> pixels."""

    def __init__(self, vertices=None, orientacao=QUAT_IDENTIDADE, escala=1.5):
        self._vertices = vertices or [
            (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)
        ]
        self._orientacao = orientacao
        self._escala = escala
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
    def test_pinca_fecha_no_limiar_de_entrada(self):
        assert esta_agarrando(mao(pinca=0.30), PARAMS, False) is True
        assert esta_agarrando(mao(pinca=0.40), PARAMS, False) is False

    def test_pinca_ja_agarrando_so_solta_no_limiar_maior(self):
        """Entre 0.35 e 0.50 o gesto continua se já estava agarrando — é o que
        impede o objeto de soltar sozinho quando a mão treme no limiar."""
        assert esta_agarrando(mao(pinca=0.42), PARAMS, True) is True
        assert esta_agarrando(mao(pinca=0.42), PARAMS, False) is False
        assert esta_agarrando(mao(pinca=0.55), PARAMS, True) is False

    def test_concha_tem_histerese_nos_dois_extremos(self):
        quase_fechada = (0.45, 0.46, 0.47, 0.44)
        assert esta_em_concha(mao(extensoes=quase_fechada), PARAMS, False) is False
        assert esta_em_concha(mao(extensoes=quase_fechada), PARAMS, True) is True

    def test_punho_nao_e_concha(self):
        assert esta_em_concha(mao(extensoes=EXT_PUNHO, pinca=0.13), PARAMS, False) is False

    def test_mao_aberta_nao_e_concha(self):
        assert esta_em_concha(mao(extensoes=EXT_ABERTA), PARAMS, False) is False

    def test_concha_com_pinca_fechada_nao_conta(self):
        """Pinça vence concha na mesma mão."""
        assert esta_em_concha(mao(extensoes=EXT_CONCHA, pinca=PINCA_FECHADA), PARAMS, False) is False


class TestEntradaNosGestos:
    def test_mao_aberta_nao_faz_nada(self):
        estado, comando = confirmar([mao()], CenaFalsa())
        assert estado.fase is Fase.OCIOSO
        assert comando.fase is Fase.OCIOSO

    def test_uma_pinca_longe_de_vertice_gira(self):
        estado, _ = confirmar([mao(x=200.0, y=200.0, pinca=PINCA_FECHADA)], CenaFalsa())
        assert estado.fase is Fase.GIRANDO_PINCA

    def test_uma_pinca_sobre_vertice_deforma(self):
        cena = CenaFalsa()
        x, y, _ = cena.vertices_tela()[1]
        estado, _ = confirmar([mao(x=x, y=y, pinca=PINCA_FECHADA)], cena)
        assert estado.fase is Fase.ARRASTANDO_VERTICE
        assert estado.vertice_ativo == 1

    def test_duas_pincas_escalam(self):
        maos = [mao(id_mao=0, x=300.0, pinca=PINCA_FECHADA),
                mao(id_mao=1, x=700.0, pinca=PINCA_FECHADA)]
        estado, _ = confirmar(maos, CenaFalsa())
        assert estado.fase is Fase.ESCALANDO
        assert {estado.id_mao_ancora, estado.id_mao_secundaria} == {0, 1}

    def test_concha_gira(self):
        estado, _ = confirmar([mao(x=200.0, y=200.0, extensoes=EXT_CONCHA)], CenaFalsa())
        assert estado.fase is Fase.GIRANDO_CONCHA

    def test_mao_recem_aparecida_nao_dispara_gesto(self):
        """Nos primeiros instantes os landmarks ainda estão convergindo."""
        recente = mao(pinca=PINCA_FECHADA, x=200.0, visivel_ha_s=0.05)
        estado, _ = confirmar([recente], CenaFalsa())
        assert estado.fase is Fase.OCIOSO

    def test_sem_maos_fica_ocioso(self):
        estado, comando = confirmar([], CenaFalsa())
        assert estado.fase is Fase.OCIOSO


class TestDebounce:
    def test_um_unico_frame_nao_dispara(self):
        m = [mao(x=200.0, y=200.0, pinca=PINCA_FECHADA)]
        estado, _ = avancar(EstadoInteracao(), m, CenaFalsa(), 1)
        assert estado.fase is Fase.OCIOSO

    def test_dois_frames_consecutivos_disparam(self):
        m = [mao(x=200.0, y=200.0, pinca=PINCA_FECHADA)]
        estado, _ = avancar(EstadoInteracao(), m, CenaFalsa(), 1)
        estado, _ = avancar(estado, m, CenaFalsa(), 2)
        assert estado.fase is Fase.GIRANDO_PINCA

    def test_frame_id_repetido_nao_conta(self):
        """A 60fps de render sobre 25 de detecção, contar frames de render
        faria o debounce filtrar só 33 ms e não pegar landmark ruim."""
        m = [mao(x=200.0, y=200.0, pinca=PINCA_FECHADA)]
        estado = EstadoInteracao()
        for _ in range(5):
            estado, _ = avancar(estado, m, CenaFalsa(), 1)  # sempre o mesmo frame_id
        assert estado.fase is Fase.OCIOSO

    def test_candidato_intermitente_reinicia_a_contagem(self):
        cena = CenaFalsa()
        agarrando = [mao(x=200.0, y=200.0, pinca=PINCA_FECHADA)]
        solta = [mao(x=200.0, y=200.0, pinca=PINCA_ABERTA)]
        estado, _ = avancar(EstadoInteracao(), agarrando, cena, 1)
        estado, _ = avancar(estado, solta, cena, 2)
        estado, _ = avancar(estado, agarrando, cena, 3)
        assert estado.fase is Fase.OCIOSO


class TestTravamentoDoGesto:
    def test_segunda_mao_nao_sequestra_um_arrasto_em_curso(self):
        """Sem travamento, encostar a outra mão no campo de visão trocaria o
        gesto no meio — falha clássica de usabilidade."""
        cena = CenaFalsa()
        x, y, _ = cena.vertices_tela()[1]
        estado, _ = confirmar([mao(id_mao=0, x=x, y=y, pinca=PINCA_FECHADA)], cena)
        assert estado.fase is Fase.ARRASTANDO_VERTICE

        duas = [mao(id_mao=0, x=x, y=y, pinca=PINCA_FECHADA),
                mao(id_mao=1, x=800.0, y=300.0, pinca=PINCA_FECHADA)]
        for i in range(5):
            estado, comando = avancar(estado, duas, cena, 10 + i)
        assert estado.fase is Fase.ARRASTANDO_VERTICE
        assert comando.escala_absoluta is None

    def test_segunda_mao_promove_orbita_para_escala(self):
        """As duas mãos quase nunca entram no quadro no mesmo frame: a
        primeira engata a órbita e, sem promoção, o gesto de duas mãos seria
        impossível de iniciar na prática."""
        cena = CenaFalsa()
        estado, _ = confirmar([mao(id_mao=0, x=300.0, y=400.0, pinca=PINCA_FECHADA)], cena)
        assert estado.fase is Fase.GIRANDO_PINCA

        duas = [mao(id_mao=0, x=300.0, y=400.0, pinca=PINCA_FECHADA),
                mao(id_mao=1, x=700.0, y=400.0, pinca=PINCA_FECHADA)]
        estado, comando = avancar(estado, duas, cena, 20)
        assert estado.fase is Fase.ESCALANDO
        assert {estado.id_mao_ancora, estado.id_mao_secundaria} == {0, 1}

    def test_segunda_mao_recem_chegada_nao_promove(self):
        cena = CenaFalsa()
        estado, _ = confirmar([mao(id_mao=0, x=300.0, y=400.0, pinca=PINCA_FECHADA)], cena)
        duas = [mao(id_mao=0, x=300.0, y=400.0, pinca=PINCA_FECHADA),
                mao(id_mao=1, x=700.0, y=400.0, pinca=PINCA_FECHADA, visivel_ha_s=0.02)]
        estado, _ = avancar(estado, duas, cena, 20)
        assert estado.fase is Fase.GIRANDO_PINCA

    def test_promocao_nao_vale_para_arrasto_de_vertice(self):
        """O arrasto é ancorado num alvo, então continua protegido — este é o
        contraste que justifica a promoção ser só da órbita."""
        cena = CenaFalsa()
        x, y, _ = cena.vertices_tela()[1]
        estado, _ = confirmar([mao(id_mao=0, x=x, y=y, pinca=PINCA_FECHADA)], cena)
        assert estado.fase is Fase.ARRASTANDO_VERTICE
        duas = [mao(id_mao=0, x=x, y=y, pinca=PINCA_FECHADA),
                mao(id_mao=1, x=800.0, y=300.0, pinca=PINCA_FECHADA)]
        estado, _ = avancar(estado, duas, cena, 20)
        assert estado.fase is Fase.ARRASTANDO_VERTICE

    def test_decisao_vertice_vs_orbita_nao_e_reavaliada(self):
        """Começou girando: passar o cursor sobre um vértice não pode virar
        deformação no meio do gesto."""
        cena = CenaFalsa()
        estado, _ = confirmar([mao(x=200.0, y=200.0, pinca=PINCA_FECHADA)], cena)
        assert estado.fase is Fase.GIRANDO_PINCA

        x, y, _ = cena.vertices_tela()[1]
        estado, comando = avancar(estado, [mao(x=x, y=y, pinca=PINCA_FECHADA)], cena, 20)
        assert estado.fase is Fase.GIRANDO_PINCA
        assert comando.vertice_movido is None

    def test_gesto_termina_quando_a_mao_solta(self):
        cena = CenaFalsa()
        estado, _ = confirmar([mao(x=200.0, y=200.0, pinca=PINCA_FECHADA)], cena)
        estado, comando = avancar(estado, [mao(x=200.0, y=200.0, pinca=PINCA_ABERTA)], cena, 20)
        assert estado.fase is Fase.OCIOSO
        assert comando.fase is Fase.OCIOSO

    def test_gesto_termina_quando_a_mao_some(self):
        cena = CenaFalsa()
        estado, _ = confirmar([mao(x=200.0, y=200.0, pinca=PINCA_FECHADA)], cena)
        estado, _ = avancar(estado, [], cena, 20)
        assert estado.fase is Fase.OCIOSO

    def test_escala_termina_se_uma_das_maos_some(self):
        cena = CenaFalsa()
        maos = [mao(id_mao=0, x=300.0, pinca=PINCA_FECHADA),
                mao(id_mao=1, x=700.0, pinca=PINCA_FECHADA)]
        estado, _ = confirmar(maos, cena)
        assert estado.fase is Fase.ESCALANDO
        estado, _ = avancar(estado, [maos[0]], cena, 20)
        assert estado.fase is Fase.OCIOSO


class TestComandoDeOrbita:
    def test_mover_a_mao_agarrada_gera_delta(self):
        cena = CenaFalsa()
        estado, _ = confirmar([mao(x=200.0, y=200.0, pinca=PINCA_FECHADA)], cena)
        estado, comando = avancar(estado, [mao(x=230.0, y=180.0, pinca=PINCA_FECHADA)], cena, 20)
        assert comando.delta_orbita_tela == pytest.approx((30.0, -20.0))

    def test_mao_parada_nao_gera_delta(self):
        cena = CenaFalsa()
        estado, _ = confirmar([mao(x=200.0, y=200.0, pinca=PINCA_FECHADA)], cena)
        estado, comando = avancar(estado, [mao(x=200.0, y=200.0, pinca=PINCA_FECHADA)], cena, 20)
        assert comando.delta_orbita_tela == pytest.approx((0.0, 0.0))


class TestComandoDeEscala:
    def test_afastar_as_maos_aumenta(self):
        cena = CenaFalsa(escala=1.5)
        maos = [mao(id_mao=0, x=400.0, pinca=PINCA_FECHADA),
                mao(id_mao=1, x=600.0, pinca=PINCA_FECHADA)]
        estado, _ = confirmar(maos, cena)
        afastadas = [mao(id_mao=0, x=300.0, pinca=PINCA_FECHADA),
                     mao(id_mao=1, x=700.0, pinca=PINCA_FECHADA)]
        _, comando = avancar(estado, afastadas, cena, 20)
        assert comando.escala_absoluta == pytest.approx(1.5 * 2.0)

    def test_juntar_as_maos_diminui(self):
        cena = CenaFalsa(escala=2.0)
        maos = [mao(id_mao=0, x=300.0, pinca=PINCA_FECHADA),
                mao(id_mao=1, x=700.0, pinca=PINCA_FECHADA)]
        estado, _ = confirmar(maos, cena)
        juntas = [mao(id_mao=0, x=400.0, pinca=PINCA_FECHADA),
                  mao(id_mao=1, x=600.0, pinca=PINCA_FECHADA)]
        _, comando = avancar(estado, juntas, cena, 20)
        assert comando.escala_absoluta == pytest.approx(1.0)

    def test_escala_e_absoluta_e_nao_acumula_deriva(self):
        """Voltar as mãos à distância inicial tem de devolver a escala inicial,
        por mais que se mexa no meio do caminho."""
        cena = CenaFalsa(escala=1.5)
        inicio = [mao(id_mao=0, x=400.0, pinca=PINCA_FECHADA),
                  mao(id_mao=1, x=600.0, pinca=PINCA_FECHADA)]
        estado, _ = confirmar(inicio, cena)
        for larg in (250.0, 350.0, 150.0, 450.0):
            estado, _ = avancar(
                estado,
                [mao(id_mao=0, x=500.0 - larg, pinca=PINCA_FECHADA),
                 mao(id_mao=1, x=500.0 + larg, pinca=PINCA_FECHADA)],
                cena, 30,
            )
        _, comando = avancar(estado, inicio, cena, 40)
        assert comando.escala_absoluta == pytest.approx(1.5)

    def test_mover_as_duas_maos_juntas_faz_pan(self):
        cena = CenaFalsa()
        maos = [mao(id_mao=0, x=400.0, y=300.0, pinca=PINCA_FECHADA),
                mao(id_mao=1, x=600.0, y=300.0, pinca=PINCA_FECHADA)]
        estado, _ = confirmar(maos, cena)
        deslocadas = [mao(id_mao=0, x=450.0, y=320.0, pinca=PINCA_FECHADA),
                      mao(id_mao=1, x=650.0, y=320.0, pinca=PINCA_FECHADA)]
        _, comando = avancar(estado, deslocadas, cena, 20)
        assert comando.delta_pan_tela == pytest.approx((50.0, 20.0))
        assert comando.escala_absoluta == pytest.approx(cena.escala_objeto())


class TestComandoDeVertice:
    def test_arrastar_leva_o_vertice_para_a_posicao_do_cursor(self):
        cena = CenaFalsa()
        x, y, _ = cena.vertices_tela()[1]
        estado, _ = confirmar([mao(x=x, y=y, pinca=PINCA_FECHADA)], cena)
        estado, comando = avancar(estado, [mao(x=x + 100.0, y=y, pinca=PINCA_FECHADA)], cena, 20)
        assert comando.vertice_movido == 1
        # vértice 1 estava em (1,0,0); cursor andou 100px = 1 unidade
        assert comando.posicao_vertice_objeto == pytest.approx((2.0, 0.0, 0.0))

    def test_offset_evita_que_o_vertice_salte_ao_ser_agarrado(self):
        """Agarrar com o cursor um pouco ao lado do vértice não pode teletransportá-lo."""
        cena = CenaFalsa()
        x, y, _ = cena.vertices_tela()[1]
        estado, _ = confirmar([mao(x=x - 20.0, y=y + 10.0, pinca=PINCA_FECHADA)], cena)
        _, comando = avancar(estado, [mao(x=x - 20.0, y=y + 10.0, pinca=PINCA_FECHADA)], cena, 20)
        assert comando.posicao_vertice_objeto == pytest.approx((1.0, 0.0, 0.0))


class TestComandoDeConcha:
    def test_girar_a_mao_gira_o_objeto_no_referencial_da_tela(self):
        cena = CenaFalsa(orientacao=quaternion_de_eixo_angulo((1.0, 0.0, 0.0), 1.2))
        parada = mao(x=200.0, y=200.0, extensoes=EXT_CONCHA, orientacao=QUAT_IDENTIDADE)
        estado, _ = confirmar([parada], cena)
        assert estado.fase is Fase.GIRANDO_CONCHA

        giro = quaternion_de_eixo_angulo((0.0, 1.0, 0.0), 0.7)
        _, comando = avancar(
            estado, [mao(x=200.0, y=200.0, extensoes=EXT_CONCHA, orientacao=giro)], cena, 20
        )
        for v in [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.3, -0.6, 0.8)]:
            esperado = aplicar_quaternion(giro, aplicar_quaternion(cena.orientacao_objeto(), v))
            assert aplicar_quaternion(comando.orientacao_absoluta, v) == pytest.approx(esperado, abs=1e-9)

    def test_mao_parada_nao_gira_o_objeto(self):
        cena = CenaFalsa(orientacao=quaternion_de_eixo_angulo((1.0, 0.0, 0.0), 1.2))
        parada = mao(x=200.0, y=200.0, extensoes=EXT_CONCHA)
        estado, _ = confirmar([parada], cena)
        _, comando = avancar(estado, [parada], cena, 20)
        for v in [(1.0, 0.0, 0.0), (0.0, 0.0, 1.0)]:
            assert aplicar_quaternion(comando.orientacao_absoluta, v) == pytest.approx(
                aplicar_quaternion(cena.orientacao_objeto(), v), abs=1e-9
            )

    def test_abrir_a_mao_encerra_o_gesto(self):
        cena = CenaFalsa()
        estado, _ = confirmar([mao(x=200.0, y=200.0, extensoes=EXT_CONCHA)], cena)
        estado, _ = avancar(estado, [mao(x=200.0, y=200.0, extensoes=EXT_ABERTA)], cena, 20)
        assert estado.fase is Fase.OCIOSO


class TestRealceDeProximidade:
    def test_indica_o_vertice_sob_a_mira_sem_agarrar(self):
        cena = CenaFalsa()
        x, y, _ = cena.vertices_tela()[2]
        _, comando = confirmar([mao(x=x, y=y, pinca=PINCA_ABERTA)], cena)
        assert comando.fase is Fase.OCIOSO
        assert comando.vertice_sob_mira == 2

    def test_sem_vertice_por_perto_nao_indica_nada(self):
        _, comando = confirmar([mao(x=50.0, y=50.0)], CenaFalsa())
        assert comando.vertice_sob_mira is None


class TestPureza:
    def test_avaliar_nao_muda_o_estado_recebido(self):
        estado = EstadoInteracao()
        avancar(estado, [mao(pinca=PINCA_FECHADA, x=200.0, y=200.0)], CenaFalsa(), 1)
        assert estado == EstadoInteracao()

    def test_mesma_entrada_produz_mesma_saida(self):
        cena = CenaFalsa()
        m = [mao(x=200.0, y=200.0, pinca=PINCA_FECHADA)]
        a = avancar(EstadoInteracao(), m, cena, 1)
        b = avancar(EstadoInteracao(), m, cena, 1)
        assert a == b

    def test_ordem_das_maos_nao_altera_o_resultado(self):
        cena = CenaFalsa()
        maos = [mao(id_mao=0, x=300.0, pinca=PINCA_FECHADA),
                mao(id_mao=1, x=700.0, pinca=PINCA_FECHADA)]
        estado_a, _ = confirmar(maos, cena)
        estado_b, _ = confirmar(list(reversed(maos)), cena)
        assert estado_a.id_mao_ancora == estado_b.id_mao_ancora
        assert estado_a.id_mao_secundaria == estado_b.id_mao_secundaria
