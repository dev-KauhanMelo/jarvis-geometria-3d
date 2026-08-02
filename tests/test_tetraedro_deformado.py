"""Testes das fórmulas genéricas (tetraedro qualquer) e do estado deformado.

O teste mais valioso aqui é o cross-validation: com o sólido regular, os
cálculos numéricos genéricos têm de bater com as fórmulas fechadas do ensino
médio. São dois caminhos de cálculo independentes chegando ao mesmo número.
"""
import pytest

from geometry.tetraedro import (
    FACES,
    Tetraedro,
    altura,
    altura_generica,
    apotema_face,
    apotema_face_generica,
    area_face_generica,
    area_total,
    area_total_generica,
    calcular_vertices,
    centroide_solido,
    comprimentos_arestas,
    e_regular,
    escalar_em_torno_do_centroide,
    volume,
    volume_generico,
)

ARESTAS_TESTE = [0.5, 1.0, 1.5, 2.0, 3.7]


class TestCrossValidationComFormulasAnaliticas:
    @pytest.mark.parametrize("a", ARESTAS_TESTE)
    def test_area_total(self, a):
        assert area_total_generica(calcular_vertices(a)) == pytest.approx(area_total(a))

    @pytest.mark.parametrize("a", ARESTAS_TESTE)
    def test_volume(self, a):
        assert volume_generico(calcular_vertices(a)) == pytest.approx(volume(a))

    @pytest.mark.parametrize("a", ARESTAS_TESTE)
    def test_altura(self, a):
        assert altura_generica(calcular_vertices(a)) == pytest.approx(altura(a))

    @pytest.mark.parametrize("a", ARESTAS_TESTE)
    def test_apotema_face(self, a):
        assert apotema_face_generica(calcular_vertices(a), FACES[0]) == pytest.approx(apotema_face(a))

    @pytest.mark.parametrize("a", ARESTAS_TESTE)
    @pytest.mark.parametrize("apice", [0, 1, 2, 3])
    def test_altura_independe_do_vertice_escolhido_no_caso_regular(self, a, apice):
        assert altura_generica(calcular_vertices(a), apice) == pytest.approx(altura(a))


class TestFuncoesGenericas:
    @pytest.mark.parametrize("a", ARESTAS_TESTE)
    def test_as_seis_arestas_sao_iguais_no_regular(self, a):
        medidas = comprimentos_arestas(calcular_vertices(a))
        assert len(medidas) == 6
        for m in medidas:
            assert m == pytest.approx(a)

    def test_volume_e_nao_negativo_independente_do_winding(self):
        v = calcular_vertices(1.0)
        trocado = [v[1], v[0], v[2], v[3]]  # inverte orientação
        assert volume_generico(trocado) == pytest.approx(volume_generico(v))
        assert volume_generico(trocado) > 0

    def test_volume_de_solido_achatado_tende_a_zero(self):
        achatado = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0.5, 0.5, 1e-9)]
        assert volume_generico(achatado) == pytest.approx(0.0, abs=1e-9)

    def test_e_regular_reconhece_o_regular(self):
        assert e_regular(calcular_vertices(1.7)) is True

    def test_e_regular_rejeita_deformado(self):
        v = calcular_vertices(1.0)
        v[0] = (v[0][0] + 0.5, v[0][1], v[0][2])
        assert e_regular(v) is False

    def test_centroide_do_regular_e_a_origem(self):
        assert centroide_solido(calcular_vertices(2.0)) == pytest.approx((0.0, 0.0, 0.0), abs=1e-9)

    @pytest.mark.parametrize("fator", [0.5, 2.0, 3.0])
    def test_escalar_preserva_centroide_e_multiplica_arestas(self, fator):
        v = calcular_vertices(1.0)
        escalado = escalar_em_torno_do_centroide(v, fator)
        assert centroide_solido(escalado) == pytest.approx(centroide_solido(v), abs=1e-9)
        for original, novo in zip(comprimentos_arestas(v), comprimentos_arestas(escalado)):
            assert novo == pytest.approx(original * fator)

    def test_area_de_face_conhecida(self):
        # triângulo retângulo de catetos 3 e 4 -> área 6
        vertices = [(0, 0, 0), (3, 0, 0), (0, 4, 0), (0, 0, 1)]
        assert area_face_generica(vertices, (0, 1, 2)) == pytest.approx(6.0)


class TestEstadoDeformado:
    def test_comeca_regular(self):
        assert Tetraedro(aresta=1.5).esta_regular() is True

    def test_mover_vertice_entra_em_estado_deformado(self):
        t = Tetraedro(aresta=1.5)
        v = t.vertices()
        assert t.mover_vertice(0, (v[0][0] + 0.8, v[0][1], v[0][2])) is True
        assert t.esta_regular() is False

    def test_vertice_movido_fica_na_posicao_pedida(self):
        t = Tetraedro(aresta=1.5)
        alvo = (2.0, 1.0, 0.5)
        assert t.mover_vertice(2, alvo) is True
        assert t.vertices()[2] == pytest.approx(alvo)

    def test_outros_vertices_nao_se_movem(self):
        t = Tetraedro(aresta=1.5)
        antes = t.vertices()
        t.mover_vertice(1, (3.0, 0.0, 0.0))
        depois = t.vertices()
        for i in (0, 2, 3):
            assert depois[i] == pytest.approx(antes[i])

    def test_movimento_degenerado_e_recusado(self):
        """Puxar um vértice para o plano dos outros três achataria o sólido —
        deve ser recusado antes de gerar normal indefinida no render."""
        t = Tetraedro(aresta=1.5)
        v = t.vertices()
        centro_face_oposta = (
            sum(v[i][0] for i in (1, 2, 3)) / 3,
            sum(v[i][1] for i in (1, 2, 3)) / 3,
            sum(v[i][2] for i in (1, 2, 3)) / 3,
        )
        assert t.mover_vertice(0, centro_face_oposta) is False
        assert t.esta_regular() is True  # nada mudou

    def test_indice_invalido_levanta_erro(self):
        t = Tetraedro(aresta=1.5)
        for indice in (-1, 4, 99):
            with pytest.raises(IndexError):
                t.mover_vertice(indice, (1.0, 1.0, 1.0))

    def test_valores_geometricos_mudam_apos_deformar(self):
        t = Tetraedro(aresta=1.5)
        volume_antes = t.volume()
        v = t.vertices()
        t.mover_vertice(0, (v[0][0] * 2.5, v[0][1] * 2.5, v[0][2] * 2.5))
        assert t.volume() != pytest.approx(volume_antes)
        assert t.volume() > 0

    def test_resetar_volta_ao_regular(self):
        t = Tetraedro(aresta=1.5)
        t.mover_vertice(0, (2.0, 2.0, 2.0))
        assert t.esta_regular() is False
        t.resetar()
        assert t.esta_regular() is True
        assert t.vertices() == pytest.approx(calcular_vertices(1.5))
        assert t.volume() == pytest.approx(volume(1.5))

    def test_metodos_despacham_para_generico_quando_deformado(self):
        t = Tetraedro(aresta=1.5)
        t.mover_vertice(0, (2.0, 1.5, 1.0))
        vertices = t.vertices()
        assert t.volume() == pytest.approx(volume_generico(vertices))
        assert t.area_total() == pytest.approx(area_total_generica(vertices))
        assert t.altura() == pytest.approx(altura_generica(vertices))


class TestEscalar:
    def test_escala_multiplica_a_aresta_no_regular(self):
        t = Tetraedro(aresta=1.0, aresta_min=0.3, aresta_max=4.0)
        t.escalar(2.0)
        assert t.aresta == pytest.approx(2.0)
        assert t.aresta_media() == pytest.approx(2.0)

    def test_respeita_limite_maximo(self):
        t = Tetraedro(aresta=3.0, aresta_min=0.3, aresta_max=4.0)
        t.escalar(10.0)
        assert t.aresta == pytest.approx(4.0)

    def test_respeita_limite_minimo(self):
        t = Tetraedro(aresta=1.0, aresta_min=0.3, aresta_max=4.0)
        t.escalar(0.01)
        assert t.aresta == pytest.approx(0.3)

    def test_fator_nao_positivo_levanta_erro(self):
        t = Tetraedro(aresta=1.0)
        for fator in (0.0, -1.0):
            with pytest.raises(ValueError):
                t.escalar(fator)

    def test_escalar_deformado_preserva_a_forma(self):
        """Escalar um sólido deformado muda o tamanho, não as proporções."""
        t = Tetraedro(aresta=1.0, aresta_max=10.0)
        t.mover_vertice(0, (1.5, 0.8, 0.3))
        proporcoes_antes = [m / t.aresta_media() for m in t.comprimentos_arestas()]
        t.escalar(2.0)
        proporcoes_depois = [m / t.aresta_media() for m in t.comprimentos_arestas()]
        assert proporcoes_depois == pytest.approx(proporcoes_antes)

    def test_escalar_deformado_preserva_centroide(self):
        t = Tetraedro(aresta=1.0, aresta_max=10.0)
        t.mover_vertice(0, (1.5, 0.8, 0.3))
        antes = t.centroide()
        t.escalar(1.7)
        assert t.centroide() == pytest.approx(antes, abs=1e-9)


class TestAjustarArestaDeformado:
    def test_ajustar_aresta_deformado_escala_sem_perder_a_forma(self):
        t = Tetraedro(aresta=1.0, aresta_max=10.0)
        t.mover_vertice(0, (1.5, 0.8, 0.3))
        assert t.esta_regular() is False
        proporcoes_antes = [m / t.aresta_media() for m in t.comprimentos_arestas()]
        t.ajustar_aresta(0.5)
        assert t.esta_regular() is False  # continua deformado
        proporcoes_depois = [m / t.aresta_media() for m in t.comprimentos_arestas()]
        assert proporcoes_depois == pytest.approx(proporcoes_antes)


class TestArestaMedia:
    @pytest.mark.parametrize("a", ARESTAS_TESTE)
    def test_no_regular_e_a_propria_aresta(self, a):
        assert Tetraedro(aresta=a).aresta_media() == pytest.approx(a)
