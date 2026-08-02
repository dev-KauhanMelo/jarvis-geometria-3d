from itertools import combinations
from math import sqrt

import pytest

from geometry.tetraedro import (
    ARESTAS_INDICES,
    FACES,
    Tetraedro,
    altura,
    apotema_face,
    area_face,
    area_total,
    calcular_normal_face,
    calcular_vertices,
    volume,
)


def _distancia(a, b):
    return sqrt(sum((a[i] - b[i]) ** 2 for i in range(3)))


class TestValoresConhecidosAresta1:
    def test_area_total(self):
        assert area_total(1) == pytest.approx(sqrt(3))

    def test_volume(self):
        assert volume(1) == pytest.approx(sqrt(2) / 12)

    def test_altura(self):
        assert altura(1) == pytest.approx(sqrt(6) / 3)

    def test_apotema_face(self):
        assert apotema_face(1) == pytest.approx(sqrt(3) / 2)


class TestValoresConhecidosAresta2:
    def test_area_total(self):
        assert area_total(2) == pytest.approx(4 * sqrt(3))

    def test_volume(self):
        assert volume(2) == pytest.approx(2 * sqrt(2) / 3)

    def test_altura(self):
        assert altura(2) == pytest.approx(2 * sqrt(6) / 3)

    def test_apotema_face(self):
        assert apotema_face(2) == pytest.approx(sqrt(3))


class TestConsistenciaEscala:
    @pytest.mark.parametrize("a", [0.5, 1.0, 1.5, 3.7])
    def test_area_total_igual_quatro_area_face(self, a):
        assert area_total(a) == pytest.approx(4 * area_face(a))

    @pytest.mark.parametrize("a", [0.5, 1.0, 1.5, 3.7])
    def test_volume_escala_com_cubo(self, a):
        assert volume(2 * a) == pytest.approx(8 * volume(a))

    @pytest.mark.parametrize("a", [0.5, 1.0, 1.5, 3.7])
    def test_area_total_escala_com_quadrado(self, a):
        assert area_total(2 * a) == pytest.approx(4 * area_total(a))


class TestCasosDeBorda:
    @pytest.mark.parametrize("a", [0, -1, -0.001])
    def test_area_face_levanta_erro(self, a):
        with pytest.raises(ValueError):
            area_face(a)

    @pytest.mark.parametrize("a", [0, -1, -0.001])
    def test_area_total_levanta_erro(self, a):
        with pytest.raises(ValueError):
            area_total(a)

    @pytest.mark.parametrize("a", [0, -1, -0.001])
    def test_volume_levanta_erro(self, a):
        with pytest.raises(ValueError):
            volume(a)

    @pytest.mark.parametrize("a", [0, -1, -0.001])
    def test_altura_levanta_erro(self, a):
        with pytest.raises(ValueError):
            altura(a)

    @pytest.mark.parametrize("a", [0, -1, -0.001])
    def test_apotema_face_levanta_erro(self, a):
        with pytest.raises(ValueError):
            apotema_face(a)

    @pytest.mark.parametrize("a", [0, -1, -0.001])
    def test_calcular_vertices_levanta_erro(self, a):
        with pytest.raises(ValueError):
            calcular_vertices(a)

    @pytest.mark.parametrize("a", [0, -1, -0.001])
    def test_tetraedro_construtor_levanta_erro(self, a):
        with pytest.raises(ValueError):
            Tetraedro(aresta=a)


class TestCalcularVertices:
    @pytest.mark.parametrize("a", [0.5, 1.0, 2.3])
    def test_retorna_quatro_pontos(self, a):
        assert len(calcular_vertices(a)) == 4

    @pytest.mark.parametrize("a", [0.5, 1.0, 2.3])
    def test_todas_distancias_par_a_par_iguais_a_aresta(self, a):
        vertices = calcular_vertices(a)
        for p1, p2 in combinations(vertices, 2):
            assert _distancia(p1, p2) == pytest.approx(a)

    @pytest.mark.parametrize("a", [0.5, 1.0, 2.3])
    def test_centroide_e_origem(self, a):
        vertices = calcular_vertices(a)
        cx = sum(v[0] for v in vertices) / 4
        cy = sum(v[1] for v in vertices) / 4
        cz = sum(v[2] for v in vertices) / 4
        assert (cx, cy, cz) == pytest.approx((0.0, 0.0, 0.0), abs=1e-9)


class TestTopologia:
    def test_faces_tem_quatro_triplas_com_indices_distintos(self):
        assert len(FACES) == 4
        for face in FACES:
            assert len(set(face)) == 3
            assert all(0 <= i <= 3 for i in face)

    def test_faces_cobrem_todas_as_triplas_possiveis(self):
        esperado = set(combinations(range(4), 3))
        assert set(FACES) == esperado

    def test_arestas_tem_seis_pares_com_indices_distintos(self):
        assert len(ARESTAS_INDICES) == 6
        for aresta in ARESTAS_INDICES:
            assert len(set(aresta)) == 2
            assert all(0 <= i <= 3 for i in aresta)

    def test_arestas_cobrem_todos_os_pares_possiveis(self):
        esperado = set(combinations(range(4), 2))
        assert set(ARESTAS_INDICES) == esperado


class TestCalcularNormalFace:
    @pytest.mark.parametrize("a", [0.5, 1.0, 2.3])
    def test_normal_e_unitaria_e_aponta_para_fora(self, a):
        vertices = calcular_vertices(a)
        centroide_solido = (
            sum(v[0] for v in vertices) / 4,
            sum(v[1] for v in vertices) / 4,
            sum(v[2] for v in vertices) / 4,
        )
        for face in FACES:
            normal = calcular_normal_face(vertices, face)
            modulo = sqrt(sum(c**2 for c in normal))
            assert modulo == pytest.approx(1.0)

            centroide_face = (
                sum(vertices[i][0] for i in face) / 3,
                sum(vertices[i][1] for i in face) / 3,
                sum(vertices[i][2] for i in face) / 3,
            )
            direcao_externa = tuple(
                centroide_face[i] - centroide_solido[i] for i in range(3)
            )
            produto_escalar = sum(normal[i] * direcao_externa[i] for i in range(3))
            assert produto_escalar > 0


class TestTetraedroAjustarAresta:
    def test_incremento_simples(self):
        t = Tetraedro(aresta=1.5, aresta_min=0.3, aresta_max=4.0)
        t.ajustar_aresta(0.5)
        assert t.aresta == pytest.approx(2.0)

    def test_decremento_simples(self):
        t = Tetraedro(aresta=1.5, aresta_min=0.3, aresta_max=4.0)
        t.ajustar_aresta(-0.5)
        assert t.aresta == pytest.approx(1.0)

    def test_respeita_limite_maximo(self):
        t = Tetraedro(aresta=3.9, aresta_min=0.3, aresta_max=4.0)
        t.ajustar_aresta(1.0)
        assert t.aresta == pytest.approx(4.0)

    def test_respeita_limite_minimo(self):
        t = Tetraedro(aresta=0.4, aresta_min=0.3, aresta_max=4.0)
        t.ajustar_aresta(-1.0)
        assert t.aresta == pytest.approx(0.3)

    def test_nunca_chega_a_zero_ou_negativo(self):
        t = Tetraedro(aresta=0.3, aresta_min=0.3, aresta_max=4.0)
        t.ajustar_aresta(-10.0)
        assert t.aresta > 0


class TestTetraedroResetar:
    def test_volta_para_aresta_padrao(self):
        t = Tetraedro(aresta=1.5, aresta_min=0.3, aresta_max=4.0)
        t.ajustar_aresta(1.0)
        assert t.aresta != pytest.approx(1.5)
        t.resetar()
        assert t.aresta == pytest.approx(1.5)

    def test_aceita_aresta_padrao_explicita(self):
        t = Tetraedro(aresta=1.5, aresta_min=0.3, aresta_max=4.0)
        t.resetar(aresta_padrao=2.5)
        assert t.aresta == pytest.approx(2.5)
