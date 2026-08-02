from math import cos, isclose, pi, radians, sin

import pytest

from geometry.transformacoes import (
    MAT3_IDENTIDADE,
    QUAT_IDENTIDADE,
    alfa_temporal,
    angulo_entre_quaternions,
    aplicar_matriz,
    aplicar_quaternion,
    base_destra,
    centroide,
    compor_rotacao_relativa,
    conjugado,
    determinante,
    eh_ortonormal,
    escalar_vetor,
    matriz4_coluna_maior_de_quaternion,
    matriz_de_colunas,
    matriz_de_quaternion,
    multiplicar_matrizes,
    multiplicar_quaternions,
    norma,
    normalizar,
    normalizar_quaternion,
    produto_escalar,
    produto_vetorial,
    quaternion_de_eixo_angulo,
    quaternion_de_euler_graus,
    quaternion_de_matriz,
    slerp,
    somar,
    subtrair,
    transpor,
)

EIXO_X = (1.0, 0.0, 0.0)
EIXO_Y = (0.0, 1.0, 0.0)
EIXO_Z = (0.0, 0.0, 1.0)


def _quats_iguais(a, b, tol=1e-9) -> bool:
    """q e -q representam a mesma rotação — compara nos dois sinais."""
    mesmo = all(isclose(x, y, abs_tol=tol) for x, y in zip(a, b))
    oposto = all(isclose(x, -y, abs_tol=tol) for x, y in zip(a, b))
    return mesmo or oposto


class TestVetores:
    def test_operacoes_basicas(self):
        assert somar((1, 2, 3), (4, 5, 6)) == (5, 7, 9)
        assert subtrair((4, 5, 6), (1, 2, 3)) == (3, 3, 3)
        assert escalar_vetor((1, 2, 3), 2) == (2, 4, 6)
        assert produto_escalar((1, 2, 3), (4, 5, 6)) == 32

    def test_produto_vetorial_segue_regra_da_mao_direita(self):
        assert produto_vetorial(EIXO_X, EIXO_Y) == pytest.approx(EIXO_Z)
        assert produto_vetorial(EIXO_Y, EIXO_Z) == pytest.approx(EIXO_X)
        assert produto_vetorial(EIXO_Z, EIXO_X) == pytest.approx(EIXO_Y)

    def test_norma_e_normalizar(self):
        assert norma((3.0, 4.0, 0.0)) == pytest.approx(5.0)
        assert norma(normalizar((3.0, 4.0, 0.0))) == pytest.approx(1.0)

    def test_normalizar_vetor_nulo_levanta_erro(self):
        with pytest.raises(ValueError):
            normalizar((0.0, 0.0, 0.0))

    def test_centroide(self):
        assert centroide([(0, 0, 0), (2, 0, 0), (0, 2, 0), (0, 0, 2)]) == pytest.approx((0.5, 0.5, 0.5))

    def test_centroide_de_sequencia_vazia_levanta_erro(self):
        with pytest.raises(ValueError):
            centroide([])


class TestMatrizes:
    def test_identidade_e_neutra(self):
        v = (1.0, 2.0, 3.0)
        assert aplicar_matriz(MAT3_IDENTIDADE, v) == pytest.approx(v)

    def test_transpor_e_involutivo(self):
        m = ((1.0, 2.0, 3.0), (4.0, 5.0, 6.0), (7.0, 8.0, 9.0))
        assert transpor(transpor(m)) == m

    def test_multiplicacao_compoe_na_ordem_correta(self):
        # aplicar(a·b, v) deve ser igual a aplicar(a, aplicar(b, v))
        a = matriz_de_quaternion(quaternion_de_eixo_angulo(EIXO_X, 0.7))
        b = matriz_de_quaternion(quaternion_de_eixo_angulo(EIXO_Y, 1.1))
        v = (0.3, -0.5, 0.9)
        assert aplicar_matriz(multiplicar_matrizes(a, b), v) == pytest.approx(
            aplicar_matriz(a, aplicar_matriz(b, v))
        )

    def test_matriz_de_colunas_poe_vetores_nas_colunas(self):
        m = matriz_de_colunas((1, 2, 3), (4, 5, 6), (7, 8, 9))
        assert aplicar_matriz(m, EIXO_X) == pytest.approx((1, 2, 3))
        assert aplicar_matriz(m, EIXO_Y) == pytest.approx((4, 5, 6))
        assert aplicar_matriz(m, EIXO_Z) == pytest.approx((7, 8, 9))

    def test_determinante_da_identidade(self):
        assert determinante(MAT3_IDENTIDADE) == pytest.approx(1.0)


class TestBaseDestra:
    def test_base_e_ortonormal_e_destra(self):
        m = base_destra((0.0, 0.0, 2.0), (0.0, 3.0, 0.0))
        assert eh_ortonormal(m)
        assert determinante(m) == pytest.approx(1.0)

    def test_terceira_coluna_e_o_ez_normalizado(self):
        m = base_destra((0.0, 0.0, 5.0), (0.0, 1.0, 0.0))
        assert aplicar_matriz(m, EIXO_Z) == pytest.approx(EIXO_Z)

    def test_referencia_nao_precisa_ser_perpendicular(self):
        # ey de referência propositalmente inclinado: Gram-Schmidt ortogonaliza
        m = base_destra((0.0, 0.0, 1.0), (0.4, 1.0, 0.7))
        assert eh_ortonormal(m)

    def test_vetores_paralelos_levantam_erro(self):
        with pytest.raises(ValueError):
            base_destra((0.0, 0.0, 1.0), (0.0, 0.0, 3.0))


class TestQuaternionBasico:
    def test_identidade_nao_rotaciona(self):
        v = (1.0, 2.0, 3.0)
        assert aplicar_quaternion(QUAT_IDENTIDADE, v) == pytest.approx(v)

    def test_rotacao_90_graus_em_z_leva_x_em_y(self):
        q = quaternion_de_eixo_angulo(EIXO_Z, pi / 2)
        assert aplicar_quaternion(q, EIXO_X) == pytest.approx(EIXO_Y, abs=1e-9)

    def test_conjugado_desfaz_a_rotacao(self):
        q = quaternion_de_eixo_angulo((1.0, 2.0, 3.0), 0.9)
        v = (0.4, -0.2, 0.7)
        assert aplicar_quaternion(conjugado(q), aplicar_quaternion(q, v)) == pytest.approx(v)

    def test_multiplicacao_compoe_rotacoes(self):
        a = quaternion_de_eixo_angulo(EIXO_X, 0.6)
        b = quaternion_de_eixo_angulo(EIXO_Y, 1.2)
        v = (0.3, 0.5, -0.8)
        assert aplicar_quaternion(multiplicar_quaternions(a, b), v) == pytest.approx(
            aplicar_quaternion(a, aplicar_quaternion(b, v))
        )

    def test_normalizar_quaternion_nulo_levanta_erro(self):
        with pytest.raises(ValueError):
            normalizar_quaternion((0.0, 0.0, 0.0, 0.0))

    def test_aplicar_quaternion_bate_com_aplicar_matriz(self):
        q = quaternion_de_eixo_angulo((0.3, -0.7, 0.2), 1.4)
        v = (1.1, -0.4, 0.6)
        assert aplicar_quaternion(q, v) == pytest.approx(aplicar_matriz(matriz_de_quaternion(q), v))


class TestConversaoMatrizQuaternion:
    @pytest.mark.parametrize("angulo", [0.0, 0.3, 1.0, 2.0, 3.0])
    @pytest.mark.parametrize("eixo", [EIXO_X, EIXO_Y, EIXO_Z, (1.0, 1.0, 0.0), (0.3, -0.6, 0.8)])
    def test_ida_e_volta_preserva_a_rotacao(self, eixo, angulo):
        q = quaternion_de_eixo_angulo(eixo, angulo)
        assert _quats_iguais(quaternion_de_matriz(matriz_de_quaternion(q)), q, tol=1e-7)

    def test_matriz_de_rotacao_e_sempre_ortonormal(self):
        q = quaternion_de_eixo_angulo((0.2, 0.9, -0.4), 2.3)
        assert eh_ortonormal(matriz_de_quaternion(q))

    @pytest.mark.parametrize("eixo", [EIXO_X, EIXO_Y, EIXO_Z, (0.6, 0.8, 0.0)])
    def test_estabilidade_proximo_de_180_graus(self, eixo):
        """Caso que a fórmula ingênua (sqrt(1+traço)) degenera: traço -> -1.
        Acontece de verdade quando o usuário vira a palma para trás."""
        q = quaternion_de_eixo_angulo(eixo, pi - 1e-7)
        recuperado = quaternion_de_matriz(matriz_de_quaternion(q))
        assert _quats_iguais(recuperado, q, tol=1e-5)

    def test_quaternion_de_matriz_nao_normalizada_ainda_normaliza(self):
        q = quaternion_de_matriz(MAT3_IDENTIDADE)
        assert norma((q[1], q[2], q[3])) == pytest.approx(0.0, abs=1e-12)
        assert abs(q[0]) == pytest.approx(1.0)


class TestEulerCompativelComOpenGL:
    def test_reproduz_composicao_do_glrotatef(self):
        """glRotatef(x,1,0,0) seguido de glRotatef(y,0,1,0) resulta em M = Rx·Ry."""
        ax, ay = 20.0, -30.0
        esperado = multiplicar_matrizes(
            matriz_de_quaternion(quaternion_de_eixo_angulo(EIXO_X, radians(ax))),
            matriz_de_quaternion(quaternion_de_eixo_angulo(EIXO_Y, radians(ay))),
        )
        obtido = matriz_de_quaternion(quaternion_de_euler_graus(ax, ay))
        for i in range(3):
            assert obtido[i] == pytest.approx(esperado[i], abs=1e-12)

    def test_angulos_zero_dao_identidade(self):
        assert _quats_iguais(quaternion_de_euler_graus(0.0, 0.0), QUAT_IDENTIDADE)


class TestSlerp:
    def test_extremos(self):
        a = quaternion_de_eixo_angulo(EIXO_X, 0.2)
        b = quaternion_de_eixo_angulo(EIXO_Y, 1.9)
        assert _quats_iguais(slerp(a, b, 0.0), a, tol=1e-9)
        assert _quats_iguais(slerp(a, b, 1.0), b, tol=1e-9)

    def test_meio_do_caminho_tem_angulo_igual_dos_dois_lados(self):
        a = QUAT_IDENTIDADE
        b = quaternion_de_eixo_angulo(EIXO_Z, 1.5)
        meio = slerp(a, b, 0.5)
        assert angulo_entre_quaternions(a, meio) == pytest.approx(
            angulo_entre_quaternions(meio, b), abs=1e-9
        )

    def test_resultado_sempre_unitario(self):
        a = quaternion_de_eixo_angulo((0.4, 0.2, -0.9), 0.4)
        b = quaternion_de_eixo_angulo((-0.1, 0.8, 0.3), 2.7)
        for t in (0.0, 0.13, 0.5, 0.87, 1.0):
            q = slerp(a, b, t)
            assert norma((q[1], q[2], q[3])) ** 2 + q[0] ** 2 == pytest.approx(1.0)

    def test_toma_o_caminho_curto_quando_sinais_sao_opostos(self):
        a = QUAT_IDENTIDADE
        b_curto = quaternion_de_eixo_angulo(EIXO_Z, 0.4)
        b_longo = tuple(-c for c in b_curto)  # mesma rotação, hemisfério oposto
        assert _quats_iguais(slerp(a, b_longo, 0.5), slerp(a, b_curto, 0.5), tol=1e-9)

    def test_t_fora_do_intervalo_e_clampado(self):
        a = QUAT_IDENTIDADE
        b = quaternion_de_eixo_angulo(EIXO_Y, 1.0)
        assert _quats_iguais(slerp(a, b, -5.0), a, tol=1e-9)
        assert _quats_iguais(slerp(a, b, 5.0), b, tol=1e-9)


class TestMatriz4ColunaMaior:
    def test_layout_column_major_de_rotacao_conhecida(self):
        """90° em torno de Z. Se sair transposto (row-major), este teste quebra —
        é o erro nº1 do pipeline fixo do OpenGL."""
        q = quaternion_de_eixo_angulo(EIXO_Z, pi / 2)
        m = matriz4_coluna_maior_de_quaternion(q)
        esperado = (
            0.0, 1.0, 0.0, 0.0,   # 1ª coluna: imagem de X, que vira +Y
            -1.0, 0.0, 0.0, 0.0,  # 2ª coluna: imagem de Y, que vira -X
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        )
        assert m == pytest.approx(esperado, abs=1e-12)

    def test_identidade_vira_matriz_identidade_4x4(self):
        m = matriz4_coluna_maior_de_quaternion(QUAT_IDENTIDADE)
        esperado = (1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1)
        assert m == pytest.approx(esperado, abs=1e-12)

    def test_tem_16_elementos(self):
        assert len(matriz4_coluna_maior_de_quaternion(QUAT_IDENTIDADE)) == 16


class TestComporRotacaoRelativa:
    """O comportamento mais crítico da Etapa 3: o delta da mão precisa agir no
    referencial da CÂMERA (multiplicação à esquerda), não no do objeto."""

    def test_mao_parada_nao_move_o_objeto(self):
        q_obj = quaternion_de_eixo_angulo((0.3, 0.5, -0.8), 1.1)
        q_mao = quaternion_de_eixo_angulo((0.7, -0.2, 0.4), 0.6)
        assert _quats_iguais(compor_rotacao_relativa(q_obj, q_mao, q_mao), q_obj, tol=1e-9)

    def test_delta_da_mao_age_no_referencial_do_mundo(self):
        """Girar a mão 90° em torno do Y da tela deve girar o objeto 90° em torno
        do Y da TELA, qualquer que seja a orientação em que o objeto já estava."""
        q_obj_inicial = quaternion_de_eixo_angulo(EIXO_X, pi / 2)
        q_mao_inicial = QUAT_IDENTIDADE
        delta = quaternion_de_eixo_angulo(EIXO_Y, pi / 2)
        q_mao_atual = delta

        resultado = compor_rotacao_relativa(q_obj_inicial, q_mao_inicial, q_mao_atual)

        for v in [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (0.3, -0.6, 0.8)]:
            esperado = aplicar_quaternion(delta, aplicar_quaternion(q_obj_inicial, v))
            assert aplicar_quaternion(resultado, v) == pytest.approx(esperado, abs=1e-9)

    def test_difere_da_composicao_a_direita(self):
        """Prova que o teste acima discrimina de fato: a forma errada (à direita)
        produz um resultado diferente, então não passaria por acidente."""
        q_obj_inicial = quaternion_de_eixo_angulo(EIXO_X, pi / 2)
        delta = quaternion_de_eixo_angulo(EIXO_Y, pi / 2)

        correto = compor_rotacao_relativa(q_obj_inicial, QUAT_IDENTIDADE, delta)
        errado = multiplicar_quaternions(q_obj_inicial, delta)  # composição à direita
        assert not _quats_iguais(correto, errado, tol=1e-6)

    def test_resultado_sempre_unitario_apos_muitas_composicoes(self):
        """A 60fps o gesto compõe milhares de vezes; sem renormalizar, deriva."""
        q = QUAT_IDENTIDADE
        passo = quaternion_de_eixo_angulo((0.2, 0.9, 0.3), 0.01)
        anterior = QUAT_IDENTIDADE
        for _ in range(5000):
            atual = multiplicar_quaternions(passo, anterior)
            q = compor_rotacao_relativa(q, anterior, atual)
            anterior = atual
        soma_quadrados = sum(c * c for c in q)
        assert soma_quadrados == pytest.approx(1.0, abs=1e-9)


class TestAnguloEntreQuaternions:
    def test_angulo_com_ele_mesmo_e_zero(self):
        q = quaternion_de_eixo_angulo((0.3, 0.4, 0.5), 1.2)
        assert angulo_entre_quaternions(q, q) == pytest.approx(0.0, abs=1e-7)

    @pytest.mark.parametrize("angulo", [0.1, 0.5, 1.0, 2.0])
    def test_recupera_o_angulo_de_uma_rotacao_simples(self, angulo):
        q = quaternion_de_eixo_angulo(EIXO_Z, angulo)
        assert angulo_entre_quaternions(QUAT_IDENTIDADE, q) == pytest.approx(angulo, abs=1e-7)

    def test_sinal_oposto_representa_a_mesma_rotacao(self):
        q = quaternion_de_eixo_angulo(EIXO_Y, 1.0)
        oposto = tuple(-c for c in q)
        assert angulo_entre_quaternions(q, oposto) == pytest.approx(0.0, abs=1e-7)


class TestAlfaTemporal:
    def test_alfa_cresce_com_o_intervalo(self):
        assert alfa_temporal(1 / 60, 0.08) < alfa_temporal(1 / 20, 0.08)

    def test_tau_zero_ou_negativo_desliga_a_suavizacao(self):
        assert alfa_temporal(0.016, 0.0) == 1.0
        assert alfa_temporal(0.016, -1.0) == 1.0

    def test_alfa_fica_no_intervalo_unitario(self):
        for dt in (0.0, 0.001, 0.016, 0.05, 1.0, 10.0):
            assert 0.0 <= alfa_temporal(dt, 0.08) <= 1.0

    def test_e_independente_da_taxa_de_amostragem(self):
        """Mesmo tempo total de suavização deve convergir ao mesmo valor,
        independente de rodar a 60fps ou a 20fps — a propriedade que o alfa
        fixo da Etapa 2 não tinha."""
        tau, alvo, total_s = 0.08, 1.0, 1.0

        def convergir(fps: int) -> float:
            dt = 1.0 / fps
            valor = 0.0
            for _ in range(int(total_s * fps)):
                valor += alfa_temporal(dt, tau) * (alvo - valor)
            return valor

        assert convergir(60) == pytest.approx(convergir(20), abs=1e-3)
