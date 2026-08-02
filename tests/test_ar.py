"""Testes do encaixe do frame da câmera na janela (funções puras, sem OpenGL).

O teste mais importante é o de não-distorção: qualquer erro na conta de
aspecto deixa a imagem esticada, e o cursor da mão deixa de coincidir com a
mão no vídeo — que é o que faria a AR parecer quebrada.
"""
import pytest

from render.ar import (
    MODO_CABER,
    MODO_PREENCHER,
    calcular_ajuste,
    mapear_uv_para_tela,
)

# (largura_frame, altura_frame, largura_janela, altura_janela)
COMBINACOES = [
    (640, 480, 1024, 768),    # 4:3 em 4:3 — encaixe exato
    (1280, 720, 1024, 768),   # 16:9 em 4:3 — frame mais largo
    (640, 480, 1920, 800),    # 4:3 em janela panorâmica
    (480, 640, 1024, 768),    # frame em pé (retrato)
    (1280, 720, 1280, 720),   # idêntico
    (800, 600, 1000, 1000),   # janela quadrada
]


class TestCalcularAjuste:
    def test_aspectos_iguais_usam_o_frame_inteiro(self):
        a = calcular_ajuste(640, 480, 1024, 768, MODO_PREENCHER)
        assert (a.u0, a.v0, a.u1, a.v1) == pytest.approx((0.0, 0.0, 1.0, 1.0))
        assert (a.x0, a.y0, a.x1, a.y1) == pytest.approx((0.0, 0.0, 1024.0, 768.0))

    def test_preencher_sempre_cobre_a_janela_inteira(self):
        for lf, af, lj, aj in COMBINACOES:
            a = calcular_ajuste(lf, af, lj, aj, MODO_PREENCHER)
            assert (a.x0, a.y0, a.x1, a.y1) == pytest.approx((0.0, 0.0, float(lj), float(aj)))

    def test_preencher_com_frame_mais_largo_corta_as_laterais(self):
        a = calcular_ajuste(1280, 720, 1024, 768, MODO_PREENCHER)
        assert a.u0 > 0.0 and a.u1 < 1.0     # cortou na horizontal
        assert (a.v0, a.v1) == pytest.approx((0.0, 1.0))  # não cortou na vertical
        assert a.u0 == pytest.approx(1.0 - a.u1)          # corte simétrico

    def test_preencher_com_frame_mais_alto_corta_topo_e_base(self):
        a = calcular_ajuste(480, 640, 1024, 768, MODO_PREENCHER)
        assert a.v0 > 0.0 and a.v1 < 1.0
        assert (a.u0, a.u1) == pytest.approx((0.0, 1.0))
        assert a.v0 == pytest.approx(1.0 - a.v1)

    def test_caber_sempre_mostra_o_frame_inteiro(self):
        for lf, af, lj, aj in COMBINACOES:
            a = calcular_ajuste(lf, af, lj, aj, MODO_CABER)
            assert (a.u0, a.v0, a.u1, a.v1) == pytest.approx((0.0, 0.0, 1.0, 1.0))

    def test_caber_nunca_extrapola_a_janela(self):
        for lf, af, lj, aj in COMBINACOES:
            a = calcular_ajuste(lf, af, lj, aj, MODO_CABER)
            assert a.x0 >= -1e-9 and a.y0 >= -1e-9
            assert a.x1 <= lj + 1e-9 and a.y1 <= aj + 1e-9

    def test_caber_com_frame_mais_largo_gera_barras_horizontais(self):
        a = calcular_ajuste(1280, 720, 1024, 768, MODO_CABER)
        assert (a.x0, a.x1) == pytest.approx((0.0, 1024.0))  # encosta nas laterais
        assert a.y0 > 0.0 and a.y1 < 768.0                    # barras em cima e embaixo
        assert a.y0 == pytest.approx(768.0 - a.y1)            # barras simétricas

    @pytest.mark.parametrize("modo", [MODO_PREENCHER, MODO_CABER])
    @pytest.mark.parametrize("lf,af,lj,aj", COMBINACOES)
    def test_nao_distorce_a_imagem(self, modo, lf, af, lj, aj):
        """Pixels de tela por pixel de imagem tem de ser igual nos dois eixos —
        se divergir, a imagem aparece esticada."""
        a = calcular_ajuste(lf, af, lj, aj, modo)
        escala_x = (a.x1 - a.x0) / ((a.u1 - a.u0) * lf)
        escala_y = (a.y1 - a.y0) / ((a.v1 - a.v0) * af)
        assert escala_x == pytest.approx(escala_y, rel=1e-9)

    def test_dimensoes_invalidas_levantam_erro(self):
        with pytest.raises(ValueError):
            calcular_ajuste(0, 480, 1024, 768)
        with pytest.raises(ValueError):
            calcular_ajuste(640, 480, 1024, 0)
        with pytest.raises(ValueError):
            calcular_ajuste(640, -1, 1024, 768)

    def test_modo_desconhecido_levanta_erro(self):
        with pytest.raises(ValueError):
            calcular_ajuste(640, 480, 1024, 768, "esticar")


class TestMapearUvParaTela:
    @pytest.mark.parametrize("modo", [MODO_PREENCHER, MODO_CABER])
    @pytest.mark.parametrize("lf,af,lj,aj", COMBINACOES)
    def test_centro_do_frame_cai_no_centro_da_janela(self, modo, lf, af, lj, aj):
        a = calcular_ajuste(lf, af, lj, aj, modo)
        x, y = mapear_uv_para_tela(0.5, 0.5, a)
        assert (x, y) == pytest.approx((lj / 2, aj / 2))

    def test_no_modo_caber_os_cantos_do_frame_batem_com_os_cantos_do_quad(self):
        a = calcular_ajuste(1280, 720, 1024, 768, MODO_CABER)
        assert mapear_uv_para_tela(0.0, 0.0, a) == pytest.approx((a.x0, a.y0))
        assert mapear_uv_para_tela(1.0, 1.0, a) == pytest.approx((a.x1, a.y1))

    def test_no_modo_preencher_a_parte_cortada_cai_fora_da_janela(self):
        a = calcular_ajuste(1280, 720, 1024, 768, MODO_PREENCHER)
        x_esquerda, _ = mapear_uv_para_tela(0.0, 0.5, a)
        x_direita, _ = mapear_uv_para_tela(1.0, 0.5, a)
        assert x_esquerda < 0.0        # borda esquerda do frame foi cortada
        assert x_direita > 1024.0      # borda direita também

    @pytest.mark.parametrize("modo", [MODO_PREENCHER, MODO_CABER])
    def test_e_monotonica_nos_dois_eixos(self, modo):
        a = calcular_ajuste(1280, 720, 1024, 768, modo)
        xs = [mapear_uv_para_tela(u, 0.5, a)[0] for u in (0.1, 0.3, 0.5, 0.7, 0.9)]
        ys = [mapear_uv_para_tela(0.5, v, a)[1] for v in (0.1, 0.3, 0.5, 0.7, 0.9)]
        assert xs == sorted(xs)
        assert ys == sorted(ys)

    def test_v_cresce_para_baixo(self):
        """Convenção do pygame: v=0 é o topo do frame e vira y menor na tela.
        Inverter isso deixaria a imagem de cabeça para baixo."""
        a = calcular_ajuste(640, 480, 1024, 768, MODO_PREENCHER)
        _, y_topo = mapear_uv_para_tela(0.5, 0.0, a)
        _, y_base = mapear_uv_para_tela(0.5, 1.0, a)
        assert y_topo < y_base
