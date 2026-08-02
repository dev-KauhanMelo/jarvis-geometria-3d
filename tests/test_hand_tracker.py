import pytest

from vision.hand_tracker import (
    ANELAR_MCP,
    ANELAR_PONTA,
    DetectorPunhoSustentado,
    INDICADOR_MCP,
    INDICADOR_PONTA,
    MEDIO_MCP,
    MEDIO_PONTA,
    MINDINHO_MCP,
    MINDINHO_PONTA,
    POLEGAR_PONTA,
    PULSO,
    abertura_mao,
    aplicar_dead_zone,
    distancia_pinca,
    mao_fechada,
    posicao_mao,
    suavizar_ema,
    tamanho_mao,
)


def _landmarks_base() -> list[tuple[float, float, float]]:
    landmarks = [(0.0, 0.0, 0.0)] * 21
    landmarks[PULSO] = (0.5, 0.9, 0.0)
    landmarks[INDICADOR_MCP] = (0.42, 0.6, 0.0)
    landmarks[MEDIO_MCP] = (0.48, 0.58, 0.0)
    landmarks[ANELAR_MCP] = (0.54, 0.6, 0.0)
    landmarks[MINDINHO_MCP] = (0.6, 0.63, 0.0)
    return landmarks


def _landmarks_mao_aberta() -> list[tuple[float, float, float]]:
    landmarks = _landmarks_base()
    landmarks[POLEGAR_PONTA] = (0.3, 0.5, 0.0)
    landmarks[INDICADOR_PONTA] = (0.4, 0.2, 0.0)
    landmarks[MEDIO_PONTA] = (0.48, 0.15, 0.0)
    landmarks[ANELAR_PONTA] = (0.56, 0.2, 0.0)
    landmarks[MINDINHO_PONTA] = (0.65, 0.3, 0.0)
    return landmarks


def _landmarks_mao_fechada() -> list[tuple[float, float, float]]:
    landmarks = _landmarks_base()
    # Pontas dos dedos dobradas bem perto do centroide da palma.
    landmarks[POLEGAR_PONTA] = (0.5, 0.62, 0.0)
    landmarks[INDICADOR_PONTA] = (0.47, 0.61, 0.0)
    landmarks[MEDIO_PONTA] = (0.5, 0.6, 0.0)
    landmarks[ANELAR_PONTA] = (0.53, 0.61, 0.0)
    landmarks[MINDINHO_PONTA] = (0.56, 0.62, 0.0)
    return landmarks


def _escalar(landmarks, fator: float) -> list[tuple[float, float, float]]:
    return [(x * fator, y * fator, z * fator) for x, y, z in landmarks]


class TestTamanhoMao:
    def test_sempre_positivo(self):
        assert tamanho_mao(_landmarks_mao_aberta()) > 0
        assert tamanho_mao(_landmarks_mao_fechada()) > 0

    def test_nunca_divide_por_zero_com_landmarks_degenerados(self):
        landmarks = _landmarks_base()
        landmarks[PULSO] = landmarks[MEDIO_MCP]  # pulso e MCP coincidentes
        assert tamanho_mao(landmarks) > 0


class TestPosicaoMao:
    def test_centroide_correto(self):
        landmarks = _landmarks_base()
        x_esperado = (0.5 + 0.42 + 0.48 + 0.54 + 0.6) / 5
        y_esperado = (0.9 + 0.6 + 0.58 + 0.6 + 0.63) / 5
        x, y = posicao_mao(landmarks)
        assert x == pytest.approx(x_esperado)
        assert y == pytest.approx(y_esperado)


class TestDistanciaPincaEAberturaMao:
    def test_pinca_e_abertura_maiores_com_mao_aberta(self):
        aberta = _landmarks_mao_aberta()
        fechada = _landmarks_mao_fechada()
        assert distancia_pinca(aberta) > distancia_pinca(fechada)
        assert abertura_mao(aberta) > abertura_mao(fechada)

    @pytest.mark.parametrize("fator", [0.5, 2.0, 5.0])
    def test_invariancia_a_escala(self, fator):
        aberta = _landmarks_mao_aberta()
        aberta_escalada = _escalar(aberta, fator)
        assert distancia_pinca(aberta) == pytest.approx(distancia_pinca(aberta_escalada))
        assert abertura_mao(aberta) == pytest.approx(abertura_mao(aberta_escalada))


class TestMaoFechada:
    def test_mao_fechada_detectada(self):
        assert mao_fechada(_landmarks_mao_fechada(), limiar=0.32) is True

    def test_mao_aberta_nao_detectada_como_fechada(self):
        assert mao_fechada(_landmarks_mao_aberta(), limiar=0.32) is False

    def test_borda_do_limiar(self):
        landmarks = _landmarks_mao_fechada()
        valor = abertura_mao(landmarks)
        assert mao_fechada(landmarks, limiar=valor) is False  # < limiar, não <=
        assert mao_fechada(landmarks, limiar=valor + 1e-9) is True


class TestSuavizarEma:
    def test_bootstrap_retorna_bruto(self):
        assert suavizar_ema(5.0, None, alfa=0.4) == 5.0

    def test_formula_conhecida(self):
        resultado = suavizar_ema(bruto=10.0, suavizado_anterior=0.0, alfa=0.25)
        assert resultado == pytest.approx(0.25 * 10.0 + 0.75 * 0.0)

    def test_converge_para_valor_alvo_constante(self):
        valor = None
        for _ in range(200):
            valor = suavizar_ema(1.0, valor, alfa=0.3)
        assert valor == pytest.approx(1.0, abs=1e-3)


class TestAplicarDeadZone:
    def test_zera_abaixo_do_limiar(self):
        assert aplicar_dead_zone(0.01, limiar=0.02) == 0.0
        assert aplicar_dead_zone(-0.01, limiar=0.02) == 0.0

    def test_preserva_acima_do_limiar(self):
        assert aplicar_dead_zone(0.05, limiar=0.02) == 0.05
        assert aplicar_dead_zone(-0.05, limiar=0.02) == -0.05

    def test_borda_exata_e_preservada(self):
        # `< limiar`, não `<=`: no valor exatamente igual ao limiar, o delta passa.
        assert aplicar_dead_zone(0.02, limiar=0.02) == 0.02
        assert aplicar_dead_zone(0.0199999, limiar=0.02) == 0.0


class _RelogioFalso:
    def __init__(self, inicio: float = 0.0) -> None:
        self._agora = inicio

    def avancar(self, segundos: float) -> None:
        self._agora += segundos

    def __call__(self) -> float:
        return self._agora


class TestDetectorPunhoSustentado:
    def test_nao_dispara_com_mao_aberta(self):
        relogio = _RelogioFalso()
        detector = DetectorPunhoSustentado(duracao_hold=1.0, relogio=relogio)
        assert detector.atualizar(False) is False

    def test_nao_dispara_antes_da_duracao(self):
        relogio = _RelogioFalso()
        detector = DetectorPunhoSustentado(duracao_hold=1.0, relogio=relogio)
        assert detector.atualizar(True) is False
        relogio.avancar(0.5)
        assert detector.atualizar(True) is False

    def test_dispara_exatamente_uma_vez_ao_atingir_duracao(self):
        relogio = _RelogioFalso()
        detector = DetectorPunhoSustentado(duracao_hold=1.0, relogio=relogio)
        detector.atualizar(True)
        relogio.avancar(1.1)
        assert detector.atualizar(True) is True
        assert detector.atualizar(True) is False  # não duplica no mesmo hold

    def test_soltar_antes_do_tempo_zera_o_acumulador(self):
        relogio = _RelogioFalso()
        detector = DetectorPunhoSustentado(duracao_hold=1.0, relogio=relogio)
        detector.atualizar(True)
        relogio.avancar(0.5)
        assert detector.atualizar(False) is False  # solta antes da duração completar
        relogio.avancar(0.6)
        assert detector.atualizar(True) is False  # reinicia a contagem, não soma com o hold anterior

    def test_permite_novo_disparo_apos_reabrir_a_mao(self):
        relogio = _RelogioFalso()
        detector = DetectorPunhoSustentado(duracao_hold=1.0, relogio=relogio)
        detector.atualizar(True)
        relogio.avancar(1.1)
        assert detector.atualizar(True) is True
        assert detector.atualizar(False) is False
        detector.atualizar(True)
        relogio.avancar(1.1)
        assert detector.atualizar(True) is True
