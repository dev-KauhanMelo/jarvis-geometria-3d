"""Suavização das mãos entre o ritmo da detecção (~25 Hz) e o do render (60 Hz).

Sem isso o snapshot só muda a cada 2-3 frames de render e o objeto anda em
degraus. A suavização usa constante de tempo (e não um alfa fixo), então o
mesmo ajuste vale independentemente de a detecção estar rápida ou lenta.

Nada de predição/extrapolação: ela mataria parte da latência, mas gera
overshoot na mudança de direção — exatamente o que se lê como "bugado".
"""
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

from geometry.transformacoes import (
    QUAT_IDENTIDADE,
    Quat,
    alfa_temporal,
    quaternion_de_matriz,
    slerp,
)
from vision.hand_tracker import (
    abertura_mao,
    distancia_pinca,
    extensoes_dedos,
    matriz_orientacao_palma,
    ponto_pinca,
    posicao_mao,
    tamanho_mao,
)


@dataclass(frozen=True)
class MaoSuave:
    """Estado de uma mão já suavizado e já em coordenadas de TELA."""

    id_mao: int
    destra: bool
    # Onde o polegar encontra o indicador. É o cursor que o usuário enxerga e
    # com o qual mira um vértice — a ponta dos dedos, não o meio da palma.
    cursor_tela: tuple[float, float]
    pinca: float
    extensoes: tuple[float, float, float, float]
    orientacao: Quat
    visivel_ha_s: float
    landmarks_tela: tuple[tuple[float, float], ...] = ()
    # Centroide da palma: com a mão fechada em garra as pontas dos dedos ficam
    # amontoadas e o cursor treme, então quem manda na translação é a palma.
    palma_tela: tuple[float, float] = (0.0, 0.0)
    abertura: float = 1.0
    # Pulso -> base do médio, em unidades normalizadas da imagem. Cresce quando
    # a mão se aproxima da câmera: é o único proxy de profundidade disponível,
    # já que o MediaPipe não dá distância absoluta.
    tamanho: float = 0.3


@dataclass
class _EstadoMao:
    cursor: Optional[tuple[float, float]] = None
    palma: Optional[tuple[float, float]] = None
    pinca: Optional[float] = None
    abertura: Optional[float] = None
    tamanho: Optional[float] = None
    extensoes: Optional[tuple[float, float, float, float]] = None
    orientacao: Quat = QUAT_IDENTIDADE
    tem_orientacao: bool = False
    visto_em: float = 0.0
    apareceu_em: float = 0.0


class SuavizadorMao:
    """Interpola no ritmo do render em direção ao último alvo da detecção.

    Mantém estado por `id_mao` para não misturar as duas mãos — é o motivo de
    a identidade estável do rastreador importar.
    """

    def __init__(
        self,
        tau_posicao: float = 0.06,
        tau_escalares: float = 0.08,
        tau_pinca: float = 0.03,
        tau_orientacao: float = 0.12,
        tau_tamanho: float = 0.25,
        tempo_esquecimento_s: float = 0.5,
    ) -> None:
        self.tau_posicao = tau_posicao
        self.tau_escalares = tau_escalares
        self.tau_pinca = tau_pinca
        self.tau_orientacao = tau_orientacao
        # O tamanho aparente da mão é a medida mais ruidosa de todas (muda com
        # o ângulo do punho, não só com a distância) e vira profundidade, que
        # é justamente onde tremer incomoda mais. Daí a constante bem maior.
        self.tau_tamanho = tau_tamanho
        self.tempo_esquecimento_s = tempo_esquecimento_s
        self._estados: dict[int, _EstadoMao] = {}

    def atualizar(
        self,
        maos_detectadas: Sequence,
        dt: float,
        mapear_para_tela: Callable[[float, float], tuple[float, float]],
        agora: float,
    ) -> tuple[MaoSuave, ...]:
        alfa_pos = alfa_temporal(dt, self.tau_posicao)
        alfa_esc = alfa_temporal(dt, self.tau_escalares)
        alfa_ori = alfa_temporal(dt, self.tau_orientacao)
        alfa_pinca = alfa_temporal(dt, self.tau_pinca)
        alfa_tam = alfa_temporal(dt, self.tau_tamanho)

        resultado: list[MaoSuave] = []
        for mao in maos_detectadas:
            estado = self._estados.get(mao.id_mao)
            if estado is None:
                estado = _EstadoMao(apareceu_em=agora)
                self._estados[mao.id_mao] = estado
            estado.visto_em = agora

            alvo_cursor = mapear_para_tela(*ponto_pinca(mao.landmarks))
            estado.cursor = _mesclar_ponto(estado.cursor, alvo_cursor, alfa_pos)

            alvo_palma = mapear_para_tela(*posicao_mao(mao.landmarks))
            estado.palma = _mesclar_ponto(estado.palma, alvo_palma, alfa_pos)

            alvo_pinca = distancia_pinca(mao.landmarks)
            estado.pinca = _mesclar(estado.pinca, alvo_pinca, alfa_pinca)

            # A abertura decide a garra, que é um gatilho como a pinça: vale a
            # mesma troca de suavidade por latência.
            estado.abertura = _mesclar(estado.abertura, abertura_mao(mao.landmarks), alfa_pinca)
            estado.tamanho = _mesclar(estado.tamanho, tamanho_mao(mao.landmarks), alfa_tam)

            # Extensões dos dedos preferem world landmarks: em 2D o
            # encurtamento por perspectiva confunde dedo dobrado com dedo
            # apontado para a câmera.
            fonte = mao.world_landmarks if mao.world_landmarks else mao.landmarks
            alvo_ext = extensoes_dedos(fonte)
            estado.extensoes = _mesclar_tupla(estado.extensoes, alvo_ext, alfa_esc)

            if mao.world_landmarks:
                try:
                    matriz = matriz_orientacao_palma(mao.world_landmarks, mao.destra)
                    alvo_q = quaternion_de_matriz(matriz)
                    estado.orientacao = (
                        alvo_q if not estado.tem_orientacao
                        else slerp(estado.orientacao, alvo_q, alfa_ori)
                    )
                    estado.tem_orientacao = True
                except ValueError:
                    pass  # mão degenerada neste frame: mantém a orientação anterior

            resultado.append(
                MaoSuave(
                    id_mao=mao.id_mao,
                    destra=mao.destra,
                    cursor_tela=estado.cursor,
                    pinca=estado.pinca,
                    extensoes=estado.extensoes,
                    orientacao=estado.orientacao,
                    visivel_ha_s=agora - estado.apareceu_em,
                    landmarks_tela=tuple(
                        mapear_para_tela(p[0], p[1]) for p in mao.landmarks
                    ),
                    palma_tela=estado.palma,
                    abertura=estado.abertura,
                    tamanho=estado.tamanho,
                )
            )

        self._esquecer_ausentes(agora)
        return tuple(resultado)

    def _esquecer_ausentes(self, agora: float) -> None:
        """Descarta o estado de mãos que sumiram, para que uma mão que volte
        não herde a posição de onde estava antes de sair do quadro."""
        expirados = [
            id_mao
            for id_mao, estado in self._estados.items()
            if agora - estado.visto_em > self.tempo_esquecimento_s
        ]
        for id_mao in expirados:
            del self._estados[id_mao]

    def esquecer(self, id_mao: int) -> None:
        self._estados.pop(id_mao, None)


def _mesclar(anterior: Optional[float], alvo: float, alfa: float) -> float:
    return alvo if anterior is None else anterior + alfa * (alvo - anterior)


def _mesclar_ponto(anterior, alvo, alfa):
    if anterior is None:
        return alvo
    return (
        anterior[0] + alfa * (alvo[0] - anterior[0]),
        anterior[1] + alfa * (alvo[1] - anterior[1]),
    )


def _mesclar_tupla(anterior, alvo, alfa):
    if anterior is None:
        return alvo
    return tuple(a + alfa * (b - a) for a, b in zip(anterior, alvo))
