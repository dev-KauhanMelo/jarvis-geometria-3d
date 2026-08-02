"""Cola entre as mãos detectadas, a cena 3D e a geometria.

Existe porque `EstadoEntrada` é sensorial (o que as mãos fazem) e não conhece
a cena, enquanto decidir "qual vértice está sob a mão" exige projetar a cena.
A fonte de entrada continua testável sem OpenGL, e o controlador —
testável com uma cena falsa — faz a ponte.
"""
import time
from typing import Callable, Optional, Sequence

from geometry.transformacoes import Quat
from interaction.gestos import (
    ComandoInteracao,
    EstadoInteracao,
    ParametrosGesto,
    avaliar_gestos,
)
from interaction.suavizacao import MaoSuave, SuavizadorMao


class CenaProjetada:
    """Implementa `ContextoCena` a partir do projetor de OpenGL e do sólido."""

    def __init__(self, projetor, tetraedro, orientacao: Quat, distancia: float = 6.0) -> None:
        self._projetor = projetor
        self._tetraedro = tetraedro
        self._orientacao = orientacao
        self._distancia = distancia

    def vertices_tela(self):
        if not self._projetor.pronto:
            return [None] * 4
        return self._projetor.projetar_varios(self._tetraedro.vertices())

    def desprojetar(self, x_px: float, y_px: float, z_janela: float):
        return self._projetor.desprojetar(x_px, y_px, z_janela)

    def orientacao_objeto(self) -> Quat:
        return self._orientacao

    def escala_objeto(self) -> float:
        return self._tetraedro.aresta

    def distancia_camera(self) -> float:
        return self._distancia


class ControladorInteracao:
    """Mantém o estado da máquina de gestos e da suavização entre frames."""

    def __init__(self, config=None, params: Optional[ParametrosGesto] = None) -> None:
        self.params = params or _params_de_config(config)
        self.suavizador = SuavizadorMao(
            tau_posicao=getattr(config, "TAU_POSICAO_MAO", 0.06),
            tau_escalares=getattr(config, "TAU_ESCALARES_MAO", 0.08),
            tau_pinca=getattr(config, "TAU_PINCA", 0.03),
            tau_orientacao=getattr(config, "TAU_ORIENTACAO_MAO", 0.12),
            tau_tamanho=getattr(config, "TAU_TAMANHO_MAO", 0.25),
        )
        self.estado = EstadoInteracao()
        self.maos_suaves: tuple[MaoSuave, ...] = ()
        self._ultimo_instante: Optional[float] = None

    def avaliar(
        self,
        maos_detectadas: Sequence,
        projetor,
        tetraedro,
        orientacao: Quat,
        mapear_para_tela: Callable[[float, float], tuple[float, float]],
        frame_id: int,
        agora: Optional[float] = None,
        distancia_camera: float = 6.0,
    ) -> ComandoInteracao:
        agora = time.monotonic() if agora is None else agora
        dt = 0.0 if self._ultimo_instante is None else max(0.0, agora - self._ultimo_instante)
        self._ultimo_instante = agora

        self.maos_suaves = self.suavizador.atualizar(
            maos_detectadas, dt, mapear_para_tela, agora
        )
        cena = CenaProjetada(projetor, tetraedro, orientacao, distancia_camera)
        self.estado, comando = avaliar_gestos(
            self.estado, self.maos_suaves, cena, self.params, frame_id
        )
        return comando

    def resetar(self) -> None:
        self.estado = EstadoInteracao()


def _params_de_config(config) -> ParametrosGesto:
    if config is None:
        return ParametrosGesto()
    padrao = ParametrosGesto()
    return ParametrosGesto(
        pinca_fecha=getattr(config, "PINCA_FECHA", padrao.pinca_fecha),
        pinca_abre=getattr(config, "PINCA_ABRE", padrao.pinca_abre),
        garra_fecha=getattr(config, "GARRA_FECHA", padrao.garra_fecha),
        garra_abre=getattr(config, "GARRA_ABRE", padrao.garra_abre),
        raio_pick_px=getattr(config, "RAIO_PICK_VERTICE_PX", padrao.raio_pick_px),
        quadros_confirmacao=getattr(
            config, "QUADROS_CONFIRMACAO_GESTO", padrao.quadros_confirmacao),
        tempo_minimo_visivel_s=getattr(
            config, "TEMPO_MINIMO_MAO_VISIVEL_S", padrao.tempo_minimo_visivel_s),
        razao_profundidade_min=getattr(
            config, "RAZAO_PROFUNDIDADE_MIN", padrao.razao_profundidade_min),
        razao_profundidade_max=getattr(
            config, "RAZAO_PROFUNDIDADE_MAX", padrao.razao_profundidade_max),
    )
