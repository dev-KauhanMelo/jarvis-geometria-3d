"""Wrapper do MediaPipe Hands (Etapa 2 — ainda não implementada).

Contrato previsto: uma classe `FonteEntradaGestos` que consome frames de
`vision/camera.py`, roda o MediaPipe Hands sobre eles e traduz landmarks em
um `render.viewer.EstadoEntrada` — posição normalizada da mão (rotação),
distância polegar-indicador (zoom), abertura geral da mão (aresta) e punho
fechado sustentado por ~1s (reset). Deve implementar o mesmo protocolo
`FonteEntrada.capturar() -> EstadoEntrada` de render/viewer.py, para que o
Viewer troque a fonte de entrada sem nenhuma alteração no núcleo de render.
Suavização (média móvel) e dead zones ficam encapsuladas aqui.
"""


class FonteEntradaGestos:
    def __init__(self, *args, **kwargs) -> None:
        raise NotImplementedError("vision/hand_tracker.py será implementado na Etapa 2")
