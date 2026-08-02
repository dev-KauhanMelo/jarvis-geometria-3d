"""Conexão e leitura do stream de vídeo (Etapa 2 — ainda não implementada).

Contrato previsto: uma classe `CameraSource` com `__enter__`/`__exit__`/`close()`
que tenta conectar à URL MJPEG do celular (ex.: http://192.168.x.x:8080/video)
e cai para `cv2.VideoCapture(0)` (webcam local) em caso de falha. Deve rodar
a leitura de frames em thread própria, expondo o frame mais recente por um
buffer thread-safe, com reconexão/backoff automático sem derrubar o processo
principal em caso de perda de conexão.
"""


class CameraSource:
    def __init__(self, *args, **kwargs) -> None:
        raise NotImplementedError("vision/camera.py será implementado na Etapa 2")
