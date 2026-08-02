"""Reconhecimento de voz e detecção do comando de ativação (Etapa 3 — ainda não implementada).

Contrato previsto: uma classe `VoiceListener` que roda em thread própria,
usando Vosk (offline, modelo pt-BR) para reconhecimento contínuo, e publica
eventos de ativação/encerramento (ex.: via `threading.Event` ou `queue.Queue`)
consumidos pelo loop principal em main.py. Frase de ativação deve conter
"jarvis" + "tetraedro"; frase de encerramento algo como "jarvis, fechar".
"""


class VoiceListener:
    def __init__(self, *args, **kwargs) -> None:
        raise NotImplementedError("voice/listener.py será implementado na Etapa 3")
