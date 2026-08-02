"""Ponto de entrada do Jarvis Tetraedro.

Arquitetura final (Etapas 2/3, ainda não implementada aqui):
    - Uma thread de voz (voice/listener.py) roda em paralelo, ouvindo
      continuamente e publicando eventos (fila/threading.Event) quando
      detecta a frase de ativação ou o comando de fechar.
    - A thread principal roda uma máquina de estados simples:
      OUVINDO -> (frase de ativação) -> ATIVO (chama viewer.executar(),
      bloqueante) -> volta a OUVINDO quando executar() retorna (ESC).
    - Ctrl+C (SIGINT) deve encerrar tudo: parar a thread de voz,
      liberar a câmera (vision/camera.py) e fechar o contexto GL.

Nesta Etapa 1, sem voz/visão, o programa apenas abre o viewer direto,
controlado por mouse (arrastar rotaciona, scroll dá zoom) e teclado
(+/- mudam a aresta, R reseta, ESC fecha).
"""
import sys

from config import Config
from geometry.tetraedro import Tetraedro
from render.viewer import Viewer


def main() -> int:
    config = Config()
    tetraedro = Tetraedro(
        aresta=config.ARESTA_INICIAL,
        aresta_min=config.ARESTA_MIN,
        aresta_max=config.ARESTA_MAX,
    )
    viewer = Viewer(tetraedro, config=config)
    try:
        viewer.executar()
    except KeyboardInterrupt:
        # viewer.executar() já libera pygame/GL no finally interno.
        print("\nInterrompido pelo usuário. Encerrando...")
    return 0


if __name__ == "__main__":
    sys.exit(main())
