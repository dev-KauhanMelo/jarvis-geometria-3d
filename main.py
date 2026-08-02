"""Ponto de entrada do Jarvis Tetraedro.

Arquitetura final (Etapa 3, ainda não implementada aqui):
    - Uma thread de voz (voice/listener.py) roda em paralelo, ouvindo
      continuamente e publicando eventos (fila/threading.Event) quando
      detecta a frase de ativação ou o comando de fechar.
    - A thread principal roda uma máquina de estados simples:
      OUVINDO -> (frase de ativação) -> ATIVO (chama viewer.executar(),
      bloqueante) -> volta a OUVINDO quando executar() retorna (ESC).
    - Ctrl+C (SIGINT) deve encerrar tudo: parar a thread de voz,
      liberar a câmera (vision/camera.py) e fechar o contexto GL.

Etapa 2 (implementada): controle por gestos de mão via webcam
(vision/hand_tracker.py), com `--mouse` como alternativa via mouse/teclado
(Etapa 1) — útil para testar sem câmera ou como salvaguarda caso o
rastreamento de mão falhe ao iniciar.
"""
import argparse
import sys

from config import Config
from geometry.tetraedro import Tetraedro
from render.viewer import FonteEntradaMouseTeclado, Viewer


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Jarvis Tetraedro")
    parser.add_argument(
        "--mouse",
        action="store_true",
        help="Usa mouse/teclado (Etapa 1) em vez de gestos de mão (Etapa 2).",
    )
    parser.add_argument(
        "--sem-janela-debug",
        action="store_true",
        help=argparse.SUPPRESS,  # mantido por compatibilidade; a janela já é opt-in
    )
    parser.add_argument(
        "--janela-debug",
        action="store_true",
        help="Abre a janela separada do OpenCV com o esqueleto da mão. "
             "Desde a Etapa 3 o vídeo aparece na janela principal, então ela "
             "só serve para depurar a detecção.",
    )
    parser.add_argument(
        "--debug-gestos",
        action="store_true",
        help="Mostra no HUD os valores ao vivo usados pelos gestos (fps de "
             "detecção, abertura dos dedos, pinça). Use para calibrar os "
             "limiares de config.py para a sua mão.",
    )
    parser.add_argument(
        "--uma-mao",
        action="store_true",
        help="Detecta só uma mão (2x mais rápido). Útil em hardware fraco.",
    )
    parser.add_argument(
        "--sem-ar",
        action="store_true",
        help="Desliga a realidade aumentada: o sólido volta a aparecer sobre "
             "fundo escuro, sem o vídeo da câmera.",
    )
    parser.add_argument(
        "--camera-url",
        default=None,
        help="Sobrescreve CAMERA_URL do config.py (URL MJPEG do celular).",
    )
    return parser.parse_args(argv)


def main() -> int:
    args = _parse_args()
    config = Config()
    if args.camera_url:
        config.CAMERA_URL = args.camera_url
    if args.sem_ar:
        config.AR_ATIVO = False
    if args.uma_mao:
        config.HAND_MAX_NUM_HANDS = 1
    if args.debug_gestos:
        config.DEBUG_GESTOS = True

    tetraedro = Tetraedro(
        aresta=config.ARESTA_INICIAL,
        aresta_min=config.ARESTA_MIN,
        aresta_max=config.ARESTA_MAX,
    )

    fonte_gestos = None
    if args.mouse:
        fonte_entrada = FonteEntradaMouseTeclado()
    else:
        try:
            from vision.hand_tracker import FonteEntradaGestos

            fonte_gestos = FonteEntradaGestos(
                config=config,
                mostrar_janela_debug=args.janela_debug,
                mostrar_diagnostico=args.debug_gestos,
            )
            fonte_entrada = fonte_gestos
        except Exception as erro:
            print(f"Falha ao iniciar rastreamento de mão ({erro}); usando mouse/teclado.", file=sys.stderr)
            fonte_entrada = FonteEntradaMouseTeclado()

    viewer = Viewer(tetraedro, fonte_entrada=fonte_entrada, config=config)
    try:
        viewer.executar()
    except KeyboardInterrupt:
        # viewer.executar() já libera pygame/GL no finally interno.
        print("\nInterrompido pelo usuário. Encerrando...")
    finally:
        if fonte_gestos is not None:
            fonte_gestos.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
