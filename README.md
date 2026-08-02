# Jarvis Tetraedro

Sistema desktop que combina ativação por voz, rastreamento de mão via webcam e renderização 3D interativa de um tetraedro regular, controlado por gestos — projeto ligado ao conteúdo de Geometria Espacial (Pirâmides) do ensino médio técnico.

## Estado atual do projeto

Construção incremental em etapas:

- **Etapa 1 (implementada)**: geometria do tetraedro + render 3D interativo, controlado por **mouse e teclado**.
- **Etapa 2 (implementada)**: rastreamento de mão via webcam (`vision/`), substituindo mouse/teclado pelos gestos (rotação, pinça para zoom, abertura da mão para a aresta, punho fechado para reset). `--mouse` continua disponível para voltar ao controle da Etapa 1.
- **Etapa 3 (planejada)**: ativação por voz em português via Vosk (`voice/`).

## Instalação no Zorin OS (Ubuntu/Debian)

### 1. Dependências do sistema

```bash
sudo apt update
sudo apt install python3-venv python3-pip libglu1-mesa
```

`libglu1-mesa` é necessário porque o render usa `gluPerspective`/`gluSphere` (biblioteca GLU), que pode não vir instalada por padrão em instalações mínimas.

### 2. Ambiente virtual Python

Este repositório já inclui uma `venv/` (Python 3.12). Se precisar recriá-la:

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
```

### 3. Dependências Python

```bash
source venv/bin/activate
pip install -r requirements.txt
```

Se alguma versão do `requirements.txt` não instalar na sua máquina, instale sem fixar versão e regrave o arquivo com o que realmente funcionou:

```bash
pip install pygame PyOpenGL numpy pytest mediapipe
pip freeze > requirements.txt
```

Depois de instalar o `mediapipe`, confira que só entrou **uma** variante do OpenCV:

```bash
pip freeze | grep -i opencv   # deve mostrar só opencv-contrib-python
```

### 4. Modelo do MediaPipe (necessário para a Etapa 2 — hand tracking)

O rastreamento de mão usa o `HandLandmarker` do MediaPipe, cujo modelo **não vem pelo pip** — baixe manualmente:

```bash
mkdir -p models
curl -L -o models/hand_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
```

Se o arquivo não existir em `models/hand_landmarker.task`, o programa cai automaticamente para o controle por mouse/teclado (com um aviso no terminal) em vez de travar.

### 5. Câmera (celular como webcam de rede)

No celular, instale um app tipo **IP Webcam** (Android) e inicie o servidor de vídeo — ele mostra uma URL tipo `http://192.168.x.x:8080/video`. Configure essa URL em `config.py` (campo `CAMERA_URL`) ou passe via linha de comando:

```bash
python3 main.py --camera-url http://192.168.x.x:8080/video
```

Se `CAMERA_URL` não for definida (ou a conexão falhar), o programa cai automaticamente para a webcam local (`/dev/video0`).

### 6. Rodar os testes (opcional, mas recomendado)

```bash
source venv/bin/activate
pytest tests/ -v
```

Os testes cobrem lógica pura (fórmulas da geometria, cálculo de gestos a partir de landmarks sintéticos, suavização, dead zone, detecção de punho sustentado, reconexão de câmera com um `VideoCapture` falso) — não abrem janela, não precisam de câmera nem GPU.

### 7. Rodar o programa

```bash
source venv/bin/activate
python3 main.py                  # gestos de mão (Etapa 2), com fallback automático para mouse
python3 main.py --mouse          # força mouse/teclado (Etapa 1), útil para testar sem câmera
python3 main.py --sem-janela-debug   # sem a janela de depuração da câmera
```

## Controles

### Gestos de mão (padrão, Etapa 2)

| Gesto | Ação |
|---|---|
| Mover a mão (eixo X/Y da câmera) | Rotaciona o tetraedro |
| Pinça (polegar + indicador) fechando/abrindo | Zoom in/out |
| Abrir/fechar a mão (dedos espalhados) | Aumenta/diminui a aresta |
| Punho fechado sustentado por ~1s | Reseta a visualização |
| Tecla `ESC` ou fechar a janela | Sai |

Uma **janela de debug** separada mostra o feed da câmera com o esqueleto da mão desenhado por cima, o estado da conexão da câmera e os valores brutos de abertura/pinça — use-a para calibrar `LIMIAR_MAO_FECHADA` e as sensibilidades em `config.py` para a sua mão/câmera.

### Mouse e teclado (`--mouse`, Etapa 1)

| Ação | Controle |
|---|---|
| Rotacionar o tetraedro | Arrastar com o botão esquerdo do mouse |
| Zoom in/out | Roda do mouse |
| Aumentar/diminuir a aresta | Teclas `+`/`-` (ou `Cima`/`Baixo`) |
| Resetar a visualização | Tecla `R` |
| Fechar a janela | Tecla `ESC` ou fechar a janela |

Em ambos os modos, `Ctrl+C` no terminal encerra tudo de forma limpa (libera câmera e contexto GL). O overlay no canto superior esquerdo da janela 3D mostra, em tempo real: aresta atual, área total, volume e altura do tetraedro.

## Solução de problemas

- **Tela preta ou erro de contexto OpenGL**: verifique se `libglu1-mesa` está instalado (passo 1) e se os drivers de vídeo estão atualizados (`glxinfo | grep "direct rendering"` deve retornar `Yes`).
- **Arrastar o mouse não rotaciona / comportamento estranho no drag, ou a janela de debug da câmera não aparece**: se estiver numa sessão Wayland, tente rodar com `SDL_VIDEODRIVER=x11 python3 main.py`.
- **Janela redimensionada distorce a imagem**: não deveria acontecer (o viewer recalcula a projeção a cada redimensionamento); se acontecer, é um bug — reporte.
- **"Falha ao iniciar rastreamento de mão" no terminal**: confira se `models/hand_landmarker.task` existe (passo 4) e se `mediapipe` está instalado; o programa cai para `--mouse` automaticamente, então continua funcionando enquanto você resolve.
- **Câmera nunca conecta / janela de debug fica preta**: confira a URL do app de webcam do celular (celular e notebook precisam estar na mesma rede Wi-Fi) e se `/dev/video0` existe como fallback (`ls /dev/video*`). O estado da câmera (`conectado`/`reconectando`/`desconectado`) aparece no canto da janela de debug.
- **Gestos não respondem ou respondem demais**: ajuste `SENSIBILIDADE_ROTACAO_GESTO`, `SENSIBILIDADE_ZOOM_GESTO`, `SENSIBILIDADE_ARESTA_GESTO`, os `DEAD_ZONE_*` e `LIMIAR_MAO_FECHADA` em `config.py`, observando os valores brutos mostrados na janela de debug.

## Estrutura de arquivos

```
jarvis-3d-geometria/
├── main.py                     # Ponto de entrada (--mouse, --camera-url, --sem-janela-debug)
├── config.py                    # Configurações (janela, aresta, câmera, sensibilidade de gestos)
├── voice/
│   └── listener.py              # Ativação por voz (Etapa 3 — ainda não implementado)
├── vision/
│   ├── camera.py                 # CameraSource: MJPEG do celular + fallback webcam local, com reconexão
│   └── hand_tracker.py           # FonteEntradaGestos: MediaPipe Hands -> EstadoEntrada
├── geometry/
│   └── tetraedro.py              # Vértices, topologia e fórmulas do tetraedro regular
├── render/
│   └── viewer.py                  # Janela OpenGL, loop de renderização e contrato FonteEntrada
├── models/                        # hand_landmarker.task (baixado manualmente, não versionado)
├── tests/
│   ├── test_tetraedro.py          # Testes de geometria (pytest)
│   ├── test_hand_tracker.py       # Testes de gestos com landmarks sintéticos (pytest)
│   └── test_camera.py             # Testes de reconexão com VideoCapture falso (pytest)
└── requirements.txt
```
