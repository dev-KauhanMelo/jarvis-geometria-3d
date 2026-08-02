# Jarvis Tetraedro

Sistema desktop que combina ativação por voz, rastreamento de mão via webcam e renderização 3D interativa de um tetraedro regular, controlado por gestos — projeto ligado ao conteúdo de Geometria Espacial (Pirâmides) do ensino médio técnico.

## Estado atual do projeto

Construção incremental em etapas:

- **Etapa 1 (implementada)**: geometria do tetraedro + render 3D interativo, controlado por **mouse e teclado**.
- **Etapa 2 (implementada)**: rastreamento de mão via webcam (`vision/`), substituindo mouse/teclado pelos gestos. `--mouse` continua disponível para voltar ao controle da Etapa 1.
- **Etapa 3 (implementada)**: **realidade aumentada e manipulação direta a duas mãos**. O sólido passa a ser desenhado sobre a imagem ao vivo da câmera, na mesma janela, e é manipulado agarrando com a pinça: girar, redimensionar com as duas mãos e **puxar um vértice para deformar** o tetraedro. A detecção saiu para uma thread própria, o que multiplicou por ~7 a fluidez do render.
- **Etapa 4 (planejada)**: ativação por voz em português via Vosk (`voice/`).

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
python3 main.py                  # AR + gestos de mão, com fallback automático para mouse
python3 main.py --mouse          # força mouse/teclado, útil para testar sem câmera
python3 main.py --debug-gestos   # mostra no HUD os valores usados para calibrar os gestos
python3 main.py --uma-mao        # detecta só uma mão (2x mais rápido, para hardware fraco)
python3 main.py --sem-ar         # desliga a AR: sólido sobre fundo escuro
python3 main.py --janela-debug   # abre a janela separada do OpenCV com o esqueleto da mão
```

## Controles

### Gestos de mão (padrão)

O modelo é **agarrar de propósito**: com a mão aberta o objeto fica parado, e você só mexe nele quando fecha a pinça. Isso permite descansar a mão na frente da câmera sem bagunçar a cena.

| Gesto | Ação |
|---|---|
| Mão aberta | Nada acontece — um **anel branco** marca a posição da mão |
| Pinça (polegar + indicador) fechada, longe de um vértice | Agarra o sólido: **mover a mão gira** |
| Pinça fechada **sobre um vértice** | **Puxa o vértice e deforma** o sólido |
| **Duas** pinças fechadas, afastando/juntando as mãos | Aumenta/diminui o tamanho |
| **Duas** pinças fechadas, movendo as mãos juntas | Desloca o sólido pela tela |
| Mão em **concha** (dedos semi-dobrados, como segurando uma esfera) | Gira o sólido acompanhando a **inclinação da palma** |
| Tecla `R` | Volta ao tetraedro regular e à posição inicial |
| Tecla `ESC` ou fechar a janela | Sai |

**Feedback visual** — o que cada cor significa:

- **anel branco**: a mão foi detectada, mas nada está agarrado
- **disco verde**: você está agarrando; o HUD escreve o que está acontecendo
- **anel âmbar num vértice**: é este que será pego se você fechar a pinça agora

Ao puxar um vértice, o sólido deixa de ser um tetraedro regular. O HUD passa a mostrar a **aresta média com o intervalo** (min/max) em vez de uma aresta única, e avisa que as fórmulas do tetraedro regular não valem mais — área, volume e altura passam a ser calculados numericamente (volume por determinante, área somando as quatro faces). `R` desfaz a deformação.

### Calibrando os gestos para a sua mão

Os limiares em `config.py` são um ponto de partida, calibrado com fotos reais processadas pelo MediaPipe. **A concha é o gesto que mais varia de pessoa para pessoa**, porque depende de quanto cada mão dobra os dedos.

```bash
python3 main.py --debug-gestos
```

O HUD passa a mostrar, ao vivo, uma linha por mão:

```
faixa da concha: 0.50–0.88 | pinça fecha < 0.35
mão 0 ext: 0.72 0.68 0.71 0.65 | pinça: 0.81 | CONCHA
```

Os quatro números de `ext` são o quanto cada dedo (indicador, médio, anelar, mindinho) está esticado: **~1,0 é dedo reto e ~0,3 é dedo totalmente dobrado**. Faça a concha, leia os quatro valores e ajuste `CONCHA_EXTENSAO_MIN`/`CONCHA_EXTENSAO_MAX` em `config.py` para que a sua concha caiba na faixa. Se a concha não dispara, o culpado mais provável é o indicador esticar mais que os outros — nesse caso, suba `CONCHA_EXTENSAO_MAX`.

Mesma ideia para a pinça: se o objeto for difícil de agarrar, suba `PINCA_FECHA`; se soltar sozinho, suba `PINCA_ABRE`.

### Mouse e teclado (`--mouse`, Etapa 1)

| Ação | Controle |
|---|---|
| Rotacionar o tetraedro | Arrastar com o botão esquerdo do mouse |
| **Pegar e arrastar um vértice (deformar)** | **Arrastar com o botão direito** |
| Zoom in/out | Roda do mouse |
| Aumentar/diminuir a aresta | Teclas `+`/`-` (ou `Cima`/`Baixo`) |
| Resetar (volta a regular) | Tecla `R` |
| Fechar a janela | Tecla `ESC` ou fechar a janela |

O botão direito cria uma "mão sintética" no cursor, que passa pela mesma máquina de gestos usada pela câmera — dá para testar a seleção e a deformação de vértices sem webcam nenhuma.

Em ambos os modos, `Ctrl+C` no terminal encerra tudo de forma limpa (libera câmera e contexto GL). O overlay no canto superior esquerdo da janela 3D mostra, em tempo real: aresta atual, área total, volume e altura do tetraedro.

## Solução de problemas

- **Tela preta ou erro de contexto OpenGL**: verifique se `libglu1-mesa` está instalado (passo 1) e se os drivers de vídeo estão atualizados (`glxinfo | grep "direct rendering"` deve retornar `Yes`).
- **Arrastar o mouse não rotaciona / comportamento estranho no drag, ou a janela de debug da câmera não aparece**: se estiver numa sessão Wayland, tente rodar com `SDL_VIDEODRIVER=x11 python3 main.py`.
- **Janela redimensionada distorce a imagem**: não deveria acontecer (o viewer recalcula a projeção a cada redimensionamento); se acontecer, é um bug — reporte.
- **"Falha ao iniciar rastreamento de mão" no terminal**: confira se `models/hand_landmarker.task` existe (passo 4) e se `mediapipe` está instalado; o programa cai para `--mouse` automaticamente, então continua funcionando enquanto você resolve.
- **Câmera nunca conecta / janela de debug fica preta**: confira a URL do app de webcam do celular (celular e notebook precisam estar na mesma rede Wi-Fi) e se `/dev/video0` existe como fallback (`ls /dev/video*`). O estado da câmera (`conectado`/`reconectando`/`desconectado`) aparece no canto da janela de debug.
- **Gestos não respondem ou respondem demais**: rode com `--debug-gestos` e ajuste os limiares em `config.py` observando os valores ao vivo (ver "Calibrando os gestos" acima).
- **A concha não é reconhecida**: é o gesto mais sensível à mão de cada pessoa. Veja a seção de calibração — normalmente basta subir `CONCHA_EXTENSAO_MAX`.
- **Difícil enxergar o sólido sobre o vídeo**: suba `AR_ESCURECIMENTO_FUNDO` em `config.py` (0 = vídeo original, 1 = fundo preto).
- **Vídeo cortado nas bordas**: é intencional — o modo `"preencher"` cobre a janela inteira sem distorcer a imagem. Para ver o quadro completo (com barras pretas), mude `AR_MODO_AJUSTE` para `"caber"`.
- **Está lento / a mão responde com atraso**: a latência depende da câmera e do MediaPipe, não do render. Melhore a **iluminação** (webcam com pouca luz cai para 15 fps por causa da auto-exposição) e, se necessário, use `--uma-mao`, que dobra a taxa de detecção. Reduzir a resolução da câmera **não** ajuda: o MediaPipe reescala internamente.

## Estrutura de arquivos

```
jarvis-3d-geometria/
├── main.py                        # Ponto de entrada (--mouse, --debug-gestos, --sem-ar, ...)
├── config.py                      # Configurações (janela, aresta, câmera, limiares de gesto, AR)
├── geometry/
│   ├── tetraedro.py               # Vértices, topologia, fórmulas do regular E do deformado
│   └── transformacoes.py          # Vetores, matrizes, quaternions, slerp (100% puro)
├── vision/
│   ├── camera.py                  # CameraSource: MJPEG do celular + fallback local, com reconexão
│   ├── rastreador.py              # Thread de detecção + SnapshotMaos + identidade estável das mãos
│   └── hand_tracker.py            # Medidas da mão (pinça, curvatura dos dedos, orientação da palma)
├── interaction/
│   ├── gestos.py                  # Máquina de estados dos gestos (pura, testável sem OpenGL)
│   ├── suavizacao.py              # Interpola detecção (~25Hz) -> render (60Hz)
│   └── controlador.py             # Cola gestos + cena + geometria
├── render/
│   ├── viewer.py                  # Janela OpenGL, loop e contrato FonteEntrada
│   ├── ar.py                      # Vídeo da câmera como fundo; matemática de encaixe
│   ├── projecao.py                # Projetar/desprojetar (seleção de vértice)
│   └── hud.py                     # Painel de texto e cursores das mãos
├── voice/
│   └── listener.py                # Ativação por voz (Etapa 4 — ainda não implementado)
├── models/                        # hand_landmarker.task (baixado manualmente, não versionado)
├── tests/                         # 351 testes; nenhum precisa de câmera, GPU ou janela
└── requirements.txt
```

## Como funciona por dentro (visão rápida)

Três threads: uma lê a câmera, uma roda a detecção do MediaPipe, e a principal renderiza. A detecção **não** fica no caminho crítico do render — ela publica o último resultado num snapshot imutável, que o render lê sem bloquear. Foi essa mudança que tirou o app dos ~24 fps (onde ele andava na velocidade do MediaPipe) para ~170 fps, mesmo passando a detectar duas mãos em vez de uma.

Entre a detecção (~25 Hz) e o render (60 Hz), as posições são interpoladas por constante de tempo, e a orientação por `slerp`. Não há predição de movimento: ela reduziria a latência, mas gera overshoot ao mudar de direção, que é exatamente o que se percebe como travamento.

A orientação do sólido é um quaternion, não ângulos de Euler — o que elimina o gimbal lock e permite o gesto da concha compor rotações a cada frame sem acumular deriva.
