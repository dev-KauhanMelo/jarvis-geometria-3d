# Jarvis Tetraedro

Sistema desktop que combina ativação por voz, rastreamento de mão via webcam e renderização 3D interativa de um tetraedro regular, controlado por gestos — projeto ligado ao conteúdo de Geometria Espacial (Pirâmides) do ensino médio técnico.

## Estado atual do projeto

Construção incremental em etapas:

- **Etapa 1 (implementada)**: geometria do tetraedro + render 3D interativo, controlado por **mouse e teclado** como substituto temporário dos gestos de mão.
- **Etapa 2 (planejada)**: rastreamento de mão via webcam (`vision/`), substituindo mouse/teclado pelos gestos descritos na especificação.
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
pip install pygame PyOpenGL numpy pytest
pip freeze > requirements.txt
```

### 4. Rodar os testes (opcional, mas recomendado)

```bash
source venv/bin/activate
pytest tests/ -v
```

Os testes cobrem só a lógica de geometria (fórmulas, vértices, topologia) — não abrem janela nem precisam de GPU.

### 5. Rodar o programa

```bash
source venv/bin/activate
python3 main.py
```

## Controles (Etapa 1 — mouse e teclado)

| Ação | Controle |
|---|---|
| Rotacionar o tetraedro | Arrastar com o botão esquerdo do mouse |
| Zoom in/out | Roda do mouse |
| Aumentar/diminuir a aresta | Teclas `+`/`-` (ou `Cima`/`Baixo`) |
| Resetar a visualização | Tecla `R` |
| Fechar a janela | Tecla `ESC` ou fechar a janela |
| Encerrar o programa a qualquer momento | `Ctrl+C` no terminal |

O overlay no canto superior esquerdo mostra, em tempo real: aresta atual, área total, volume e altura do tetraedro.

## Solução de problemas

- **Tela preta ou erro de contexto OpenGL**: verifique se `libglu1-mesa` está instalado (passo 1) e se os drivers de vídeo estão atualizados (`glxinfo | grep "direct rendering"` deve retornar `Yes`).
- **Arrastar o mouse não rotaciona / comportamento estranho no drag**: se estiver numa sessão Wayland, tente rodar com `SDL_VIDEODRIVER=x11 python3 main.py`.
- **Janela redimensionada distorce a imagem**: não deveria acontecer (o viewer recalcula a projeção a cada redimensionamento); se acontecer, é um bug — reporte.
- **Texto do overlay ilegível ou de cabeça para baixo**: reporte também — foi testado nesta configuração, mas pode variar entre drivers.

## Estrutura de arquivos

```
jarvis-3d-geometria/
├── main.py                  # Ponto de entrada
├── config.py                 # Configurações (tamanho da janela, aresta inicial, etc.)
├── voice/
│   └── listener.py           # Ativação por voz (Etapa 3 — ainda não implementado)
├── vision/
│   ├── camera.py              # Captura de vídeo (Etapa 2 — ainda não implementado)
│   └── hand_tracker.py        # Rastreamento de mão (Etapa 2 — ainda não implementado)
├── geometry/
│   └── tetraedro.py           # Vértices, topologia e fórmulas do tetraedro regular
├── render/
│   └── viewer.py               # Janela OpenGL e loop de renderização
├── tests/
│   └── test_tetraedro.py       # Testes de geometria (pytest)
└── requirements.txt
```
