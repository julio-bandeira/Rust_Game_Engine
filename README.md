# Rust_Game_Engine
Study of the fundamentals of a game engine using Rust (winit + wgpu)

```
┌──────────────────────────────┐
│   Frame Loop (uma iteração)  │
└───────┬──────────────────────┘
        │
        ▼
┌───────────────────────────┐
│  Lista de Objetos na cena │
└───────┬───────────────────┘
        │
        ▼
┌──────────────────────────────────────────────────────────────┐
│ Para cada objeto:                                            │
│                                                              │
│ 1 - Carrega dados do objeto (posição, rotação, escala, ...)  | 
│ 2 - Atualiza uniformes / bind group                          │
│ 3 - Atualiza buffers                                         │
└─────────┬────────────────────────────────────────────────────┘
          │
          ▼
┌───────────────────────────────────────────────────────────┐
│ Shader (único) na GPU                                     │
│                                                           │
│ - Vertex Shader: calcula posições dos vértices do objeto  │
│ - Fragment Shader: calcula cores/pixels                   │
└─────────┬─────────────────────────────────────────────────┘
          │
          ▼
┌──────────────────────────────────┐
│ Saída no framebuffer / surface   │
└─────────┬────────────────────────┘
          │
          ▼
┌─────────────────────────────┐
│ Objeto renderizado na tela  │
└─────────────────────────────┘
```

Frame Loop:
Cada iteração do loop principal processa todos os objetos que devem ser desenhados naquele frame.

Lista de Objetos:
Pode ser uma simples Vec<GameObject> ou algo mais complexo.

Processamento por objeto:

Buffers → geometria do objeto (vértices, índices)

Uniformes / Bind groups → posição, rotação, escala, cor, texturas, etc.

Todos os objetos podem usar o mesmo shader, mudando apenas os dados que alimentam a GPU.

Shader:

O vertex shader pega os vértices e aplica transformações.

O fragment shader calcula cor, luz, textura, etc.

Surface / Framebuffer:
A GPU escreve o resultado final pixel a pixel na tela.