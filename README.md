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

-------------------

+----------------------+
|      Main / App      | <---- inicia tudo, cria janela e contexto WGPU
+----------------------+
           |
           v
+----------------------+
|     WgpuContext      | <---- inicializa GPU, pipeline, surface, shaders
+----------------------+
           |
           |--- cria RenderPipeline (vertex + fragment shader)
           |--- cria Depth Texture
           |--- cria CameraUniform buffer
           |--- cria Instance buffer
           v
+----------------------+
|      Camera          |
+----------------------+
           |
           |--- CameraController atualiza posição da câmera
           |--- CameraUniform recebe view_proj matrix
           v
+----------------------+
|     Resource Loader  | <---- abstração para ler arquivos do disco
+----------------------+
           |
           |--- load_string()  -> lê OBJ ou MTL
           |--- load_binary()  -> lê imagens
           |--- load_texture() -> cria Texture (wgpu::Texture + Sampler)
           |--- load_model()   -> cria Model (meshes + materials)
           v
+----------------------+
|      Model / Mesh    |
+----------------------+
           |
           |--- Mesh: buffers de vértices e índices
           |--- Material: bind group com Texture + Sampler
           v
+----------------------+
|     Instance         |
+----------------------+
           |
           |--- Cada instância tem posição + rotação
           |--- Converte para InstanceRaw (matriz 4x4) para GPU
           v
+----------------------+
| GPU Buffers          |
+----------------------+
           |
           |--- Vertex Buffer (posição, tex_coords, normal)
           |--- Index Buffer
           |--- Instance Buffer (matriz 4x4)
           |--- CameraUniform Buffer (matriz view_proj)
           v
+----------------------+
| Render Pass          |
+----------------------+
           |
           |--- set_pipeline()
           |--- set_vertex_buffer()
           |--- set_index_buffer()
           |--- set_bind_group() (Material + Camera)
           |--- draw_indexed() ou draw_indexed_instanced()
           v
+----------------------+
| Frame Output         |
+----------------------+
           |
           v
+----------------------+
| Surface.present()    |
+----------------------+

