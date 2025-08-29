use image::GenericImageView;
use wgpu::util::DeviceExt;

// Classe para criar texturas
//----------------------------------------------------------------------------------
struct Texture {
    texture: wgpu::Texture,
    view: wgpu::TextureView,
    sampler: wgpu::Sampler
}

impl Texture {
    // Decodifica bytes de imagem na memória e cria uma textura na GPU chamando from_image
    fn from_bytes(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        bytes: &[u8],
        label: &str
    ) -> anyhow::Result<Self> {
        // Tenta decodificar os bytes em memória (bytes) como uma imagem (PNG, JPEG etc.) 
        let texture_image = image::load_from_memory(bytes)?;
        
        // Chama a função from_image
        Self::from_image(device, queue, &texture_image, Some(label))
    }

    // Cria uma textura na GPU a partir de uma imagem da CPU, configurando seus pixels, view e sampler
    fn from_image(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        texture_image: &image::DynamicImage,
        label: Option<&str>
    ) -> anyhow::Result<Self> {
        // Converte a imagem (texture_image) para o formato de pixels RGBA com 8 bits por canal
        let rgba = texture_image.to_rgba8();

        // Pega a largura e a altura da imagem (texture_image) e retorna como uma tupla (u32, u32).
        let dimensions = texture_image.dimensions();

        // Cria um objeto Extent3d do wgpu que define as dimensões da textura a partir do tamanho da imagem.
        let size = wgpu::Extent3d{
            width: dimensions.0, // Largura
            height: dimensions.1, // Altura
            depth_or_array_layers: 1 // Profundidade / Camadas
        };

        // Cria uma textura 2D na GPU (com formato RGBA8 e suporte para ser usada em shaders e receber dados copiados da CPU).
        let texture = device.create_texture(
            &wgpu::TextureDescriptor {
                label: label,
                size: size,
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8UnormSrgb,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[]
            }
        );

        // Copia os pixels da imagem RGBA da CPU (rgba) para a textura criada na GPU (texture).
        queue.write_texture(
            wgpu::TexelCopyTextureInfoBase {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO, // Preenchendo-a a partir do nível 0 e origem (0,0).
                aspect: wgpu::TextureAspect::All
            },
            &rgba,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(4 * dimensions.0),
                rows_per_image: Some(dimensions.1)
            },
            size
        );

        // Cria uma "view" da textura na GPU, que é usada para acessar ou renderizar a textura em shaders.
        let view = texture.create_view(
            &wgpu::TextureViewDescriptor::default()
        );

        // Cria um sampler na GPU, que define como a textura será amostrada nos shaders (como filtrar, interpolar e lidar com coordenadas fora do limite).
        let sampler = device
            .create_sampler(
                &wgpu::SamplerDescriptor {
                    label: label,
                    address_mode_u: wgpu::AddressMode::ClampToEdge,
                    address_mode_v: wgpu::AddressMode::ClampToEdge,
                    address_mode_w: wgpu::AddressMode::ClampToEdge,
                    mag_filter: wgpu::FilterMode::Linear,
                    min_filter: wgpu::FilterMode::Nearest,
                    mipmap_filter: wgpu::FilterMode::Nearest,
                    //lod_min_clamp: (),
                    //lod_max_clamp: (),
                    //compare: (),
                    //anisotropy_clamp: (),
                    //border_color: (),
                    ..Default::default()
                }
            );
        
        // Retorna com sucesso (Ok) uma instância da struct contendo a textura, a view e o sampler criados
        Ok(
            Self {
                texture: texture,
                view: view,
                sampler: sampler
            }
        )
    }
}
//----------------------------------------------------------------------------------

// Classe para criar vertices
//----------------------------------------------------------------------------------
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Vertex {
    position: [f32; 3],
    tex_coords: [f32; 2]
    //color: [f32; 3]
}

impl Vertex {
    // É um array de descrições de atributos do vértice.
    // Aqui você diz como a GPU deve interpretar cada campo da sua struct Vertex.
    const ATTRIBUTES: [wgpu::VertexAttribute; 2] = wgpu::vertex_attr_array![
        0 => Float32x3,
        1 => Float32x2
        //1 => Float32x3
    ];

    // Retorna uma VertexBufferLayout, que descreve como os vértices estão organizados no buffer
    fn desc<'a>() -> wgpu::VertexBufferLayout<'a> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Self>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &Self::ATTRIBUTES
        }
    }
}

// Define um array constante de vértices (Vertex) com posições 3D e coordenadas de textura
const VERTICES: &[Vertex] = &[
    Vertex { position: [ -0.5, 0.5, 0.0], tex_coords: [0.0, 0.0] },//color: [ 1.0, 0.0, 0.0] }, //A
    Vertex { position: [ -0.5, -0.5, 0.0], tex_coords: [0.0, 1.0] },//color: [ 0.0, 1.0, 0.0] }, //B
    Vertex { position: [ 0.5, -0.5, 0.0], tex_coords: [1.0, 1.0] },//color: [ 0.0, 0.0, 1.0] }, //C
    Vertex { position: [ 0.5, 0.5, 0.0], tex_coords: [1.0, 0.0] }//color: [ 0.0, 1.0, 0.0] } //D
];

// Define a ordem de desenho dos vértices (Nesse Caso forma um quadrado)
const INDICES: &[u16] = &[
    0, 1, 2,
    0, 2, 3,
];

//----------------------------------------------------------------------------------

// Classe de contexto do Wgpu
//----------------------------------------------------------------------------------
struct WgpuContext<'window_lifetime> {
    surface: wgpu::Surface<'window_lifetime>,
    adapter: wgpu::Adapter,
    device: wgpu::Device,
    vertex_buffer: wgpu::Buffer,
    num_vertices: u32,
    index_buffer: wgpu::Buffer,
    num_indices: u32,
    queue: wgpu::Queue,
    diffuse_bind_group: wgpu::BindGroup,
    diffuse_texture: Texture,
    surface_configuration: wgpu::SurfaceConfiguration,
    render_pipeline: wgpu::RenderPipeline
}

impl<'window_lifetime> WgpuContext<'window_lifetime> {
    // Chama a função new_async de forma assincrona
    fn new(window: std::sync::Arc<winit::window::Window>) -> WgpuContext<'window_lifetime> {
        pollster::block_on(WgpuContext::new_async(window))
    }

    // Inicializa o contexto do Wgpu
    async fn new_async(window: std::sync::Arc<winit::window::Window>) -> WgpuContext<'window_lifetime> {
        //  Representa a conexão com a API gráfica (Vulkan, DirectX, Metal ou WebGPU no navegador).
        let instance = wgpu::Instance::default();

        // Cria a superficie que será desenhada
        let surface = instance
            .create_surface(std::sync::Arc::clone(&window))
            .unwrap();

        // Requisita o adaptador que é basicamente uma placa de vídeo ou backend gráfico disponível no sistema  
        let adapter = instance
            .request_adapter(
                &wgpu::RequestAdapterOptionsBase {
                    power_preference: wgpu::PowerPreference::HighPerformance,
                    force_fallback_adapter: false,
                    compatible_surface: Some(&surface)
                }
            )
            .await
            .expect("Failled to find apropriate adapter");

        // São a interface que você vai realmente usar no seu código.
        let (
            // É a "conexão lógica" com a GPU, usada para criar recursos (buffers, texturas, pipelines...).
            device,
            // É a fila de comandos para enviar trabalho à GPU (ex.: desenhar, copiar memória).
            queue
        ) = adapter
            .request_device(
                &wgpu::wgt::DeviceDescriptor {
                    label: None,
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::defaults(),
                    memory_hints: wgpu::MemoryHints::Performance,
                    trace: wgpu::Trace::Off
                }
            )
            .await
            .expect("Failled to create device");

        // Cria um vertex buffer (um pedaço de memória na GPU que armazena os vértices que você vai desenhar)
        let vertex_buffer = device
            .create_buffer_init(
                &wgpu::util::BufferInitDescriptor {
                    label: Some("Vertex Buffer"),
                    contents: bytemuck::cast_slice(VERTICES),
                    usage: wgpu::BufferUsages::VERTEX
                }
            );
        
        // Informa o tamanho da lista de vertices
        let num_vertices = VERTICES.len() as u32;
        
        // Cria um index buffer (um pedaço de memória na GPU que armazena os indices dos vértices que você vai desenhar)
        let index_buffer = device
            .create_buffer_init(
                &wgpu::util::BufferInitDescriptor {
                    label: Some("Indices Buffer"),
                    contents: bytemuck::cast_slice(INDICES),
                    usage: wgpu::BufferUsages::INDEX
                }
            );
        
        // Informa o tamanho da lista de indices
        let num_indices = INDICES.len() as u32;

        // Inclui o arquivo como um array de bytes no binário em tempo de compilação, permitindo acessá-lo diretamente na memória.
        let diffuse_bytes = include_bytes!("assets/tree.png");

        // Cria uma textura na GPU a partir dos bytes da imagem (diffuse_bytes)
        let diffuse_texture = Texture::from_bytes(&device, &queue, diffuse_bytes, "assets/tree.png").unwrap();

        // Cria um layout de bind group na GPU, definindo como uma textura 2D e seu sampler serão acessados pelos shaders de fragmento.
        let texture_bind_group_layout = device
            .create_bind_group_layout(
                &wgpu::BindGroupLayoutDescriptor {
                    label: Some("Texture Bind Group Layout"),
                    entries: &[
                        wgpu::BindGroupLayoutEntry{
                            binding: 0,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Texture {
                                sample_type: wgpu::TextureSampleType::Float {
                                    filterable: true
                                },
                                view_dimension: wgpu::TextureViewDimension::D2,
                                multisampled: false
                            },
                            count: None
                        },
                        wgpu::BindGroupLayoutEntry{
                            binding: 1,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Sampler(
                                wgpu::SamplerBindingType::Filtering
                            ),
                            count: None
                        }
                    ]
                }
            );
        
        // Cria um bind group que associa a textura e o sampler específicos ao layout definido, permitindo que os shaders usem esses recursos na renderização.
        let diffuse_bind_group = device.create_bind_group(
            &wgpu::BindGroupDescriptor {
                label: Some("Diffuse Bind Group"),
                layout: &texture_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry{
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&diffuse_texture.view)
                    },
                    wgpu::BindGroupEntry{
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&diffuse_texture.sampler)
                    }
                ]
            }
        );

        // Coleta o tamanho atual da janela
        let window_size = window.inner_size();

        // Define os parâmetros da superficie
        let surface_configuration = surface
            .get_default_config(
                &adapter,
                window_size.width.max(1),
                window_size.height.max(1)
            )
            .unwrap();

        // Aplica a configuração de superfice definida
        surface.configure(&device, &surface_configuration);

        // Carrega o código WGSL (<archive_name>.wgsl) e compila para rodar na GPU.
        let shader = device
            .create_shader_module(
                wgpu::ShaderModuleDescriptor {
                    label: Some("Shader"),
                    source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into())
                }
            );
        
        // Define como os shaders vão receber recursos externos (texturas, buffers uniformes, samplers, etc.).
        let render_pipeline_layout = device
            .create_pipeline_layout(
                &wgpu::PipelineLayoutDescriptor {
                    label: Some("Render Pipeline Layout"),
                    bind_group_layouts: &[&texture_bind_group_layout],
                    push_constant_ranges: &[]
                }
            );
        
        // Cria o pipeline gráfico. Esse é o coração do WGPU, define como desenhar.
        let render_pipeline = device
            .create_render_pipeline(
                &wgpu::RenderPipelineDescriptor {
                    label: Some("Render Pipeline"),
                    layout: Some(&render_pipeline_layout),
                    vertex: wgpu::VertexState {
                        module: &shader,
                        entry_point: Some("vs_main"),
                        compilation_options: wgpu::PipelineCompilationOptions::default(),
                        buffers: &[
                            Vertex::desc() // Informa o buffer para o wgsl
                        ]
                    },
                    primitive: wgpu::PrimitiveState::default(), // Triangle List
                    depth_stencil: None,
                    multisample: wgpu::MultisampleState::default(),
                    fragment: Some(
                        wgpu::FragmentState {
                            module: &shader,
                            entry_point: Some("fs_main"),
                            compilation_options: wgpu::PipelineCompilationOptions::default(),
                            targets: &[
                                Some(
                                    wgpu::ColorTargetState {
                                        format: surface_configuration.format,
                                        blend: Some(wgpu::BlendState::REPLACE),
                                        write_mask: wgpu::ColorWrites::ALL
                                    }
                                )
                            ]
                        }
                    ),
                    multiview: None,
                    cache: None
                }
            );

        WgpuContext {
            surface,
            adapter,
            device,
            vertex_buffer,
            num_vertices,
            index_buffer,
            num_indices,
            queue,
            diffuse_bind_group,
            diffuse_texture,
            surface_configuration,
            render_pipeline
        }
    }

    // Redimensiona a resolução da janela
    fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        self.surface_configuration.width = new_size.width.max(1);
        self.surface_configuration.height = new_size.height.max(1);
        self.surface.configure(&self.device, &self.surface_configuration);
    }

    // Renderiza a imagem na janela
    fn draw(&mut self) {
        // Pega a textura da superfície (a "tela" onde vai desenhar)
        let frame = self
            .surface
            .get_current_texture()
            .expect("Failed to acquire next surface texture");
            //swap chain == surface
        
        // Cria uma view dessa textura: A view é como você enxerga e acessa a textura
        let view = frame
            .texture
            .create_view(
                &wgpu::TextureViewDescriptor::default()
            );
        
        // Cria um command encoder: O encoder é um bloco de comandos que você vai enviar para a GPU.
        let mut encoder = self
            .device
            .create_command_encoder(
                &wgpu::wgt::CommandEncoderDescriptor {
                    label: Some("Render Encoder")
                }
            );
        
        // Escopo não é necessário, é só para separar o que será desenhado
        {
            // Começa um render pass: Aqui você diz: “vou desenhar nessa textura (view)”.
            let mut render_pass = encoder
            .begin_render_pass(
                &wgpu::RenderPassDescriptor {
                    label: Some("Render Pass"),
                    color_attachments: &[
                        Some(
                            wgpu::RenderPassColorAttachment {
                                view: &view,
                                depth_slice: None,
                                resolve_target: None,
                                ops: wgpu::Operations {
                                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                                    store: wgpu::StoreOp::default()
                                }
                            }
                        )
                    ],
                    depth_stencil_attachment: None,
                    //timestamp_writes: (),
                    //occlusion_query_set: ()
                    ..Default::default()
                }
            );

            // Configura o pipeline de renderização: Aqui você diz qual pipeline usar.
            render_pass.set_pipeline(&self.render_pipeline);

            //
            render_pass.set_bind_group(0, &self.diffuse_bind_group, &[]);

            // Define o vertex buffer enviado para GPU -- talvez mudar commentario --
            render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));

            // Define o index buffer enviado para GPU -- talvez mudar commentario --
            render_pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint16);

            // Desenha os vértices: Esse comando dispara o vertex shader e o fragment shader do seu arquivo <archive_name>.wgsl.
            //render_pass.draw(0..self.num_vertices, 0..1); // sem index
            render_pass.draw_indexed(0..self.num_indices, 0, 0..1); // com index
        }

        // Envia os comandos para execução pela GPU.
        self.queue.submit(
            Some(
                // Fecha o bloco de comandos.
                encoder.finish()
            )
        );

        // Exibe a textura resultante na janela.
        frame.present();
    }
}
//----------------------------------------------------------------------------------

// Classe da Aplicação do Winit
//----------------------------------------------------------------------------------
#[derive(Default)]
struct WinitApplication<'window_lifetime> {
    window: Option<std::sync::Arc<winit::window::Window>>,
    wgpu_context: Option<WgpuContext<'window_lifetime>>
}

impl<'window_lifetime> winit::application::ApplicationHandler for WinitApplication<'window_lifetime> {
    // Chamada quando a aplicação é retomada e cria a janela e o contexto WGPU se ainda não existirem.
    fn resumed(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        if self.window.is_none() {
            // Declara a janela de forma assincrona
            let new_window = std::sync::Arc::new(
                event_loop
                .create_window(
                    winit::window::Window::default_attributes()
                    .with_title("Rust_Game_Engine")
                )
                .unwrap()
            );

            // Define os valores da classe WinitApplication
            self.window = Some(new_window.clone());
            self.wgpu_context = Some(WgpuContext::new(new_window.clone()));
        }
    }

    // Eventos que ocorrem na janela
    fn window_event(
        &mut self,
        event_loop: &winit::event_loop::ActiveEventLoop,
        window_id: winit::window::WindowId,
        event: winit::event::WindowEvent,
    ) {
        match event {
            // Evento quando a janela é fechada (X da janela)
            winit::event::WindowEvent::CloseRequested => {
                // Encerra o loop da janela
                event_loop.exit();
            },
            // Evento quando as teclas são acionadas
            winit::event::WindowEvent::KeyboardInput {
                event: winit::event::KeyEvent {
                    physical_key: winit::keyboard::PhysicalKey::Code(
                        code
                    ),
                    state,
                    ..
                },
                ..
            } => {
                match (code, state.is_pressed()) {
                    // Ao pressionar Escape (Esc)
                    (winit::keyboard::KeyCode::Escape, true) => {
                        // Encerra o loop da janela
                        event_loop.exit();
                    },
                    _ => ()
                }
            },
            // Evento quando a tela é redimensionada
            winit::event::WindowEvent::Resized(new_size) => {
                if let (
                    Some(window),
                    Some(wgpu_context)
                ) = (
                    self.window.as_ref(),
                    self.wgpu_context.as_mut()
                ) {
                    wgpu_context.resize(new_size);
                }
            }
            // Evento quando o redesenho é requisitado
            winit::event::WindowEvent::RedrawRequested => {
                if let (
                    Some(window),
                    Some(wgpu_context)
                ) = (
                    self.window.as_ref(),
                    self.wgpu_context.as_mut()
                ) {
                    // Chama a função para desenhar na tela
                    wgpu_context.draw();

                    // Solicita uma nova requisição de desenho (atualiza o loop continuamente)
                    window.request_redraw();
                }
            },
            _ => ()
        }
    }
}
//----------------------------------------------------------------------------------

// Inicia o evento de loop da janela
fn run_window() -> Result<(), winit::error::EventLoopError> {
    // Inicia o evento de loop
    let event_loop = winit::event_loop::EventLoop::new().unwrap();

    // Configura para que a janela atualize continuamente
    event_loop.set_control_flow(winit::event_loop::ControlFlow::Poll);

    // Declara a aplicação
    let mut winit_application = WinitApplication::default();

    // Executa a aplicação no evento de loop
    event_loop.run_app(&mut winit_application)
}

// Função principal
fn main() -> Result<(), winit::error::EventLoopError> {
    run_window()
}