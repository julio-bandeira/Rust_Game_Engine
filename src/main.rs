// Classe de contexto do Wgpu
//----------------------------------------------------------------------------------
struct WgpuContext<'window_lifetime> {
    surface: wgpu::Surface<'window_lifetime>,
    adapter: wgpu::Adapter,
    device: wgpu::Device,
    queue: wgpu::Queue,
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
                    bind_group_layouts: &[],
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
                        buffers: &[] // sem buffer de vértices, usamos vertex_index
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
            queue,
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

            // Desenha os vértices: Esse comando dispara o vertex shader e o fragment shader do seu arquivo <archive_name>.wgsl.
            render_pass.draw(0..3, 0..1);
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
                    (winit::keyboard::KeyCode::Escape, true) => {
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
    let mut wAplication = WinitApplication::default();

    // Executa a aplicação no evento de loop
    event_loop.run_app(&mut wAplication)
}

// Função principal
fn main() -> Result<(), winit::error::EventLoopError> {
    run_window()
}