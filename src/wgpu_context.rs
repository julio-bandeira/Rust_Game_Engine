pub struct WgpuContext<'app_lifetime> {
    pub surface: wgpu::Surface<'app_lifetime>,
    pub adapter: wgpu::Adapter,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub surface_configuration: wgpu::SurfaceConfiguration
}

impl<'app_lifetime> WgpuContext<'app_lifetime> {
    // Chama a função new_async de forma assincrona
    pub fn new(window: std::sync::Arc<winit::window::Window>) -> WgpuContext<'app_lifetime> {
        pollster::block_on(WgpuContext::new_async(window))
    }

    async fn new_async(window: std::sync::Arc<winit::window::Window>) -> WgpuContext<'app_lifetime> {
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
                    trace: wgpu::Trace::Off,
                    ..Default::default()
                }
            )
            .await
            .expect("Failled to create device");

        // Coleta o tamanho atual da janela
        let window_size = window.inner_size();

        // Define os parâmetros da superficie
        let mut surface_configuration = surface
            .get_default_config(
                &adapter,
                window_size.width.max(1),
                window_size.height.max(1)
            )
            .unwrap();

        // Consulta à GPU quais formatos, modos de apresentação e alpha ela suporta
        let capabilities = surface.get_capabilities(&adapter);

        // Escolhe o modo de apresentação (VSYNC on/off)
        let present_mode = if capabilities
            // Verifica se a GPU + backend + SO suportam o modo Immediate (VSYNC desligado)
            .present_modes
            .contains(&wgpu::PresentMode::Immediate)
        {
            // Se suportar, usa Immediate → FPS livre (sem VSYNC)
            wgpu::PresentMode::Immediate
        } else {
            // Caso contrário, usa Fifo → modo garantido (VSYNC ligado)
            wgpu::PresentMode::Fifo
        };

        // Ajusta o present mode (VSYNC on/off)
        surface_configuration.present_mode = present_mode;

        // (Opcional) escolher alpha_mode explicitamente
        surface_configuration.alpha_mode = wgpu::CompositeAlphaMode::Auto;

        // (Opcional) uso da superfície
        surface_configuration.usage = wgpu::TextureUsages::RENDER_ATTACHMENT;

        WgpuContext {
            surface: surface,
            adapter: adapter,
            device: device,
            queue: queue,
            surface_configuration: surface_configuration
        }
    }

    // Redimensiona a resolução da janela
    pub fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        self.surface_configuration.width = new_size.width.max(1);
        self.surface_configuration.height = new_size.height.max(1);
        self.surface.configure(&self.device, &self.surface_configuration);
    }

    // Renderiza a imagem na janela
    pub fn render(&mut self) {
        // Pega a textura da superfície (a "tela" onde vai desenhar)
        let frame = match self.surface.get_current_texture() {
            Ok(frame) => frame,

            Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                // fallback de segurança
                self.surface.configure(&self.device, &self.surface_configuration);
                return;
            }

            Err(wgpu::SurfaceError::OutOfMemory) => {
                panic!("Out of memory");
            }

            Err(e) => {
                eprintln!("Surface error: {:?}", e);
                return;
            }
        };
        
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
                &wgpu::CommandEncoderDescriptor {
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
                                    store: wgpu::StoreOp::Store
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