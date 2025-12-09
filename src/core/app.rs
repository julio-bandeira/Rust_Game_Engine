use std::default;

use wgpu::{
    SurfaceError
};
use winit::{
    application::ApplicationHandler,
    event::{
        WindowEvent
    },
    event_loop::{
        ActiveEventLoop,
        ControlFlow,
        EventLoop
    },
    window::{
        Window,
        WindowId
    }
};
use log::info;

pub struct App<'a> {
    surface: wgpu::Surface<'a>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    window: Window
}

impl<'a> App<'a> {
    // Criação da engine (janela + wgpu)
    async fn new(event_loop: &ActiveEventLoop) -> Self {
        // Criação da janela
        let window = event_loop
            .create_window(
                Window::default_attributes()
                    .with_title("Rust Game Engine")
            )
            .expect("Falha ao criar janela");

        // Instancia WGPU
        let instance = wgpu::Instance::new(
            &wgpu::InstanceDescriptor::default()
        );

        // Criação da surface a partir da janela
        let surface = instance
            .create_surface(&window)
            .expect("Falha ao criar surface");

        // Escolha do adaptador
        let adapter = instance
            .request_adapter(
                &wgpu::RequestAdapterOptionsBase {
                    power_preference: wgpu::PowerPreference::HighPerformance,
                    force_fallback_adapter: false,
                    compatible_surface: Some(&surface)
                }
            )
            .await
            .expect("Falha ao escolher adaptador GPU");

        // Criação do device e queue
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: None,
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                    ..Default::default()
                }
            )
            .await
            .expect("Falha ao criar device");

        // Configuração da surface
        let size = window.inner_size();

        // Define formatos suportados da GPU
        let caps = surface.get_capabilities(&adapter);
        let format = caps.formats[0];

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width: size.width,
            height: size.height,
            present_mode: wgpu::PresentMode::Fifo,
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2
        };

        surface.configure(&device, &config);

        info!("Engine inicializada com sucesso!");

        Self {
            surface,
            device,
            queue,
            config,
            window
        }
    }

    // Resize da janela
    fn resize(&mut self, width: u32, height: u32) {
        if width > 0 && height > 0 {
            self.config.width = width;
            self.config.height = height;
            self.surface.configure(&self.device, &self.config);
        }
    }

    // Render (limpa tela)
    fn render(&mut self) -> Result<(), SurfaceError> {
        let frame = self.surface.get_current_texture()?;
        let view = frame.texture.create_view(
            &wgpu::TextureViewDescriptor::default()
        );

        let mut enconder = self.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder")
            }
        );

        {
            let _rpass = enconder.begin_render_pass(
                &wgpu::RenderPassDescriptor {
                    label: Some("Rnder Pass"),
                    color_attachments: &[
                        Some(
                            wgpu::RenderPassColorAttachment {
                                view: &view,
                                resolve_target: None,
                                ops: wgpu::Operations {
                                    load: wgpu::LoadOp::Clear(
                                        wgpu::Color {
                                            r: 0.05,
                                            g: 0.05,
                                            b: 0.15,
                                            a: 1.0
                                        }
                                    ),
                                    store: wgpu::StoreOp::Store,
                                },
                                depth_slice: None
                            }
                        )
                    ],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None
                }
            );

        }
        
        self.queue.submit(
            Some(
                enconder.finish()
            )
        );
        frame.present();

        Ok(())
    }

    // Função chamada pelo main
    pub async fn run() {
        let event_loop = EventLoop::new().unwrap();

        let mut app_holder: Option<App> = None;

        event_loop.run_app(
            &mut MyAppHandler {
                app: &mut app_holder
            }
        );
    }
}

// Handler que integra winit -> nossa struct App
struct MyAppHandler<'a> {
    app: &'a mut Option<App>
}

impl<'a> ApplicationHandler for MyAppHandler<'a> {
    // Inicialização do app dentro do event_loop
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.app.is_none() {
            *self.app = Some(pollster::block_on(App::new(event_loop)));
        }
    }

    // Eventos de janela
    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        window_id: WindowId,
        event: WindowEvent,
    ) {
        let app = self.app.as_mut().unwrap();

        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => {
                app.resize(size.width, size.height);
            },
            WindowEvent::RedrawRequested => {
                match app.render() {
                    Ok(_) => {},
                    Err(SurfaceError::Lost) => {
                        app.resize(app.config.width, app.config.height);
                    },
                    Err(SurfaceError::OutOfMemory) => event_loop.exit(),
                    Err(e) => eprintln!("Erro ao renderizar: {:?}", e)
                }
            },
            _ => ()
        }
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        if let Some(app) = self.app.as_ref() {
            app.window.request_redraw();
        }
    }
}