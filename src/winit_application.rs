// Classe da Aplicação do Winit
//----------------------------------------------------------------------------------
use crate::wgpu_context::WgpuContext;

#[derive(Default)]
pub struct WinitApplication<'window_lifetime> {
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
        if let (
            Some(window),
            Some(wgpu_context)
        ) = (
            self.window.as_ref(),
            self.wgpu_context.as_mut()
        ) {
            //
            wgpu_context.input(&event);
            //
            wgpu_context.update();
        }

        //
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