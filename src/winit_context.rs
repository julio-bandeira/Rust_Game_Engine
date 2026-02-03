#[derive(Default)]
pub struct WinitContext<'app_lifetime> {
    pub window: Option<std::sync::Arc<winit::window::Window>>,
    pub wgpu_context: Option<crate::wgpu_context::WgpuContext<'app_lifetime>>,
    pub config_controller: Option<crate::config_controller::ConfigController>,
    pub scene_controller: Option<crate::scene_controller::SceneController>,
    pub last_frame: Option<std::time::Instant>
}

impl<'app_lifetime> winit::application::ApplicationHandler for WinitContext<'app_lifetime> {
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
            self.wgpu_context = Some(crate::wgpu_context::WgpuContext::new(new_window.clone()));
            self.config_controller = Some(crate::config_controller::ConfigController::new());
            self.scene_controller = Some(crate::scene_controller::SceneController::new());

            self.last_frame = Some(std::time::Instant::now());

            // Aplica as configurações
            if let (
                Some(config_controller),
                Some(window),
                Some(wgpu_context),
                Some(scene_controller)
            ) = (
                self.config_controller.as_ref(),
                self.window.as_ref(),
                self.wgpu_context.as_mut(),
                self.scene_controller.as_mut()
            ) {
                config_controller.apply_window(window);
                config_controller.apply_gpu(wgpu_context);
                //scene_controller.set_scene(Box::new(crate::scenes::gameplay_scene::GameplayScene::new()), wgpu_context);
                scene_controller.set_scene(Box::new(crate::scenes::main_menu_scene::MainMenuScene::new()), wgpu_context);
            }
        }
    }

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
            },
            _ => ()
        }
        // Aplica as configurações
        if let (
            Some(window),
            Some(scene_controller)
        ) = (
            self.window.as_ref(),
            self.scene_controller.as_mut()
        ) {
            scene_controller.handle_event(&event);
        }
    }

    // Aqui controla o ciclo de atualização da aplicação (update e render)
    fn about_to_wait(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        let now = std::time::Instant::now();
        let delta = now - self.last_frame.unwrap();

        self.last_frame = Some(now);

        //let fps = 1.0 / delta.as_secs_f32();
        //println!("FPS: {:.2}", fps);

        if let (
            Some(window),
            Some(wgpu_context),
            Some(scene_controller)
        ) = (
            self.window.as_ref(),
            self.wgpu_context.as_mut(),
            self.scene_controller.as_mut()
        ) {
            // Chama a função para atualizar e desenhar a cena na tela
            scene_controller.update(delta.as_secs_f32(), wgpu_context);
            scene_controller.render(wgpu_context);
    
            // Solicita uma nova requisição de desenho (atualiza o loop continuamente)S
            window.request_redraw();
        }
    }
}