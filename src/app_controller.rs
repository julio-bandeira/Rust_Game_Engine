pub struct AppController<'app_lifetime> {
    pub winit_context: crate::winit_context::WinitContext<'app_lifetime>
}

impl<'app_lifetime> AppController<'app_lifetime> {
    pub fn new() -> AppController<'app_lifetime> {
        // Declara o contexto da aplicação
        let winit_context = crate::winit_context::WinitContext::default();

        AppController {
            winit_context: winit_context
        }
    }

    pub fn run(&mut self) -> Result<(), winit::error::EventLoopError> {
        // Inicia o evento de loop
        let event_loop = winit::event_loop::EventLoop::new().unwrap();

        // Configura para que a janela atualize continuamente
        event_loop.set_control_flow(winit::event_loop::ControlFlow::Poll);

        // Executa a aplicação no evento de loop
        event_loop.run_app(&mut self.winit_context)
    }
}