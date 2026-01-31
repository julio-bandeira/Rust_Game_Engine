pub struct MainMenuScene;

impl MainMenuScene {
    pub fn new() -> Self {
        Self
    }
}

impl crate::scene::Scene for MainMenuScene {
    fn on_enter(&mut self, _wgpu: &mut crate::wgpu_context::WgpuContext) {
        println!("Entrou no Main Menu");
    }

    fn on_exit(&mut self, _wgpu: &mut crate::wgpu_context::WgpuContext) {
        println!("Saiu do Main Menu");
    }

    fn update(&mut self, _dt: f32) {
        // animações, lógica
    }

    fn render(&mut self, wgpu: &mut crate::wgpu_context::WgpuContext) {
        //wgpu.clear_screen(); // fundo preto, por enquanto
        wgpu.render();
    }

    fn handle_event(&mut self, event: &winit::event::WindowEvent) {
        match event {
            winit::event::WindowEvent::KeyboardInput { .. } => {
                println!("Input no menu");
            }
            _ => {}
        }
    }
}
