pub struct GameplayScene;

impl GameplayScene {
    pub fn new() -> Self {
        Self
    }
}

impl crate::scene::Scene for GameplayScene {
    fn on_enter(&mut self, _wgpu: &mut crate::wgpu_context::WgpuContext) {
        println!("Entrou na Gameplay");
    }

    fn on_exit(&mut self, _wgpu: &mut crate::wgpu_context::WgpuContext) {
        println!("Saiu da Gameplay");
    }

    fn update(&mut self, _dt: f32) {
        // animações, lógica
    }

    fn render(&mut self, wgpu: &mut crate::wgpu_context::WgpuContext) {
        //wgpu.clear_screen(); // fundo preto, por enquanto
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
