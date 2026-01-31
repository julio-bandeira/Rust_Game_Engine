// Sistema de controle de cenas
pub struct SceneController {
    current_scene: Option<Box<dyn crate::scene::Scene>>,
}

impl SceneController {

    // Declara o controle de cenas
    pub fn new() -> Self {
        Self {
            current_scene: None,
        }
    }

    // Altera a cena atual
    pub fn set_scene(
        &mut self,
        mut new_scene: Box<dyn crate::scene::Scene>,
        wgpu: &mut crate::wgpu_context::WgpuContext,
    ) {
        if let Some(scene) = self.current_scene.as_mut() {
            scene.on_exit(wgpu);
        }

        new_scene.on_enter(wgpu);
        self.current_scene = Some(new_scene);
    }

    // Atualiza informações da cena atual
    pub fn update(&mut self, delta_time: f32) {
        if let Some(scene) = self.current_scene.as_mut() {
            scene.update(delta_time);
        }
    }

    // Renderiza informações da cena atual
    pub fn render(&mut self, wgpu: &mut crate::wgpu_context::WgpuContext) {
        if let Some(scene) = self.current_scene.as_mut() {
            scene.render(wgpu);
        }
    }

    // Processa os eventos durante a cena
    pub fn handle_event(&mut self, event: &winit::event::WindowEvent) {
        if let Some(scene) = self.current_scene.as_mut() {
            scene.handle_event(event);
        }
    }
    
}
