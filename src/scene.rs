// Trait para padronizar os comando de gerenciamento das cenas
pub trait Scene {

    // Quando a cena passa a existir
    fn on_enter(&mut self, wgpu: &mut crate::wgpu_context::WgpuContext);

    // Quando a cena deixa de existir
    fn on_exit(&mut self, wgpu: &mut crate::wgpu_context::WgpuContext);

    // Atualizar informações da cena
    fn update(&mut self, delta_time: f32);

    // Renderizar tela
    fn render(&mut self, wgpu: &mut crate::wgpu_context::WgpuContext);

    // Processar os eventos em tela
    fn handle_event(&mut self, event: &winit::event::WindowEvent);
    
}