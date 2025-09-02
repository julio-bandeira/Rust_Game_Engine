use crate::winit_application::WinitApplication;

mod winit_application;
mod wgpu_context;
mod vertex;
mod texture;
mod camera;
mod camera_uniform;
mod camera_controller;
mod instance;

// Inicia o evento de loop da janela
fn run_window() -> Result<(), winit::error::EventLoopError> {
    // Inicia o evento de loop
    let event_loop = winit::event_loop::EventLoop::new().unwrap();

    // Configura para que a janela atualize continuamente
    event_loop.set_control_flow(winit::event_loop::ControlFlow::Poll);

    // Declara a aplicação
    let mut winit_application = WinitApplication::default();

    // Executa a aplicação no evento de loop
    event_loop.run_app(&mut winit_application)
}

// Função principal
fn main() -> Result<(), winit::error::EventLoopError> {
    run_window()
}