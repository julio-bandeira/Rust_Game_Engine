mod app_controller;
mod config_controller;
mod winit_context;
mod wgpu_context;

fn main() {
    // Instância o controle da aplicação
    let mut app = crate::app_controller::AppController::new();

    // Inicia o loop da aplicação
    app.run().unwrap();
}