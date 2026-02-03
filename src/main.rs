mod app_controller;
mod config_controller;
mod winit_context;
mod wgpu_context;
mod scene_controller;
mod scene;
mod scenes;

mod render_pipeline_controller;
mod render;
mod resource;

fn main() {
    // Instância o controle da aplicação
    let mut app = crate::app_controller::AppController::new();

    // Inicia o loop da aplicação
    app.run().unwrap();
}