mod core;

use core::app::App;
use env_logger;

fn main() {
    // Inicializar logs (wgpu usa bastante)
    env_logger::init();

    // Cria a aplicação e roda até fechar
    pollster::block_on(App::run());
}
