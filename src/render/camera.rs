use cgmath::InnerSpace;
use cgmath::SquareMatrix;

// Uniform da câmera contendo a matriz view-proj enviada ao shader.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CameraUniform {
    pub view_proj: [[f32; 4]; 4]
}

impl CameraUniform {
    // Inicializa uniform com identidade.
    pub fn new() -> Self {
        Self {
            view_proj: cgmath::Matrix4::identity().into()
        }
    }

    // Atualiza a matriz view_proj a partir de uma Camera.
    pub fn update_view_proj(&mut self, camera: &Camera) {
        self.view_proj = camera.build_view_projection_matrix().into();
    }
}

// Controlador simples de câmera que lida com entrada e atualiza posição/olhar.
pub struct CameraController {
    pub speed: f32,
    pub is_forward_pressed: bool,
    pub is_backward_pressed: bool,
    pub is_left_pressed: bool,
    pub is_right_pressed: bool,
}

impl CameraController {
    // Cria um controlador com velocidade fornecida.
    pub fn new(speed: f32) -> Self {
        Self {
            speed: speed,
            is_forward_pressed: false,
            is_backward_pressed: false,
            is_left_pressed: false,
            is_right_pressed: false
        }
    }

    // Processa eventos de teclado e atualiza flags. Retorna true se evento foi consumido.
    pub fn process_events(&mut self, event: &winit::event::WindowEvent) -> bool {
        match event {
            winit::event::WindowEvent::KeyboardInput {
                event:
                    winit::event::KeyEvent {
                        state,
                        physical_key: winit::keyboard::PhysicalKey::Code(keycode),
                        ..
                    },
                ..
            } => {
                let is_pressed = *state == winit::event::ElementState::Pressed;
                match keycode {
                    winit::keyboard::KeyCode::KeyW | winit::keyboard::KeyCode::ArrowUp => {
                        self.is_forward_pressed = is_pressed;
                        true
                    }
                    winit::keyboard::KeyCode::KeyA | winit::keyboard::KeyCode::ArrowLeft => {
                        self.is_left_pressed = is_pressed;
                        true
                    }
                    winit::keyboard::KeyCode::KeyS | winit::keyboard::KeyCode::ArrowDown => {
                        self.is_backward_pressed = is_pressed;
                        true
                    }
                    winit::keyboard::KeyCode::KeyD | winit::keyboard::KeyCode::ArrowRight => {
                        self.is_right_pressed = is_pressed;
                        true
                    }
                    _ => false,
                }
            },
            _ => false
        }
    }

    // Atualiza posição da câmera com base nas flags de teclado (movimento orbitante).
    pub fn update_camera(&self, camera: &mut Camera) {
        let forward = camera.target - camera.eye;
        let forward_norm = forward.normalize();
        let forward_mag = forward.magnitude();

        // Impede ruído quando muito próximo do centro
        if self.is_forward_pressed && forward_mag > self.speed {
            camera.eye += forward_norm * self.speed;
        }
        if self.is_backward_pressed {
            camera.eye -= forward_norm * self.speed;
        }

        let right = forward_norm.cross(camera.up);

        // Recalcula forward e magnitude após mudanças
        let forward = camera.target - camera.eye;
        let forward_mag = forward.magnitude();

        if self.is_right_pressed {
            // Rotaciona ao redor do alvo mantendo a mesma distância
            camera.eye = camera.target - (forward + right * self.speed).normalize() * forward_mag;
        }
        if self.is_left_pressed {
            camera.eye = camera.target - (forward - right * self.speed).normalize() * forward_mag;
        }
    }
}

// Representa a câmera (eye, target, up) e parâmetros de projeção.
pub struct Camera {
    pub eye: cgmath::Point3<f32>,
    pub target: cgmath::Point3<f32>,
    pub up: cgmath::Vector3<f32>,
    pub aspect: f32,
    pub fovy: f32,
    pub znear: f32,
    pub zfar: f32
}

impl Camera {
    // Matriz para converter coordenadas OpenGL para espaço esperado pelo WGPU (clip space fix).
    #[rustfmt::skip]
    pub const OPENGL_TO_WGPU_MATRIX: cgmath::Matrix4<f32> = cgmath::Matrix4::from_cols(
        cgmath::Vector4::new(1.0, 0.0, 0.0, 0.0),
        cgmath::Vector4::new(0.0, 1.0, 0.0, 0.0),
        cgmath::Vector4::new(0.0, 0.0, 0.5, 0.0),
        cgmath::Vector4::new(0.0, 0.0, 0.5, 1.0)
    );

    // Constrói a matriz view-projection usada nos shaders.
    pub fn build_view_projection_matrix(&self) -> cgmath::Matrix4<f32> {
        let view = cgmath::Matrix4::look_at_rh(self.eye, self.target, self.up);
        let proj = cgmath::perspective(cgmath::Deg(self.fovy), self.aspect, self.znear, self.zfar);

        return Self::OPENGL_TO_WGPU_MATRIX * proj * view
    }
}
