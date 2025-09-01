// Classe para criar CameraUniform <mudar comentário depois>
//----------------------------------------------------------------------------------
use cgmath::SquareMatrix;
use crate::camera::Camera;

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CameraUniform {
    pub view_proj: [[f32; 4]; 4]
}

impl CameraUniform {
    //
    pub fn new() -> Self {
        Self {
            view_proj: cgmath::Matrix4::identity().into()
        }
    }

    //
    pub fn update_view_proj(&mut self, camera: &Camera) {
        self.view_proj = camera.build_view_projection_matrix().into();
    }
}
//----------------------------------------------------------------------------------