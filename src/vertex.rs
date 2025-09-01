// Classe para criar vertices
//----------------------------------------------------------------------------------
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Vertex {
    pub position: [f32; 3],
    pub tex_coords: [f32; 2]
    //color: [f32; 3]
}

impl Vertex {
    // É um array de descrições de atributos do vértice.
    // Aqui você diz como a GPU deve interpretar cada campo da sua struct Vertex.
    const ATTRIBUTES: [wgpu::VertexAttribute; 2] = wgpu::vertex_attr_array![
        0 => Float32x3,
        1 => Float32x2
        //1 => Float32x3
    ];

    // Retorna uma VertexBufferLayout, que descreve como os vértices estão organizados no buffer
    pub fn desc<'a>() -> wgpu::VertexBufferLayout<'a> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Self>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &Self::ATTRIBUTES
        }
    }
}
//----------------------------------------------------------------------------------