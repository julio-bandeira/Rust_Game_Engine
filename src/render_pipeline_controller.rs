#[derive(Hash, Eq, PartialEq, Debug, Clone, Copy)]
pub enum RenderPipelineType {
    Mesh3D,
    Sprite2D,
    Ui
}

pub struct RenderPipelineManager {
    pipelines: std::collections::HashMap<RenderPipelineType, wgpu::RenderPipeline>
}

impl RenderPipelineManager {
    pub fn new() -> RenderPipelineManager {
        RenderPipelineManager { pipelines: std::collections::HashMap::new() }
    }

    pub fn load_pipelines(
        &mut self,
        device: &wgpu::Device,
        surface_config: &wgpu::SurfaceConfiguration,
        mesh_layout: &wgpu::PipelineLayout,
        mesh_shader: &wgpu::ShaderModule,
    ) {

        // Mesh3D
        self.pipelines.insert(
            RenderPipelineType::Mesh3D,
            device.create_render_pipeline(
            &wgpu::RenderPipelineDescriptor {
                    label: Some("Mesh3D Pipeline"),
                    layout: Some(mesh_layout),

                    vertex: wgpu::VertexState {
                        module: mesh_shader,
                        entry_point: Some("vs_main"),
                        compilation_options: Default::default(),
                        buffers: &[
                            crate::render::model::ModelVertex::desc(),
                            crate::render::instance::InstanceRaw::desc(),
                        ],
                    },

                    primitive: wgpu::PrimitiveState::default(),

                    depth_stencil: Some(wgpu::DepthStencilState {
                        format: crate::render::texture::Texture::DEPTH_FORMAT,
                        depth_write_enabled: true,
                        depth_compare: wgpu::CompareFunction::LessEqual,
                        stencil: Default::default(),
                        bias: Default::default(),
                    }),

                    multisample: Default::default(),

                    fragment: Some(wgpu::FragmentState {
                        module: mesh_shader,
                        entry_point: Some("fs_main"),
                        compilation_options: Default::default(),
                        targets: &[Some(wgpu::ColorTargetState {
                            format: surface_config.format,
                            blend: Some(wgpu::BlendState::REPLACE),
                            write_mask: wgpu::ColorWrites::ALL,
                        })],
                    }),

                    multiview: None,
                    cache: None,
                }
            )
        );
    }

    pub fn get(&self, render_type: RenderPipelineType) -> &wgpu::RenderPipeline {
        self.pipelines.get(&render_type).expect("Pipeline não carregado")
    }
}