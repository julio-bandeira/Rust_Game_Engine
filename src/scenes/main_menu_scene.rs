use wgpu::util::DeviceExt;
use cgmath::prelude::*;

pub struct MainMenuScene {
    pub depth_texture: Option<crate::render::texture::Texture>,
    pub render_pipeline: Option<wgpu::RenderPipeline>,

    pub camera: Option<crate::render::camera::Camera>,
    pub camera_uniform: Option<crate::render::camera::CameraUniform>,
    pub camera_buffer: Option<wgpu::Buffer>,
    pub camera_bind_group: Option<wgpu::BindGroup>,
    pub camera_controller: Option<crate::render::camera::CameraController>,

    pub obj_model: Option<crate::render::model::Model>,
    pub instances: Option<Vec<crate::render::instance::Instance>>,
    pub instance_buffer: Option<wgpu::Buffer>,
}

impl MainMenuScene {
    pub fn new() -> Self {
        Self {
            depth_texture: None,
            render_pipeline: None,
            camera: None,
            camera_uniform: None,
            camera_buffer: None,
            camera_bind_group: None,
            camera_controller: None,
            obj_model: None,
            instances: None,
            instance_buffer: None
        }
    }
}

impl crate::scene::Scene for MainMenuScene {
    fn on_enter(&mut self, _wgpu: &mut crate::wgpu_context::WgpuContext) {
        let mut _camera = crate::render::camera::Camera {
                eye: (0.0, 1.0, 2.0).into(),
                target: (0.0, 0.0, 0.0).into(),
                up: cgmath::Vector3::unit_y(),
                aspect: _wgpu.surface_configuration.width as f32 / _wgpu.surface_configuration.height as f32,
                fovy: 45.0,
                znear: 0.1,
                zfar: 100.0
            };
        //
        let mut _camera_uniform = crate::render::camera::CameraUniform::new();
        
        //
        _camera_uniform.update_view_proj(&_camera);
        
        //
        self.camera = Some(_camera);

        self.camera_uniform = Some(_camera_uniform);

        //
        let mut _camera_buffer = _wgpu.device.create_buffer_init(
                &wgpu::util::BufferInitDescriptor{
                    label: Some("Camera Buffer"),
                    contents: bytemuck::cast_slice(&[self.camera_uniform.unwrap()]),
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST
                }
            );

        //
        let camera_bind_group_layout = _wgpu.device
            .create_bind_group_layout(
                &wgpu::BindGroupLayoutDescriptor {
                    label: Some("Camera Bind Group Layout"),
                    entries: &[
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::VERTEX,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Uniform,
                                has_dynamic_offset: false,
                                min_binding_size: None
                            },
                            count: None
                        }
                    ]
                }
            );

        //
        self.camera_bind_group = Some(
            _wgpu.device
                .create_bind_group(
                    &wgpu::BindGroupDescriptor {
                        label: Some("Camera Bind Group"),
                        layout: &camera_bind_group_layout,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: _camera_buffer.as_entire_binding()
                            }
                        ]
                    }
                )
        );

        self.camera_buffer = Some(_camera_buffer);
        
        //
        self.camera_controller = Some(crate::render::camera::CameraController::new(0.2));

        // Carrega o código WGSL (<archive_name>.wgsl) e compila para rodar na GPU.
        let shader = _wgpu.device
            .create_shader_module(
                wgpu::ShaderModuleDescriptor {
                    label: Some("Shader"),
                    source: wgpu::ShaderSource::Wgsl(include_str!("../../shaders/shader.wgsl").into())
                }
            );
        
        //
        self.depth_texture = Some(crate::render::texture::Texture::create_depth_texture(&_wgpu.device, &_wgpu.surface_configuration, "depth_texture"));
        
        //
        const NUM_INSTANCES_PER_ROW: u32 = 4;

        //
        const SPACE_BETWEEN: f32 = 3.0;
        let _instances = (0..NUM_INSTANCES_PER_ROW).flat_map(|z| {
            (0..NUM_INSTANCES_PER_ROW).map(move |x| {
                let x = SPACE_BETWEEN * (x as f32 - NUM_INSTANCES_PER_ROW as f32 / 2.0);
                let z = SPACE_BETWEEN * (z as f32 - NUM_INSTANCES_PER_ROW as f32 / 2.0);

                let position = cgmath::Vector3 { x, y: 0.0, z };

                let rotation = if position.is_zero() {
                    cgmath::Quaternion::from_axis_angle(cgmath::Vector3::unit_z(), cgmath::Deg(0.0))
                } else {
                    cgmath::Quaternion::from_axis_angle(position.normalize(), cgmath::Deg(45.0))
                };

                crate::render::instance::Instance {
                    position, rotation,
                }
            })
        }).collect::<Vec<_>>();

        //
        let instance_data = _instances.iter().map(crate::render::instance::Instance::to_raw).collect::<Vec<_>>();

        self.instances = Some(_instances);

        //
        self.instance_buffer = Some(
            _wgpu.device.create_buffer_init(
                &wgpu::util::BufferInitDescriptor {
                    label: Some("Instance Buffer"),
                    contents: bytemuck::cast_slice(&instance_data),
                    usage: wgpu::BufferUsages::VERTEX
                }
            )
        );

        // Cria um layout de bind group na GPU, definindo como uma textura 2D e seu sampler serão acessados pelos shaders de fragmento.
        let texture_bind_group_layout = _wgpu.device
            .create_bind_group_layout(
                &wgpu::BindGroupLayoutDescriptor {
                    label: Some("Texture Bind Group Layout"),
                    entries: &[
                        wgpu::BindGroupLayoutEntry{
                            binding: 0,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Texture {
                                sample_type: wgpu::TextureSampleType::Float {
                                    filterable: true
                                },
                                view_dimension: wgpu::TextureViewDimension::D2,
                                multisampled: false
                            },
                            count: None
                        },
                        wgpu::BindGroupLayoutEntry{
                            binding: 1,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Sampler(
                                wgpu::SamplerBindingType::Filtering
                            ),
                            count: None
                        }
                    ]
                }
            );

        //
        self.obj_model = Some(
            crate::resource::load_model("cube.obj", &_wgpu.device, &_wgpu.queue, &texture_bind_group_layout).unwrap()
        );

        // Define como os shaders vão receber recursos externos (texturas, buffers uniformes, samplers, etc.).
        let render_pipeline_layout = _wgpu.device
            .create_pipeline_layout(
                &wgpu::PipelineLayoutDescriptor {
                    label: Some("Render Pipeline Layout"),
                    bind_group_layouts: &[
                        &texture_bind_group_layout,
                        &camera_bind_group_layout
                    ],
                    push_constant_ranges: &[]
                }
            );
        
        // Cria o pipeline gráfico. Esse é o coração do WGPU, define como desenhar.
        self.render_pipeline = Some(
            _wgpu.device
                .create_render_pipeline(
                    &wgpu::RenderPipelineDescriptor {
                        label: Some("Render Pipeline"),
                        layout: Some(&render_pipeline_layout),
                        vertex: wgpu::VertexState {
                            module: &shader,
                            entry_point: Some("vs_main"),
                            compilation_options: wgpu::PipelineCompilationOptions::default(),
                            buffers: &[
                                //Vertex::desc(), // Informa o buffer para o wgsl
                                crate::render::model::ModelVertex::desc(), // Informa o buffer para o wgsl
                                crate::render::instance::InstanceRaw::desc() // Informa o buffer para o wgsl
                            ]
                        },
                        primitive: wgpu::PrimitiveState::default(), // Triangle List
                        depth_stencil: Some(
                            wgpu::DepthStencilState {
                                format: crate::render::texture::Texture::DEPTH_FORMAT,
                                depth_write_enabled: true,
                                depth_compare: wgpu::CompareFunction::LessEqual,
                                stencil: wgpu::StencilState::default(),
                                bias: wgpu::DepthBiasState::default()
                            }
                        ),
                        multisample: wgpu::MultisampleState::default(),
                        fragment: Some(
                            wgpu::FragmentState {
                                module: &shader,
                                entry_point: Some("fs_main"),
                                compilation_options: wgpu::PipelineCompilationOptions::default(),
                                targets: &[
                                    Some(
                                        wgpu::ColorTargetState {
                                            format: _wgpu.surface_configuration.format,
                                            blend: Some(wgpu::BlendState::REPLACE),
                                            write_mask: wgpu::ColorWrites::ALL
                                        }
                                    )
                                ]
                            }
                        ),
                        multiview: None,
                        cache: None
                    }
                )
        );

        //if let (
        //    Some(camera),
        //    Some(camera_uniform),
        //    Some(camera_buffer),
        //    Some(camera_bind_group),
        //    Some(camera_controller),
        //    Some(obj_model),
        //    Some(instances),
        //    Some(instance_buffer)
        //) = (
        //    self.camera.as_mut(),
        //    self.camera_uniform.as_mut(),
        //    self.camera_buffer.as_mut(),
        //    self.camera_bind_group.as_mut(),
        //    self.camera_controller.as_mut(),
        //    self.obj_model.as_mut(),
        //    self.instances.as_mut(),
        //    self.instance_buffer.as_mut()
        //) {
        //
        //    camera = ;
        //    camera_uniform = ;
        //    camera_buffer = ;
        //    camera_bind_group = ;
        //    camera_controller = ;
        //    obj_model = ;
        //    instances = ;
        //    instance_buffer = ;
        //}
        println!("Entrou na Gameplay");
    }

    fn on_exit(&mut self, _wgpu: &mut crate::wgpu_context::WgpuContext) {
        println!("Saiu da Gameplay");
    }

    fn update(&mut self, _dt: f32, _wgpu: &mut crate::wgpu_context::WgpuContext) {
        // animações, lógica
        if let (
            Some(camera),
            Some(camera_controller),
            Some(camera_uniform),
            Some(camera_buffer)
        ) = (
            self.camera.as_mut(),
            self.camera_controller.as_mut(),
            self.camera_uniform.as_mut(),
            self.camera_buffer.as_mut()
        ) {
            //
            camera_controller.update_camera(camera);
            //
            camera_uniform.update_view_proj(&camera);
            //
            _wgpu.queue
                .write_buffer(
                    &camera_buffer,
                    0,
                    bytemuck::cast_slice(&[camera_uniform.clone()])
                );
        }
    }

    fn render(&mut self, _wgpu: &mut crate::wgpu_context::WgpuContext) {
        //wgpu.clear_screen(); // fundo preto, por enquanto
        // Pega a textura da superfície (a "tela" onde vai desenhar)
        let frame = _wgpu
            .surface
            .get_current_texture()
            .expect("Failed to acquire next surface texture");
            //swap chain == surface
        
        // Cria uma view dessa textura: A view é como você enxerga e acessa a textura
        let view = frame
            .texture
            .create_view(
                &wgpu::TextureViewDescriptor::default()
            );
        
        // Cria um command encoder: O encoder é um bloco de comandos que você vai enviar para a GPU.
        let mut encoder = _wgpu
            .device
            .create_command_encoder(
                &wgpu::wgt::CommandEncoderDescriptor {
                    label: Some("Render Encoder")
                }
            );
        
        if let (
            Some(camera_bind_group),
            Some(depth_texture),
            Some(instances),
            Some(instance_buffer),
            Some(render_pipeline),
            Some(obj_model)
        ) = (
            self.camera_bind_group.as_ref(),
            self.depth_texture.as_ref(),
            self.instances.as_ref(),
            self.instance_buffer.as_ref(),
            self.render_pipeline.as_ref(),
            self.obj_model.as_ref()
        ) {
            // Escopo não é necessário, é só para separar o que será desenhado
            {
                // Começa um render pass: Aqui você diz: “vou desenhar nessa textura (view)”.
                let mut render_pass = encoder
                .begin_render_pass(
                    &wgpu::RenderPassDescriptor {
                        label: Some("Render Pass"),
                        color_attachments: &[
                            Some(
                                wgpu::RenderPassColorAttachment {
                                    view: &view,
                                    depth_slice: None,
                                    resolve_target: None,
                                    ops: wgpu::Operations {
                                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                                        store: wgpu::StoreOp::default()
                                    }
                                }
                            )
                        ],
                        depth_stencil_attachment: Some(
                            wgpu::RenderPassDepthStencilAttachment {
                                view: &depth_texture.view,
                                depth_ops: Some(
                                    wgpu::Operations {
                                        load: wgpu::LoadOp::Clear(1.0),
                                        store: wgpu::StoreOp::Store
                                    }
                                ),
                                stencil_ops: None
                            }
                        ),
                        //timestamp_writes: (),
                        //occlusion_query_set: ()
                        ..Default::default()
                    }
                );
    
                // Configura o pipeline de renderização: Aqui você diz qual pipeline usar.
                render_pass.set_pipeline(&render_pipeline);
    
                //
                //render_pass.set_bind_group(0, &self.diffuse_bind_group, &[]);
                
                //
                //render_pass.set_bind_group(1, &self.camera_bind_group, &[]);
    
                // Define o vertex buffer enviado para GPU -- talvez mudar commentario --
                //render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
    
                /*
                    //
                    render_pass.set_vertex_buffer(1, self.instance_buffer.slice(..));
    
                    // Define o index buffer enviado para GPU -- talvez mudar commentario --
                    render_pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint16);
    
                    // Desenha os vértices: Esse comando dispara o vertex shader e o fragment shader do seu arquivo <archive_name>.wgsl.
                    //render_pass.draw(0..self.num_vertices, 0..1); // sem index
                    //render_pass.draw_indexed(0..self.num_indices, 0, 0..1); // com index
                    render_pass.draw_indexed(0..self.num_indices, 0, 0..self.instances.len() as _); // com index
                */
    
                //
                render_pass.set_vertex_buffer(1, instance_buffer.slice(..));
    
                use crate::render::model::DrawModel;
                //render_pass.draw_mesh_instanced(&self.obj_model.meshes[0], 0..self.instances.len() as u32);
                //let mesh = &self.obj_model.meshes[0];
                //let material = &self.obj_model.materials[mesh.material];
                //render_pass.draw_mesh_instanced(mesh, material, 0..self.instances.len() as u32, &self.camera_bind_group);
                render_pass.draw_model_instanced(&obj_model, 0..instances.len() as u32, &camera_bind_group);
            }
        }

        // Envia os comandos para execução pela GPU.
        _wgpu.queue.submit(
            Some(
                // Fecha o bloco de comandos.
                encoder.finish()
            )
        );

        // Exibe a textura resultante na janela.
        frame.present();
    }

    fn handle_event(&mut self, event: &winit::event::WindowEvent) {
        if let Some(camera_controller) = self.camera_controller.as_mut() {
            camera_controller.process_events(event);
        }

        match event {
            winit::event::WindowEvent::KeyboardInput { .. } => {
                println!("Input na Gameplay");
            }
            _ => {}
        }
    }
}
