// Classe de contexto do Wgpu
//----------------------------------------------------------------------------------
use wgpu::util::DeviceExt;
use crate::model::Model;
use crate::model::Vertex;
//use crate::vertex::Vertex;
use crate::texture::Texture;
use crate::camera::Camera;
use crate::camera_uniform::CameraUniform;
use crate::camera_controller::CameraController;
use crate::instance::Instance;
use crate::instance::InstanceRaw;
use cgmath::prelude::*;
use crate::model::ModelVertex;

// Define um array constante de vértices (Vertex) com posições 3D e coordenadas de textura
//const VERTICES: &[Vertex] = &[
//    Vertex { position: [ -0.5, 0.5, 0.0], tex_coords: [0.0, 0.0] },//color: [ 1.0, 0.0, 0.0] }, //A
//    Vertex { position: [ -0.5, -0.5, 0.0], tex_coords: [0.0, 1.0] },//color: [ 0.0, 1.0, 0.0] }, //B
//    Vertex { position: [ 0.5, -0.5, 0.0], tex_coords: [1.0, 1.0] },//color: [ 0.0, 0.0, 1.0] }, //C
//    Vertex { position: [ 0.5, 0.5, 0.0], tex_coords: [1.0, 0.0] }//color: [ 0.0, 1.0, 0.0] } //D
//];

// Define a ordem de desenho dos vértices (Nesse Caso forma um quadrado)
//const INDICES: &[u16] = &[
//    0, 1, 2,
//    0, 2, 3,
//];

//const NUM_INSTANCE_PER_RAW: u32 = 10;
//const INSTANCE_DISPLACEMENT: cgmath::Vector3<f32> = cgmath::Vector3::new(
//    NUM_INSTANCE_PER_RAW as f32 * 0.5,
//    0.0,
//    NUM_INSTANCE_PER_RAW as f32 * 0.5
//);

pub struct WgpuContext<'window_lifetime> {
    pub surface: wgpu::Surface<'window_lifetime>,
    pub adapter: wgpu::Adapter,
    pub device: wgpu::Device,
    //pub vertex_buffer: wgpu::Buffer,
    //pub num_vertices: u32,
    //pub index_buffer: wgpu::Buffer,
    //pub num_indices: u32,
    pub queue: wgpu::Queue,
    //pub diffuse_bind_group: wgpu::BindGroup,
    //pub diffuse_texture: Texture,
    pub surface_configuration: wgpu::SurfaceConfiguration,
    pub camera: Camera,
    pub camera_uniform: CameraUniform,
    pub camera_buffer: wgpu::Buffer,
    pub camera_bind_group: wgpu::BindGroup,
    pub camera_controller: CameraController,
    pub render_pipeline: wgpu::RenderPipeline,
    pub obj_model: Model,
    pub instances: Vec<Instance>,
    pub instance_buffer: wgpu::Buffer,
    pub depth_texture: Texture
}

impl<'window_lifetime> WgpuContext<'window_lifetime> {
    // Chama a função new_async de forma assincrona
    pub fn new(window: std::sync::Arc<winit::window::Window>) -> WgpuContext<'window_lifetime> {
        pollster::block_on(WgpuContext::new_async(window))
    }

    // Inicializa o contexto do Wgpu
    async fn new_async(window: std::sync::Arc<winit::window::Window>) -> WgpuContext<'window_lifetime> {
        //  Representa a conexão com a API gráfica (Vulkan, DirectX, Metal ou WebGPU no navegador).
        let instance = wgpu::Instance::default();

        // Cria a superficie que será desenhada
        let surface = instance
            .create_surface(std::sync::Arc::clone(&window))
            .unwrap();

        // Requisita o adaptador que é basicamente uma placa de vídeo ou backend gráfico disponível no sistema  
        let adapter = instance
            .request_adapter(
                &wgpu::RequestAdapterOptionsBase {
                    power_preference: wgpu::PowerPreference::HighPerformance,
                    force_fallback_adapter: false,
                    compatible_surface: Some(&surface)
                }
            )
            .await
            .expect("Failled to find apropriate adapter");

        // São a interface que você vai realmente usar no seu código.
        let (
            // É a "conexão lógica" com a GPU, usada para criar recursos (buffers, texturas, pipelines...).
            device,
            // É a fila de comandos para enviar trabalho à GPU (ex.: desenhar, copiar memória).
            queue
        ) = adapter
            .request_device(
                &wgpu::wgt::DeviceDescriptor {
                    label: None,
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::defaults(),
                    memory_hints: wgpu::MemoryHints::Performance,
                    trace: wgpu::Trace::Off
                }
            )
            .await
            .expect("Failled to create device");

        // Cria um vertex buffer (um pedaço de memória na GPU que armazena os vértices que você vai desenhar)
        //let vertex_buffer = device
        //    .create_buffer_init(
        //        &wgpu::util::BufferInitDescriptor {
        //            label: Some("Vertex Buffer"),
        //            contents: bytemuck::cast_slice(VERTICES),
        //            usage: wgpu::BufferUsages::VERTEX
        //        }
        //    );
        
        // Informa o tamanho da lista de vertices
        //let num_vertices = VERTICES.len() as u32;
        
        // Cria um index buffer (um pedaço de memória na GPU que armazena os indices dos vértices que você vai desenhar)
        //let index_buffer = device
        //    .create_buffer_init(
        //        &wgpu::util::BufferInitDescriptor {
        //            label: Some("Indices Buffer"),
        //            contents: bytemuck::cast_slice(INDICES),
        //            usage: wgpu::BufferUsages::INDEX
        //        }
        //    );
        
        // Informa o tamanho da lista de indices
        //let num_indices = INDICES.len() as u32;

        // Inclui o arquivo como um array de bytes no binário em tempo de compilação, permitindo acessá-lo diretamente na memória.
        //let diffuse_bytes = include_bytes!("assets/textures/tree.png");

        // Cria uma textura na GPU a partir dos bytes da imagem (diffuse_bytes)
        //let diffuse_texture = Texture::from_bytes(&device, &queue, diffuse_bytes, "assets/textures/tree.png").unwrap();

        // Cria um layout de bind group na GPU, definindo como uma textura 2D e seu sampler serão acessados pelos shaders de fragmento.
        let texture_bind_group_layout = device
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
        
        // Cria um bind group que associa a textura e o sampler específicos ao layout definido, permitindo que os shaders usem esses recursos na renderização.
        //let diffuse_bind_group = device.create_bind_group(
        //    &wgpu::BindGroupDescriptor {
        //        label: Some("Diffuse Bind Group"),
        //        layout: &texture_bind_group_layout,
        //        entries: &[
        //            wgpu::BindGroupEntry{
        //                binding: 0,
        //                resource: wgpu::BindingResource::TextureView(&diffuse_texture.view)
        //            },
        //            wgpu::BindGroupEntry{
        //                binding: 1,
        //                resource: wgpu::BindingResource::Sampler(&diffuse_texture.sampler)
        //            }
        //        ]
        //    }
        //);

        // Coleta o tamanho atual da janela
        let window_size = window.inner_size();

        // Define os parâmetros da superficie
        let surface_configuration = surface
            .get_default_config(
                &adapter,
                window_size.width.max(1),
                window_size.height.max(1)
            )
            .unwrap();

        // Aplica a configuração de superfice definida
        surface.configure(&device, &surface_configuration);

        // Carrega o código WGSL (<archive_name>.wgsl) e compila para rodar na GPU.
        let shader = device
            .create_shader_module(
                wgpu::ShaderModuleDescriptor {
                    label: Some("Shader"),
                    source: wgpu::ShaderSource::Wgsl(include_str!("assets/shaders/shader.wgsl").into())
                }
            );

        //
        let camera = Camera {
            eye: (0.0, 1.0, 2.0).into(),
            target: (0.0, 0.0, 0.0).into(),
            up: cgmath::Vector3::unit_y(),
            aspect: surface_configuration.width as f32 / surface_configuration.height as f32,
            fovy: 45.0,
            znear: 0.1,
            zfar: 100.0
        };

        //
        let mut camera_uniform = CameraUniform::new();

        //
        camera_uniform.update_view_proj(&camera);

        //
        let camera_buffer = device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor{
                label: Some("Camera Buffer"),
                contents: bytemuck::cast_slice(&[camera_uniform]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST
            }
        );

        //
        let camera_bind_group_layout = device
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
        let camera_bind_group = device
            .create_bind_group(
                &wgpu::BindGroupDescriptor {
                    label: Some("Camera Bind Group"),
                    layout: &camera_bind_group_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: camera_buffer.as_entire_binding()
                        }
                    ]
                }
            );
        
        //
        let camera_controller = CameraController::new(0.2);

        //
        let depth_texture = Texture::create_depth_texture(&device, &surface_configuration, "depth_texture");
        
        /*
            //
            let instances = (0..NUM_INSTANCE_PER_RAW).flat_map(
                |z| {
                    (0..NUM_INSTANCE_PER_RAW).map(
                        move |x| {
                            let position = cgmath::Vector3 {
                                x: x as f32,
                                y: 0.0,
                                z: z as f32
                            } - INSTANCE_DISPLACEMENT;

                            let rotation = if position.is_zero() {
                                cgmath::Quaternion::from_axis_angle(cgmath::Vector3::unit_z(), cgmath::Deg(0.0))
                            }else {
                                cgmath::Quaternion::from_axis_angle(position.normalize(), cgmath::Deg(45.0))
                            };

                            Instance {
                                position,
                                rotation
                            }
                        }
                    )
                }
            ).collect::<Vec<_>>();

            //
            let instance_data = instances.iter().map(Instance::to_raw).collect::<Vec<_>>();

            //
            let instance_buffer = device.create_buffer_init(
                &wgpu::util::BufferInitDescriptor {
                    label: Some("Instance Buffer"),
                    contents: bytemuck::cast_slice(&instance_data),
                    usage: wgpu::BufferUsages::VERTEX
                }
            );
        */

        //
        const NUM_INSTANCES_PER_ROW: u32 = 10;
        //const INSTANCE_DISPLACEMENT: cgmath::Vector3<f32> = cgmath::Vector3::new(
        //    NUM_INSTANCES_PER_ROW as f32 * 0.5,
        //    0.0,
        //    NUM_INSTANCES_PER_ROW as f32 * 0.5
        //);

        //
        const SPACE_BETWEEN: f32 = 3.0;
        let instances = (0..NUM_INSTANCES_PER_ROW).flat_map(|z| {
            (0..NUM_INSTANCES_PER_ROW).map(move |x| {
                let x = SPACE_BETWEEN * (x as f32 - NUM_INSTANCES_PER_ROW as f32 / 2.0);
                let z = SPACE_BETWEEN * (z as f32 - NUM_INSTANCES_PER_ROW as f32 / 2.0);

                let position = cgmath::Vector3 { x, y: 0.0, z };

                let rotation = if position.is_zero() {
                    cgmath::Quaternion::from_axis_angle(cgmath::Vector3::unit_z(), cgmath::Deg(0.0))
                } else {
                    cgmath::Quaternion::from_axis_angle(position.normalize(), cgmath::Deg(45.0))
                };

                Instance {
                    position, rotation,
                }
            })
        }).collect::<Vec<_>>();

        //
        let instance_data = instances.iter().map(Instance::to_raw).collect::<Vec<_>>();

        //
        let instance_buffer = device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("Instance Buffer"),
                contents: bytemuck::cast_slice(&instance_data),
                usage: wgpu::BufferUsages::VERTEX
            }
        );

        //
        let obj_model = crate::resource::load_model("cube.obj", &device, &queue, &texture_bind_group_layout)
            .await
            .unwrap();

        // Define como os shaders vão receber recursos externos (texturas, buffers uniformes, samplers, etc.).
        let render_pipeline_layout = device
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
        let render_pipeline = device
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
                            ModelVertex::desc(), // Informa o buffer para o wgsl
                            InstanceRaw::desc() // Informa o buffer para o wgsl
                        ]
                    },
                    primitive: wgpu::PrimitiveState::default(), // Triangle List
                    depth_stencil: Some(
                        wgpu::DepthStencilState {
                            format: Texture::DEPTH_FORMAT,
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
                                        format: surface_configuration.format,
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
            );

        WgpuContext {
            surface,
            adapter,
            device,
            //vertex_buffer,
            //num_vertices,
            //index_buffer,
            //num_indices,
            queue,
            //diffuse_bind_group,
            //diffuse_texture,
            surface_configuration,
            camera,
            camera_uniform,
            camera_buffer,
            camera_bind_group,
            camera_controller,
            render_pipeline,
            obj_model,
            instances,
            instance_buffer,
            depth_texture
        }
    }

    // Redimensiona a resolução da janela
    pub fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        self.surface_configuration.width = new_size.width.max(1);
        self.surface_configuration.height = new_size.height.max(1);
        self.depth_texture = Texture::create_depth_texture(&self.device, &self.surface_configuration, "depth_texture");
        self.surface.configure(&self.device, &self.surface_configuration);
    }

    //
    pub fn input(&mut self, event: &winit::event::WindowEvent) -> bool {
        //
        self.camera_controller.process_events(event)
    }

    //
    pub fn update(&mut self) {
        //
        self.camera_controller.update_camera(&mut self.camera);
        //
        self.camera_uniform.update_view_proj(&self.camera);
        //
        self.queue
            .write_buffer(
                &self.camera_buffer,
                0,
                bytemuck::cast_slice(&[self.camera_uniform])
            );
    }

    // Renderiza a imagem na janela
    pub fn draw(&mut self) {
        // Pega a textura da superfície (a "tela" onde vai desenhar)
        let frame = self
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
        let mut encoder = self
            .device
            .create_command_encoder(
                &wgpu::wgt::CommandEncoderDescriptor {
                    label: Some("Render Encoder")
                }
            );
        
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
                            view: &self.depth_texture.view,
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
            render_pass.set_pipeline(&self.render_pipeline);

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
            render_pass.set_vertex_buffer(1, self.instance_buffer.slice(..));

            use crate::model::DrawModel;
            //render_pass.draw_mesh_instanced(&self.obj_model.meshes[0], 0..self.instances.len() as u32);
            //let mesh = &self.obj_model.meshes[0];
            //let material = &self.obj_model.materials[mesh.material];
            //render_pass.draw_mesh_instanced(mesh, material, 0..self.instances.len() as u32, &self.camera_bind_group);
            render_pass.draw_model_instanced(&self.obj_model, 0..self.instances.len() as u32, &self.camera_bind_group);
        }

        // Envia os comandos para execução pela GPU.
        self.queue.submit(
            Some(
                // Fecha o bloco de comandos.
                encoder.finish()
            )
        );

        // Exibe a textura resultante na janela.
        frame.present();
    }
}
//----------------------------------------------------------------------------------