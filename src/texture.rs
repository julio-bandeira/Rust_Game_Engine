use image::GenericImageView;

// Classe para criar texturas
//----------------------------------------------------------------------------------
pub struct Texture {
    pub texture: wgpu::Texture,
    pub view: wgpu::TextureView,
    pub sampler: wgpu::Sampler
}

impl Texture {
    //
    pub const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float; // 1.
    
    //
    pub fn create_depth_texture(device: &wgpu::Device, config: &wgpu::SurfaceConfiguration, label: &str) -> Self {
        let size = wgpu::Extent3d { // 2.
            width: config.width.max(1),
            height: config.height.max(1),
            depth_or_array_layers: 1,
        };
        let desc = wgpu::TextureDescriptor {
            label: Some(label),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: Self::DEPTH_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT // 3.
                | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        };
        let texture = device.create_texture(&desc);

        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let sampler = device.create_sampler(
            &wgpu::SamplerDescriptor { // 4.
                address_mode_u: wgpu::AddressMode::ClampToEdge,
                address_mode_v: wgpu::AddressMode::ClampToEdge,
                address_mode_w: wgpu::AddressMode::ClampToEdge,
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                mipmap_filter: wgpu::FilterMode::Nearest,
                compare: Some(wgpu::CompareFunction::LessEqual), // 5.
                lod_min_clamp: 0.0,
                lod_max_clamp: 100.0,
                ..Default::default()
            }
        );

        Self { texture, view, sampler }
    }

    // Decodifica bytes de imagem na memória e cria uma textura na GPU chamando from_image
    pub fn from_bytes(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        bytes: &[u8],
        label: &str
    ) -> anyhow::Result<Self> {
        // Tenta decodificar os bytes em memória (bytes) como uma imagem (PNG, JPEG etc.) 
        let texture_image = image::load_from_memory(bytes)?;
        
        // Chama a função from_image
        Self::from_image(device, queue, &texture_image, Some(label))
    }

    // Cria uma textura na GPU a partir de uma imagem da CPU, configurando seus pixels, view e sampler
    fn from_image(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        texture_image: &image::DynamicImage,
        label: Option<&str>
    ) -> anyhow::Result<Self> {
        // Converte a imagem (texture_image) para o formato de pixels RGBA com 8 bits por canal
        let rgba = texture_image.to_rgba8();

        // Pega a largura e a altura da imagem (texture_image) e retorna como uma tupla (u32, u32).
        let dimensions = texture_image.dimensions();

        // Cria um objeto Extent3d do wgpu que define as dimensões da textura a partir do tamanho da imagem.
        let size = wgpu::Extent3d{
            width: dimensions.0, // Largura
            height: dimensions.1, // Altura
            depth_or_array_layers: 1 // Profundidade / Camadas
        };

        // Cria uma textura 2D na GPU (com formato RGBA8 e suporte para ser usada em shaders e receber dados copiados da CPU).
        let texture = device.create_texture(
            &wgpu::TextureDescriptor {
                label: label,
                size: size,
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8UnormSrgb,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[]
            }
        );

        // Copia os pixels da imagem RGBA da CPU (rgba) para a textura criada na GPU (texture).
        queue.write_texture(
            wgpu::TexelCopyTextureInfoBase {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO, // Preenchendo-a a partir do nível 0 e origem (0,0).
                aspect: wgpu::TextureAspect::All
            },
            &rgba,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(4 * dimensions.0),
                rows_per_image: Some(dimensions.1)
            },
            size
        );

        // Cria uma "view" da textura na GPU, que é usada para acessar ou renderizar a textura em shaders.
        let view = texture.create_view(
            &wgpu::TextureViewDescriptor::default()
        );

        // Cria um sampler na GPU, que define como a textura será amostrada nos shaders (como filtrar, interpolar e lidar com coordenadas fora do limite).
        let sampler = device
            .create_sampler(
                &wgpu::SamplerDescriptor {
                    label: label,
                    address_mode_u: wgpu::AddressMode::ClampToEdge,
                    address_mode_v: wgpu::AddressMode::ClampToEdge,
                    address_mode_w: wgpu::AddressMode::ClampToEdge,
                    mag_filter: wgpu::FilterMode::Linear,
                    min_filter: wgpu::FilterMode::Nearest,
                    mipmap_filter: wgpu::FilterMode::Nearest,
                    //lod_min_clamp: (),
                    //lod_max_clamp: (),
                    //compare: (),
                    //anisotropy_clamp: (),
                    //border_color: (),
                    ..Default::default()
                }
            );
        
        // Retorna com sucesso (Ok) uma instância da struct contendo a textura, a view e o sampler criados
        Ok(
            Self {
                texture: texture,
                view: view,
                sampler: sampler
            }
        )
    }
}
//----------------------------------------------------------------------------------