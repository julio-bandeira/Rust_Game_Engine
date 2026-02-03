use image::GenericImageView;

// Wrapper simples para textura GPU contendo texture, view e sampler.
pub struct Texture {
    pub texture: wgpu::Texture,
    pub view: wgpu::TextureView,
    pub sampler: wgpu::Sampler
}

impl Texture {
    // Formato de profundidade utilizado (Depth32Float).
    pub const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float; // 1.
    
    // Cria a textura de profundidade usada no depth stencil attachment.
    pub fn create_depth_texture(device: &wgpu::Device, config: &wgpu::SurfaceConfiguration, label: &str) -> Self {
        let size = wgpu::Extent3d {
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
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        };
        let texture = device.create_texture(&desc);

        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let sampler = device.create_sampler(
            &wgpu::SamplerDescriptor {
                address_mode_u: wgpu::AddressMode::ClampToEdge,
                address_mode_v: wgpu::AddressMode::ClampToEdge,
                address_mode_w: wgpu::AddressMode::ClampToEdge,
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                mipmap_filter: wgpu::FilterMode::Nearest,
                compare: Some(wgpu::CompareFunction::LessEqual),
                lod_min_clamp: 0.0,
                lod_max_clamp: 100.0,
                ..Default::default()
            }
        );

        Self { texture, view, sampler }
    }

    // Decodifica bytes de imagem e cria uma Texture (chama from_image).
    pub fn from_bytes(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        bytes: &[u8],
        label: &str
    ) -> anyhow::Result<Self> {
        // Decodifica bytes em DynamicImage (pode ser PNG/JPEG/etc.)
        let texture_image = image::load_from_memory(bytes)?;
        
        // Chama a função que cria a textura a partir da DynamicImage
        Self::from_image(device, queue, &texture_image, Some(label))
    }

    // Cria textura GPU a partir de uma DynamicImage: copia pixels, cria view e sampler.
    pub fn from_image(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        texture_image: &image::DynamicImage,
        label: Option<&str>
    ) -> anyhow::Result<Self> {
        // Converte para RGBA8 para garantir layout conhecido.
        let rgba = texture_image.to_rgba8();

        // Obtém dimensões da imagem
        let dimensions = texture_image.dimensions();

        // Define tamanho da textura GPU
        let size = wgpu::Extent3d{
            width: dimensions.0,
            height: dimensions.1,
            depth_or_array_layers: 1
        };

        // Cria a textura com formato sRGB (Rgba8UnormSrgb)
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

        // Copia os pixels para a textura GPU
        queue.write_texture(
            wgpu::TexelCopyTextureInfoBase {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
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

        // Cria view padrão
        let view = texture.create_view(
            &wgpu::TextureViewDescriptor::default()
        );

        // Cria sampler com filtros básicos
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
                    ..Default::default()
                }
            );
        
        // Retorna a textura empacotada
        Ok(
            Self {
                texture: texture,
                view: view,
                sampler: sampler
            }
        )
    }
}