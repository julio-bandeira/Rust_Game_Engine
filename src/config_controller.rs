#[derive(Debug, Clone)]
pub struct ConfigController {
    window_mode: WindowMode,
    resolution: Resolution,
    vsync: VSyncMode,
    frame_limit: FrameLimit,
    ui_scale: UiScaleMode,
    anti_aliasing: AntiAliasing,
    render_scale: RenderScale,

    color_settings: ColorSettings,
    color_format: ColorFormat,
    display_target: DisplayTarget,
    aspect_ratio: AspectRatio
}

impl ConfigController {
    pub fn new() -> ConfigController {
        ConfigController {
            window_mode: WindowMode::BorderlessFullscreen,
            resolution: Resolution { width: 2560, height: 1440 },
            vsync: VSyncMode::On,
            frame_limit: FrameLimit::Unlimited,
            ui_scale: UiScaleMode::Manual(1.0),
            anti_aliasing: AntiAliasing::Off,
            render_scale: RenderScale { scale: 1.0 },

            color_settings: ColorSettings { gamma: 2.2, brightness: 1.0, contrast: 1.0 },
            color_format: ColorFormat::SRgb,
            display_target: DisplayTarget { monitor_index: 0 },
            aspect_ratio: AspectRatio::Auto
        }
    }

    pub fn apply_window(
        &self,
        window: &winit::window::Window,
    ) {
        match self.window_mode {
            WindowMode::Windowed => {
                window.set_fullscreen(None);
                window.request_inner_size(
                    winit::dpi::PhysicalSize::new(
                        self.resolution.width,
                        self.resolution.height,
                    )
                ).unwrap();
            }
            WindowMode::BorderlessFullscreen => {
                window.set_fullscreen(Some(
                    winit::window::Fullscreen::Borderless(None)
                ));
            }
            WindowMode::ExclusiveFullscreen => {
                // opcional / avançado
            }
        }
    }

    pub fn apply_gpu(
        &self,
        gpu: &mut crate::wgpu_context::WgpuContext,
    ) {
        gpu.surface_configuration.width = self.resolution.width.max(1);
        gpu.surface_configuration.height = self.resolution.height.max(1);

        gpu.surface_configuration.present_mode =
            self.vsync.present_mode(&gpu.surface, &gpu.adapter);

        gpu.surface.configure(
            &gpu.device,
            &gpu.surface_configuration,
        );
    }
}

// Modo de janela
#[derive(Debug, Clone)]
pub enum WindowMode {
    Windowed,
    BorderlessFullscreen,
    ExclusiveFullscreen,
}

// Resolução
#[derive(Debug, Clone, Copy)]
pub struct Resolution {
    width: u32,
    height: u32,
}

// Taxa de atualização
#[derive(Debug, Clone)]
pub enum FrameLimit {
    Unlimited,
    Fps(u32), // 30, 60, 120, 144...
}

// Sincronização vertical
#[derive(Debug, Clone)]
pub enum VSyncMode {
    On,
    Off,
    Adaptive
}

impl VSyncMode {
    pub fn present_mode(
        &self,
        surface: &wgpu::Surface,
        adapter: &wgpu::Adapter
    ) -> wgpu::PresentMode {
        let capabilities = surface.get_capabilities(adapter);

        match self {
            VSyncMode::On => {
                wgpu::PresentMode::Fifo
            },
            VSyncMode::Off => {
                if capabilities.present_modes.contains(&wgpu::PresentMode::Immediate) {
                    wgpu::PresentMode::Immediate
                }else {
                    wgpu::PresentMode::Fifo
                }
            },
            VSyncMode::Adaptive => {
                if capabilities.present_modes.contains(&wgpu::PresentMode::Mailbox) {
                    wgpu::PresentMode::Mailbox
                }else {
                    wgpu::PresentMode::Fifo
                }
            }
        }
    }
}

// Escala de UI / DPI
#[derive(Debug, Clone)]
pub enum UiScaleMode {
    Auto,
    Manual(f32),
}

// Espaço de cor / Gamma
#[derive(Debug, Clone)]
pub struct ColorSettings {
    gamma: f32,       // 2.2 padrão
    brightness: f32,  // 1.0 padrão
    contrast: f32,    // 1.0 padrão
}

// Formato de pixel (interno)
//Normalmente invisível pro jogador, mas importante na engine.
#[derive(Debug, Clone)]
pub enum ColorFormat {
    SRgb,
    Linear,
}

// Anti-Aliasing (MSAA)
#[derive(Debug, Clone)]
pub enum AntiAliasing {
    Off,
    MSAAx2,
    MSAAx4,
    MSAAx8,
}

impl AntiAliasing {
    pub fn sample_count(&self) -> u32 {
        match self {
            AntiAliasing::Off => 1,
            AntiAliasing::MSAAx2 => 2,
            AntiAliasing::MSAAx4 => 4,
            AntiAliasing::MSAAx8 => 8,
        }
    }
}

// Escala de render (Render Scale)
// Muito usado hoje (DLSS / FSR / resolução dinâmica).
#[derive(Debug, Clone, Copy)]
pub struct RenderScale {
    scale: f32, // 0.5 → 50%, 1.0 → 100%
}

// Monitor / Display alvo
#[derive(Debug, Clone)]
pub struct DisplayTarget {
    monitor_index: usize,
}

// Aspect Ratio
#[derive(Debug, Clone)]
pub enum AspectRatio {
    Auto,
    Ratio16x9,
    Ratio21x9,
    Ratio4x3,
}