//
struct CameraUniform {
    view_proj: mat4x4<f32>
};

//
@group(1) @binding(0)
var<uniform> camera: CameraUniform;

//
struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) tex_coords: vec2<f32>
    //@location(1) color: vec3<f32>
};

struct VertexOutput {
    @builtin(position) vertex_position: vec4<f32>,
    @location(1) tex_coords: vec2<f32>
    //@location(1) color: vec4<f32>
};

@vertex
//fn vs_main(@builtin(vertex_index) in_vertex_index: u32) -> VertexOutput {
fn vs_main(model: VertexInput) -> VertexOutput {
    var output: VertexOutput;
    output.tex_coords = model.tex_coords;
    //output.color = vec4<f32>(model.color, 1.0);
    //output.vertex_position = vec4<f32>(model.position, 1.0);
    output.vertex_position = camera.view_proj * vec4<f32>(model.position, 1.0);
    return output;
}

// Fragment Shader
@group(0) @binding(0)
var t_diffuse: texture_2d<f32>;
@group(0) @binding(1)
var s_diffuse: sampler;

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return textureSample(t_diffuse, s_diffuse, in.tex_coords);
    //return in.color;
}