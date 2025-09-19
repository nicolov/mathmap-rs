struct OutputBuffer {
    pixels: array<vec4<f32>>,
};

struct Params {
    size: vec2<u32>,
};

@group(0) @binding(0)
var<storage, read_write> output: OutputBuffer;

@group(0) @binding(1)
var<uniform> params: Params;
