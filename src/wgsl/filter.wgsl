fn rgbColor(r: f32, g: f32, b: f32) -> vec4<f32> {
    return vec4<f32>(r, g, b, 1.0);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) GlobalInvocationID : vec3<u32>) {
    let idx = GlobalInvocationID.x;
    let pixel_count = params.size.x * params.size.y;
    if (idx >= pixel_count) {
        return;
    }
