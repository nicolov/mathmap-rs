fn rgbColor(r: f32, g: f32, b: f32) -> vec4<f32> {
    return vec4<f32>(r, g, b, 1.0);
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.size.x || gid.y >= params.size.y) { return; }
    let idx: u32 = gid.x + gid.y * params.size.x;
    // output.pixels[idx] = vec4<f32>(1.0, 0.0, 0.0, 1.0);
