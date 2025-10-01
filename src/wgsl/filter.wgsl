fn FN_rgbColor(r: f32, g: f32, b: f32) -> vec4<f32> {
    return vec4<f32>(r, g, b, 1.0);
}

fn FN_grayColor(x: f32) -> vec4<f32> {
    return vec4<f32>(x, x, x, 1.0);
}

fn FN___add(x: f32, y: f32) -> f32 {
	return x + y;
}

fn FN___div(x: f32, y: f32) -> f32 {
	return x / y;
}

fn FN___mul(x: f32, y: f32) -> f32 {
	return x * y;
}

fn FN___sub(x: f32, y: f32) -> f32 {
	return x - y;
}

fn FN_sin(x: f32) -> f32 {
	return sin(x);
}

fn to_ra(x: f32, y: f32) -> vec2<f32> {
    let r = sqrt(x * x + y * y);
    var a = atan2(y, x);
    if (a < 0.0) {
        // Shift from [-pi, pi) to [0, 2*pi].
        a = a + 2.0 * 3.141592653589793;
    }
    return vec2<f32>(r, a);
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.size.x || gid.y >= params.size.y) { return; }
    let idx: u32 = gid.x + gid.y * params.size.x;

	let cx: f32 = (f32(params.size.x) - 1.0) / 2.0;
	let cy: f32 = (f32(params.size.y) - 1.0) / 2.0;

	let x: f32 = (f32(gid.x) - cx) / cx;
	let y: f32 = (f32(gid.y) - cy) / cy;

	let ra = to_ra(x, y);
	let r = ra[0];
	let a = ra[1];

	let pi = 3.141592653589793;

	let t = 0.0;
