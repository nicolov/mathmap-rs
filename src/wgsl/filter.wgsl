fn FN_rgbColor(r: f32, g: f32, b: f32) -> vec4<f32> {
    return vec4<f32>(r, g, b, 1.0);
}

fn FN_grayColor(x: f32) -> vec4<f32> {
    return vec4<f32>(x, x, x, 1.0);
}

fn FN_abs_quat(x: vec4<f32>) -> f32 {
    return x[0]*x[0] + x[1]*x[1] + x[2]*x[2] + x[3]*x[3];
}

fn FN___mod(x: f32, y: f32) -> f32 {
    return x % y;
}

fn FN___neg(x: f32) -> f32 {
    return -x;
}

fn FN___or(x: i32, y: i32) -> i32 {
    if (x != 0) || (y != 0) {
        return 1;
    } else {
        return 0;
    }
}

fn FN___and(x: i32, y: i32) -> i32 {
    if (x != 0) && (y != 0) {
        return 1;
    } else {
        return 0;
    }
}

fn FN___less(x: f32, y: f32) -> i32 {
    if (x < y) {
        return 1;
    } else {
        return 0;
    }
}

fn FN___lessequal(x: f32, y: f32) -> i32 {
    if (x <= y) {
        return 1;
    } else {
        return 0;
    }
}

fn FN___pow(x: f32, y: f32) -> f32 {
    return pow(x, y);
}

fn FN_mul_quat_quat(x: vec4<f32>, y: vec4<f32>) -> vec4<f32> {
    return vec4<f32>(
        x[0] * y[0] - x[1] * y[1] - x[2] * y[2] - x[3] * y[3],
        x[0] * y[1] + x[1] * y[0] + x[2] * y[3] - x[3] * y[2],
        x[0] * y[2] - x[1] * y[3] + x[2] * y[0] + x[3] * y[1],
        x[0] * y[3] + x[1] * y[2] - x[2] * y[1] + x[3] * y[0]
    );
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
    let y2: f32 = (f32(gid.y) - cy) / cy;
    // Flip Y so +y is up (cartesian).
    let y = -y2;

    let xy = vec2<f32>(x, y);

    let ra = to_ra(x, y);
    let r = ra[0];
    let a = ra[1];

    let pi = 3.141592653589793;

    let t = 0.0;
