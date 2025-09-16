use js_sys::Error as JsError;
use wasm_bindgen::prelude::*;

#[derive(Debug)]
struct JsMathMapError(mathmap::MathMapError);

impl From<JsMathMapError> for JsValue {
    fn from(e: JsMathMapError) -> JsValue {
        let msg = match e.0 {
            mathmap::MathMapError::Syntax(e) => format!("{}", e),
            mathmap::MathMapError::Runtime(e) => format!("{}", e),
        };
        JsValue::from(JsError::new(&msg))
    }
}

#[wasm_bindgen]
pub fn make_image(script: &str) -> Result<Vec<u8>, JsValue> {
    let im_w = 256;
    let im_h = im_w;

    let mut buffers = mathmap::exec_mathmap_script(script.to_string(), im_w, im_h, 1)
        .map_err(|e| JsValue::from(JsMathMapError(e)))?;

    let buffer = buffers
        .next()
        .ok_or_else(|| JsValue::from(js_sys::Error::new("no buffers returned")))?
        .map_err(|e| JsValue::from(JsMathMapError(e)))?
        .into_raw();

    Ok(buffer)
}

#[wasm_bindgen]
pub fn compile_to_wgsl(script: &str) -> Result<String, JsValue> {
    let wgsl = r#"
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

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) GlobalInvocationID : vec3<u32>) {
    let idx = GlobalInvocationID.x;
    let pixel_count = params.size.x * params.size.y;
    if (idx >= pixel_count) {
        return;
    }
    output.pixels[idx] = vec4<f32>(1.0, 0.0, 0.0, 1.0);
}
"#;

    Ok(wgsl.to_string())
}
