use image::{Rgba, RgbaImage};
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub fn make_image(script: &str) -> Vec<u8> {
    let im_w = 256;
    let im_h = 256;

    let mut buffers = mathmap::exec_mathmap_script(script.to_string(), im_w, im_h, 1).unwrap();

    buffers.next().unwrap().into_raw()
}
