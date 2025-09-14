use wasm_bindgen::prelude::*;
use image::{RgbaImage, Rgba};

#[wasm_bindgen]
pub fn make_image(width: u32, height: u32) -> Vec<u8> {
    let mut img = RgbaImage::new(width, height);

    for y in 0..height {
        for x in 0..width {
            let r = (x * 255 / width) as u8;
            let g = (y * 255 / height) as u8;
            img.put_pixel(x, y, Rgba([r, g, 128, 255]));
        }
    }

    img.into_raw()
}
