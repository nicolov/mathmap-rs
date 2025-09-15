mod ast;
mod interpreter;
mod lexer;

pub use ast::ParseError;

#[derive(Debug)]
pub enum MathMapError {
    Parse(ParseError),
}

pub fn exec_mathmap_file(
    srcpath: &str,
    im_w: u32,
    im_h: u32,
    num_frames: i64,
) -> Result<impl Iterator<Item = image::ImageBuffer<image::Rgba<u8>, Vec<u8>>>, MathMapError> {
    let src = std::fs::read_to_string(srcpath).unwrap();
    exec_mathmap_script(src, im_w, im_h, num_frames)
}

pub fn exec_mathmap_script(
    src: String,
    im_w: u32,
    im_h: u32,
    num_frames: i64,
) -> Result<impl Iterator<Item = image::ImageBuffer<image::Rgba<u8>, Vec<u8>>>, MathMapError> {
    let module = ast::parse_module(&src).map_err(MathMapError::Parse)?;
    println!("{:#?}", module);

    let mut filters = module.filters;
    let filter = filters.remove(0);

    let render_fn = move |t: f32| {
        image::ImageBuffer::<image::Rgba<u8>, _>::from_fn(im_w, im_h, |x, y| {
            // Scale x from -1 to 1.
            let cx = (im_w as f32 - 1.0) / 2.0;
            let cy = (im_h as f32 - 1.0) / 2.0;

            let xf = (x as f32 - cx) / cx;
            let yf = (y as f32 - cy) / cy;
            // Flip Y so +y is up (cartesian).
            let yf = -yf;

            let value = interpreter::eval_filter(&filter, xf, yf, t);

            if let interpreter::Value::Tuple(_, data) = value {
                let r = data[0] * 255.0;
                let g = data[1] * 255.0;
                let b = data[2] * 255.0;
                let a = data[3] * 255.0;
                image::Rgba([r as u8, g as u8, b as u8, a as u8])
            } else {
                panic!("not a tuple");
            }
        })
    };

    Ok((0..num_frames).map(move |i| {
        let t = i as f32 / num_frames as f32;
        render_fn(t)
    }))
}
