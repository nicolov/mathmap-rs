mod ast;
mod err;
mod interpreter;
mod lexer;
mod sema;
mod wgsl;

pub use err::MathMapError;
pub use err::SyntaxError;
pub use err::TypeError;

pub fn exec_mathmap_file(
    srcpath: &str,
    im_w: u32,
    im_h: u32,
    num_frames: i64,
) -> Result<
    impl Iterator<Item = Result<image::ImageBuffer<image::Rgba<u8>, Vec<u8>>, MathMapError>>,
    MathMapError,
> {
    let src = std::fs::read_to_string(srcpath).unwrap();
    exec_mathmap_script(src, im_w, im_h, num_frames)
}

pub fn exec_mathmap_script(
    src: String,
    im_w: u32,
    im_h: u32,
    num_frames: i64,
) -> Result<
    impl Iterator<Item = Result<image::ImageBuffer<image::Rgba<u8>, Vec<u8>>, MathMapError>>,
    MathMapError,
> {
    let module = ast::parse_module(&src).map_err(MathMapError::Syntax)?;
    println!("{:#?}", module);

    let mut filters = module.filters;
    let filter = filters.remove(0);

    let render_fn =
        move |t: f32| -> Result<image::ImageBuffer<image::Rgba<u8>, Vec<u8>>, MathMapError> {
            let mut buf = image::ImageBuffer::new(im_w, im_h);

            for (x, y, p) in buf.enumerate_pixels_mut() {
                // Scale x from -1 to 1.
                let cx = (im_w as f32 - 1.0) / 2.0;
                let cy = (im_h as f32 - 1.0) / 2.0;

                let xf = (x as f32 - cx) / cx;
                let yf = (y as f32 - cy) / cy;
                // Flip Y so +y is up (cartesian).
                let yf = -yf;

                match interpreter::eval_filter(&filter, xf, yf, t) {
                    Ok(value) => {
                        if let interpreter::Value::Tuple(_, data) = value {
                            let r = data[0] * 255.0;
                            let g = data[1] * 255.0;
                            let b = data[2] * 255.0;
                            let a = data[3] * 255.0;
                            *p = image::Rgba([r as u8, g as u8, b as u8, a as u8])
                        } else {
                            return Err(err::RuntimeError::with_pos(
                                "filter did not return a tuple",
                                0,
                                0,
                            )
                            .into());
                        }
                    }
                    Err(e) => {
                        return Err(e.into());
                    }
                }
            }

            Ok(buf)
        };

    Ok((0..num_frames).map(move |i| {
        let t = i as f32 / num_frames as f32;
        render_fn(t)
    }))
}

pub fn compile_script_to_wgsl(src: &str) -> Result<String, MathMapError> {
    let module = ast::parse_module(src).map_err(MathMapError::Syntax)?;
    println!("{:#?}", module);

    let mut filters = module.filters;
    let filter = filters.remove(0);

    wgsl::compile_filter(&filter)
}
