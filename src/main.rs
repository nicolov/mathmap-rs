struct CliArgs {
    srcpath: String,
    num_frames: i64,
    wgsl: bool,
}

fn parse_args() -> Result<CliArgs, lexopt::Error> {
    use lexopt::prelude::*;

    let mut parser = lexopt::Parser::from_env();

    let mut srcpath: Option<String> = None;
    let mut num_frames = 1;
    let mut wgsl = false;

    while let Some(arg) = parser.next()? {
        match arg {
            Value(x) => {
                srcpath = Some(x.string()?);
            }
            Long("wgsl") => {
                wgsl = true;
            }
            Long("num-frames") => {
                num_frames = parser.value()?.parse()?;
            }
            Long("help") => {
                std::process::exit(0);
            }
            _ => return Err(arg.unexpected()),
        }
    }

    Ok(CliArgs {
        srcpath: srcpath.ok_or("missing argument SRCPATH")?,
        num_frames,
        wgsl,
    })
}

fn main() -> anyhow::Result<()> {
    let args = parse_args()?;

    if args.wgsl {
        // Compile to wgsl and return.
        let src = std::fs::read_to_string(&args.srcpath)?;
        let out = mathmap::compile_script_to_wgsl(&src)?;
        println!("{}", out);
        return Ok(());
    }

    let im_w = 256;
    let im_h = 256;

    let mut buffers = mathmap::exec_mathmap_file(&args.srcpath, im_w, im_h, args.num_frames)?;

    if args.num_frames == 1 {
        let im = buffers
            .next()
            .ok_or_else(|| anyhow::anyhow!("no buffers returned"))??;
        im.save("out/out.png")?;
    } else {
        let gif_file = std::fs::File::create("out/out.gif").unwrap();
        let mut gif_encoder = image::codecs::gif::GifEncoder::new(gif_file);
        gif_encoder
            .set_repeat(image::codecs::gif::Repeat::Infinite)
            .unwrap();

        for im in buffers {
            let im = im?;
            let frame =
                image::Frame::from_parts(im, 0, 0, image::Delay::from_numer_denom_ms(50, 1));
            gif_encoder.encode_frame(frame).unwrap();
        }
    }

    Ok(())
}
