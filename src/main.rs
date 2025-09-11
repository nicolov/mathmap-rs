mod ast;
mod interpreter;
mod lexer;

struct CliArgs {
    srcpath: String,
    num_frames: i64,
}

fn parse_args() -> Result<CliArgs, lexopt::Error> {
    use lexopt::prelude::*;

    let mut parser = lexopt::Parser::from_env();

    let mut srcpath: Option<String> = None;
    let mut num_frames = 1;

    while let Some(arg) = parser.next()? {
        match arg {
            Value(x) => {
                srcpath = Some(x.string()?);
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
    })
}

fn main() -> Result<(), lexopt::Error> {
    let args = parse_args()?;

    let im_w = 256;
    let im_h = 256;

    let mut buffers = mathmap::exec_mathmap_script(&args.srcpath, im_w, im_h, args.num_frames)?;

    if args.num_frames == 1 {
        buffers.next().unwrap().save("out/out.png").unwrap();
    } else {
        let gif_file = std::fs::File::create("out/out.gif").unwrap();
        let mut gif_encoder = image::codecs::gif::GifEncoder::new(gif_file);
        gif_encoder
            .set_repeat(image::codecs::gif::Repeat::Infinite)
            .unwrap();

        for im in buffers {
            let frame =
                image::Frame::from_parts(im, 0, 0, image::Delay::from_numer_denom_ms(50, 1));
            gif_encoder.encode_frame(frame).unwrap();
        }
    }

    Ok(())
}
