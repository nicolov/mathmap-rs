// A transpiler to WebGPU Shading Language (WGSL).

#![allow(dead_code)]

use crate::{MathMapError, ast};

struct LineWriter {
    buf: String,
    indent_level: usize,
}

impl LineWriter {
    fn new() -> Self {
        Self {
            buf: String::new(),
            indent_level: 0,
        }
    }

    fn line(&mut self, text: &str) {
        for _ in 0..self.indent_level {
            self.buf.push_str("  ");
        }
        self.buf.push_str(&text);
        self.buf.push('\n');
    }

    fn indent(&mut self) {
        self.indent_level += 1;
    }

    fn dedent(&mut self) {
        self.indent_level -= 1;
    }

    fn finish(self) -> String {
        self.buf
    }
}

const MODULE_PREAMBLE: &str = r#"
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
"#;

const FILTER_PREAMBLE: &str = r#"
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) GlobalInvocationID : vec3<u32>) {
    let idx = GlobalInvocationID.x;
    let pixel_count = params.size.x * params.size.y;
    if (idx >= pixel_count) {
        return;
    }
"#;

struct WgslCompiler {
    writer: LineWriter,
}

impl WgslCompiler {
    fn new() -> Self {
        Self {
            writer: LineWriter::new(),
        }
    }

    fn compile_filter(&mut self, filter: &ast::Filter) {
        self.writer.buf.push_str(MODULE_PREAMBLE);
        self.writer.buf.push_str(FILTER_PREAMBLE);

        self.writer.indent();

        self.writer
            .line("output.pixels[idx] = vec4<f32>(0.0, 1.0, 0.0, 1.0);");
        self.writer.dedent();
        self.writer.line("}");
    }
}

pub fn compile_filter(filter: &ast::Filter) -> Result<String, MathMapError> {
    let mut compiler = WgslCompiler::new();
    compiler.compile_filter(filter);
    Ok(compiler.writer.finish())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn filter() -> Result<(), Box<dyn std::error::Error>> {
        let input = "filter red ()
            rgbColor(1, 0, 0)
        end";
        let ast = ast::parse_module(input)?;
        let filter = &ast.filters[0];
        let out = compile_filter(filter);
        println!("{:?}", out);
        Ok(())
    }
}
