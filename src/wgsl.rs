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

fn rgbColor(r: f32, g: f32, b: f32) -> vec4<f32> {
    return vec4<f32>(r, g, b, 1.0);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) GlobalInvocationID : vec3<u32>) {
    let idx = GlobalInvocationID.x;
    let pixel_count = params.size.x * params.size.y;
    if (idx >= pixel_count) {
        return;
    }
"#;

const LOCAL_VAR_PREFIX: &str = "local_";

struct WgslCompiler {
    writer: LineWriter,
    // TODO: Make a proper symbol table to keep track of variables.
    local_var_idx: usize,
}

impl WgslCompiler {
    fn new() -> Self {
        Self {
            writer: LineWriter::new(),
            local_var_idx: 0,
        }
    }

    fn compile_expr(&mut self, expr: &ast::Expression) -> usize {
        let expr_idx = self.local_var_idx;

        match expr {
            ast::Expression::FunctionCall { name, args, .. } => {
                let mut s = format!(
                    "var {}_{} : vec4<f32> = ",
                    LOCAL_VAR_PREFIX, self.local_var_idx
                );
                self.local_var_idx += 1;

                s.push_str(&format!("{}(", name));

                let mut it = args.iter().peekable();
                while let Some(arg) = it.next() {
                    let arg_idx = self.compile_expr(arg);
                    s.push_str(&format!("{}_{}", LOCAL_VAR_PREFIX, arg_idx));
                    if it.peek().is_some() {
                        s.push_str(", ");
                    }
                }
                s.push_str(")");
                s.push_str(";");
                self.writer.line(&s);
            }
            ast::Expression::IntConst { value, .. } => {
                let mut s = format!("var {}_{} : f32 = ", LOCAL_VAR_PREFIX, self.local_var_idx);
                self.local_var_idx += 1;

                s.push_str(&format!("{}", value));
                s.push_str(";");
                self.writer.line(&s);
            }
            _ => {}
        }

        expr_idx
    }

    fn compile_filter(&mut self, filter: &ast::Filter) {
        self.writer.buf.push_str(MODULE_PREAMBLE);
        self.writer.buf.push_str(FILTER_PREAMBLE);

        self.writer.indent();

        let mut last_expr_idx = 0;

        for expr in &filter.exprs {
            last_expr_idx = self.compile_expr(expr);
        }

        // Assign the last expression to the output buffer.
        self.writer.line(&format!(
            "output.pixels[idx] = {}_{};",
            LOCAL_VAR_PREFIX, last_expr_idx,
        ));
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

        if let Ok(out) = out {
            println!("{}", out);
        }

        Ok(())
    }
}
