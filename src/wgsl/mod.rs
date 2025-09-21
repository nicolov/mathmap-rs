// A transpiler to WebGPU Shading Language (WGSL).

#![allow(dead_code)]

use crate::{MathMapError, TypeError, ast, sema};

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
            self.buf.push_str("    ");
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

const MODULE_PREAMBLE: &str = include_str!("module.wgsl");

const FILTER_PREAMBLE: &str = include_str!("filter.wgsl");

const LOCAL_VAR_PREFIX: &str = "l";

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

    fn var_decl(&mut self, ty: &sema::Type) -> (String, usize) {
        let idx = self.local_var_idx;
        let s = format!("var {}_{} : {} = ", LOCAL_VAR_PREFIX, idx, ty.as_wgsl());
        self.local_var_idx += 1;
        (s, idx)
    }

    fn compile_expr(&mut self, expr: &ast::Expression) -> Result<usize, TypeError> {
        match expr {
            ast::Expression::FunctionCall { name, args, .. } => {
                // Compile args and keep track of their variable idx.
                let args_idxs: Vec<_> = args
                    .into_iter()
                    .map(|arg_expr| self.compile_expr(arg_expr))
                    .collect::<Result<_, _>>()?;

                let (mut decl, idx) = self.var_decl(&sema::Type::Tuple(4));
                decl.push_str(name);
                decl.push_str("(");
                decl.push_str(
                    &args_idxs
                        .iter()
                        .map(|x| format!("{}_{}", LOCAL_VAR_PREFIX, x))
                        .collect::<Vec<_>>()
                        .join(", "),
                );
                decl.push_str(");");
                self.writer.line(&decl);
                Ok(idx)
            }
            ast::Expression::IntConst { value, ty } => {
                let (mut s, idx) = self.var_decl(&ty);
                s.push_str(&format!("{}", value));
                s.push_str(";");
                self.writer.line(&s);
                Ok(idx)
            }
            ast::Expression::FloatConst { value, ty } => {
                let (mut s, idx) = self.var_decl(&ty);
                s.push_str(&format!("{}", value));
                s.push_str(";");
                self.writer.line(&s);
                Ok(idx)
            }
            ast::Expression::Cast { tag, expr, ty } => {
                let expr_idx = self.compile_expr(expr)?;
                let (mut s, idx) = self.var_decl(&ty);
                s.push_str(
                    format!("{}({}_{});", ty.as_wgsl(), LOCAL_VAR_PREFIX, expr_idx).as_str(),
                );
                self.writer.line(&s);
                Ok(idx)
            }
            _ => Err(TypeError::with_pos(
                format!("unimplemented expression {:?}", expr),
                0,
                0,
            )),
        }
    }

    fn compile_filter(&mut self, filter: &ast::Filter) -> Result<(), MathMapError> {
        self.writer.buf.push_str(MODULE_PREAMBLE);
        self.writer.buf.push_str(FILTER_PREAMBLE);

        self.writer.indent();

        let mut last_expr_idx = 0;

        for expr in &filter.exprs {
            last_expr_idx = self.compile_expr(expr)?;
        }

        // Assign the last expression to the output buffer.
        self.writer.line(&format!(
            "output.pixels[idx] = {}_{};",
            LOCAL_VAR_PREFIX, last_expr_idx,
        ));
        self.writer.dedent();
        self.writer.line("}");

        Ok(())
    }
}

pub fn compile_filter(filter: &ast::Filter) -> Result<String, MathMapError> {
    let mut filt = filter.clone();
    let mut sema = crate::sema::SemanticAnalyzer::new();
    sema.analyze_filter(&mut filt)?;
    let mut compiler = WgslCompiler::new();
    compiler.compile_filter(&filt)?;
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

    #[test]
    fn filter_with_cast() -> Result<(), Box<dyn std::error::Error>> {
        let input = "filter red ()
            rgbColor(1, 0, 0)
        end";
        let ast = ast::parse_module(input)?;
        let filter = &ast.filters[0];
        let out = compile_filter(filter)?;

        Ok(())
    }
}
