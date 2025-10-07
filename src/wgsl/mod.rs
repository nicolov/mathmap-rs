// A transpiler to WebGPU Shading Language (WGSL).

#![allow(dead_code)]

use crate::ast::Expression;
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

    fn is_wgsl_operator(&self, name: &str) -> Option<char> {
        match name {
            "__add" => Some('+'),
            "__sub" => Some('-'),
            "__mul" => Some('*'),
            "__div" => Some('/'),
            _ => None,
        }
    }

    fn is_wgsl_intrinsic(&self, name: &str) -> bool {
        match name {
            "sin" => true,
            "abs" => true,
            _ => false,
        }
    }

    fn var_name(&self, idx: usize) -> String {
        format!("{}_{}", LOCAL_VAR_PREFIX, idx)
    }

    fn var_decl(&mut self, ty: &sema::Type) -> (String, usize) {
        let idx = self.local_var_idx;
        let s = format!("var {}_{} : {} = ", LOCAL_VAR_PREFIX, idx, ty.as_wgsl());
        self.local_var_idx += 1;
        (s, idx)
    }

    fn var_decl2(&mut self, ty: &sema::Type) -> (String, usize) {
        let idx = self.local_var_idx;
        let s = format!("var {}_{} : {}", LOCAL_VAR_PREFIX, idx, ty.as_wgsl());
        self.local_var_idx += 1;
        (s, idx)
    }

    fn compile_expr(&mut self, expr: &ast::Expression) -> Result<usize, TypeError> {
        match expr {
            ast::Expression::FunctionCall { name, args, ty } => {
                // Compile args and keep track of their variable idx.
                let args_idxs: Vec<_> = args
                    .into_iter()
                    .map(|arg_expr| self.compile_expr(arg_expr))
                    .collect::<Result<_, _>>()?;

                let (mut decl, idx) = self.var_decl(ty);

                // Special handling for wgsl operators so we don't have to duplicate them in the wgsl preamble
                // (which would also require name mangling because wgsl doesn't have overloading).
                if let Some(op) = self.is_wgsl_operator(name) {
                    let op_call = format!(
                        "{} {} {}",
                        self.var_name(args_idxs[0]),
                        op,
                        self.var_name(args_idxs[1])
                    );
                    decl.push_str(&op_call);
                } else {
                    if !self.is_wgsl_intrinsic(name) {
                        // Mangle the function name because wgsl doesn't allow leading double underscore..
                        decl.push_str("FN_");
                    }
                    decl.push_str(name);
                    decl.push_str("(");
                    decl.push_str(
                        &args_idxs
                            .iter()
                            .map(|x| format!("{}_{}", LOCAL_VAR_PREFIX, x))
                            .collect::<Vec<_>>()
                            .join(", "),
                    );
                    decl.push_str(")");
                }

                decl.push_str(";");
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
            ast::Expression::Assignment { name, value, ty } => {
                let value_idx = self.compile_expr(value)?;

                // Same logic as self.var_decl, but use the existing name to make
                // debugging easier.
                let idx = self.local_var_idx;
                let s = format!("var {}_{} : {} = ", LOCAL_VAR_PREFIX, idx, ty.as_wgsl());
                self.local_var_idx += 1;

                let s = format!(
                    "var {} : {} = {}_{};",
                    name,
                    ty.as_wgsl(),
                    LOCAL_VAR_PREFIX,
                    value_idx
                );
                self.writer.line(&s);

                Ok(idx)
            }
            ast::Expression::Variable { name, ty } => {
                let (mut s, idx) = self.var_decl(&ty);
                s.push_str(name);
                s.push_str(";");
                self.writer.line(&s);
                Ok(idx)
            }
            ast::Expression::If {
                condition,
                then,
                else_,
                ty,
            } => {
                let (mut s, eval_idx) = self.var_decl2(&ty);
                s.push_str(";");
                self.writer.line(&s);

                let cond_idx = self.compile_expr(condition)?;

                let branch = format!("if ({} != 0) {{", self.var_name(cond_idx));
                self.writer.line(&branch);
                self.writer.indent();
                let then_idx = self.compile_expr_block(then)?;

                let assign_s =
                    format!("{} = {};", self.var_name(eval_idx), self.var_name(then_idx));
                self.writer.line(&assign_s);
                self.writer.dedent();

                if !else_.is_empty() {
                    self.writer.line("} else {");
                    self.writer.indent();
                    let else_idx = self.compile_expr_block(else_)?;
                    let assign_s =
                        format!("{} = {};", self.var_name(eval_idx), self.var_name(else_idx));
                    self.writer.line(&assign_s);
                    self.writer.dedent();
                    self.writer.line("}");
                } else {
                    self.writer.line("}");
                }

                Ok(eval_idx)
            }
            ast::Expression::Index {
                expr: array,
                index,
                ty,
            } => {
                let array_idx = self.compile_expr(array)?;
                let index_idx = self.compile_expr(index)?;

                let (mut s, idx) = self.var_decl(&ty);

                s.push_str(&format!(
                    "{}[{}];",
                    self.var_name(array_idx),
                    self.var_name(index_idx)
                ));
                self.writer.line(&s);

                Ok(idx)
            }
            _ => Err(TypeError::with_pos(
                format!("unimplemented wgsl expr {:?}", expr),
                0,
                0,
            )),
        }
    }

    fn compile_expr_block(&mut self, exprs: &Vec<Expression>) -> Result<usize, TypeError> {
        let mut last_expr_idx = 0;

        for expr in exprs {
            last_expr_idx = self.compile_expr(expr)?;
        }

        Ok(last_expr_idx)
    }

    fn compile_filter(&mut self, filter: &ast::Filter) -> Result<(), MathMapError> {
        self.writer.buf.push_str(MODULE_PREAMBLE);
        self.writer.buf.push_str(FILTER_PREAMBLE);

        self.writer.indent();

        let last_expr_idx = self.compile_expr_block(&filter.exprs)?;

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
    use ast::Parser;
    use sema::SemanticAnalyzer;
    use std::error::Error;

    fn compile_expr(src: &str) -> Result<String, Box<dyn Error>> {
        let mut parser = Parser::new(src);
        let mut ast = parser.parse_expr_block()?;
        let mut sema = SemanticAnalyzer::new();
        sema.analyze_expr_block(&mut ast)?;

        let mut compiler = WgslCompiler::new();
        for expr in &mut ast {
            compiler.compile_expr(&expr)?;
        }
        Ok(compiler.writer.finish())
    }

    #[test]
    fn filter() -> Result<(), Box<dyn std::error::Error>> {
        let input = "filter red ()
            rgbColor(1, 0, 0)
        end";
        let ast = ast::parse_module(input)?;
        let filter = &ast.filters[0];
        compile_filter(filter)?;

        Ok(())
    }

    #[test]
    fn filter_with_cast() -> Result<(), Box<dyn std::error::Error>> {
        let input = "filter red ()
            rgbColor(1, 0, 0)
        end";
        let ast = ast::parse_module(input)?;
        let filter = &ast.filters[0];
        compile_filter(filter)?;

        Ok(())
    }

    #[test]
    fn assign() -> Result<(), Box<dyn std::error::Error>> {
        let input = "z = 1";
        compile_expr(input)?;

        Ok(())
    }

    #[test]
    fn variable() -> Result<(), Box<dyn std::error::Error>> {
        let input = "z = 2; 1 + z";
        compile_expr(input)?;

        if let Ok(out) = compile_expr(input) {
            println!("{}", out);
        }

        Ok(())
    }
}
