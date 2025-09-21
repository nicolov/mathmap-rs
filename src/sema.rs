// Semantic analysis and type checking.

#![allow(dead_code)]

use crate::ast::{self, TupleTag};
use crate::err::TypeError;
use ast::Expression;
use std::collections::HashMap;

#[derive(Debug, Clone, PartialEq, Default)]
pub enum Type {
    #[default]
    Unknown,
    Int,
    // todo: add TupleTag.
    Tuple(usize),
}

impl Type {
    pub fn as_wgsl(&self) -> &str {
        match self {
            Type::Int => "i32",
            Type::Tuple(1) => "f32",
            Type::Tuple(4) => "vec4<f32>",
            _ => todo!(),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct FuncParam {
    pub name: String,
    pub ty: Type,
}

#[derive(Debug, Clone, PartialEq)]
pub struct FuncSignature {
    pub name: String,
    pub params: Vec<FuncParam>,
    pub ret: Type,
}

#[derive(Debug, Clone, PartialEq)]
pub struct FuncDef {
    pub signature: FuncSignature,
    // todo: store bodies of user-defined funcs here?
}

#[derive(Debug, Clone, PartialEq)]
pub struct FunctionTable {
    pub functions: HashMap<String, Vec<FuncDef>>,
}

// The result of overload resolution is the selected signature + any implicit casts
// that were inserted.
#[derive(Debug, Clone, PartialEq)]
struct OverloadResolutionResult<'a> {
    def: &'a FuncDef,
    casts: Vec<Option<Type>>,
}

impl FunctionTable {
    pub fn new() -> Self {
        let mut fns = HashMap::new();

        fns.insert(
            "rgbColor".to_string(),
            vec![FuncDef {
                signature: FuncSignature {
                    name: "rgbColor".to_string(),
                    params: vec![
                        FuncParam {
                            name: "r".to_string(),
                            ty: Type::Tuple(1),
                        },
                        FuncParam {
                            name: "g".to_string(),
                            ty: Type::Tuple(1),
                        },
                        FuncParam {
                            name: "b".to_string(),
                            ty: Type::Tuple(1),
                        },
                    ],
                    ret: Type::Tuple(4),
                },
            }],
        );

        fns.insert(
            "__add".to_string(),
            vec![
                FuncDef {
                    signature: FuncSignature {
                        name: "__add".to_string(),
                        params: vec![
                            FuncParam {
                                name: "x".to_string(),
                                ty: Type::Tuple(1),
                            },
                            FuncParam {
                                name: "y".to_string(),
                                ty: Type::Tuple(1),
                            },
                        ],
                        ret: Type::Tuple(1),
                    },
                },
                FuncDef {
                    signature: FuncSignature {
                        name: "__add".to_string(),
                        params: vec![
                            FuncParam {
                                name: "x".to_string(),
                                ty: Type::Int,
                            },
                            FuncParam {
                                name: "y".to_string(),
                                ty: Type::Int,
                            },
                        ],
                        ret: Type::Int,
                    },
                },
            ],
        );

        Self { functions: fns }
    }

    fn lookup(&self, name: &str, arg_tys: &[Type]) -> Result<OverloadResolutionResult, TypeError> {
        let candidates = self.functions.get(name).ok_or(TypeError::with_pos(
            format!("unimplemented function {:?}", name),
            0,
            0,
        ))?;

        let mut matches: Vec<(OverloadResolutionResult, usize)> = Vec::new();

        // Go through all candidates and rank them by how many implicit casts
        // they would need.
        for cand in candidates {
            if cand.signature.params.len() != arg_tys.len() {
                continue;
            }

            let mut casts = Vec::new();
            let mut num_casts = 0;
            let mut compatible = true;

            for (current_ty, param_ty) in arg_tys.iter().zip(cand.signature.params.iter()) {
                if current_ty == &param_ty.ty {
                    casts.push(None);
                } else if current_ty == &Type::Int && param_ty.ty == Type::Tuple(1) {
                    casts.push(Some(Type::Tuple(1)));
                    num_casts += 1;
                } else {
                    compatible = false;
                    break;
                }
            }

            if compatible {
                matches.push((OverloadResolutionResult { def: cand, casts }, num_casts));
            }
        }

        if matches.is_empty() {
            return Err(TypeError::with_pos(
                format!("no matching overload for function {:?}", name),
                0,
                0,
            ));
        } else {
            // Pick the best match.
            matches.sort_by_key(|(_, num_casts)| *num_casts);
            let (best, best_cost) = &matches[0];

            if matches.len() > 1 && matches[1].1 == *best_cost {
                return Err(TypeError::with_pos(
                    format!("ambiguous overload for function {:?}", name),
                    0,
                    0,
                ));
            }

            return Ok(best.clone());
        }
    }
}

pub struct SemanticAnalyzer {
    func_table: FunctionTable,
}

impl SemanticAnalyzer {
    pub fn new() -> Self {
        Self {
            func_table: FunctionTable::new(),
        }
    }

    fn analyze_expr(&mut self, expr: &mut Expression) -> Result<(), TypeError> {
        match expr {
            // Types of int and float literals are already filled in by the parser.
            Expression::IntConst { .. } => Ok(()),
            Expression::FloatConst { .. } => Ok(()),
            // Expression::TupleConst { .. } => Ok(()),
            Expression::FunctionCall { name, args, ty } => {
                for arg in &mut *args {
                    self.analyze_expr(arg)?;
                }

                // Find the matching function with overloads, etc..
                let func = self.func_table.lookup(
                    name.as_str(),
                    args.iter().map(|x| x.ty()).collect::<Vec<_>>().as_slice(),
                )?;

                // Insert cast AST nodes following overload resolution.
                for (i, cast) in func.casts.iter().enumerate() {
                    if let Some(cast) = cast {
                        if let Type::Tuple(..) = cast {
                            let node = Expression::Cast {
                                tag: TupleTag::Nil,
                                expr: Box::new(args[i].clone()),
                                ty: (*cast).clone(),
                            };
                            args[i] = node;
                        } else {
                            return Err(TypeError::with_pos(
                                "for now only casts to float tuples are implemented",
                                0,
                                0,
                            )
                            .into());
                        }
                    }
                }

                // Annotate the return type.
                *ty = func.def.signature.ret.clone();
                Ok(())
            }
            _ => {
                todo!();
            }
        }
    }

    pub fn analyze_filter(&mut self, filter: &mut ast::Filter) -> Result<(), TypeError> {
        if filter.exprs.is_empty() {
            return Err(TypeError::with_pos("empty filter", 0, 0));
        }
        for expr in &mut filter.exprs {
            self.analyze_expr(expr)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ast::Expression as E;
    use ast::Parser;
    use std::error::Error;

    fn analyze_expr(src: &str) -> Result<Expression, Box<dyn Error>> {
        let mut parser = Parser::new(src);
        let mut ast = parser.parse_expression(1)?;
        let mut sema = SemanticAnalyzer::new();
        sema.analyze_expr(&mut ast)?;
        Ok(ast)
    }

    #[test]
    fn int_constant() -> Result<(), Box<dyn Error>> {
        let expr = analyze_expr("1")?;
        if let E::IntConst { ty, .. } = expr {
            assert_eq!(ty, Type::Int);
        } else {
            panic!("expected int constant");
        }
        Ok(())
    }

    #[test]
    fn add_ints() -> Result<(), Box<dyn Error>> {
        let expr = analyze_expr("1 + 2")?;
        if let E::FunctionCall { ty, .. } = expr {
            assert_eq!(ty, Type::Int);
        } else {
            panic!("expected function call");
        }
        Ok(())
    }

    #[test]
    fn add_int_and_float() -> Result<(), Box<dyn Error>> {
        let expr = analyze_expr("1 + 2.0")?;
        if let E::FunctionCall { ty, .. } = &expr {
            assert_eq!(*ty, Type::Tuple(1));
        } else {
            panic!("expected function call");
        }

        let expected_ast = E::FunctionCall {
            name: "__add".to_string(),
            args: vec![
                E::Cast {
                    tag: TupleTag::Nil,
                    expr: Box::new(E::int_(1)),
                    ty: Type::Tuple(1),
                },
                E::float_(2.0),
            ],
            ty: Type::Tuple(1),
        };

        assert_eq!(expr, expected_ast);
		Ok(())
    }
}

#[cfg(test)]
mod fn_table_tests {
    use super::*;
    use std::error::Error;

    fn _num_casts(res: &OverloadResolutionResult) -> usize {
        res.casts.iter().filter(|x| x.is_some()).count()
    }

    #[test]
    fn not_found() -> Result<(), Box<dyn Error>> {
        let fns = FunctionTable::new();
        let hello_fn = fns.lookup("__not_existing__", &vec![]);
        assert!(matches!(hello_fn, Err(TypeError(_))));
        Ok(())
    }

    #[test]
    fn lookup_no_matching_overload() -> Result<(), Box<dyn Error>> {
        let fns = FunctionTable::new();
        let hello_fn = fns.lookup("rgbColor", &vec![]);
        assert!(matches!(hello_fn, Err(TypeError(_))));
        Ok(())
    }

    #[test]
    fn lookup_simple() -> Result<(), Box<dyn Error>> {
        let fns = FunctionTable::new();
        let hello_fn = fns.lookup(
            "rgbColor",
            &vec![Type::Tuple(1), Type::Tuple(1), Type::Tuple(1)],
        )?;

        let OverloadResolutionResult { def, .. } = &hello_fn;
        assert_eq!(def.signature.name, "rgbColor");
        assert_eq!(def.signature.params.len(), 3);
        assert_eq!(def.signature.ret, Type::Tuple(4));
        assert_eq!(_num_casts(&hello_fn), 0);

        Ok(())
    }

    #[test]
    fn lookup_cast() -> Result<(), Box<dyn Error>> {
        let fns = FunctionTable::new();
        let hello_fn = fns.lookup("rgbColor", &vec![Type::Int, Type::Tuple(1), Type::Int])?;

        let OverloadResolutionResult { def, .. } = &hello_fn;
        assert_eq!(def.signature.name, "rgbColor");
        assert_eq!(def.signature.params.len(), 3);
        assert_eq!(def.signature.ret, Type::Tuple(4));
        assert_eq!(_num_casts(&hello_fn), 2);

        Ok(())
    }
}
