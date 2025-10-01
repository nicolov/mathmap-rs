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
    // TupleVar is for polymorphic function defs, eg abs(tuple<N>) -> tuple<N>.
    TupleVar(char),
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
    // Actual return type where typevars were replaced with concrete tuple lengths.
    ret_ty: Type,
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
            "grayColor".to_string(),
            vec![FuncDef {
                signature: FuncSignature {
                    name: "grayColor".to_string(),
                    params: vec![FuncParam {
                        name: "x".to_string(),
                        ty: Type::Tuple(1),
                    }],
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
                                ty: Type::TupleVar('N'),
                            },
                            FuncParam {
                                name: "y".to_string(),
                                ty: Type::TupleVar('N'),
                            },
                        ],
                        ret: Type::TupleVar('N'),
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

        fns.insert(
            "__div".to_string(),
            vec![
                // div is floating point only, we rely on implicit type casts.
                FuncDef {
                    signature: FuncSignature {
                        name: "__div".to_string(),
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
            ],
        );

        fns.insert(
            "abs".to_string(),
            vec![FuncDef {
                signature: FuncSignature {
                    name: "abs".to_string(),
                    params: vec![FuncParam {
                        name: "x".to_string(),
                        ty: Type::TupleVar('N'),
                    }],
                    ret: Type::TupleVar('N'),
                },
            }],
        );

        Self { functions: fns }
    }

    fn check_candidate<'a>(
        &self,
        cand: &'a FuncDef,
        arg_tys: &[Type],
    ) -> Option<(OverloadResolutionResult<'a>, usize)> {
        if cand.signature.params.len() != arg_tys.len() {
            return None;
        }

        let mut casts = Vec::new();
        let mut cost = 0;
        let mut compatible = true;

        // Mapping from typevar chars to the actual length of the tuple.
        let mut typevars = HashMap::new();

        // Match the actual argument type with the corresponding parameter in the function signature.
        for (current_ty, param_ty) in arg_tys.iter().zip(cand.signature.params.iter()) {
            // todo: simplify by handling type promotion (first) and broadcasting (second) in two separate steps.
            match (current_ty, &param_ty.ty) {
                (Type::Int, Type::Int) => {
                    casts.push(None);
                }
                (Type::Tuple(1), Type::Tuple(1)) => {
                    // todo: make this generic with Tuple(N).
                    casts.push(None);
                }
                (Type::Int, Type::Tuple(1)) => {
                    // todo: also promote an int to a Tuple(N) when the signature is
                    // polymorphic.
                    casts.push(Some(Type::Tuple(1)));
                    cost += 1;
                }
                (Type::Int, Type::TupleVar(tv)) => {
                    match typevars.get(&tv) {
                        Some(n2) => {
                            // We already have a substitution for this typevar, and it must be 1 to unify
                            // with a single integer.
                            if *n2 != 1 {
                                compatible = false;
                                break;
                            }
                        }
                        None => {
                            // Record this unification.
                            typevars.insert(tv, 1);
                        }
                    }
                    casts.push(Some(Type::Tuple(1)));
                    cost += 1;
                }
                (Type::Tuple(n), Type::TupleVar(tv)) => {
                    match typevars.get(&tv) {
                        Some(n2) => {
                            // We already have a substitution for this typevar, and
                            // it must match the concrete type, otherwise we skip this
                            // overload candidate.
                            if n == n2 {
                                casts.push(None);
                            } else {
                                compatible = false;
                                break;
                            }
                        }
                        None => {
                            // Record this unification.
                            typevars.insert(tv, *n);
                        }
                    }
                }
                _ => {
                    compatible = false;
                    break;
                }
            }
        }

        if compatible {
            let ret_ty = match &cand.signature.ret {
                Type::TupleVar(tv) => match typevars.get(&tv) {
                    Some(n) => Type::Tuple(*n),
                    None => {
                        // This should never happen if the signatures are well formed, but don't really check them yet..
                        panic!("unknown typevar {}", &tv.clone());
                    }
                },
                x => x.clone(),
            };

            Some((
                OverloadResolutionResult {
                    def: cand,
                    casts,
                    ret_ty,
                },
                cost,
            ))
        } else {
            None
        }
    }

    fn lookup(&self, name: &str, arg_tys: &[Type]) -> Result<OverloadResolutionResult<'_>, TypeError> {
        let candidates = self.functions.get(name).ok_or(TypeError::with_pos(
            format!("unimplemented function {:?}", name),
            0,
            0,
        ))?;

        let mut matches: Vec<(OverloadResolutionResult, usize)> = Vec::new();

        for cand in candidates {
            if let Some((res, cost)) = self.check_candidate(cand, arg_tys) {
                matches.push((res, cost));
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
                    format!("ambiguous overload for function {:?}, {:?}", name, matches),
                    0,
                    0,
                ));
            }

            return Ok(best.clone());
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
struct SymbolTable {
    vars: HashMap<String, Type>,
}

impl SymbolTable {
    fn new() -> Self {
        let mut vars = HashMap::new();
        vars.insert("x".to_string(), Type::Tuple(1));
        vars.insert("y".to_string(), Type::Tuple(1));
        vars.insert("xy".to_string(), Type::Tuple(2));

        Self { vars: vars }
    }
}

pub struct SemanticAnalyzer {
    func_table: FunctionTable,
    vars: SymbolTable,
}

impl SemanticAnalyzer {
    pub fn new() -> Self {
        Self {
            func_table: FunctionTable::new(),
            vars: SymbolTable::new(),
        }
    }

    pub fn analyze_expr(&mut self, expr: &mut Expression) -> Result<(), TypeError> {
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

                // Annotate the return type with the (potentially polymorphic) return type.
                *ty = func.ret_ty.clone();
                Ok(())
            }
            Expression::Assignment { name, value, ty } => {
                self.analyze_expr(value)?;
                *ty = value.ty();
                self.vars.vars.insert(name.clone(), value.ty());
                Ok(())
            }
            Expression::Variable { name, ty, .. } => {
                if let Some(sym_ty) = self.vars.vars.get(name) {
                    *ty = sym_ty.clone();
                    Ok(())
                } else {
                    Err(TypeError::with_pos(
                        format!("variable {} not found", name),
                        0,
                        0,
                    ))
                }
            }
            Expression::TupleConst { tag, values, .. } => {
                // Check that the number of values matches the tag.
                if values.len() != tag.len() {
                    return Err(TypeError::with_pos(
                        format!(
                            "expected {} values for {:?}, got {}",
                            tag.len(),
                            tag,
                            values.len()
                        ),
                        0,
                        0,
                    ));
                }

                for i in 0..values.len() {
                    let value = &mut values[i];
                    self.analyze_expr(value)?;

                    match value.ty() {
                        Type::Int => {
                            // Add cast node.
                            *value = Expression::cast_with_ty_(
                                TupleTag::Nil,
                                value.clone(),
                                Type::Tuple(1),
                            )
                        }
                        Type::Tuple(1) => {}
                        Type::Tuple(n) => {
                            return Err(TypeError::with_pos(
                                format!("expected scalar in tuple literal, got a {}-tuple", n),
                                0,
                                0,
                            ));
                        }
                        _ => {
                            return Err(TypeError::with_pos(
                                format!("unexpected type {:?}", value.ty()),
                                0,
                                0,
                            ));
                        }
                    }
                }

                Ok(())
            }
            _ => Err(TypeError::with_pos(
                format!("sema unimplemented expr {:?}", expr),
                0,
                0,
            )),
        }
    }

    pub fn analyze_expr_block(&mut self, exprs: &mut Vec<Expression>) -> Result<(), TypeError> {
        for expr in exprs {
            self.analyze_expr(expr)?;
        }
        Ok(())
    }

    pub fn analyze_filter(&mut self, filter: &mut ast::Filter) -> Result<(), TypeError> {
        if filter.exprs.is_empty() {
            return Err(TypeError::with_pos("empty filter", 0, 0));
        }
        self.analyze_expr_block(&mut filter.exprs)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ast::Expression as E;
    use ast::Parser;
    use std::error::Error;

    fn analyze_expr(src: &str) -> Result<Vec<Expression>, Box<dyn Error>> {
        let mut parser = Parser::new(src);
        let mut ast = parser.parse_expr_block()?;
        let mut sema = SemanticAnalyzer::new();
        sema.analyze_expr_block(&mut ast)?;
        Ok(ast)
    }

    #[test]
    fn int_constant() -> Result<(), Box<dyn Error>> {
        let expr = &analyze_expr("1")?[0];
        if let E::IntConst { ty, .. } = expr {
            assert_eq!(*ty, Type::Int);
        } else {
            panic!("expected int constant");
        }
        Ok(())
    }

    #[test]
    fn add_ints() -> Result<(), Box<dyn Error>> {
        let expr = &analyze_expr("1 + 2")?[0];
        if let E::FunctionCall { ty, .. } = expr {
            assert_eq!(*ty, Type::Int);
        } else {
            panic!("expected function call");
        }
        Ok(())
    }

    #[test]
    fn add_int_and_float() -> Result<(), Box<dyn Error>> {
        let expr = &analyze_expr("1 + 2.0")?[0];
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

        assert_eq!(*expr, expected_ast);
        Ok(())
    }

    #[test]
    fn assignment() -> Result<(), Box<dyn Error>> {
        let expr = &analyze_expr("x = 1")?[0];
        if let E::Assignment { name, value, ty } = expr {
            assert_eq!(name, "x");
            assert_eq!(value.ty(), Type::Int);
            assert_eq!(*ty, Type::Int);
        } else {
            panic!("expected assignment");
        }
        Ok(())
    }

    #[test]
    fn polymorphic() -> Result<(), Box<dyn Error>> {
        let expr = &analyze_expr("abs(xy)")?[0];
        if let E::FunctionCall { name, args, ty } = expr {
            assert_eq!(name, "abs");
            assert_eq!(args.len(), 1);
            assert_eq!(args[0].ty(), Type::Tuple(2));
            assert_eq!(*ty, Type::Tuple(2));
        } else {
            panic!("expected function call");
        }
        Ok(())
    }

    #[test]
    fn polymorphic_binary() -> Result<(), Box<dyn Error>> {
        let expr = &analyze_expr("xy + xy")?[0];
        if let E::FunctionCall { name, args, ty } = expr {
            assert_eq!(name, "__add");
            assert_eq!(args.len(), 2);
            assert_eq!(args[0].ty(), Type::Tuple(2));
            assert_eq!(args[1].ty(), Type::Tuple(2));
            assert_eq!(*ty, Type::Tuple(2));
        } else {
            panic!("expected function call");
        }
        Ok(())
    }

    #[test]
    fn tuple_const() -> Result<(), Box<dyn Error>> {
        let expr = &analyze_expr("xy:[1.0, 2.0]")?[0];
        if let E::TupleConst { tag, values, ty } = expr {
            assert_eq!(*tag, TupleTag::Xy);
            assert_eq!(values.len(), 2);
            assert_eq!(*ty, Type::Tuple(2));
        } else {
            panic!("expected tuple const");
        }
        Ok(())
    }

    #[test]
    fn tuple_const_promotion() -> Result<(), Box<dyn Error>> {
        let expr = &analyze_expr("xy:[1, 2.0]")?[0];

        let expected_ast = E::TupleConst {
            tag: TupleTag::Xy,
            values: vec![
                E::cast_with_ty_(TupleTag::Nil, E::int_(1), Type::Tuple(1)),
                E::float_(2.0),
            ],
            ty: Type::Tuple(2),
        };
        assert_eq!(*expr, expected_ast);
        Ok(())
    }

    #[test]
    fn tuple_const_wrong_len() -> Result<(), Box<dyn Error>> {
        if let Err(e) = analyze_expr("rgba:[1, 2, 3]") {
            if let Some(TypeError(tye)) = e.downcast_ref::<TypeError>() {
                assert_eq!(tye.message, "expected 4 values for Rgba, got 3");
                Ok(())
            } else {
                panic!("expected type error");
            }
        } else {
            panic!("expected the parser to fail");
        }
    }

    #[test]
    fn tuple_const_wrong_inner() -> Result<(), Box<dyn Error>> {
        if let Err(e) = analyze_expr("xy:[1.0, xy:[2.0, 3.0]]") {
            if let Some(TypeError(tye)) = e.downcast_ref::<TypeError>() {
                assert_eq!(
                    tye.message,
                    "expected scalar in tuple literal, got a 2-tuple"
                );
            } else {
                panic!("expected type error");
            }
        }
        Ok(())
    }

    #[test]
    fn broadcasting_incompatible() -> Result<(), Box<dyn Error>> {
        if let Err(e) = analyze_expr("xy:[1, 2] + rgba:[1, 2, 3, 4]") {
            if let Some(TypeError(tye)) = e.downcast_ref::<TypeError>() {
                assert_eq!(tye.message, "no matching overload for function \"__add\"");
                Ok(())
            } else {
                panic!("expected type error");
            }
        } else {
            panic!("expected the parser to fail");
        }
    }

    #[test]
    #[ignore]
    fn broadcasting() -> Result<(), Box<dyn Error>> {
        let expr = &analyze_expr("xy:[1.0, 2.0] + 1.0")?[0];

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
