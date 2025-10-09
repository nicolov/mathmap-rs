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
            Type::Tuple(2) => "vec2<f32>",
            Type::Tuple(3) => "vec3<f32>",
            Type::Tuple(4) => "vec4<f32>",
            _ => todo!("type {:?} can not be converted to wgsl", self),
        }
    }

    pub fn is_scalar(&self) -> bool {
        self.len() == 1
    }

    pub fn len(&self) -> usize {
        match self {
            Type::Int => 1,
            Type::Tuple(n) => *n,
            Type::TupleVar(_) => todo!(),
            _ => {
                todo!("don't know how to get len of {:?}", self);
            }
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

        // todo: find a better abstraction to define function signatures.
        let mut def_int_float_binary = |name: &str| {
            fns.insert(
                name.to_string(),
                vec![
                    FuncDef {
                        signature: FuncSignature {
                            name: name.to_string(),
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
                            name: name.to_string(),
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
        };

        def_int_float_binary("__add");
        def_int_float_binary("__sub");
        def_int_float_binary("__mul");
        def_int_float_binary("__mod");
        def_int_float_binary("__pow");

        let mut def_comparison = |name: &str| {
            fns.insert(
                name.to_string(),
                vec![FuncDef {
                    signature: FuncSignature {
                        name: name.to_string(),
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
                        ret: Type::Int,
                    },
                }],
            );
        };

        def_comparison("__less");
        def_comparison("__lessequal");

        let mut def_unary = |name: &str| {
            fns.insert(
                name.to_string(),
                vec![FuncDef {
                    signature: FuncSignature {
                        name: name.to_string(),
                        params: vec![FuncParam {
                            name: "x".to_string(),
                            ty: Type::TupleVar('N'),
                        }],
                        ret: Type::TupleVar('N'),
                    },
                }],
            );
        };

        def_unary("__neg");

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

        let mut def_float_unary = |name: &str| {
            fns.insert(
                name.to_string(),
                vec![FuncDef {
                    signature: FuncSignature {
                        name: name.to_string(),
                        params: vec![FuncParam {
                            name: "x".to_string(),
                            ty: Type::TupleVar('N'),
                        }],
                        ret: Type::TupleVar('N'),
                    },
                }],
            );
        };

        // todo: abs does vector magnitude for all tuple types right now but should
        // only do that for quaternions and complex numbers, and do elementwise abs
        // otherwise.
        def_float_unary("abs");
        def_float_unary("sin");

        let mut def_bool_binary = |name: &str| {
            fns.insert(
                name.to_string(),
                vec![FuncDef {
                    signature: FuncSignature {
                        name: name.to_string(),
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
                }],
            );
        };

        def_bool_binary("__and");
        def_bool_binary("__or");

        Self { functions: fns }
    }

    fn check_candidate<'a>(
        &self,
        cand: &'a FuncDef,
        arg_tys: &[Type],
    ) -> Option<(OverloadResolutionResult<'a>, usize)> {
        // Remember:
        // * parameter: placeholder in the function definition,
        // * argument: actual value passed to the function.
        if cand.signature.params.len() != arg_tys.len() {
            return None;
        }

        // Constrain typevars according to (non-scalar) arguments. Leave scalars alone for now.
        // eg. this is a mapping 'N' -> 3, etc..
        let mut typevar_constraints = HashMap::new();
        for (param, arg) in cand.signature.params.iter().zip(arg_tys.iter()) {
            if let Type::TupleVar(tv) = param.ty {
                // Ignore scalars for now.
                if let Type::Tuple(k) = arg
                    && *k > 1
                {
                    // If there was a previous binding for this typevar, check that it's consistent.
                    // Otherwise, create a new one.
                    if let Some(prev) = typevar_constraints.get(&tv) {
                        if *prev != *k {
                            return None;
                        }
                    } else {
                        typevar_constraints.insert(tv, *k);
                    }
                }
            }
        }

        // Constraint any unbound typevars to 1 to allow broadcasting later.
        for p in cand.signature.params.iter() {
            if let Type::TupleVar(tv) = p.ty {
                if !typevar_constraints.contains_key(&tv) {
                    typevar_constraints.insert(tv, 1);
                }
            }
        }

        let apply_typevars = |ty: &Type| match ty {
            Type::TupleVar(tv) => match typevar_constraints.get(&tv) {
                Some(n) => Type::Tuple(*n),
                None => {
                    panic!("unresolved typevar {}", &tv.clone());
                }
            },
            x => x.clone(),
        };

        // Apply typevar constraints to params.
        let params2 = cand
            .signature
            .params
            .iter()
            .map(|p| apply_typevars(&p.ty))
            .collect::<Vec<_>>();

        let mut total_cost = 0usize;
        let mut casts = Vec::new();
        for (param, arg) in params2.iter().zip(arg_tys.iter()) {
            // First, try if args match params exactly. This has zero cost.
            let perfect_match = match (param, arg) {
                (Type::Int, Type::Int) => true,
                (Type::Tuple(n), Type::Tuple(m)) => n == m,
                _ => false,
            };

            if perfect_match {
                casts.push(None);
                continue;
            }

            // Then try with type casts (with cost 1).
            let cast = match (param, arg) {
                (Type::Tuple(1), Type::Int) => Some(Type::Tuple(1)),
                _ => None,
            };

            if let Some(cast) = cast {
                casts.push(Some(cast));
                total_cost += 1;
                continue;
            }

            // Finally, try broadcasting (with cost 2).
            let bcast = match (param, arg) {
                // param is a (resolved) tuple and arg is a scalar, so we just promote arg to the param.
                (Type::Tuple(n), a) if a.is_scalar() && *n > 1 => Some(Type::Tuple(*n)),
                _ => None,
            };

            if let Some(_bcast) = bcast {
                total_cost += 2;

                // Since all tuples are float, any int -> tuple broadcast will also need a type cast.
                if matches!(arg, Type::Int) {
                    casts.push(Some(Type::Tuple(1)));
                    total_cost += 1;
                } else {
                    casts.push(None);
                }

                continue;
            }

            // If we get here, the candidate is not compatible.
            return None;
        }

        let ret_ty = apply_typevars(&cand.signature.ret);

        Some((
            OverloadResolutionResult {
                def: cand,
                casts,
                ret_ty,
            },
            total_cost,
        ))
    }

    fn lookup(
        &self,
        name: &str,
        arg_tys: &[Type],
    ) -> Result<OverloadResolutionResult<'_>, TypeError> {
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
                format!(
                    "no matching overload for function {:?} with args {:?}",
                    name, arg_tys
                ),
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

        vars.insert("r".to_string(), Type::Tuple(1));
        vars.insert("a".to_string(), Type::Tuple(1));

        vars.insert("t".to_string(), Type::Tuple(1));

        vars.insert("pi".to_string(), Type::Tuple(1));

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
            Expression::If {
                condition,
                then,
                else_,
                ty,
            } => {
                self.analyze_expr(condition)?;
                if !matches!(condition.ty(), Type::Int | Type::Tuple(1)) {
                    return Err(TypeError::with_pos(
                        format!("if condition must be int, found {:?}", condition.ty()),
                        0,
                        0,
                    ));
                }

                self.analyze_expr_block(then)?;
                let then_ty = then.last().unwrap().ty();

                if !else_.is_empty() {
                    self.analyze_expr_block(else_)?;
                    let else_ty = else_.last().unwrap().ty();

                    if then_ty != else_ty {
                        return Err(TypeError::with_pos(
                            format!(
                                "then and else branches must have the same type, got {:?} and {:?}",
                                then_ty, else_ty
                            ),
                            0,
                            0,
                        ));
                    }
                }

                *ty = then_ty;

                Ok(())
            }
            Expression::Index {
                expr: array,
                index,
                ty,
            } => {
                self.analyze_expr(array)?;
                self.analyze_expr(index)?;

                if !matches!(array.ty(), Type::Tuple(_)) {
                    return Err(TypeError::with_pos(
                        format!("index target must be a tuple, found {:?}", array.ty()),
                        0,
                        0,
                    ));
                }

                if !matches!(index.ty(), Type::Int) {
                    return Err(TypeError::with_pos(
                        format!("index variable must be an int, found {:?}", index.ty()),
                        0,
                        0,
                    ));
                }

                // Tuple items are always float.
                *ty = Type::Tuple(1);
                Ok(())
            }
            Expression::Cast { expr, ty, tag } => {
                self.analyze_expr(expr)?;
                // We don't have tags in the sema type system yet, so just check that the arity
                // matches.
                if expr.ty().len() != tag.len() {
                    return Err(TypeError::with_pos(
                        format!(
                            "expected {} values for {:?}, got {}",
                            tag.len(),
                            tag,
                            expr.ty().len()
                        ),
                        0,
                        0,
                    ));
                }
                *ty = expr.ty();
                Ok(())
            }
            Expression::While {
                condition,
                body,
                ty,
            } => {
                self.analyze_expr(condition)?;
                if !matches!(condition.ty(), Type::Int) {
                    return Err(TypeError::with_pos(
                        format!("while condition must be int, found {:?}", condition.ty()),
                        0,
                        0,
                    ));
                }
                self.analyze_expr_block(body)?;

                // A while loop always evaluates to 0 according to the docs.
                *ty = Type::Int;
                Ok(())
            }
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
    fn self_assignment() -> Result<(), Box<dyn Error>> {
        let expr = &analyze_expr("i = 0; i = i +1")?[1];
        if let E::Assignment { name, value, ty } = expr {
            assert_eq!(name, "i");
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
                assert_eq!(
                    tye.message,
                    "no matching overload for function \"__add\" with args [Tuple(2), Tuple(4)]"
                );
                Ok(())
            } else {
                panic!("expected type error");
            }
        } else {
            panic!("expected the parser to fail");
        }
    }

    #[test]
    fn broadcasting() -> Result<(), Box<dyn Error>> {
        let expr = &analyze_expr("xy:[1.0, 2.0] + 1.0")?[0];

        let expected_ast = E::FunctionCall {
            name: "__add".to_string(),
            args: vec![
                E::tuple_(TupleTag::Xy, vec![E::float_(1.0), E::float_(2.0)]),
                E::float_(1.0),
            ],
            ty: Type::Tuple(2),
        };
        assert_eq!(*expr, expected_ast);

        Ok(())
    }

    #[test]
    fn broadcasting_with_cast() -> Result<(), Box<dyn Error>> {
        let expr = &analyze_expr("xy:[1.0, 2.0] + 1")?[0];

        let expected_ast = E::FunctionCall {
            name: "__add".to_string(),
            args: vec![
                E::tuple_(TupleTag::Xy, vec![E::float_(1.0), E::float_(2.0)]),
                E::cast_with_ty_(TupleTag::Nil, E::int_(1), Type::Tuple(1)),
            ],
            ty: Type::Tuple(2),
        };
        assert_eq!(*expr, expected_ast);

        Ok(())
    }

    #[test]
    fn if_ok() -> Result<(), Box<dyn Error>> {
        let expr = &analyze_expr("if 1 then 2 else 3 end")?[0];
        assert_eq!(expr.ty(), Type::Int);
        Ok(())
    }

    #[test]
    fn if_different_branch_types() -> Result<(), Box<dyn Error>> {
        if let Err(e) = analyze_expr("if 1 then 2 else 3.0 end") {
            if let Some(TypeError(tye)) = e.downcast_ref::<TypeError>() {
                assert_eq!(
                    tye.message,
                    "then and else branches must have the same type, got Int and Tuple(1)"
                );
                Ok(())
            } else {
                panic!("expected type error");
            }
        } else {
            panic!("expected the parser to fail");
        }
    }

    #[test]
    fn if_wrong_branch_type() -> Result<(), Box<dyn Error>> {
        if let Err(e) = analyze_expr("if xy then 2 else 3 end") {
            if let Some(TypeError(tye)) = e.downcast_ref::<TypeError>() {
                assert_eq!(tye.message, "if condition must be int, found Tuple(2)");
                Ok(())
            } else {
                panic!("expected type error");
            }
        } else {
            panic!("expected the parser to fail");
        }
    }

    #[test]
    fn index_wrong_array() -> Result<(), Box<dyn Error>> {
        if let Err(e) = analyze_expr("a = 1; a[1]") {
            if let Some(TypeError(tye)) = e.downcast_ref::<TypeError>() {
                assert_eq!(tye.message, "index target must be a tuple, found Int");
                Ok(())
            } else {
                panic!("expected type error");
            }
        } else {
            panic!("expected the parser to fail");
        }
    }

    #[test]
    fn index_wrong_index() -> Result<(), Box<dyn Error>> {
        if let Err(e) = analyze_expr("a = xy:[1, 2]; a[1.0]") {
            if let Some(TypeError(tye)) = e.downcast_ref::<TypeError>() {
                assert_eq!(tye.message, "index variable must be an int, found Tuple(1)");
                Ok(())
            } else {
                panic!("expected type error");
            }
        } else {
            panic!("expected the parser to fail");
        }
    }

    #[test]
    fn index_ok() -> Result<(), Box<dyn Error>> {
        let expr = &analyze_expr("a = xy:[1, 2]; a[1]")?[1];
        assert_eq!(expr.ty(), Type::Tuple(1));
        Ok(())
    }

    #[test]
    fn cast_ok() -> Result<(), Box<dyn Error>> {
        let expr = &analyze_expr("a = xy:[1, 2]; b = ri:a")?[1];
        assert_eq!(expr.ty(), Type::Tuple(2));
        Ok(())
    }

    #[test]
    fn cast_wrong_arity() -> Result<(), Box<dyn Error>> {
        if let Err(e) = analyze_expr("a = xy:[1, 2]; b = rgba:a") {
            if let Some(TypeError(tye)) = e.downcast_ref::<TypeError>() {
                assert_eq!(tye.message, "expected 4 values for Rgba, got 2");
                Ok(())
            } else {
                panic!("expected type error");
            }
        } else {
            panic!("expected the parser to fail");
        }
    }

    #[test]
    fn while_ok() -> Result<(), Box<dyn Error>> {
        let expr = &analyze_expr("i = 0; while i < 10 do i = i + 1; end")?;
        assert_eq!(expr.last().unwrap().ty(), Type::Int);
        Ok(())
    }

    #[test]
    fn while_wrong_condition() -> Result<(), Box<dyn Error>> {
        if let Err(e) = analyze_expr("i = 0; while 1.1 do i = i + 1; end") {
            if let Some(TypeError(tye)) = e.downcast_ref::<TypeError>() {
                assert_eq!(tye.message, "while condition must be int, found Tuple(1)");
                Ok(())
            } else {
                panic!("expected type error");
            }
        } else {
            panic!("expected the parser to fail");
        }
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
