// Semantic analysis and type checking.

#![allow(dead_code)]

mod builtins;

use crate::ast::{self, TagVar, TupleTag};
use crate::err::TypeError;
use ast::Expression;
use std::collections::HashMap;
use std::fmt;

use self::builtins::FunctionTable;

type ArityVar = char;

#[derive(Debug, Clone, PartialEq)]
pub enum Arity {
    Sized(usize),
    Var(ArityVar),
}

#[derive(Clone, PartialEq, Default)]
pub enum Type {
    #[default]
    Unknown,
    Int,
    Tuple(TupleTag, Arity),
}

impl Type {
    pub fn as_wgsl(&self) -> &str {
        match self {
            Type::Int => "i32",
            Type::Tuple(_, Arity::Sized(1)) => "f32",
            Type::Tuple(_, Arity::Sized(2)) => "vec2<f32>",
            Type::Tuple(_, Arity::Sized(3)) => "vec3<f32>",
            Type::Tuple(_, Arity::Sized(4)) => "vec4<f32>",
            _ => todo!("type {:?} can not be converted to wgsl", self),
        }
    }

    pub fn is_scalar(&self) -> bool {
        self.len() == 1
    }

    pub fn len(&self) -> usize {
        match self {
            Type::Int => 1,
            Type::Tuple(_, Arity::Sized(n)) => *n,
            _ => {
                todo!("don't know how to get len of {:?}", self);
            }
        }
    }

    // todo: figure out which of these helpers are needed.
    pub fn tuple_sized(n: usize) -> Self {
        Self::Tuple(TupleTag::Nil, Arity::Sized(n))
    }

    pub fn scalar() -> Self {
        Self::Tuple(TupleTag::Nil, Arity::Sized(1))
    }

    pub fn tuple_tag(tag: TupleTag, n: usize) -> Self {
        Self::Tuple(tag, Arity::Sized(n))
    }

    pub fn rgba() -> Self {
        Self::tuple_tag(TupleTag::Rgba, 4)
    }

    pub fn ri() -> Self {
        Self::tuple_tag(TupleTag::Ri, 2)
    }

    pub fn xy() -> Self {
        Self::tuple_tag(TupleTag::Xy, 2)
    }

    pub fn quat() -> Self {
        Self::tuple_tag(TupleTag::Quat, 4)
    }

    pub fn tuplevar(tagvar: TagVar, nvar: ArityVar) -> Self {
        Self::Tuple(TupleTag::Var(tagvar), Arity::Var(nvar))
    }
}

impl fmt::Display for Type {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Type::Unknown => write!(f, "?"),
            Type::Int => write!(f, "int"),
            Type::Tuple(TupleTag::Var(tv), Arity::Sized(n)) => {
                write!(f, "?{}:{}", tv, n)
            }
            Type::Tuple(TupleTag::Var(tv), Arity::Var(av)) => {
                write!(f, "?{}:?{}", tv, av)
            }
            Type::Tuple(tag, Arity::Sized(n)) => {
                write!(f, "{}:{}", tag, n)
            }
            Type::Tuple(tag, Arity::Var(av)) => {
                write!(f, "{}:?{}", tag, av)
            }
        }
    }
}

impl fmt::Debug for Type {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self}")
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct FuncParam {
    pub name: String,
    pub ty: Type,
}

pub fn func_param(name: &str, ty: Type) -> FuncParam {
    return FuncParam {
        name: name.to_string(),
        ty: ty,
    };
}

#[derive(Clone, PartialEq)]
pub struct FuncSignature {
    pub name: String,
    pub params: Vec<FuncParam>,
    pub ret: Type,
}

impl fmt::Display for FuncSignature {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: (", self.name)?;

        for (idx, param) in self.params.iter().enumerate() {
            if idx > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{}", param.ty)?;
        }

        write!(f, ") -> {}", self.ret)
    }
}

impl fmt::Debug for FuncSignature {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct FuncDef {
    pub signature: FuncSignature,
    // todo: store bodies of user-defined funcs here?
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

#[derive(Debug, Clone, PartialEq)]
struct SymbolTable {
    vars: HashMap<String, Type>,
}

impl SymbolTable {
    fn new() -> Self {
        let mut vars = HashMap::new();
        vars.insert("x".to_string(), Type::scalar());
        vars.insert("y".to_string(), Type::scalar());
        vars.insert("xy".to_string(), Type::xy());

        vars.insert("r".to_string(), Type::scalar());
        vars.insert("a".to_string(), Type::scalar());

        vars.insert("t".to_string(), Type::scalar());

        vars.insert("pi".to_string(), Type::scalar());

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
                                ty: cast.clone(),
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
                // Rewrite the function name with the selected overload: this allows name
                // mangling since WGSL doesn't allow overloading.
                *name = func.def.signature.name.clone();

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
            Expression::TupleConst {
                tag, values, ty, ..
            } => {
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
                                Type::scalar(),
                            )
                        }
                        Type::Tuple(_, Arity::Sized(1)) => {}
                        Type::Tuple(_, Arity::Sized(n)) => {
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

                *ty = Type::Tuple(*tag, Arity::Sized(values.len()));

                Ok(())
            }
            Expression::If {
                condition,
                then,
                else_,
                ty,
            } => {
                self.analyze_expr(condition)?;
                if !matches!(condition.ty(), Type::Int | Type::Tuple(_, Arity::Sized(1))) {
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

                if !matches!(array.ty(), Type::Tuple(..)) {
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
                *ty = Type::scalar();
                Ok(())
            }
            Expression::Cast { expr, ty, tag } => {
                self.analyze_expr(expr)?;
                if expr.ty().len() != tag.len() {
                    return Err(TypeError::with_pos(
                        format!(
                            "can not cast tuple of {} values to tuple {} of {} values",
                            expr.ty().len(),
                            tag,
                            tag.len()
                        ),
                        0,
                        0,
                    ));
                }
                *ty = Type::Tuple(*tag, Arity::Sized(expr.ty().len()));
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
            assert_eq!(*ty, Type::scalar());
        } else {
            panic!("expected function call");
        }

        let expected_ast = E::FunctionCall {
            name: "__add".to_string(),
            args: vec![
                E::Cast {
                    tag: TupleTag::Nil,
                    expr: Box::new(E::int_(1)),
                    ty: Type::scalar(),
                },
                E::float_(2.0),
            ],
            ty: Type::scalar(),
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
            assert_eq!(args[0].ty(), Type::xy());
            assert_eq!(*ty, Type::xy());
        } else {
            panic!("expected function call");
        }
        Ok(())
    }

    #[test]
    fn quat_abs() -> Result<(), Box<dyn Error>> {
        let expr = &analyze_expr("abs(quat:[1, 2, 3, 4])")?[0];
        if let E::FunctionCall { name, args, ty } = expr {
            assert_eq!(name, "abs_quat");
            assert_eq!(args.len(), 1);
            assert_eq!(args[0].ty(), Type::quat());
            assert_eq!(*ty, Type::scalar());
        } else {
            panic!("expected function call");
        }
        Ok(())
    }

    #[test]
    fn polymorphic_binary() -> Result<(), Box<dyn Error>> {
        let expr = &analyze_expr("xy + xy")?[0];

        let expected_ast = E::FunctionCall {
            name: "__add".to_string(),
            args: vec![
                E::variable_ty_("xy", Type::xy()),
                E::variable_ty_("xy", Type::xy()),
            ],
            ty: Type::xy(),
        };

        assert_eq!(*expr, expected_ast);

        Ok(())
    }

    #[test]
    fn add_ri() -> Result<(), Box<dyn Error>> {
        let expr = &analyze_expr("ri:xy + ri:xy")?[0];
        if let E::FunctionCall { name, args, ty } = expr {
            assert_eq!(name, "add_ri_ri");
            assert_eq!(args.len(), 2);
            assert_eq!(args[0].ty(), Type::ri());
            assert_eq!(args[1].ty(), Type::ri());
            assert_eq!(*ty, Type::ri());
        } else {
            panic!("expected function call");
        }
        Ok(())
    }

    #[test]
    fn add_wrong_tag() -> Result<(), Box<dyn Error>> {
        if let Err(e) = analyze_expr("ri:xy + xy") {
            if let Some(TypeError(tye)) = e.downcast_ref::<TypeError>() {
                assert_eq!(
                    tye.message,
                    "no matching overload for function \"__add\" with args [ri:2, xy:2]"
                );
                Ok(())
            } else {
                panic!("expected type error");
            }
        } else {
            panic!("expected typechecking to fail");
        }
    }

    #[test]
    fn tuple_const() -> Result<(), Box<dyn Error>> {
        let expr = &analyze_expr("xy:[1.0, 2.0]")?[0];
        if let E::TupleConst { tag, values, ty } = expr {
            assert_eq!(*tag, TupleTag::Xy);
            assert_eq!(values.len(), 2);
            assert_eq!(*ty, Type::Tuple(TupleTag::Xy, Arity::Sized(2)));
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
                E::cast_with_ty_(TupleTag::Nil, E::int_(1), Type::scalar()),
                E::float_(2.0),
            ],
            ty: Type::Tuple(TupleTag::Xy, Arity::Sized(2)),
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
            panic!("expected typechecking to fail");
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
                    "no matching overload for function \"__add\" with args [xy:2, rgba:4]"
                );
                Ok(())
            } else {
                panic!("expected type error");
            }
        } else {
            panic!("expected typechecking to fail");
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
            ty: Type::Tuple(TupleTag::Xy, Arity::Sized(2)),
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
                E::cast_with_ty_(TupleTag::Nil, E::int_(1), Type::scalar()),
            ],
            ty: Type::Tuple(TupleTag::Xy, Arity::Sized(2)),
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
                    "then and else branches must have the same type, got int and nil:1"
                );
                Ok(())
            } else {
                panic!("expected type error");
            }
        } else {
            panic!("expected typechecking to fail");
        }
    }

    #[test]
    fn if_wrong_branch_type() -> Result<(), Box<dyn Error>> {
        if let Err(e) = analyze_expr("if xy then 2 else 3 end") {
            if let Some(TypeError(tye)) = e.downcast_ref::<TypeError>() {
                assert_eq!(tye.message, "if condition must be int, found xy:2");
                Ok(())
            } else {
                panic!("expected type error");
            }
        } else {
            panic!("expected typechecking to fail");
        }
    }

    #[test]
    fn index_wrong_array() -> Result<(), Box<dyn Error>> {
        if let Err(e) = analyze_expr("a = 1; a[1]") {
            if let Some(TypeError(tye)) = e.downcast_ref::<TypeError>() {
                assert_eq!(tye.message, "index target must be a tuple, found int");
                Ok(())
            } else {
                panic!("expected type error");
            }
        } else {
            panic!("expected typechecking to fail");
        }
    }

    #[test]
    fn index_wrong_index() -> Result<(), Box<dyn Error>> {
        if let Err(e) = analyze_expr("a = xy:[1, 2]; a[1.0]") {
            if let Some(TypeError(tye)) = e.downcast_ref::<TypeError>() {
                assert_eq!(tye.message, "index variable must be an int, found nil:1");
                Ok(())
            } else {
                panic!("expected type error");
            }
        } else {
            panic!("expected typechecking to fail");
        }
    }

    #[test]
    fn index_ok() -> Result<(), Box<dyn Error>> {
        let expr = &analyze_expr("a = xy:[1, 2]; a[1]")?[1];
        assert_eq!(expr.ty(), Type::scalar());
        Ok(())
    }

    #[test]
    fn cast_ok() -> Result<(), Box<dyn Error>> {
        let expr = &analyze_expr("a = xy:[1, 2]; b = ri:a")?[1];
        assert_eq!(expr.ty(), Type::Tuple(TupleTag::Ri, Arity::Sized(2)));
        Ok(())
    }

    #[test]
    fn cast_wrong_arity() -> Result<(), Box<dyn Error>> {
        if let Err(e) = analyze_expr("a = xy:[1, 2]; b = rgba:a") {
            if let Some(TypeError(tye)) = e.downcast_ref::<TypeError>() {
                assert_eq!(
                    tye.message,
                    "can not cast tuple of 2 values to tuple rgba of 4 values"
                );
                Ok(())
            } else {
                panic!("expected type error");
            }
        } else {
            panic!("expected typechecking to fail");
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
                assert_eq!(tye.message, "while condition must be int, found nil:1");
                Ok(())
            } else {
                panic!("expected type error");
            }
        } else {
            panic!("expected typechecking to fail");
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
        let hello_fn = fns.lookup("rgbColor", &vec![Type::scalar(); 3])?;

        let OverloadResolutionResult { def, .. } = &hello_fn;
        assert_eq!(def.signature.name, "rgbColor");
        assert_eq!(def.signature.params.len(), 3);
        assert_eq!(
            def.signature.ret,
            Type::Tuple(TupleTag::Rgba, Arity::Sized(4))
        );
        assert_eq!(_num_casts(&hello_fn), 0);

        Ok(())
    }

    #[test]
    fn lookup_cast() -> Result<(), Box<dyn Error>> {
        let fns = FunctionTable::new();
        let hello_fn = fns.lookup("rgbColor", &vec![Type::Int, Type::scalar(), Type::Int])?;

        let OverloadResolutionResult { def, .. } = &hello_fn;
        assert_eq!(def.signature.name, "rgbColor");
        assert_eq!(def.signature.params.len(), 3);
        assert_eq!(
            def.signature.ret,
            Type::Tuple(TupleTag::Rgba, Arity::Sized(4))
        );
        assert_eq!(_num_casts(&hello_fn), 2);

        Ok(())
    }
}
