#![allow(dead_code)]

use crate::ast;

use core::panic;
use std::collections::HashMap;

#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    Tuple(TupleTag, Vec<f32>),
    Int(i64),
}

impl Value {
    pub fn len(&self) -> usize {
        match self {
            Value::Tuple(_, data) => data.len(),
            Value::Int(_) => 1,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TupleTag {
    Nil,   // Default tag for single numbers
    Rgba,  // RGBA color
    Hsva,  // HSVA color
    Ri,    // Complex number
    Xy,    // Cartesian coordinates
    Ra,    // Polar coordinates
    V2,    // 2D vector
    V3,    // 3D vector
    M2x2,  // 2x2 matrix
    M3x3,  // 3x3 matrix
    Quat,  // Non-commutative quaternion
    Cquat, // Commutative quaternion
    Hyper, // Hypercomplex number
}

#[derive(Debug, Clone)]
struct Environment {
    values: HashMap<String, Value>,
}

impl Environment {
    fn new() -> Self {
        Self {
            values: HashMap::new(),
        }
    }
}

fn promote_scalar_to_float(value: Value) -> f32 {
    match value {
        Value::Tuple(_, data) => {
            assert!(data.len() == 1);
            data[0]
        }
        Value::Int(x) => x as f32,
    }
}

fn needs_float(lhs: &Value, rhs: &Value) -> bool {
    match (lhs, rhs) {
        (Value::Int(..), Value::Int(..)) => false,
        _ => true,
    }
}

fn eval_binary_op<FloatOp, IntOp>(
    lhs: &Value,
    rhs: &Value,
    float_op: FloatOp,
    int_op: IntOp,
) -> Value
where
    FloatOp: Fn(f32, f32) -> f32,
    IntOp: Fn(i64, i64) -> i64,
{
    // Handle broadcasting. Vector types are always float, so we only need to promote
    // the scalar side.
    match (lhs.len(), rhs.len()) {
        (1, 1) => {
            if needs_float(lhs, rhs) {
                let a = promote_scalar_to_float(lhs.clone());
                let b = promote_scalar_to_float(rhs.clone());
                Value::Tuple(TupleTag::Nil, vec![float_op(a, b)])
            } else {
                if let (Value::Int(x), Value::Int(y)) = (lhs, rhs) {
                    Value::Int(int_op(*x, *y))
                } else {
                    panic!();
                }
            }
        }
        (1, _) => {
            if needs_float(lhs, rhs) {
                let a = promote_scalar_to_float(lhs.clone());
                if let Value::Tuple(tag, data) = rhs.clone() {
                    Value::Tuple(tag, data.iter().map(|x| float_op(a, *x)).collect())
                } else {
                    panic!();
                }
            } else {
                todo!("broadcasting not implemented for int arguments.")
            }
        }
        (_, 1) => {
            if needs_float(lhs, rhs) {
                let b = promote_scalar_to_float(rhs.clone());
                if let Value::Tuple(tag, data) = lhs.clone() {
                    Value::Tuple(tag, data.iter().map(|x| float_op(*x, b)).collect())
                } else {
                    panic!();
                }
            } else {
                todo!("broadcasting not implemented for int arguments.");
            }
        }
        (lhslen, rhslen) => {
            if lhslen != rhslen {
                panic!("mismatched vector lengths");
            }

            if let (Value::Tuple(lhs_tag, lhs_data), Value::Tuple(rhs_tag, rhs_data)) = (lhs, rhs) {
                if lhs_tag != rhs_tag {
                    panic!("mismatched tuple tags");
                }

                Value::Tuple(
                    *lhs_tag,
                    lhs_data
                        .iter()
                        .zip(rhs_data)
                        .map(|(x, y)| float_op(*x, *y))
                        .collect(),
                )
            } else {
                panic!("unexpected")
            }
        }
    }
}

fn eval_function_call(name: &str, args: &Vec<ast::Expression>, env: &mut Environment) -> Value {
    let args = args
        .iter()
        .map(|arg| eval_expression(arg, env))
        .collect::<Vec<_>>();

    match name {
        "rgbColor" => {
            assert!(args.len() == 3);

            let r = promote_scalar_to_float(args[0].clone());
            let g = promote_scalar_to_float(args[1].clone());
            let b = promote_scalar_to_float(args[2].clone());

            Value::Tuple(TupleTag::Rgba, vec![r, g, b, 1.0])
        }
        "rgbaColor" => {
            assert!(args.len() == 4);

            let r = promote_scalar_to_float(args[0].clone());
            let g = promote_scalar_to_float(args[1].clone());
            let b = promote_scalar_to_float(args[2].clone());
            let a = promote_scalar_to_float(args[3].clone());

            Value::Tuple(TupleTag::Rgba, vec![r, g, b, a])
        }
        "grayColor" => {
            assert!(args.len() == 1);

            let x = promote_scalar_to_float(args[0].clone());

            Value::Tuple(TupleTag::Rgba, vec![x, x, x, 1.0])
        }
        "__add" => {
            assert!(args.len() == 2);
            eval_binary_op(&args[0], &args[1], std::ops::Add::add, std::ops::Add::add)
        }
        "__sub" => {
            assert!(args.len() == 2);
            eval_binary_op(&args[0], &args[1], std::ops::Sub::sub, std::ops::Sub::sub)
        }
        "__mul" => {
            assert!(args.len() == 2);
            eval_binary_op(&args[0], &args[1], std::ops::Mul::mul, std::ops::Mul::mul)
        }
        "__div" => {
            assert!(args.len() == 2);
            eval_binary_op(&args[0], &args[1], std::ops::Div::div, std::ops::Div::div)
        }
        "__mod" => {
            assert!(args.len() == 2);
            eval_binary_op(&args[0], &args[1], std::ops::Rem::rem, std::ops::Rem::rem)
        }
        "__less" => {
            assert!(args.len() == 2);

            match (&args[0], &args[1]) {
                (Value::Int(x), Value::Int(y)) => Value::Int(if x < y { 1 } else { 0 }),
                _ => {
                    let a = promote_scalar_to_float(args[0].clone());
                    let b = promote_scalar_to_float(args[1].clone());
                    let res = a < b;
                    Value::Int(if res { 1 } else { 0 })
                }
            }
        }
        "__neg" => {
            assert!(args.len() == 1);

            let a = args[0].clone();

            match a {
                Value::Tuple(tag, data) => Value::Tuple(tag, data.iter().map(|x| -x).collect()),
                Value::Int(x) => Value::Int(-x),
            }
        }
        "abs" => {
            assert!(args.len() == 1);

            let a = args[0].clone();

            match a {
                Value::Tuple(tag, data) => {
                    Value::Tuple(tag, data.iter().map(|x| x.abs()).collect())
                }
                Value::Int(x) => Value::Int(x.abs()),
            }
        }
        "sin" => {
            assert!(args.len() == 1);

            let a = args[0].clone();

            match a {
                Value::Tuple(tag, data) => {
                    Value::Tuple(tag, data.iter().map(|x| x.sin()).collect())
                }
                Value::Int(x) => Value::Tuple(TupleTag::Nil, vec![f32::from(x as f32).sin()]),
            }
        }
        _ => panic!("unimplemented function {}", name),
    }
}

fn eval_expression(expr: &ast::Expression, env: &mut Environment) -> Value {
    match expr {
        ast::Expression::IntConst { value } => Value::Int(*value),
        ast::Expression::FloatConst { value } => Value::Tuple(TupleTag::Nil, vec![*value as f32]),
        ast::Expression::FunctionCall { name, args } => eval_function_call(&name, &args, env),
        ast::Expression::Variable { name } => {
            if let Some(val) = env.values.get(name) {
                return val.clone();
            } else {
                panic!("variable {} not found", name);
            }
        }
        ast::Expression::If {
            condition,
            then,
            else_,
        } => {
            let cond_result = eval_expression(condition, env);

            if let Value::Int(x) = cond_result {
                if x != 0 {
                    eval_expression(then, env)
                } else if let Some(else_expr) = else_ {
                    eval_expression(else_expr, env)
                } else {
                    todo!();
                }
            } else {
                panic!("condition is not an int");
            }
        }
        ast::Expression::Assignment { name, value } => {
            let value = eval_expression(value, env);
            env.values.insert(name.clone(), value.clone());
            value
        }
        ast::Expression::Index { expr, index } => {
            let expr = eval_expression(expr, env);
            let index = eval_expression(index, env);
            if let Value::Tuple(_, data) = expr {
                if let Value::Int(idx) = index {
                    if idx < 0 || idx >= data.len() as i64 {
                        panic!(
                            "index out of bounds, got {} but tuple has {} elements",
                            idx,
                            data.len()
                        );
                    }
                    Value::Tuple(TupleTag::Nil, vec![data[idx as usize].clone()])
                } else {
                    panic!("index must be int, got {:?}", index);
                }
            } else {
                panic!("only tuples can be indexed, got {:?}", expr);
            }
        }
    }
}

fn eval_filter_impl(filter: &ast::Filter, env: &mut Environment) -> Value {
    let mut exprs = filter.exprs.iter().peekable();

    while let Some(expr) = exprs.next() {
        let result = eval_expression(expr, env);

        if exprs.peek().is_none() {
            return result;
        }
    }

    panic!("unreachable");
}

fn to_ra(x: f32, y: f32) -> (f32, f32) {
    let r = x.hypot(y);
    let mut a = y.atan2(x);
    if a < 0.0 {
        // Shift from [-pi, pi) to [0, 2*pi].
        a += 2.0 * std::f32::consts::PI;
    }
    (r, a)
}

pub fn eval_filter(filter: &ast::Filter, x: f32, y: f32, t: f32) -> Value {
    let mut env = Environment::new();

    let mut bind_float_scalar = |name: &str, value: f32| {
        let tup = Value::Tuple(TupleTag::Nil, vec![value]);
        env.values.insert(name.to_string(), tup);
    };

    bind_float_scalar("x", x);
    bind_float_scalar("y", y);
    bind_float_scalar("t", t);
    bind_float_scalar("pi", std::f32::consts::PI);

    let (r, a) = to_ra(x, y);
    bind_float_scalar("r", r);
    bind_float_scalar("a", a);

    eval_filter_impl(filter, &mut env)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::Parser;

    #[test]
    fn test_float_promotion() {
        let input = "1 + 2.0";
        let mut parser = Parser::new(input);
        let ast = parser.parse_expression(1).unwrap();
        let mut env = Environment::new();
        let val = eval_expression(&ast, &mut env);

        let val_expected = Value::Tuple(TupleTag::Nil, vec![3.0]);
        assert_eq!(val, val_expected);
    }

    #[test]
    fn test_int_op() {
        let input = "1 + 2";
        let mut parser = Parser::new(input);
        let ast = parser.parse_expression(1).unwrap();
        let mut env = Environment::new();
        let val = eval_expression(&ast, &mut env);
        let val_expected = Value::Int(3);
        assert_eq!(val, val_expected);
    }

    #[test]
    fn test_broadcast() {
        let input = "1 + rgba:[2, 3, 4, 1]";
        let mut parser = Parser::new(input);
        let ast = parser.parse_expression(1).unwrap();
        let mut env = Environment::new();
        let val = eval_expression(&ast, &mut env);
        let val_expected = Value::Tuple(TupleTag::Rgba, vec![3.0, 4.0, 5.0, 2.0]);
        assert_eq!(val, val_expected);
    }

    #[test]
    fn test_broadcast_2() {
        let input = "rgba:[2, 3, 4, 0.5] * 0.5";
        let mut parser = Parser::new(input);
        let ast = parser.parse_expression(1).unwrap();
        let mut env = Environment::new();
        let val = eval_expression(&ast, &mut env);
        let val_expected = Value::Tuple(TupleTag::Rgba, vec![1.0, 1.5, 2.0, 0.25]);
        assert_eq!(val, val_expected);
    }

    #[test]
    fn test_elementwise() {
        let input = "rgba:[1, 2, 3, 4] + rgba:[5, 6, 7, 8]";
        let mut parser = Parser::new(input);
        let ast = parser.parse_expression(1).unwrap();
        let mut env = Environment::new();
        let val = eval_expression(&ast, &mut env);
        let val_expected = Value::Tuple(TupleTag::Rgba, vec![6.0, 8.0, 10.0, 12.0]);
        assert_eq!(val, val_expected);
    }

    #[test]
    fn test_index() {
        let input = "(rgba:[1, 2, 3, 4])[1]";
        let mut parser = Parser::new(input);
        let ast = parser.parse_expression(1).unwrap();
        let mut env = Environment::new();
        let val = eval_expression(&ast, &mut env);
        let val_expected = Value::Tuple(TupleTag::Nil, vec![2.0]);
        assert_eq!(val, val_expected);
    }
}
