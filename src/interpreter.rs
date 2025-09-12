#![allow(dead_code)]

use crate::ast;
use crate::ast::TupleTag;

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
    always_as_float: bool,
) -> Value
where
    FloatOp: Fn(f32, f32) -> f32,
    IntOp: Fn(i64, i64) -> i64,
{
    // Handle broadcasting. Vector types are always float, so we only need to promote
    // the scalar side.
    match (lhs.len(), rhs.len()) {
        (1, 1) => {
            if always_as_float || needs_float(lhs, rhs) {
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

fn quat_mul(a: &Vec<f32>, b: &Vec<f32>) -> Vec<f32> {
    let w = a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3];
    let x = a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2];
    let y = a[0] * b[2] + a[2] * b[0] - a[1] * b[3] + a[3] * b[1];
    let z = a[0] * b[3] + a[3] * b[0] + a[1] * b[2] - a[2] * b[1];
    vec![w, x, y, z]
}

fn eval_function_call(name: &str, args: &Vec<ast::Expression>, env: &mut Environment) -> Value {
    // Logical and needs special logic for short-circuiting.
    if name == "__and" {
        assert!(args.len() == 2);

        if let Value::Int(x) = eval_expression(&args[0], env) {
            if x != 0 {
                if let Value::Int(y) = eval_expression(&args[1], env) {
                    if y != 0 {
                        return Value::Int(1);
                    } else {
                        return Value::Int(0);
                    }
                } else {
                    panic!("and operator expects int arguments, found {:?}", args[1]);
                }
            } else {
                return Value::Int(0);
            }
        } else {
            panic!("and operator expects int arguments, found {:?}", args[0]);
        }
    }

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
            eval_binary_op(
                &args[0],
                &args[1],
                std::ops::Add::add,
                std::ops::Add::add,
                false,
            )
        }
        "__sub" => {
            assert!(args.len() == 2);
            eval_binary_op(
                &args[0],
                &args[1],
                std::ops::Sub::sub,
                std::ops::Sub::sub,
                false,
            )
        }
        "__mul" => {
            assert!(args.len() == 2);

            match (&args[0], &args[1]) {
                (Value::Tuple(TupleTag::Quat, q1), Value::Tuple(TupleTag::Quat, q2)) => {
                    Value::Tuple(TupleTag::Quat, quat_mul(q1, q2))
                }
                _ => eval_binary_op(
                    &args[0],
                    &args[1],
                    std::ops::Mul::mul,
                    std::ops::Mul::mul,
                    false,
                ),
            }
        }
        "__div" => {
            assert!(args.len() == 2);
            eval_binary_op(
                &args[0],
                &args[1],
                std::ops::Div::div,
                std::ops::Div::div,
                true,
            )
        }
        "__mod" => {
            assert!(args.len() == 2);
            eval_binary_op(
                &args[0],
                &args[1],
                std::ops::Rem::rem,
                std::ops::Rem::rem,
                false,
            )
        }
        "__or" => {
            assert!(args.len() == 2);
            // Only supports scalar ints for now, should we promote floats?
            match (&args[0], &args[1]) {
                (Value::Int(x), Value::Int(y)) => {
                    Value::Int(if *x != 0 || *y != 0 { 1 } else { 0 })
                }
                _ => panic!(
                    "__or only supports int inputs, found {:?} and {:?}",
                    args[0], args[1]
                ),
            }
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
        "__lessequal" => {
            assert!(args.len() == 2);

            match (&args[0], &args[1]) {
                (Value::Int(x), Value::Int(y)) => Value::Int(if x <= y { 1 } else { 0 }),
                _ => {
                    let a = promote_scalar_to_float(args[0].clone());
                    let b = promote_scalar_to_float(args[1].clone());
                    let res = a <= b;
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
                Value::Tuple(TupleTag::Quat, data) => {
                    let mag = data.iter().map(|x| x * x).sum::<f32>().sqrt();
                    Value::Tuple(TupleTag::Nil, vec![mag])
                }
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
        "log" => {
            assert!(args.len() == 1);

            let a = args[0].clone();

            match a {
                Value::Tuple(tag, data) => Value::Tuple(tag, data.iter().map(|x| x.ln()).collect()),
                Value::Int(x) => Value::Tuple(TupleTag::Nil, vec![f32::from(x as f32).ln()]),
            }
        }
        "min" => {
            assert!(args.len() == 2);
            eval_binary_op(
                &args[0],
                &args[1],
                |x, y| x.min(y),
                |x, y| x.min(y),
                false,
            )
        }
        "max" => {
            assert!(args.len() == 2);
            eval_binary_op(
                &args[0],
                &args[1],
                |x, y| x.max(y),
                |x, y| x.max(y),
                false,
            )
        }
        _ => panic!("unimplemented function {}", name),
    }
}

fn eval_expression(expr: &ast::Expression, env: &mut Environment) -> Value {
    match expr {
        ast::Expression::IntConst { value } => Value::Int(*value),
        ast::Expression::FloatConst { value } => Value::Tuple(TupleTag::Nil, vec![*value as f32]),
        ast::Expression::TupleConst { tag, values } => {
            // Here we assume that all expressions inside a tuple literal evaluate to a scalar
            // (that we promote to float to store in the tuple).
            Value::Tuple(
                *tag,
                values
                    .iter()
                    .map(|x| promote_scalar_to_float(eval_expression(x, env)))
                    .collect(),
            )
        }
        ast::Expression::Cast { tag, expr } => {
            let expr_val = eval_expression(&expr, env);
            if let Value::Tuple(_, data) = expr_val {
                Value::Tuple(*tag, data)
            } else {
                panic!(
                    "cast expression must evaluate to a tuple, got {:?}",
                    expr_val
                );
            }
        }
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
                    eval_expr_block(then, env)
                } else if !else_.is_empty() {
                    eval_expr_block(else_, env)
                } else {
                    todo!();
                }
            } else {
                panic!("condition is not an int");
            }
        }
        ast::Expression::While { condition, body } => {
            loop {
                let cond_result = eval_expression(condition, env);

                if let Value::Int(x) = cond_result {
                    if x != 0 {
                        eval_expr_block(body, env);
                    } else {
                        break;
                    }
                } else {
                    panic!("condition is not an int");
                }
            }

            // A while loop always evaluates to 0 according to the docs.
            Value::Int(0)
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

fn eval_expr_block(exprs: &Vec<ast::Expression>, env: &mut Environment) -> Value {
    assert!(!exprs.is_empty());

    let mut result: Option<Value> = None;

    for expr in exprs {
        result = Some(eval_expression(expr, env));
    }

    result.unwrap()
}

fn eval_filter_impl(filter: &ast::Filter, env: &mut Environment) -> Value {
    return eval_expr_block(&filter.exprs, env);
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

    env.values
        .insert("xy".to_string(), Value::Tuple(TupleTag::Xy, vec![x, y]));

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

    #[test]
    fn test_less() {
        let input = "1 < 1";
        let mut parser = Parser::new(input);
        let ast = parser.parse_expression(1).unwrap();
        let mut env = Environment::new();
        let val = eval_expression(&ast, &mut env);
        let val_expected = Value::Int(0);
        assert_eq!(val, val_expected);
    }

    #[test]
    fn test_lessequal() {
        let input = "1 <= 1";
        let mut parser = Parser::new(input);
        let ast = parser.parse_expression(1).unwrap();
        let mut env = Environment::new();
        let val = eval_expression(&ast, &mut env);
        let val_expected = Value::Int(1);
        assert_eq!(val, val_expected);
    }

    #[test]
    fn test_or() {
        let input = "0 || 1";
        let mut parser = Parser::new(input);
        let ast = parser.parse_expression(1).unwrap();
        let mut env = Environment::new();
        let val = eval_expression(&ast, &mut env);
        let val_expected = Value::Int(1);
        assert_eq!(val, val_expected);
    }

    #[test]
    fn test_and() {
        let input = "1 && 2";
        let mut parser = Parser::new(input);
        let ast = parser.parse_expression(1).unwrap();
        let mut env = Environment::new();
        let val = eval_expression(&ast, &mut env);
        let val_expected = Value::Int(1);
        assert_eq!(val, val_expected);
    }

    #[test]
    fn test_tuple_build() {
        let input = "rgba:[1, 2, 3, 4]";
        let mut parser = Parser::new(input);
        let ast = parser.parse_expression(1).unwrap();
        let mut env = Environment::new();
        let val = eval_expression(&ast, &mut env);
        let val_expected = Value::Tuple(TupleTag::Rgba, vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(val, val_expected);
    }

    #[test]
    fn test_if() {
        let input = "if x < 100 then y = 10; y else 200 end";
        let mut parser = Parser::new(input);
        let ast = parser.parse_expression(1).unwrap();
        let mut env = Environment::new();
        env.values.insert("x".to_string(), Value::Int(1));
        let val = eval_expression(&ast, &mut env);
        let val_expected = Value::Int(10);
        assert_eq!(val, val_expected);
    }

    #[test]
    fn test_while() {
        let input = "i = 0; while i < 10 do i = i + 1; end; i";
        let mut parser = Parser::new(input);
        let exprs = parser.parse_expr_block().unwrap();
        let mut env = Environment::new();
        let val = eval_expr_block(&exprs, &mut env);
        let val_expected = Value::Int(10);
        assert_eq!(val, val_expected);
    }

    #[test]
    fn test_cast() {
        let input = "ri:xy";
        let mut parser = Parser::new(input);
        let ast = parser.parse_expression(1).unwrap();
        let mut env = Environment::new();
        env.values
            .insert("xy".to_string(), Value::Tuple(TupleTag::Xy, vec![1.0, 2.0]));
        let val = eval_expression(&ast, &mut env);
        dbg!(&val);
        let val_expected = Value::Tuple(TupleTag::Ri, vec![1.0, 2.0]);
        assert_eq!(val, val_expected);
    }
}
