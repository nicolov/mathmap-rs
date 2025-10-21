#![allow(dead_code)]

use crate::ast;
use crate::ast::TupleTag;
use crate::err::RuntimeError;

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

fn promote_scalar_to_float(value: &Value) -> Result<f32, RuntimeError> {
    match value {
        Value::Tuple(_, data) => {
            if data.len() == 1 {
                Ok(data[0])
            } else {
                Err(RuntimeError::with_pos(
                    format!("expected scalar tuple len 1, got {}", data.len()),
                    0,
                    0,
                ))
            }
        }
        Value::Int(x) => Ok(*x as f32),
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
) -> Result<Value, RuntimeError>
where
    FloatOp: Fn(f32, f32) -> f32,
    IntOp: Fn(i64, i64) -> i64,
{
    // Handle broadcasting. Vector types are always float, so we only need to promote
    // the scalar side. Promote/cast first, broadcast later, to avoid broadcasting the
    // wrong type.
    match (lhs.len(), rhs.len()) {
        (1, 1) => {
            if always_as_float || needs_float(lhs, rhs) {
                let a = promote_scalar_to_float(lhs)?;
                let b = promote_scalar_to_float(rhs)?;
                Ok(Value::Tuple(TupleTag::Nil, vec![float_op(a, b)]))
            } else {
                if let (Value::Int(x), Value::Int(y)) = (lhs, rhs) {
                    Ok(Value::Int(int_op(*x, *y)))
                } else {
                    Err(RuntimeError::with_pos(
                        "internal type mismatch in integer op",
                        0,
                        0,
                    ))
                }
            }
        }
        (1, _) => {
            if needs_float(lhs, rhs) {
                let a = promote_scalar_to_float(lhs)?;
                if let Value::Tuple(tag, data) = rhs.clone() {
                    Ok(Value::Tuple(
                        tag,
                        data.iter().map(|x| float_op(a, *x)).collect(),
                    ))
                } else {
                    Err(RuntimeError::with_pos(
                        "unexpected non-tuple RHS in broadcasting",
                        0,
                        0,
                    ))
                }
            } else {
                Err(RuntimeError::with_pos(
                    "broadcasting not implemented for int arguments",
                    0,
                    0,
                ))
            }
        }
        (_, 1) => {
            if needs_float(lhs, rhs) {
                let b = promote_scalar_to_float(rhs)?;
                if let Value::Tuple(tag, data) = lhs.clone() {
                    Ok(Value::Tuple(
                        tag,
                        data.iter().map(|x| float_op(*x, b)).collect(),
                    ))
                } else {
                    Err(RuntimeError::with_pos(
                        "unexpected non-tuple LHS in broadcasting",
                        0,
                        0,
                    ))
                }
            } else {
                Err(RuntimeError::with_pos(
                    "broadcasting not implemented for int arguments",
                    0,
                    0,
                ))
            }
        }
        (lhslen, rhslen) => {
            if lhslen != rhslen {
                return Err(RuntimeError::with_pos(
                    format!("mismatched vector lengths: {} vs {}", lhslen, rhslen),
                    0,
                    0,
                ));
            }

            if let (Value::Tuple(lhs_tag, lhs_data), Value::Tuple(rhs_tag, rhs_data)) = (lhs, rhs) {
                if lhs_tag != rhs_tag {
                    return Err(RuntimeError::with_pos("mismatched tuple tags", 0, 0));
                }

                Ok(Value::Tuple(
                    *lhs_tag,
                    lhs_data
                        .iter()
                        .zip(rhs_data)
                        .map(|(x, y)| float_op(*x, *y))
                        .collect(),
                ))
            } else {
                Err(RuntimeError::with_pos(
                    "unexpected non-tuple operands in vector op",
                    0,
                    0,
                ))
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

fn eval_function_call(
    name: &str,
    args: &Vec<ast::Expression>,
    env: &mut Environment,
) -> Result<Value, RuntimeError> {
    // Logical and needs special logic for short-circuiting.
    if name == "__and" {
        assert!(args.len() == 2);

        if let Value::Int(x) = eval_expression(&args[0], env)? {
            if x != 0 {
                if let Value::Int(y) = eval_expression(&args[1], env)? {
                    if y != 0 {
                        return Ok(Value::Int(1));
                    } else {
                        return Ok(Value::Int(0));
                    }
                } else {
                    return Err(RuntimeError::with_pos(
                        format!("and operator expects int arguments, found {:?}", args[1]),
                        0,
                        0,
                    ));
                }
            } else {
                return Ok(Value::Int(0));
            }
        } else {
            return Err(RuntimeError::with_pos(
                format!("and operator expects int arguments, found {:?}", args[0]),
                0,
                0,
            ));
        }
    }

    let args = args
        .iter()
        .map(|arg| eval_expression(arg, env))
        .collect::<Result<Vec<_>, _>>()?;

    match name {
        "rgbColor" => {
            assert!(args.len() == 3);

            let r = promote_scalar_to_float(&args[0])?;
            let g = promote_scalar_to_float(&args[1])?;
            let b = promote_scalar_to_float(&args[2])?;

            Ok(Value::Tuple(TupleTag::Rgba, vec![r, g, b, 1.0]))
        }
        "rgbaColor" => {
            assert!(args.len() == 4);

            let r = promote_scalar_to_float(&args[0])?;
            let g = promote_scalar_to_float(&args[1])?;
            let b = promote_scalar_to_float(&args[2])?;
            let a = promote_scalar_to_float(&args[3])?;

            Ok(Value::Tuple(TupleTag::Rgba, vec![r, g, b, a]))
        }
        "grayColor" => {
            assert!(args.len() == 1);

            let x = promote_scalar_to_float(&args[0])?;

            Ok(Value::Tuple(TupleTag::Rgba, vec![x, x, x, 1.0]))
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
                    Ok(Value::Tuple(TupleTag::Quat, quat_mul(q1, q2)))
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
        "__pow" => {
            assert!(args.len() == 2);
            eval_binary_op(
                &args[0],
                &args[1],
                |x, y| x.powf(y),
                |x, y| x.pow(y.try_into().unwrap()),
                false,
            )
        }
        "__or" => {
            assert!(args.len() == 2);
            // Only supports scalar ints for now, should we promote floats?
            match (&args[0], &args[1]) {
                (Value::Int(x), Value::Int(y)) => {
                    Ok(Value::Int(if *x != 0 || *y != 0 { 1 } else { 0 }))
                }
                _ => Err(RuntimeError::with_pos(
                    format!(
                        "__or only supports int inputs, found {:?} and {:?}",
                        args[0], args[1]
                    ),
                    0,
                    0,
                )),
            }
        }
        "__less" => {
            assert!(args.len() == 2);

            match (&args[0], &args[1]) {
                (Value::Int(x), Value::Int(y)) => Ok(Value::Int(if x < y { 1 } else { 0 })),
                _ => {
                    let a = promote_scalar_to_float(&args[0])?;
                    let b = promote_scalar_to_float(&args[1])?;
                    let res = a < b;
                    Ok(Value::Int(if res { 1 } else { 0 }))
                }
            }
        }
        "__lessequal" => {
            assert!(args.len() == 2);

            match (&args[0], &args[1]) {
                (Value::Int(x), Value::Int(y)) => Ok(Value::Int(if x <= y { 1 } else { 0 })),
                _ => {
                    let a = promote_scalar_to_float(&args[0])?;
                    let b = promote_scalar_to_float(&args[1])?;
                    let res = a <= b;
                    Ok(Value::Int(if res { 1 } else { 0 }))
                }
            }
        }
        "__neg" => {
            assert!(args.len() == 1);

            let a = args[0].clone();

            match a {
                Value::Tuple(tag, data) => Ok(Value::Tuple(tag, data.iter().map(|x| -x).collect())),
                Value::Int(x) => Ok(Value::Int(-x)),
            }
        }
        "abs" => {
            assert!(args.len() == 1);

            let a = args[0].clone();

            match a {
                Value::Tuple(TupleTag::Quat, data) => {
                    let mag = data.iter().map(|x| x * x).sum::<f32>().sqrt();
                    Ok(Value::Tuple(TupleTag::Nil, vec![mag]))
                }
                Value::Tuple(tag, data) => {
                    Ok(Value::Tuple(tag, data.iter().map(|x| x.abs()).collect()))
                }
                Value::Int(x) => Ok(Value::Int(x.abs())),
            }
        }
        "sin" => {
            assert!(args.len() == 1);

            let a = args[0].clone();

            match a {
                Value::Tuple(tag, data) => {
                    Ok(Value::Tuple(tag, data.iter().map(|x| x.sin()).collect()))
                }
                Value::Int(x) => Ok(Value::Tuple(TupleTag::Nil, vec![f32::from(x as f32).sin()])),
            }
        }
        "log" => {
            assert!(args.len() == 1);

            let a = args[0].clone();

            match a {
                Value::Tuple(tag, data) => {
                    Ok(Value::Tuple(tag, data.iter().map(|x| x.ln()).collect()))
                }
                Value::Int(x) => Ok(Value::Tuple(TupleTag::Nil, vec![f32::from(x as f32).ln()])),
            }
        }
        "sqrt" => {
            assert!(args.len() == 1);
            let a = args[0].clone();
            match a {
                Value::Tuple(tag, data) => {
                    Ok(Value::Tuple(tag, data.iter().map(|x| x.sqrt()).collect()))
                }
                Value::Int(x) => Ok(Value::Tuple(
                    TupleTag::Nil,
                    vec![f32::from(x as f32).sqrt()],
                )),
            }
        }
        "min" => {
            assert!(args.len() == 2);
            eval_binary_op(&args[0], &args[1], |x, y| x.min(y), |x, y| x.min(y), false)
        }
        "max" => {
            assert!(args.len() == 2);
            eval_binary_op(&args[0], &args[1], |x, y| x.max(y), |x, y| x.max(y), false)
        }
        _ => Err(RuntimeError::with_pos(
            format!("unimplemented function {}", name),
            0,
            0,
        )),
    }
}

fn eval_expression(expr: &ast::Expression, env: &mut Environment) -> Result<Value, RuntimeError> {
    match expr {
        ast::Expression::IntConst { value, .. } => Ok(Value::Int(*value)),
        ast::Expression::FloatConst { value, .. } => {
            Ok(Value::Tuple(TupleTag::Nil, vec![*value as f32]))
        }
        ast::Expression::TupleConst { tag, values, .. } => {
            // Here we assume that all expressions inside a tuple literal evaluate to a scalar
            // (that we promote to float to store in the tuple).
            let mut out: Vec<f32> = Vec::with_capacity(values.len());
            for x in values {
                let v = eval_expression(x, env)?;
                out.push(promote_scalar_to_float(&v)?);
            }
            Ok(Value::Tuple(*tag, out))
        }
        ast::Expression::Cast { tag, expr, .. } => {
            let expr_val = eval_expression(&expr, env)?;
            if let Value::Tuple(_, data) = expr_val {
                Ok(Value::Tuple(*tag, data))
            } else {
                Err(RuntimeError::with_pos(
                    format!(
                        "cast expression must evaluate to a tuple, got {:?}",
                        expr_val
                    ),
                    0,
                    0,
                ))
            }
        }
        ast::Expression::FunctionCall { name, args, .. } => eval_function_call(&name, &args, env),
        ast::Expression::Variable { name, .. } => {
            if let Some(val) = env.values.get(name) {
                Ok(val.clone())
            } else {
                Err(RuntimeError::with_pos(
                    format!("variable {} not found", name),
                    0,
                    0,
                ))
            }
        }
        ast::Expression::If {
            condition,
            then,
            else_,
            ..
        } => {
            let cond_result = eval_expression(condition, env)?;

            if let Value::Int(x) = cond_result {
                if x != 0 {
                    eval_expr_block(then, env)
                } else if !else_.is_empty() {
                    eval_expr_block(else_, env)
                } else {
                    Err(RuntimeError::with_pos("empty else branch", 0, 0))
                }
            } else {
                Err(RuntimeError::with_pos("condition is not an int", 0, 0))
            }
        }
        ast::Expression::While {
            condition, body, ..
        } => {
            loop {
                let cond_result = eval_expression(condition, env)?;

                if let Value::Int(x) = cond_result {
                    if x != 0 {
                        let _ = eval_expr_block(body, env)?;
                    } else {
                        break;
                    }
                } else {
                    return Err(RuntimeError::with_pos("condition is not an int", 0, 0));
                }
            }

            // A while loop always evaluates to 0 according to the docs.
            Ok(Value::Int(0))
        }
        ast::Expression::Assignment { name, value, .. } => {
            let value = eval_expression(value, env)?;
            env.values.insert(name.clone(), value.clone());
            Ok(value)
        }
        ast::Expression::Index { expr, index, .. } => {
            let expr = eval_expression(expr, env)?;
            let index = eval_expression(index, env)?;
            if let Value::Tuple(_, data) = expr {
                if let Value::Int(idx) = index {
                    if idx < 0 || idx >= data.len() as i64 {
                        return Err(RuntimeError::with_pos(
                            format!(
                                "index out of bounds, got {} but tuple has {} elements",
                                idx,
                                data.len()
                            ),
                            0,
                            0,
                        ));
                    }
                    Ok(Value::Tuple(
                        TupleTag::Nil,
                        vec![data[idx as usize].clone()],
                    ))
                } else {
                    Err(RuntimeError::with_pos("index must be int", 0, 0))
                }
            } else {
                Err(RuntimeError::with_pos("only tuples can be indexed", 0, 0))
            }
        }
    }
}

fn eval_expr_block(
    exprs: &Vec<ast::Expression>,
    env: &mut Environment,
) -> Result<Value, RuntimeError> {
    if exprs.is_empty() {
        return Err(RuntimeError::with_pos("empty block", 0, 0));
    }

    let mut result: Option<Value> = None;

    for expr in exprs {
        result = Some(eval_expression(expr, env)?);
    }

    result.ok_or_else(|| RuntimeError::with_pos("empty block", 0, 0))
}

fn eval_filter_impl(filter: &ast::Filter, env: &mut Environment) -> Result<Value, RuntimeError> {
    eval_expr_block(&filter.exprs, env)
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

pub fn eval_filter(filter: &ast::Filter, x: f32, y: f32, t: f32) -> Result<Value, RuntimeError> {
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
    fn test_float_promotion() -> Result<(), Box<dyn std::error::Error>> {
        let input = "1 + 2.0";
        let mut parser = Parser::new(input);
        let ast = parser.parse_expression(1)?;
        let mut env = Environment::new();
        let val = eval_expression(&ast, &mut env)?;

        let val_expected = Value::Tuple(TupleTag::Nil, vec![3.0]);
        assert_eq!(val, val_expected);
        Ok(())
    }

    #[test]
    fn test_int_op() -> Result<(), Box<dyn std::error::Error>> {
        let input = "1 + 2";
        let mut parser = Parser::new(input);
        let ast = parser.parse_expression(1)?;
        let mut env = Environment::new();
        let val = eval_expression(&ast, &mut env)?;
        let val_expected = Value::Int(3);
        assert_eq!(val, val_expected);
        Ok(())
    }

    #[test]
    fn test_broadcast() -> Result<(), Box<dyn std::error::Error>> {
        let input = "1 + rgba:[2, 3, 4, 1]";
        let mut parser = Parser::new(input);
        let ast = parser.parse_expression(1)?;
        let mut env = Environment::new();
        let val = eval_expression(&ast, &mut env)?;
        let val_expected = Value::Tuple(TupleTag::Rgba, vec![3.0, 4.0, 5.0, 2.0]);
        assert_eq!(val, val_expected);
        Ok(())
    }

    #[test]
    fn test_broadcast_2() -> Result<(), Box<dyn std::error::Error>> {
        let input = "rgba:[2, 3, 4, 0.5] * 0.5";
        let mut parser = Parser::new(input);
        let ast = parser.parse_expression(1)?;
        let mut env = Environment::new();
        let val = eval_expression(&ast, &mut env)?;
        let val_expected = Value::Tuple(TupleTag::Rgba, vec![1.0, 1.5, 2.0, 0.25]);
        assert_eq!(val, val_expected);
        Ok(())
    }

    #[test]
    fn test_elementwise() -> Result<(), Box<dyn std::error::Error>> {
        let input = "rgba:[1, 2, 3, 4] + rgba:[5, 6, 7, 8]";
        let mut parser = Parser::new(input);
        let ast = parser.parse_expression(1)?;
        let mut env = Environment::new();
        let val = eval_expression(&ast, &mut env)?;
        let val_expected = Value::Tuple(TupleTag::Rgba, vec![6.0, 8.0, 10.0, 12.0]);
        assert_eq!(val, val_expected);
        Ok(())
    }

    #[test]
    fn test_index() -> Result<(), Box<dyn std::error::Error>> {
        let input = "(rgba:[1, 2, 3, 4])[1]";
        let mut parser = Parser::new(input);
        let ast = parser.parse_expression(1)?;
        let mut env = Environment::new();
        let val = eval_expression(&ast, &mut env)?;
        let val_expected = Value::Tuple(TupleTag::Nil, vec![2.0]);
        assert_eq!(val, val_expected);
        Ok(())
    }

    #[test]
    fn test_less() -> Result<(), Box<dyn std::error::Error>> {
        let input = "1 < 1";
        let mut parser = Parser::new(input);
        let ast = parser.parse_expression(1)?;
        let mut env = Environment::new();
        let val = eval_expression(&ast, &mut env)?;
        let val_expected = Value::Int(0);
        assert_eq!(val, val_expected);
        Ok(())
    }

    #[test]
    fn test_lessequal() -> Result<(), Box<dyn std::error::Error>> {
        let input = "1 <= 1";
        let mut parser = Parser::new(input);
        let ast = parser.parse_expression(1)?;
        let mut env = Environment::new();
        let val = eval_expression(&ast, &mut env)?;
        let val_expected = Value::Int(1);
        assert_eq!(val, val_expected);
        Ok(())
    }

    #[test]
    fn test_or() -> Result<(), Box<dyn std::error::Error>> {
        let input = "0 || 1";
        let mut parser = Parser::new(input);
        let ast = parser.parse_expression(1)?;
        let mut env = Environment::new();
        let val = eval_expression(&ast, &mut env)?;
        let val_expected = Value::Int(1);
        assert_eq!(val, val_expected);
        Ok(())
    }

    #[test]
    fn test_and() -> Result<(), Box<dyn std::error::Error>> {
        let input = "1 && 2";
        let mut parser = Parser::new(input);
        let ast = parser.parse_expression(1)?;
        let mut env = Environment::new();
        let val = eval_expression(&ast, &mut env)?;
        let val_expected = Value::Int(1);
        assert_eq!(val, val_expected);
        Ok(())
    }

    #[test]
    fn test_tuple_build() -> Result<(), Box<dyn std::error::Error>> {
        let input = "rgba:[1, 2, 3, 4]";
        let mut parser = Parser::new(input);
        let ast = parser.parse_expression(1)?;
        let mut env = Environment::new();
        let val = eval_expression(&ast, &mut env)?;
        let val_expected = Value::Tuple(TupleTag::Rgba, vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(val, val_expected);
        Ok(())
    }

    #[test]
    fn test_if() -> Result<(), Box<dyn std::error::Error>> {
        let input = "if x < 100 then y = 10; y else 200 end";
        let mut parser = Parser::new(input);
        let ast = parser.parse_expression(1)?;
        let mut env = Environment::new();
        env.values.insert("x".to_string(), Value::Int(1));
        let val = eval_expression(&ast, &mut env)?;
        let val_expected = Value::Int(10);
        assert_eq!(val, val_expected);
        Ok(())
    }

    #[test]
    fn test_while() -> Result<(), Box<dyn std::error::Error>> {
        let input = "i = 0; while i < 10 do i = i + 1; end; i";
        let mut parser = Parser::new(input);
        let exprs = parser.parse_expr_block()?;
        let mut env = Environment::new();
        let val = eval_expr_block(&exprs, &mut env)?;
        let val_expected = Value::Int(10);
        assert_eq!(val, val_expected);
        Ok(())
    }

    #[test]
    fn test_cast() -> Result<(), Box<dyn std::error::Error>> {
        let input = "ri:xy";
        let mut parser = Parser::new(input);
        let ast = parser.parse_expression(1)?;
        let mut env = Environment::new();
        env.values
            .insert("xy".to_string(), Value::Tuple(TupleTag::Xy, vec![1.0, 2.0]));
        let val = eval_expression(&ast, &mut env)?;
        let val_expected = Value::Tuple(TupleTag::Ri, vec![1.0, 2.0]);
        assert_eq!(val, val_expected);
        Ok(())
    }

    #[test]
    fn test_pow() -> Result<(), Box<dyn std::error::Error>> {
        let input = "2 ^ 3 ^ 1.5";
        let mut parser = Parser::new(input);
        let ast = parser.parse_expression(1)?;
        let mut env = Environment::new();
        let val = eval_expression(&ast, &mut env)?;
        let val_expected = Value::Tuple(TupleTag::Nil, vec![2.0f32.powf(3.0f32.powf(1.5f32))]);
        assert_eq!(val, val_expected);
        Ok(())
    }

    #[test]
    fn test_runtime_error() -> Result<(), Box<dyn std::error::Error>> {
        let input = "randomfn(1)";
        let mut parser = Parser::new(input);
        let ast = parser.parse_expression(1)?;
        let mut env = Environment::new();
        if let Err(RuntimeError(e)) = eval_expression(&ast, &mut env) {
            assert!(e.message.contains("unimplemented function randomfn"));
            assert_eq!(e.line, 0);
            assert_eq!(e.column, 0);
            Ok(())
        } else {
            panic!("expected the parser to fail with RuntimeError");
        }
    }
}
