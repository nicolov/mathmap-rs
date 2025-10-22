use super::{func_param, Arity, FuncDef, FuncSignature, OverloadResolutionResult, Type};
use crate::ast::TupleTag;
use crate::err::TypeError;
use std::collections::HashMap;

#[derive(Debug, Clone, PartialEq)]
pub struct FunctionTable {
    pub functions: HashMap<String, Vec<FuncDef>>,
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
                        func_param("r", Type::scalar()),
                        func_param("b", Type::scalar()),
                        func_param("b", Type::scalar()),
                    ],
                    ret: Type::rgba(),
                },
            }],
        );

        fns.insert(
            "grayColor".to_string(),
            vec![FuncDef {
                signature: FuncSignature {
                    name: "grayColor".to_string(),
                    params: vec![func_param("c", Type::scalar())],
                    ret: Type::rgba(),
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
                                func_param("x", Type::tuplevar('T', 'N')),
                                func_param("y", Type::tuplevar('T', 'N')),
                            ],
                            ret: Type::tuplevar('T', 'N'),
                        },
                    },
                    FuncDef {
                        signature: FuncSignature {
                            name: name.to_string(),
                            params: vec![func_param("x", Type::Int), func_param("y", Type::Int)],
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

        // For add:
        // [x] (ri:2, ri:2) -> ri:2
        // [ ] (ri:2, ?:1) -> ri:2
        // [ ] (?:1, ri:2) -> ri:2
        // [ ] (?t:1, ?t:1) -> ?t:1
        // [ ] (?t:?l, ?:1) -> ?t:?l
        // [x] (?t:?l, ?t:?l) -> ?t:?l

        let add = "__add".to_string();

        // [x] (ri:2, ri:2) -> ri:2
        fns.get_mut(&add).unwrap().push(FuncDef {
            signature: FuncSignature {
                name: "add_ri_ri".to_string(),
                params: vec![func_param("x", Type::ri()), func_param("y", Type::ri())],
                ret: Type::ri(),
            },
        });

        fns.get_mut("__mul").unwrap().push(FuncDef {
            signature: FuncSignature {
                name: "mul_quat_quat".to_string(),
                params: vec![func_param("x", Type::quat()), func_param("y", Type::quat())],
                ret: Type::quat(),
            },
        });

        let mut def_comparison = |name: &str| {
            fns.insert(
                name.to_string(),
                vec![FuncDef {
                    signature: FuncSignature {
                        name: name.to_string(),
                        params: vec![
                            func_param("x", Type::scalar()),
                            func_param("y", Type::scalar()),
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
                        params: vec![func_param("x", Type::tuplevar('T', 'N'))],
                        ret: Type::tuplevar('T', 'N'),
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
                            func_param("x", Type::tuplevar('T', 'N')),
                            func_param("y", Type::tuplevar('T', 'N')),
                        ],
                        ret: Type::tuplevar('T', 'N'),
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
                        params: vec![func_param("x", Type::tuplevar('T', 'N'))],
                        ret: Type::tuplevar('T', 'N'),
                    },
                }],
            );
        };

        // todo: abs does vector magnitude for all tuple types right now but should
        // only do that for quaternions and complex numbers, and do elementwise abs
        // otherwise.
        def_float_unary("abs");
        def_float_unary("sin");

        fns.get_mut("abs").unwrap().push(FuncDef {
            signature: FuncSignature {
                name: "abs_quat".to_string(),
                params: vec![func_param("x", Type::quat())],
                ret: Type::scalar(),
            },
        });

        fns.insert(
            "sqrt".to_string(),
            vec![FuncDef {
                signature: FuncSignature {
                    name: "sqrt".to_string(),
                    params: vec![func_param("x", Type::scalar())],
                    ret: Type::scalar(),
                },
            }],
        );

        let mut def_bool_binary = |name: &str| {
            fns.insert(
                name.to_string(),
                vec![FuncDef {
                    signature: FuncSignature {
                        name: name.to_string(),
                        params: vec![func_param("x", Type::Int), func_param("y", Type::Int)],
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

        println!("checking {:?}", cand.signature);

        // Constrain typevars according to (non-scalar) arguments. Leave scalars alone for now.
        // eg. this is a mapping 'N' -> 3, etc..
        let mut arity_constraints: HashMap<char, usize> = HashMap::new();
        let mut tag_constraints: HashMap<char, TupleTag> = HashMap::new();
        for (param, arg) in cand.signature.params.iter().zip(arg_tys.iter()) {
            if let Type::Tuple(param_tag, param_arity) = &param.ty {
                // Handle tags: if both tags are concrete and they don't match, reject the candidate.
                // For now we do it here, but maybe we want to move it below close to the broadcasting and
                // type promotion logic so we can do some kind of tag promotion too.
                // We don't do this for scalars (which always have nil type) to allow broadcasting.
                if let Type::Tuple(arg_tag, arg_arity) = arg {
                    if !matches!(arg_arity, Arity::Sized(1)) {
                        if !matches!(param_tag, TupleTag::Var(_))
                            && !matches!(arg_tag, TupleTag::Var(_))
                            && *param_tag != *arg_tag
                        {
                            println!(
                                "  reject candidate because for {}, {} != {}",
                                param.name, param_tag, arg_tag
                            );
                            return None;
                        }
                    }
                }

                // Handle tags: if the param has a tag var, and the arg has a known tag, then unify.
                // We don't do this for scalars (which always have nil type) to allow broadcasting.
                if let TupleTag::Var(param_tv) = param_tag {
                    if let Type::Tuple(arg_tag, arg_arity) = arg {
                        if !matches!(arg_arity, Arity::Sized(1)) {
                            if let Some(prev) = tag_constraints.get(&param_tv) {
                                if *prev != *arg_tag {
                                    println!(
                                        "  reject candidate because for {} {} != {}",
                                        param_tv, prev, arg_tag
                                    );
                                    return None;
                                }
                            } else {
                                tag_constraints.insert(*param_tv, *arg_tag);
                            }
                        }
                    }
                }

                // Handle arity: if the param has a arity var, and the arg has known arity, then unify.
                // We don't do this for scalars (which always have nil type) to allow broadcasting.
                if let Arity::Var(tv) = param_arity {
                    // Ignore scalars for now.
                    if let Type::Tuple(_, Arity::Sized(k)) = arg
                        && *k > 1
                    {
                        // If there was a previous binding for this typevar, check that it's consistent.
                        // Otherwise, create a new one.
                        if let Some(prev) = arity_constraints.get(&tv) {
                            if *prev != *k {
                                println!("  reject candidate because {} != {}", prev, k);
                                return None;
                            }
                        } else {
                            arity_constraints.insert(*tv, *k);
                        }
                    }
                }
            }
        }

        // Constrain any unbound arity vars to 1 to allow broadcasting later, and
        // constrain any unbound tag vars to nil. We only do this after the main loop
        // (ie after we've seen all params) so that more specific constraints from other
        // args would always win.
        for p in cand.signature.params.iter() {
            if let Type::Tuple(_, Arity::Var(av)) = p.ty {
                if !arity_constraints.contains_key(&av) {
                    arity_constraints.insert(av, 1);
                }
            }

            if let Type::Tuple(TupleTag::Var(tv), _) = p.ty {
                if !tag_constraints.contains_key(&tv) {
                    tag_constraints.insert(tv, TupleTag::Nil);
                }
            }
        }

        let apply_typevars = |ty: &Type| match ty {
            Type::Tuple(tag, arity) => {
                // Apply tag substitutions.
                let tag = match tag {
                    TupleTag::Var(tv) => match tag_constraints.get(&tv) {
                        Some(n) => *n,
                        None => {
                            panic!("unresolved tag var {}", &tv.clone());
                        }
                    },
                    x => x.clone(),
                };

                // Apply arity substitutions.
                let arity = match arity {
                    Arity::Var(tv) => match arity_constraints.get(&tv) {
                        Some(n) => Arity::Sized(*n),
                        None => {
                            panic!("unresolved arity var {}", &tv.clone());
                        }
                    },
                    x => x.clone(),
                };

                Type::Tuple(tag, arity)
            }
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
                (Type::Tuple(_, Arity::Sized(n)), Type::Tuple(_, Arity::Sized(m))) => n == m,
                _ => false,
            };

            if perfect_match {
                casts.push(None);
                continue;
            }

            // Then try with type casts (with cost 1).
            let cast = match (param, arg) {
                (Type::Tuple(_, Arity::Sized(1)), Type::Int) => Some(Type::scalar()),
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
                (Type::Tuple(_, Arity::Sized(n)), a) if a.is_scalar() && *n > 1 => {
                    Some(Type::tuple_sized(*n))
                }
                _ => None,
            };

            println!("  broadcast {:?} {:?}", param, arg);

            if let Some(_bcast) = bcast {
                total_cost += 2;

                // Since all tuples are float, any int -> tuple broadcast will also need a type cast.
                if matches!(arg, Type::Int) {
                    casts.push(Some(Type::scalar()));
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

        println!("  ret {:?}", ret_ty);

        Some((
            OverloadResolutionResult {
                def: cand,
                casts,
                ret_ty,
            },
            total_cost,
        ))
    }

    pub fn lookup(
        &self,
        name: &str,
        arg_tys: &[Type],
    ) -> Result<OverloadResolutionResult<'_>, TypeError> {
        let candidates = self.functions.get(name).ok_or(TypeError::with_pos(
            format!("unimplemented function {:?}", name),
            0,
            0,
        ))?;

        let mut matches: Vec<(OverloadResolutionResult, usize, usize)> = Vec::new();

        for cand in candidates {
            if let Some((res, cost)) = self.check_candidate(cand, arg_tys) {
                // Count how many typevars are in the function signature: this is used to break ties
                // by preferring more specific matches. Note that this could be done at compile time
                // instead.
                let num_typevars: usize = cand
                    .signature
                    .params
                    .iter()
                    .map(|p| match p.ty {
                        Type::Tuple(_, Arity::Var(_)) => 1,
                        _ => 0,
                    })
                    .sum();
                matches.push((res, cost, num_typevars));
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
            matches.sort_by_key(|&(_, num_casts, num_typevars)| (num_casts, num_typevars));
            let (best, best_cost, best_num_typevars) = &matches[0];

            if matches.len() > 1 && matches[1].1 == *best_cost && matches[1].2 == *best_num_typevars
            {
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
