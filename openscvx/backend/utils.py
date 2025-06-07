# utils.py

from expr import Const, Add, Mul, MatMul, Neg

def pretty_print(expr):
    print(expr.pretty())

def evaluate(expr, values):
    if hasattr(expr, 'name'):
        return values[expr.name]

    if isinstance(expr, Const):
        return expr.value

    if isinstance(expr, Add):
        return evaluate(expr.a, values) + evaluate(expr.b, values)

    if isinstance(expr, Mul):
        return evaluate(expr.a, values) * evaluate(expr.b, values)

    if isinstance(expr, MatMul):
        return evaluate(expr.a, values) @ evaluate(expr.b, values)

    if isinstance(expr, Neg):
        return -evaluate(expr.a, values)

    raise NotImplementedError(f"Evaluation not implemented for {type(expr)}")
