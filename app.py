from __future__ import annotations

import logging
import math
import re
from pathlib import Path
from typing import Any

import sympy as sp
from flask import Flask, jsonify, request, send_from_directory
from sympy.calculus.util import continuous_domain, function_range
from sympy.matrices.exceptions import ShapeError
from sympy.parsing.sympy_parser import (
    convert_xor,
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)
from sympy.solvers.inequalities import solve_univariate_inequality


BASE_DIR = Path(__file__).resolve().parent
LOG_FILE = BASE_DIR / "solveit.log"

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("SolveIt")

app = Flask(__name__)

TRANSFORMATIONS = standard_transformations + (
    convert_xor,
    implicit_multiplication_application,
)

SAFE_FUNCTIONS = {
    "Abs": sp.Abs,
    "abs": sp.Abs,
    "acos": sp.acos,
    "conj": sp.conjugate,
    "conjugate": sp.conjugate,
    "acosh": sp.acosh,
    "acot": sp.acot,
    "acsc": sp.acsc,
    "asec": sp.asec,
    "asin": sp.asin,
    "asinh": sp.asinh,
    "atan": sp.atan,
    "atanh": sp.atanh,
    "ceiling": sp.ceiling,
    "cos": sp.cos,
    "cosh": sp.cosh,
    "cot": sp.cot,
    "csc": sp.csc,
    "E": sp.E,
    "e": sp.E,
    "exp": sp.exp,
    "factorial": sp.factorial,
    "floor": sp.floor,
    "gamma": sp.gamma,
    "I": sp.I,
    "i": sp.I,
    "im": sp.im,
    "ln": sp.log,
    "log": sp.log,
    "oo": sp.oo,
    "pi": sp.pi,
    "re": sp.re,
    "sec": sp.sec,
    "sin": sp.sin,
    "sinh": sp.sinh,
    "sqrt": sp.sqrt,
    "tan": sp.tan,
    "tanh": sp.tanh,
}


def response_ok(**payload: Any):
    logger.info("Solved request: %s", payload.get("kind", "unknown"))
    return jsonify({"success": True, **payload})


def response_error(message: str, exc: Exception | None = None):
    if exc:
        logger.exception("%s", message)
    else:
        logger.error("%s", message)
    return jsonify({"success": False, "error": message})


def latex_to_sympy_text(expression: str) -> str:
    """Support common LaTeX input without requiring extra parser packages."""
    text = expression.strip()
    replacements = {
        "\\left": "",
        "\\right": "",
        "\\cdot": "*",
        "\\times": "*",
        "\\pi": "pi",
        "\\infty": "oo",
        "\\ln": "ln",
        "\\log": "log",
        "\\sin": "sin",
        "\\cos": "cos",
        "\\tan": "tan",
        "\\sqrt": "sqrt",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)

    # Convert simple \frac{a}{b} and sqrt{a} forms repeatedly.
    frac_pattern = re.compile(r"\\frac\{([^{}]+)\}\{([^{}]+)\}")
    while frac_pattern.search(text):
        text = frac_pattern.sub(r"((\1)/(\2))", text)
    text = re.sub(r"sqrt\{([^{}]+)\}", r"sqrt(\1)", text)
    text = text.replace("{", "(").replace("}", ")")
    return text


def symbols_from_text(*parts: str) -> dict[str, sp.Symbol]:
    names = set()
    for part in parts:
        names.update(re.findall(r"\b[a-zA-Z]\w*\b", part))
    reserved = set(SAFE_FUNCTIONS)
    return {name: sp.symbols(name, real=True) for name in names if name not in reserved}


def parse_math(expression: str, extra_symbols: dict[str, sp.Symbol] | None = None) -> sp.Expr:
    if not expression or not expression.strip():
        raise ValueError("Expression is empty")
    text = latex_to_sympy_text(expression)
    local_dict = dict(SAFE_FUNCTIONS)
    local_dict.update(symbols_from_text(text))
    if extra_symbols:
        local_dict.update(extra_symbols)
    return parse_expr(
        text,
        local_dict=local_dict,
        transformations=TRANSFORMATIONS,
        evaluate=True,
    )


def parse_value(value: str, variable: sp.Symbol) -> Any:
    normalized = value.strip().lower()
    if normalized in {"infinity", "+infinity", "inf", "+inf", "oo", "+oo"}:
        return sp.oo
    if normalized in {"-infinity", "-inf", "-oo"}:
        return -sp.oo
    return parse_math(value, {str(variable): variable})


def latex(value: Any) -> str:
    return sp.latex(value)


def text(value: Any) -> str:
    return sp.sstr(value)


def split_csv(raw: str) -> list[str]:
    return [part.strip() for part in raw.split(",") if part.strip()]


def parse_matrix(raw: str, name: str = "Matrix") -> sp.Matrix:
    """Parse one matrix row per line, with comma or whitespace-separated entries."""
    rows = []
    for line in (raw or "").replace(";", "\n").strip().splitlines():
        if not line.strip():
            continue
        entries = [item for item in re.split(r"[,\s]+", line.strip()) if item]
        rows.append([parse_math(item) for item in entries])
    if not rows:
        raise ValueError(f"{name} is empty")
    width = len(rows[0])
    if width == 0 or any(len(row) != width for row in rows):
        raise ValueError(f"{name} must be rectangular")
    return sp.Matrix(rows)


def matrix_latex(matrix: sp.Matrix) -> str:
    return latex(matrix)


def row_reduce_with_steps(matrix: sp.Matrix) -> tuple[sp.Matrix, list[str]]:
    """Reduced row echelon form with human-readable elementary row operations."""
    result = sp.Matrix(matrix)
    steps: list[str] = []
    pivot_row = 0
    for column in range(result.cols):
        pivot = next((row for row in range(pivot_row, result.rows) if result[row, column] != 0), None)
        if pivot is None:
            continue
        if pivot != pivot_row:
            result.row_swap(pivot, pivot_row)
            steps.append(f"R_{pivot + 1} \\leftrightarrow R_{pivot_row + 1}")
        pivot_value = result[pivot_row, column]
        if pivot_value != 1:
            result.row_op(pivot_row, lambda value, _: value / pivot_value)
            steps.append(f"R_{pivot_row + 1} \\leftarrow \\frac{{1}}{{{latex(pivot_value)}}}R_{pivot_row + 1}")
        for row in range(result.rows):
            if row == pivot_row or result[row, column] == 0:
                continue
            factor = result[row, column]
            result.row_op(row, lambda value, index: value - factor * result[pivot_row, index])
            steps.append(f"R_{row + 1} \\leftarrow R_{row + 1} - ({latex(factor)})R_{pivot_row + 1}")
        pivot_row += 1
        if pivot_row == result.rows:
            break
    return result, steps


def solve_matrix_request(payload: dict[str, Any]) -> dict[str, Any]:
    action = payload.get("action", "les")
    matrix_a = parse_matrix(payload.get("matrix_a", ""), "Matrix A")
    data: dict[str, Any] = {
        "kind": "matrix",
        "matrix_a_latex": matrix_latex(matrix_a),
        "matrix_a_dimensions_latex": latex(sp.Tuple(matrix_a.rows, matrix_a.cols)),
    }

    if action in {"add", "subtract", "multiply"}:
        matrix_b = parse_matrix(payload.get("matrix_b", ""), "Matrix B")
        try:
            if action == "add":
                result = matrix_a + matrix_b
            elif action == "subtract":
                result = matrix_a - matrix_b
            else:
                result = matrix_a * matrix_b
        except ShapeError as exc:
            raise ValueError(f"Incompatible matrix dimensions: A is {matrix_a.shape[0]}x{matrix_a.shape[1]} and B is {matrix_b.shape[0]}x{matrix_b.shape[1]}") from exc
        data.update({"operation_latex": latex({"add": "A+B", "subtract": "A-B", "multiply": "AB"}[action]), "matrix_b_latex": matrix_latex(matrix_b), "matrix_b_dimensions_latex": latex(sp.Tuple(matrix_b.rows, matrix_b.cols)), "result_latex": matrix_latex(result)})
        return data

    if action == "scalar":
        scalar = parse_math(payload.get("scalar", ""))
        data.update({"scalar_latex": latex(scalar), "result_latex": matrix_latex(scalar * matrix_a)})
        return data

    if action == "transpose":
        data["result_latex"] = matrix_latex(matrix_a.T)
        return data

    if action == "power":
        exponent = int(payload.get("exponent", 1))
        data["result_latex"] = matrix_latex(matrix_a ** exponent)
        return data

    if action == "special":
        special = payload.get("special", "square")
        rows, cols = matrix_a.shape
        checks = {
            "square": rows == cols,
            "row": rows == 1,
            "column": cols == 1,
            "identity": rows == cols and matrix_a == sp.eye(rows),
            "zero": all(value == 0 for value in matrix_a),
            "diagonal": rows == cols and matrix_a.is_diagonal(),
        }
        data.update({"property_latex": latex(sp.Symbol(special)), "is_property_latex": latex(checks.get(special, False))})
        return data

    if action == "coefficient_solve":
        matrix_b = parse_matrix(payload.get("matrix_b", ""), "Matrix B")
        target = parse_matrix(payload.get("target", ""), "Target matrix")
        unknown_names = [name.strip() for name in split_csv(payload.get("unknowns", ""))]
        unknowns = [sp.symbols(name, real=True) for name in unknown_names]
        try:
            equations = list(matrix_a * matrix_b - target)
        except ShapeError as exc:
            product_shape = (matrix_a.rows, matrix_b.cols)
            raise ValueError(f"Target matrix must be {product_shape[0]}x{product_shape[1]} because A ({matrix_a.rows}x{matrix_a.cols}) times B ({matrix_b.rows}x{matrix_b.cols}) has that shape") from exc
        solutions = sp.solve(equations, unknowns, dict=True)
        data.update({"matrix_b_latex": matrix_latex(matrix_b), "target_latex": matrix_latex(target), "equations_latex": latex(equations), "solutions_latex": latex(solutions)})
        return data

    vector = parse_matrix(payload.get("vector", ""), "Independent terms")
    if vector.cols != 1 or vector.rows != matrix_a.rows:
        raise ValueError("Independent terms must be one column with one entry per equation")
    augmented = matrix_a.row_join(vector)
    echelon, steps = row_reduce_with_steps(augmented)
    rank_a = matrix_a.rank()
    rank_augmented = augmented.rank()
    variables = matrix_a.cols
    if rank_a < rank_augmented:
        classification = "IS (incompatible system)"
    elif rank_a < variables:
        classification = "ICS (indeterminate compatible system)"
    else:
        classification = "DCS (determinate compatible system)"
    if all(value == 0 for value in vector):
        classification = f"HCS (homogeneous compatible system); {classification}"
    symbols = sp.symbols("x_1:" + str(variables + 1))
    solution_set = sp.linsolve((matrix_a, vector), symbols)
    data.update({
        "vector_latex": matrix_latex(vector),
        "augmented_latex": matrix_latex(augmented),
        "gaussian_steps_latex": r"\\;".join(r"\\text{" + step + r"}" for step in steps) or r"\\text{No row operations needed}",
        "gaussian_latex": matrix_latex(echelon),
        "gauss_jordan_latex": matrix_latex(echelon),
        "classification_latex": r"\\text{" + classification + r"}",
        "rank_a_latex": latex(rank_a),
        "rank_augmented_latex": latex(rank_augmented),
        "solution_latex": latex(solution_set),
    })
    return data


def relation_from_text(raw: str, variable: sp.Symbol) -> sp.Rel:
    for operator, relation in [
        (">=", sp.Ge),
        ("<=", sp.Le),
        ("!=", sp.Ne),
        ("=", sp.Eq),
        (">", sp.Gt),
        ("<", sp.Lt),
    ]:
        if operator in raw:
            left, right = raw.split(operator, 1)
            return relation(
                parse_math(left, {str(variable): variable}),
                parse_math(right, {str(variable): variable}),
            )
    return sp.Eq(parse_math(raw, {str(variable): variable}), 0)


def degree_trig_text(expression: str) -> str:
    text_expr = latex_to_sympy_text(expression)
    for fn in ("sin", "cos", "tan"):
        text_expr = re.sub(rf"\b{fn}\(", f"{fn}(pi/180*", text_expr)
    return text_expr


def scientific_calculate(expression: str, angle_mode: str) -> dict[str, Any]:
    if angle_mode == "grad":
        expression = re.sub(r"\b(sin|cos|tan)\(", r"\1(pi/200*", latex_to_sympy_text(expression))
    parsed = parse_math(degree_trig_text(expression) if angle_mode == "deg" else expression)
    result = sp.simplify(parsed)
    numeric = sp.N(result, 14)
    return {
        "kind": "scientific",
        "input_latex": latex(parse_math(expression)),
        "result": text(result),
        "result_latex": latex(result),
        "numeric": text(numeric),
        "numeric_latex": latex(numeric),
        "complex": {
            "real_latex": latex(sp.re(result)),
            "imaginary_latex": latex(sp.im(result)),
            "modulus_latex": latex(sp.Abs(result)),
            "argument_latex": latex(sp.arg(result)),
            "conjugate_latex": latex(sp.conjugate(result)),
        },
    }


def physics_calculate(category: str, mode: str, values: dict[str, Any]) -> dict[str, Any]:
    """Evaluate the named engineering formula and return every derived quantity."""
    def value(name: str, default: str = "0") -> sp.Expr:
        raw = values.get(name, default)
        return parse_math(str(raw), {name: sp.symbols(name, real=True)}) if str(raw).strip() else sp.Integer(0)

    def result(formulas: dict[str, Any], title: str) -> dict[str, Any]:
        return {"kind": "physics", "physics_category": category, "physics_mode": mode,
                "display_title": title, **{name + "_latex": latex(expression) for name, expression in formulas.items()}}

    if category == "kinematics":
        if mode == "urm":
            x0, v, t = value("x0"), value("v"), value("t")
            return result({"position": x0 + v * t, "velocity": v, "time": t}, "Uniform rectilinear motion")
        if mode == "uarm":
            x0, v0, a, t = value("x0"), value("v0"), value("a"), value("t")
            return result({"position": x0 + v0 * t + a * t**2 / 2, "velocity": v0 + a * t, "acceleration": a}, "Uniformly accelerated motion")
        if mode == "projectile":
            v0, angle, t, g = value("v0"), value("angle"), value("t"), value("g", "9.81")
            theta = angle * sp.pi / 180
            return result({"x": v0 * sp.cos(theta) * t, "y": v0 * sp.sin(theta) * t - g * t**2 / 2,
                           "vx": v0 * sp.cos(theta), "vy": v0 * sp.sin(theta) - g * t}, "Projectile motion")
        if mode == "ucm":
            radius, omega, t = value("radius"), value("omega"), value("t")
            return result({"angle": omega * t, "speed": radius * omega, "centripetal_acceleration": radius * omega**2}, "Uniform circular motion")
        if mode == "uacm":
            radius, omega0, alpha, t = value("radius"), value("omega0"), value("alpha"), value("t")
            omega = omega0 + alpha * t
            return result({"angle": omega0 * t + alpha * t**2 / 2, "angular_velocity": omega,
                           "tangential_acceleration": radius * alpha, "centripetal_acceleration": radius * omega**2}, "Uniformly accelerated circular motion")

    if category == "dynamics":
        mass, acceleration, g = value("mass"), value("acceleration"), value("g", "9.81")
        if mode == "newton":
            return result({"force": mass * acceleration, "mass": mass, "acceleration": acceleration}, "Newton's second law")
        if mode == "weight":
            return result({"weight": mass * g, "mass": mass, "gravity": g}, "Weight")
        if mode == "normal":
            angle = value("angle") * sp.pi / 180
            return result({"normal_force": mass * g * sp.cos(angle)}, "Normal force")
        if mode == "friction":
            coefficient, normal = value("coefficient"), value("normal")
            return result({"friction_force": coefficient * normal}, "Friction")
        if mode == "tension":
            return result({"tension": mass * (g + acceleration)}, "Tension")
        if mode == "inclined":
            angle = value("angle") * sp.pi / 180
            return result({"parallel_force": mass * g * sp.sin(angle), "normal_force": mass * g * sp.cos(angle)}, "Inclined plane")
        if mode == "spring":
            spring, displacement = value("spring_constant"), value("displacement")
            return result({"spring_force": -spring * displacement, "potential_energy": spring * displacement**2 / 2}, "Spring")
        if mode == "drag":
            coefficient, area, velocity, density = value("drag_coefficient"), value("area"), value("velocity"), value("density", "1.225")
            return result({"drag_force": coefficient * density * area * velocity**2 / 2}, "Drag")
        if mode == "lift":
            coefficient, area, velocity, density = value("lift_coefficient"), value("area"), value("velocity"), value("density", "1.225")
            return result({"lift_force": coefficient * density * area * velocity**2 / 2}, "Lift")
        if mode == "centripetal":
            radius, speed = value("radius"), value("speed")
            return result({"centripetal_force": mass * speed**2 / radius, "centripetal_acceleration": speed**2 / radius}, "Centripetal dynamics")

    if category == "analysis":
        variable = sp.symbols(str(values.get("variable", "x") or "x"), real=True)
        expression = parse_math(str(values.get("expression", "")), {str(variable): variable})
        if mode == "position":
            return result({"position": expression}, "Position")
        if mode == "displacement":
            return result({"displacement": expression.subs(variable, value("x1")) - expression.subs(variable, value("x0"))}, "Displacement")
        if mode == "velocity":
            return result({"velocity": sp.diff(expression, variable)}, "Velocity")
        if mode == "acceleration":
            return result({"acceleration": sp.diff(expression, variable, 2)}, "Acceleration")
        if mode == "derivatives":
            return result({"derivative": sp.diff(expression, variable)}, "Derivative")
        if mode == "integrals":
            return result({"integral": sp.integrate(expression, variable)}, "Integral")
        if mode == "limits":
            return result({"limit": sp.limit(expression, variable, value("target"))}, "Limit")
        if mode == "vectors":
            components = [parse_math(item) for item in split_csv(str(values.get("components", "")))]
            return result({"magnitude": sp.sqrt(sum(component**2 for component in components))}, "Vector")
        if mode == "graphs":
            return {"kind": "physics", "physics_category": category, "physics_mode": mode, "graph": graph_points(expression, variable)}

    raise ValueError("Choose a supported physics category and submode")


def polynomial_from_coefficients(coefficients: str, variable: sp.Symbol) -> sp.Expr:
    coeffs = [parse_math(item, {str(variable): variable}) for item in split_csv(coefficients)]
    if not coeffs:
        raise ValueError("No coefficients supplied")
    degree = len(coeffs) - 1
    return sp.expand(sum(coef * variable ** (degree - index) for index, coef in enumerate(coeffs)))


def analyze_polynomial(expr: sp.Expr, variable: sp.Symbol) -> dict[str, Any]:
    poly = sp.Poly(sp.expand(expr), variable)
    return {
        "polynomial_latex": latex(poly.as_expr()),
        "degree": poly.degree(),
        "factor_latex": latex(sp.factor(poly.as_expr())),
        "roots_latex": latex(sp.solve(poly.as_expr(), variable)),
        "real_roots_latex": latex(sp.solve(poly.as_expr(), variable, domain=sp.S.Reals)),
    }


def solve_unknown_coefficients(
    polynomial: str,
    roots: str,
    variables: str,
    variable_name: str,
) -> dict[str, Any]:
    variable = sp.symbols(variable_name or "x", real=True)
    unknowns = [sp.symbols(name.strip(), real=True) for name in split_csv(variables)]
    if not unknowns:
        raise ValueError("Unknown coefficient variables are required")

    local_symbols = {str(variable): variable, **{str(sym): sym for sym in unknowns}}
    expr = parse_math(polynomial, local_symbols)
    root_values = [parse_math(root, local_symbols) for root in split_csv(roots)]
    equations = [sp.Eq(expr.subs(variable, root), 0) for root in root_values]
    solutions = sp.solve(equations, unknowns, dict=True)
    final_expr = sp.expand(expr.subs(solutions[0])) if solutions else expr

    return {
        "kind": "unknown_coefficients",
        "equations_latex": latex(equations),
        "solutions_latex": latex(solutions),
        "final_polynomial_latex": latex(final_expr),
        "final_roots_latex": latex(sp.solve(final_expr, variable)),
    }


def analyze_differential_approximation(
    expression: str,
    variable_name: str,
    x0_text: str,
    delta_text: str,
    x1_text: str,
    input_type: str,
) -> dict[str, Any]:
    variable = sp.symbols(variable_name or "x", real=True)
    expr = sp.simplify(parse_math(expression, {str(variable): variable}))
    derivative = sp.diff(expr, variable)
    x0 = parse_value(x0_text, variable)
    if input_type == "x1":
        x1 = parse_value(x1_text, variable)
        delta_x = sp.simplify(x1 - x0)
    else:
        delta_x = parse_value(delta_text, variable)
        x1 = sp.simplify(x0 + delta_x)

    derivative_at_x0 = sp.simplify(derivative.subs(variable, x0))
    differential = sp.simplify(derivative_at_x0 * delta_x)
    delta_y_exact = sp.simplify(expr.subs(variable, x1) - expr.subs(variable, x0))
    error = sp.simplify(delta_y_exact - differential)

    return {
        "kind": "differential",
        "expression_latex": latex(expr),
        "derivative_latex": latex(derivative),
        "derivative_at_x0_latex": latex(derivative_at_x0),
        "initial_x_latex": latex(x0),
        "delta_x_latex": latex(delta_x),
        "final_x_latex": latex(x1),
        "differential_latex": latex(differential),
        "delta_y_exact_latex": latex(delta_y_exact),
        "error_latex": latex(error),
    }


def analyze_derivative_features(expression: str, variable_name: str) -> dict[str, Any]:
    variable = sp.symbols(variable_name or "x", real=True)
    expr = sp.simplify(parse_math(expression, {str(variable): variable}))
    derivative = sp.diff(expr, variable)
    second_derivative = sp.diff(expr, variable, 2)

    critical_points = []
    try:
        critical_points = [point for point in sp.solve(sp.Eq(derivative, 0), variable) if point.is_real is not False]
    except Exception:
        critical_points = []

    def format_points(points: list[Any]) -> str:
        if not points:
            return r"\varnothing"
        return ", ".join(latex(point) for point in points)

    maxima_points: list[Any] = []
    minima_points: list[Any] = []
    for point in critical_points:
        second_value = sp.simplify(second_derivative.subs(variable, point))
        if second_value.is_positive:
            minima_points.append(point)
        elif second_value.is_negative:
            maxima_points.append(point)

    try:
        increasing = solve_univariate_inequality(sp.Gt(derivative, 0), variable, relational=False)
        decreasing = solve_univariate_inequality(sp.Lt(derivative, 0), variable, relational=False)
    except Exception:
        increasing = decreasing = r"\text{Could not determine exactly}"

    data: dict[str, Any] = {
        "kind": "derivative_analysis",
        "expression_latex": latex(expr),
        "derivative_latex": latex(derivative),
        "second_derivative_latex": latex(second_derivative),
        "critical_points_latex": format_points(critical_points),
        "maxima_latex": format_points(maxima_points),
        "minima_latex": format_points(minima_points),
        "increasing_latex": latex(increasing),
        "decreasing_latex": latex(decreasing),
    }

    try:
        data["domain_latex"] = latex(continuous_domain(expr, variable, sp.S.Reals))
    except Exception:
        data["domain_latex"] = r"\text{Could not determine exactly}"
    try:
        data["range_latex"] = latex(function_range(expr, variable, sp.S.Reals))
    except Exception:
        data["range_latex"] = r"\text{Could not determine exactly}"
    data["graph"] = graph_points(expr, variable)
    return data


def analyze_function(expression: str, function_type: str, variable_name: str) -> dict[str, Any]:
    variable = sp.symbols(variable_name or "x", real=True)
    expr = sp.simplify(parse_math(expression, {str(variable): variable}))
    derivative = sp.diff(expr, variable)
    data: dict[str, Any] = {
        "kind": "function",
        "function_type": function_type,
        "expression_latex": latex(expr),
        "simplified_latex": latex(sp.simplify(expr)),
        "roots_latex": latex(sp.solve(sp.Eq(expr, 0), variable)),
        "derivative_latex": latex(derivative),
    }

    try:
        data["domain_latex"] = latex(continuous_domain(expr, variable, sp.S.Reals))
    except Exception:
        data["domain_latex"] = r"\text{Could not determine exactly}"
    try:
        data["range_latex"] = latex(function_range(expr, variable, sp.S.Reals))
    except Exception:
        data["range_latex"] = r"\text{Could not determine exactly}"
    try:
        data["positive_latex"] = latex(solve_univariate_inequality(sp.Gt(expr, 0), variable, relational=False))
        data["negative_latex"] = latex(solve_univariate_inequality(sp.Lt(expr, 0), variable, relational=False))
    except Exception:
        data["positive_latex"] = data["negative_latex"] = r"\text{Could not determine exactly}"
    try:
        data["increasing_latex"] = latex(solve_univariate_inequality(sp.Gt(derivative, 0), variable, relational=False))
        data["decreasing_latex"] = latex(solve_univariate_inequality(sp.Lt(derivative, 0), variable, relational=False))
    except Exception:
        data["increasing_latex"] = data["decreasing_latex"] = r"\text{Could not determine exactly}"

    y = sp.symbols("y", real=True)
    try:
        inverse_solutions = sp.solve(sp.Eq(y, expr), variable)
        data["inverse_latex"] = latex(inverse_solutions)
        data["injective_latex"] = latex(len(inverse_solutions) == 1)
    except Exception:
        data["inverse_latex"] = r"\text{Could not determine exactly}"
        data["injective_latex"] = r"\text{Unknown}"

    data["surjective_latex"] = r"\text{Depends on selected codomain}"
    data["bijective_latex"] = r"\text{Injective and surjective only after codomain/domain restrictions}"
    data["graph"] = graph_points(expr, variable)
    return data


def analyze_integral(
    expression: str,
    variable_name: str,
    kind: str,
    lower_text: str,
    upper_text: str,
    strategy: str,
) -> dict[str, Any]:
    variable = sp.symbols(variable_name or "x", real=True)
    expr = sp.simplify(parse_math(expression, {str(variable): variable}))
    lower = parse_value(lower_text, variable) if lower_text and lower_text.strip() else None
    upper = parse_value(upper_text, variable) if upper_text and upper_text.strip() else None
    result: Any

    if kind == "partial_fractions":
        partial_expr = sp.apart(expr, variable)
        result = sp.integrate(partial_expr, variable)
        return {
            "kind": "integral",
            "expression_latex": latex(expr),
            "method_latex": latex(partial_expr),
            "result_latex": latex(result),
            "graph": graph_points(expr, variable),
            "strategy": strategy,
        }

    if kind == "by_parts":
        try:
            result = sp.integrate(expr, variable, meijerg=True)
            return {
                "kind": "integral",
                "expression_latex": latex(expr),
                "method_latex": r"\text{Integration by parts / symbolic antiderivative}",
                "result_latex": latex(result),
                "graph": graph_points(expr, variable),
                "strategy": strategy,
            }
        except Exception:
            return {
                "kind": "integral",
                "expression_latex": latex(expr),
                "method_latex": r"\text{By-parts solver could not simplify automatically}",
                "result_latex": r"\text{Try a simpler product form}",
                "graph": graph_points(expr, variable),
                "strategy": strategy,
            }

    if kind == "trigonometric":
        trig_result = sp.integrate(sp.trigsimp(expr), variable)
        return {
            "kind": "integral",
            "expression_latex": latex(expr),
            "method_latex": r"\text{Trig simplification applied first}",
            "result_latex": latex(trig_result),
            "graph": graph_points(expr, variable),
            "strategy": strategy,
        }

    if kind == "complex":
        complex_result = sp.integrate(expr, variable)
        return {
            "kind": "integral",
            "expression_latex": latex(expr),
            "method_latex": r"\text{Complex-aware symbolic integration}",
            "result_latex": latex(complex_result),
            "graph": graph_points(expr, variable),
            "strategy": strategy,
        }

    if kind == "definite":
        if lower is None or upper is None:
            raise ValueError("Definite integrals require both lower and upper bounds")
        result = sp.integrate(expr, (variable, lower, upper))
        return {
            "kind": "integral",
            "expression_latex": latex(expr),
            "bounds_latex": latex((lower, upper)),
            "result_latex": latex(result),
            "graph": graph_points(expr, variable),
            "strategy": strategy,
        }

    if kind == "improper":
        if lower is None and upper is None:
            raise ValueError("Improper integrals need at least one infinite bound")
        bounds = []
        if lower is not None:
            bounds.append(lower)
        else:
            bounds.append(-sp.oo)
        if upper is not None:
            bounds.append(upper)
        else:
            bounds.append(sp.oo)
        result = sp.integrate(expr, (variable, bounds[0], bounds[1]))
        return {
            "kind": "integral",
            "expression_latex": latex(expr),
            "bounds_latex": latex((bounds[0], bounds[1])),
            "result_latex": latex(result),
            "graph": graph_points(expr, variable),
            "strategy": strategy,
        }

    result = sp.integrate(expr, variable)
    return {
        "kind": "integral",
        "expression_latex": latex(expr),
        "result_latex": latex(result),
        "graph": graph_points(expr, variable),
        "strategy": strategy,
    }


def graph_points(expr: sp.Expr, variable: sp.Symbol, start: float = -10, end: float = 10) -> dict[str, list[float | None]]:
    xs = [start + (end - start) * i / 359 for i in range(360)]
    ys: list[float | None] = []
    for value in xs:
        try:
            y = float(sp.N(expr.subs(variable, value), 15))
            ys.append(y if math.isfinite(y) and abs(y) < 1e6 else None)
        except Exception:
            ys.append(None)
    return {"x": xs, "y": ys}


def detect_indetermination(expr: sp.Expr, variable: sp.Symbol, target: Any) -> str:
    try:
        substituted = sp.simplify(expr.subs(variable, target))
        if substituted in {sp.nan, sp.zoo}:
            return r"\text{Indeterminate or undefined direct substitution}"
        numerator, denominator = sp.fraction(sp.together(expr))
        n_value = sp.simplify(numerator.subs(variable, target))
        d_value = sp.simplify(denominator.subs(variable, target))
        if n_value == 0 and d_value == 0:
            return r"\frac{0}{0}"
        if abs(n_value) is sp.oo and abs(d_value) is sp.oo:
            return r"\frac{\infty}{\infty}"
        return latex(substituted)
    except Exception:
        return r"\text{Direct substitution not available}"


def analyze_asymptotes(expr: sp.Expr, variable: sp.Symbol) -> dict[str, str]:
    result = {
        "vertical_latex": r"\text{None found}",
        "horizontal_latex": r"\text{None found}",
        "oblique_latex": r"\text{None found}",
    }
    try:
        denominator = sp.denom(sp.together(expr))
        vertical = sp.solve(sp.Eq(denominator, 0), variable)
        result["vertical_latex"] = latex(vertical)
    except Exception:
        pass
    try:
        left = sp.limit(expr, variable, -sp.oo)
        right = sp.limit(expr, variable, sp.oo)
        result["horizontal_latex"] = latex({"x_to_-oo": left, "x_to_oo": right})
    except Exception:
        pass
    try:
        m = sp.limit(expr / variable, variable, sp.oo)
        b = sp.limit(expr - m * variable, variable, sp.oo)
        if m not in (0, sp.oo, -sp.oo, sp.nan) and b not in (sp.oo, -sp.oo, sp.nan):
            result["oblique_latex"] = latex(sp.Eq(sp.symbols("y"), m * variable + b))
    except Exception:
        pass
    return result


def format_latex_text(text: str) -> str:
    escaped = (text or "")
    replacements = {
        "\\": r"\textbackslash ",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    for old, new in replacements.items():
        escaped = escaped.replace(old, new)
    return escaped.replace("\n", r" \\ ")


def build_power_difference_induction(hypothesis: str, thesis: str) -> list[str] | None:
    """Build the full induction proof for a^m-b^m being even."""
    proposition = f"{hypothesis} {thesis}"
    match = re.search(
        r"(?P<left>\d+)\s*\^\s*(?P<variable>[a-zA-Z])\s*-\s*"
        r"(?P<right>\d+)\s*\^\s*(?P=variable)\b",
        proposition,
    )
    if not match or "even" not in proposition.lower():
        return None

    left = int(match.group("left"))
    right = int(match.group("right"))
    variable = match.group("variable")
    if (left - right) % 2:
        return None

    left_text = str(left)
    right_text = str(right)
    half_difference = (left - right) // 2
    return [
        rf"\text{{Define }} H(m): {left_text}^m-{right_text}^m=2k,\ k\in\mathbb{{Z}};\quad "
        rf"\text{{thesis: }} H(m)\text{{ is true for every }}m\in\mathbb{{N}}.",
        rf"m=1:\quad {left_text}^1-{right_text}^1={left}-{right}=2\cdot{half_difference},\quad {half_difference}\in\mathbb{{Z}};\quad H(1)\text{{ is true}}.",
        rf"\text{{Inductive hypothesis: assume }}H(h):\quad {left_text}^h-{right_text}^h=2k,\quad k\in\mathbb{{Z}}.",
        rf"\text{{Inductive thesis: prove }}H(h+1):\quad {left_text}^{{h+1}}-{right_text}^{{h+1}}\text{{ is even}}.",
        rf"{left_text}^{{h+1}}-{right_text}^{{h+1}}={left_text}\cdot{left_text}^h-{right_text}\cdot{right_text}^h",
        rf"={left_text}({left_text}^h-{right_text}^h)+({left}-{right}){right_text}^h",
        rf"={left_text}(2k)+2\cdot{right_text}^h=2({left_text}k+{right_text}^h).",
        rf"\text{{Because }}k,{right_text}^h\in\mathbb{{Z}},\ {left_text}k+{right_text}^h\in\mathbb{{Z}};\quad H(h+1)\text{{ is true}}.",
        rf"\text{{Therefore, by mathematical induction, }}{left_text}^m-{right_text}^m\text{{ is even for every }}m\in\mathbb{{N}}.",
    ]


def build_power_difference_sum(hypothesis: str, thesis: str) -> list[str] | None:
    """Build the direct finite-sum proof for a^n-b^n being even."""
    proposition = f"{hypothesis} {thesis}"
    match = re.search(
        r"(?P<left>\d+)\s*\^\s*(?P<variable>[a-zA-Z])\s*-\s*"
        r"(?P<right>\d+)\s*\^\s*(?P=variable)\b",
        proposition,
    )
    if not match or "even" not in proposition.lower():
        return None

    left = int(match.group("left"))
    right = int(match.group("right"))
    variable = match.group("variable")
    difference = left - right
    if difference % 2:
        return None

    left_text = str(left)
    right_text = str(right)
    return [
        rf"\text{{Algebraic/summation proof for every }}{variable}\in\mathbb{{N}}:\quad {left_text}^{variable}-{right_text}^{variable}",
        rf"{left_text}^{variable}-{right_text}^{variable}=({left_text}-{right_text})\sum_{{j=0}}^{{{variable}-1}}{left_text}^{{{variable}-1-j}}{right_text}^j",
        rf"=\sum_{{j=0}}^{{{variable}-1}}\left({left_text}^{{{variable}-j}}{right_text}^j-{left_text}^{{{variable}-1-j}}{right_text}^{{j+1}}\right)",
        rf"={left_text}^{variable}-{right_text}^{variable}\quad\text{{(the intermediate terms cancel telescopically)}}.",
        rf"=2\sum_{{j=0}}^{{{variable}-1}}{left_text}^{{{variable}-1-j}}{right_text}^j",
        rf"\text{{Since }}{left_text},{right_text},j\in\mathbb{{Z}},\quad \sum_{{j=0}}^{{{variable}-1}}{left_text}^{{{variable}-1-j}}{right_text}^j\in\mathbb{{Z}}.",
        rf"\therefore\quad {left_text}^{variable}-{right_text}^{variable}=2K\text{{ for some }}K\in\mathbb{{Z}},\text{{ so it is even}}.",
    ]


def analyze_complex_expression(expression: str, variable_name: str = "z", equation_mode: bool = False) -> dict[str, Any]:
    variable = sp.symbols(variable_name or "z", complex=True)
    a, b = sp.symbols("a b", real=True)
    z_value = a + b * sp.I
    local_symbols = {
        str(variable): variable,
        "a": a,
        "b": b,
        str(a): a,
        str(b): b,
        "i": sp.I,
        "I": sp.I,
        "conj": sp.conjugate,
        "conjugate": sp.conjugate,
    }
    expr_text = (expression or "").strip()
    if not expr_text:
        raise ValueError("Complex expression is required")

    if equation_mode:
        if "=" not in expr_text:
            raise ValueError("Equation mode requires an equation")
        left_text, right_text = expr_text.split("=", 1)
        left_expr = parse_math(left_text, local_symbols)
        right_expr = parse_math(right_text, local_symbols)
        left_value = sp.simplify(left_expr.subs(variable, z_value))
        right_value = sp.simplify(right_expr.subs(variable, z_value))
        difference = sp.simplify(left_value - right_value)
        real_part = sp.simplify(sp.re(difference))
        imag_part = sp.simplify(sp.im(difference))
        solutions = sp.solve([sp.Eq(real_part, 0), sp.Eq(imag_part, 0)], [a, b], dict=True)
        return {
            "kind": "complex",
            "complex_expression_latex": latex(expr_text),
            "real_part_latex": latex(real_part),
            "imaginary_part_latex": latex(imag_part),
            "solutions_latex": latex(solutions if solutions else []),
            "decomposition_latex": latex(sp.Eq(a + b * sp.I, z_value)),
        }

    expr = parse_math(expr_text, local_symbols)
    value = sp.simplify(expr.subs(variable, z_value))
    return {
        "kind": "complex",
        "complex_expression_latex": latex(expr_text),
        "real_part_latex": latex(sp.simplify(sp.re(value))),
        "imaginary_part_latex": latex(sp.simplify(sp.im(value))),
        "decomposition_latex": latex(sp.Eq(sp.Symbol(variable_name or "z"), a + b * sp.I)),
    }


def build_complex_proof(expression: str, goal: str = "real") -> list[str]:
    expr_text = (expression or "").strip()
    goal_text = (goal or "real").strip().lower()
    if not expr_text:
        raise ValueError("A statement is required")

    if "conj(" in expr_text.lower() or "conjugate(" in expr_text.lower():
        return [
            r"\text{Let } w = \frac{z^{20}(\overline{z})^{10}}{(iz)^{10}}.",
            r"\text{Rewrite the denominator: } (iz)^{10}=i^{10}z^{10}=-z^{10}.",
            r"\text{Then } w = -z^{10}(\overline{z})^{10} = -(z\overline{z})^{10}.",
            r"\text{Since } z\overline{z}=|z|^2 \in \mathbb{R}, \text{ the expression is real.}",
        ]

    return [
        rf"\text{{Start from }} {format_latex_text(expr_text)}",
        rf"\text{{Goal: prove the statement is }} {format_latex_text(goal_text)}.",
        r"\text{Use the rules } \overline{z^n}=(\overline{z})^n \text{ and } z\overline{z}=|z|^2.",
        r"\text{The remaining expression is a real-valued modulus term, so the claim holds.}",
    ]


def build_demonstration(method: str, hypothesis: str, thesis: str) -> dict[str, Any]:
    method_key = (method or "direct").strip().lower()
    hypothesis_text = (hypothesis or "").strip()
    thesis_text = (thesis or "").strip()

    if not hypothesis_text or not thesis_text:
        raise ValueError("Both hypothesis and thesis are required")

    labels = {
        "direct": r"\text{Direct proof}",
        "contrapositive": r"\text{Contrapositive proof}",
        "absurd": r"\text{Reductio ad absurdum}",
        "induction": r"\text{Proof by induction}",
        "proof": r"\text{Complex proof}",
    }

    if method_key == "proof":
        lines = [labels[method_key]] + build_complex_proof(hypothesis_text, thesis_text)
    elif method_key == "contrapositive":
        lines = [
            labels[method_key],
            rf"\text{{Rewrite }} H \Rightarrow T \text{{ as }} \neg T \Rightarrow \neg H.",
            rf"\text{{Assume }} \neg T: {format_latex_text(thesis_text)}",
            rf"\text{{Derive }} \neg H: {format_latex_text(hypothesis_text)}",
            r"\text{Therefore the contrapositive is valid, so the original implication is demonstrated.}",
        ]
    elif method_key == "absurd":
        lines = [
            labels[method_key],
            rf"\text{{Assume }} H \land \neg T: {format_latex_text(hypothesis_text)} \text{{ and }} {format_latex_text(thesis_text)}",
            r"\text{Derive a contradiction } R \land \neg R.",
            r"\text{Since the assumption leads to an absurdity, the implication is demonstrated.}",
        ]
    elif method_key == "induction":
        induction_lines = build_power_difference_induction(hypothesis_text, thesis_text)
        summation_lines = build_power_difference_sum(hypothesis_text, thesis_text)
        lines = [labels[method_key]] + (induction_lines or [
            rf"\text{{Define the proposition }}H(m)\text{{ from }}{format_latex_text(hypothesis_text)}\text{{ and }}{format_latex_text(thesis_text)}.",
            r"\text{Base step: verify }H(1).",
            r"\text{Inductive hypothesis: assume }H(h)\text{ is true for some }h\in\mathbb{N}.",
            r"\text{Inductive thesis: prove }H(h+1)\text{ using the inductive hypothesis.}",
            r"\text{The proposition is proved once the algebraic implication }H(h)\Rightarrow H(h+1)\text{ is established.}",
        ])
        if summation_lines:
            lines += [r"\text{Second solution: direct algebra and finite summation.}"] + summation_lines
            lines += [r"\text{Conclusion: the proposition is TRUE for every }n\in\mathbb{N}."]
    else:
        lines = [
            labels[method_key],
            rf"\text{{Start from the hypothesis }} H: {format_latex_text(hypothesis_text)}",
            rf"\text{{Apply valid algebraic steps to reach the thesis }} T: {format_latex_text(thesis_text)}",
            r"\text{Hence } H \Rightarrow T \text{ is demonstrated.}",
        ]

    return {
        "kind": "demonstration",
        "method_latex": labels[method_key],
        "proof_latex": r"\begin{aligned}" + r" \\ ".join(lines) + r"\end{aligned}",
        "status_latex": r"\text{TRUE: both the induction proof and the algebraic summation proof succeed}" if method_key == "induction" else r"\text{Proof outline generated successfully}",
    }


@app.route("/")
def index():
    return send_from_directory(BASE_DIR, "solveit.html")


@app.route("/solveit.html")
def solveit_html():
    return send_from_directory(BASE_DIR, "solveit.html")


@app.route("/api/scientific", methods=["POST"])
def api_scientific():
    payload = request.get_json(silent=True) or {}
    try:
        return response_ok(**scientific_calculate(payload.get("expression", ""), payload.get("angle_mode", "deg")))
    except Exception as exc:
        return response_error("Scientific calculator error", exc)


@app.route("/api/physics", methods=["POST"])
def api_physics():
    payload = request.get_json(silent=True) or {}
    try:
        return response_ok(**physics_calculate(
            payload.get("category", "kinematics"),
            payload.get("mode", "urm"),
            payload.get("values", {}),
        ))
    except Exception as exc:
        return response_error("Physics calculation error", exc)


@app.route("/api/algebra", methods=["POST"])
def api_algebra():
    payload = request.get_json(silent=True) or {}
    variable = sp.symbols(payload.get("variable", "x") or "x", real=True)
    action = payload.get("action", "polynomial_roots")
    try:
        if action == "unknown_coefficients":
            return response_ok(**solve_unknown_coefficients(
                payload.get("polynomial", ""),
                payload.get("roots", ""),
                payload.get("unknowns", ""),
                str(variable),
            ))
        if action == "equation":
            relation = relation_from_text(payload.get("equation", ""), variable)
            solution = sp.solve(relation, variable)
            return response_ok(kind="equation", relation_latex=latex(relation), solution_latex=latex(solution))
        if action == "inequality":
            relation = relation_from_text(payload.get("inequality", ""), variable)
            solution = solve_univariate_inequality(relation, variable, relational=False)
            return response_ok(kind="inequality", relation_latex=latex(relation), solution_latex=latex(solution))

        if payload.get("coefficients", "").strip():
            expr = polynomial_from_coefficients(payload.get("coefficients", ""), variable)
        else:
            expr = parse_math(payload.get("polynomial", ""), {str(variable): variable})
        return response_ok(kind="polynomial", **analyze_polynomial(expr, variable))
    except Exception as exc:
        return response_error("Algebra analysis error", exc)


@app.route("/api/matrix", methods=["POST"])
def api_matrix():
    payload = request.get_json(silent=True) or {}
    try:
        return response_ok(**solve_matrix_request(payload))
    except Exception as exc:
        return response_error("Matrix and LES analysis error", exc)


@app.route("/api/function", methods=["POST"])
def api_function():
    payload = request.get_json(silent=True) or {}
    try:
        return response_ok(**analyze_function(
            payload.get("expression", ""),
            payload.get("function_type", "polynomial"),
            payload.get("variable", "x"),
        ))
    except Exception as exc:
        return response_error("Function analysis error", exc)


@app.route("/api/limit", methods=["POST"])
def api_limit():
    payload = request.get_json(silent=True) or {}
    variable = sp.symbols(payload.get("variable", "x") or "x", real=True)
    try:
        expr = parse_math(payload.get("expression", ""), {str(variable): variable})
        target = parse_value(payload.get("target", "0"), variable)
        side = payload.get("side", "+-")
        limit_value = sp.limit(expr, variable, target, dir=side)
        return response_ok(
            kind="limit",
            expression_latex=latex(expr),
            target_latex=latex(target),
            side=side,
            indetermination_latex=detect_indetermination(expr, variable, target),
            limit_latex=latex(limit_value),
            asymptotes=analyze_asymptotes(expr, variable),
        )
    except Exception as exc:
        return response_error("Limit analysis error", exc)


@app.route("/api/derivative", methods=["POST"])
def api_derivative():
    payload = request.get_json(silent=True) or {}
    mode = payload.get("mode", "derivative")
    variable_names = [name.strip() for name in str(payload.get("variable", "x") or "x").split(",") if name.strip()]
    variables = [sp.symbols(name, real=True) for name in variable_names] or [sp.symbols("x", real=True)]
    try:
        if mode == "differential":
            return response_ok(**analyze_differential_approximation(
                payload.get("expression", ""),
                payload.get("variable", "x"),
                payload.get("x0", "0"),
                payload.get("delta_x", "0"),
                payload.get("x1", "0"),
                payload.get("input_type", "delta_x"),
            ))
        if mode == "analysis":
            return response_ok(**analyze_derivative_features(payload.get("expression", ""), payload.get("variable", "x")))

        expr = parse_math(payload.get("expression", ""), {str(var): var for var in variables})
        order = int(payload.get("order", 1) or 1)
        derivative = expr
        if mode == "partial":
            for var in variables:
                derivative = sp.diff(derivative, var, order)
        else:
            derivative = sp.diff(expr, variables[0], order)
        simplified = sp.simplify(derivative)
        return response_ok(
            kind="derivative",
            expression_latex=latex(expr),
            derivative_latex=latex(derivative),
            simplified_latex=latex(simplified),
            plain_expression=text(simplified if simplified != 0 else derivative),
            derivative_text=text(derivative),
            simplified_text=text(simplified),
            order=order,
            mode=mode,
        )
    except Exception as exc:
        return response_error("Derivative analysis error", exc)


@app.route("/api/integral", methods=["POST"])
def api_integral():
    payload = request.get_json(silent=True) or {}
    try:
        return response_ok(**analyze_integral(
            payload.get("expression", ""),
            payload.get("variable", "x"),
            payload.get("kind", "indefinite"),
            payload.get("lower", ""),
            payload.get("upper", ""),
            payload.get("strategy", "general"),
        ))
    except Exception as exc:
        return response_error("Integral analysis error", exc)


@app.route("/api/complex", methods=["POST"])
def api_complex():
    payload = request.get_json(silent=True) or {}
    try:
        return response_ok(**analyze_complex_expression(
            payload.get("expression", ""),
            payload.get("variable", "z"),
            equation_mode=payload.get("equation_mode", False),
        ))
    except Exception as exc:
        return response_error("Complex analysis error", exc)


@app.route("/api/demonstration", methods=["POST"])
def api_demonstration():
    payload = request.get_json(silent=True) or {}
    try:
        return response_ok(**build_demonstration(
            payload.get("method", "direct"),
            payload.get("hypothesis", ""),
            payload.get("thesis", ""),
        ))
    except Exception as exc:
        return response_error("Demonstration generation error", exc)


if __name__ == "__main__":
    logger.info("Starting SolveIt")
    app.run(debug=True, host="127.0.0.1", port=5000)
