"""Answer values, conservative consistency checks, and option-label remapping."""

from __future__ import annotations

import ast
import math
import re
import unicodedata
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from fractions import Fraction
from typing import Any


LETTERS = "ABCD"
ANSWER_CONTRACT_VERSION = 4
_OPTION_PREFIX = re.compile(
    r"^\s*(?:(?:option|choice)\s*)?(?:\([A-D]\)|\[[A-D]\]|[A-D]\s*[.)\]:-])\s*",
    re.IGNORECASE,
)
_NUMBER = r"[+-]?(?:\d+(?:\.\d+)?|\.\d+)(?:\s*/\s*[+-]?\d+)?"


def strip_option_prefix(value: Any) -> str:
    """Remove model-authored A./B)/C: style labels from an option value.

    The application owns the displayed option letters and may shuffle options.
    Keeping labels inside the option text creates duplicate-looking choices and can
    make an otherwise correct answer key point at the wrong displayed value.
    """
    text = unicodedata.normalize("NFC", "" if value is None else str(value)).strip()
    return _OPTION_PREFIX.sub("", text, count=1).strip()


def normalized_answer(value: Any) -> str:
    text = strip_option_prefix(value)
    text = text.replace("−", "-").replace("×", "*").replace("÷", "/")
    text = re.sub(r"\\(?:dfrac|tfrac|frac)\s*\{([^{}]+)\}\s*\{([^{}]+)\}", r"\1/\2", text)
    text = re.sub(r"\\(?:text|mathrm)\{([^{}]+)\}", r"\1", text)
    text = text.replace(r"\(", "").replace(r"\)", "").replace("$", "")
    return re.sub(r"\s+", "", text).casefold().rstrip(".;")


def numeric_answer(value: Any) -> tuple[Fraction, str] | None:
    text = normalized_answer(value)
    match = re.fullmatch(rf"({_NUMBER})(%|[a-z]+(?:[²³]|\^[23])?)?", text)
    if not match:
        return None
    try:
        number = Fraction(match.group(1))
    except (ValueError, ZeroDivisionError):
        return None
    unit = match.group(2) or ""
    return (number / 100, "") if unit == "%" else (number, unit)


def _finite_set(value: Any) -> frozenset[str] | None:
    text = normalized_answer(value)
    if text in {"∅", r"\emptyset", r"\varnothing", "emptyset", "{}"}:
        return frozenset()
    match = re.fullmatch(r"\{([^{}]*)\}", text)
    if not match:
        return None
    elements: set[str] = set()
    for element in match.group(1).split(","):
        number = numeric_answer(element)
        if number is not None and number[1] == "":
            elements.add(str(number[0]))
        elif re.fullmatch(r"[a-z]", element):
            elements.add(element)
        else:
            return None
    return frozenset(elements)


def answer_values_equal(first: Any, second: Any) -> bool:
    left, right = normalized_answer(first), normalized_answer(second)
    if not left or not right:
        return False
    if left == right:
        return True
    left_set, right_set = _finite_set(first), _finite_set(second)
    if left_set is not None and right_set is not None:
        return left_set == right_set
    left_number, right_number = numeric_answer(first), numeric_answer(second)
    return left_number is not None and left_number == right_number


def matching_answer_indices(value: Any, options: list[Any]) -> list[int]:
    return [index for index, option in enumerate(options) if answer_values_equal(value, option)]


def answer_letter_from_value(value: Any, options: list[Any]) -> str:
    matches = matching_answer_indices(value, options)
    if len(options) != 4 or len(matches) != 1:
        raise ValueError("correct_answer must match exactly one displayed option")
    return LETTERS[matches[0]]


_OPTION_REFERENCE = re.compile(r"\boptions?\s+([A-D])\b", re.IGNORECASE)
_OPTION_LIST = re.compile(
    r"\boptions?\s+((?:\(?[A-D]\b\)?)(?:(?:\s*[,/]\s*(?:and\s+|or\s+)?|\s+(?:and|or)\s+)\(?[A-D]\b\)?)*)",
    re.IGNORECASE,
)
_ORDINAL_OPTION = re.compile(r"\b(first|second|third|fourth)\s+(?:option|choice)\b", re.IGNORECASE)
_CORRECT_REFERENCE = re.compile(
    r"\b(?:correct\s+answer|correct\s+option)\s*(?:is|:|=)?\s*(?:option\s*)?\(?((?-i:[A-D]))\b",
    re.IGNORECASE,
)
_LETTER_CORRECT = re.compile(
    r"\b([A-D])\s+(?:is|would\s+be)\s+(?:the\s+)?correct(?:\s+answer|\s+option)?\b",
    re.IGNORECASE,
)
_OPTION_CORRECT = re.compile(
    r"\boption\s*([A-D])\s+(?:(?:is\s+)?(?:the\s+)?correct\b|correctly\b)",
    re.IGNORECASE,
)


def remap_explanation_options(explanation: str, letter_map: dict[str, str]) -> str:
    spans: dict[tuple[int, int], str] = {}
    for match in _OPTION_LIST.finditer(explanation):
        for letter in re.finditer(r"\b[A-D]\b", match.group(1), re.IGNORECASE):
            span = (match.start(1) + letter.start(), match.start(1) + letter.end())
            spans[span] = letter_map[letter.group(0).upper()]
    for match in _ORDINAL_OPTION.finditer(explanation):
        original = LETTERS[("first", "second", "third", "fourth").index(match.group(1).lower())]
        spans[match.span()] = f"option {letter_map[original]}"
    for pattern in (_OPTION_REFERENCE, _CORRECT_REFERENCE, _LETTER_CORRECT):
        for match in pattern.finditer(explanation):
            spans[match.span(1)] = letter_map[match.group(1).upper()]
    for (start, end), replacement in sorted(spans.items(), reverse=True):
        explanation = explanation[:start] + replacement + explanation[end:]
    return explanation


def _conclusion_indices(explanation: str, options: list[Any]) -> set[int]:
    conclusions = re.findall(
        r"\b(?:correct\s+)?(?:answer|result|value)\s*(?:is|:|=)\s*([^\n]+)",
        explanation,
        flags=re.IGNORECASE,
    )
    terminal = re.search(
        r"(?:\bgives\b|\bequals\b|=)\s*([^=\n]+?)\s*[.!]?$", explanation,
        flags=re.IGNORECASE,
    )
    if terminal:
        conclusions.append(terminal.group(1))
    matches: set[int] = set()
    for conclusion in conclusions:
        # Only explicit conclusions count; a value mentioned as a distractor or
        # an intermediate quantity is not evidence for the keyed answer.
        for index, option in enumerate(options):
            candidate = normalized_answer(conclusion)
            answer = normalized_answer(option)
            if candidate == answer or (
                candidate.startswith(answer)
                and re.match(r"^(?:\.(?!\d)|;|,?because)", candidate[len(answer):])
            ):
                matches.add(index)
        number = re.match(rf"^\s*({_NUMBER})\s*(%|[a-z]+(?:[²³]|\^[23])?)?\s*[.!]?\s*$", conclusion)
        if number:
            matches.update(matching_answer_indices(number.group(0), options))
    return matches


def _arithmetic_result(stem: str) -> Fraction | None:
    match = re.fullmatch(
        r"\s*(?:what is|calculate|evaluate|compute|find the value of)\s+([\d\s.+*/()^×÷−-]+)\s*[?.!]?\s*",
        stem,
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    expression = match.group(1).strip().rstrip(".")
    if len(expression) > 200:
        return None
    expression = expression.replace("×", "*").replace("÷", "/").replace("−", "-").replace("^", "**")
    try:
        tree = ast.parse(expression, mode="eval")
        if sum(1 for _ in ast.walk(tree)) > 64:
            return None

        def evaluate(node: ast.AST) -> Fraction:
            if isinstance(node, ast.Constant) and type(node.value) in {int, float}:
                return Fraction(str(node.value))
            if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
                value = evaluate(node.operand)
                return -value if isinstance(node.op, ast.USub) else value
            if not isinstance(node, ast.BinOp):
                raise ValueError("Unsupported expression")
            left, right = evaluate(node.left), evaluate(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                return left / right
            if isinstance(node.op, ast.Pow) and right.denominator == 1 and abs(right) <= 12 and abs(left) <= 10**6:
                return left ** int(right)
            raise ValueError("Unsupported expression")

        return evaluate(tree.body)
    except (ValueError, SyntaxError, ZeroDivisionError, OverflowError, RecursionError):
        return None


def _evaluate_set_expression(
    expression: str,
    sets: dict[str, frozenset[str] | None],
) -> frozenset[str] | None:
    """Evaluate a conservative finite-set expression used in school MCQs.

    Supported operators are union (∪), intersection (∩), and set difference
    (``\\`` or ``\\setminus``), with parentheses. Any other token makes the
    expression unsupported rather than guessed.
    """
    text = str(expression or "").strip()
    text = text.replace(r"\setminus", "\\").replace("−", "\\")
    text = re.sub(r"\s+", "", text)
    if not text or any(ch not in "ABCDEFGHIJKLMNOPQRSTUVWXYZ()∪∩\\" for ch in text):
        return None

    tokens = list(text)
    position = 0

    def parse_factor() -> frozenset[str]:
        nonlocal position
        if position >= len(tokens):
            raise ValueError("missing set operand")
        token = tokens[position]
        if token == "(":
            position += 1
            value = parse_union()
            if position >= len(tokens) or tokens[position] != ")":
                raise ValueError("unbalanced set expression")
            position += 1
            return value
        if token.isalpha() and token.isupper():
            position += 1
            value = sets.get(token)
            if value is None:
                raise ValueError("undefined set")
            return value
        raise ValueError("unsupported set operand")

    def parse_intersection_difference() -> frozenset[str]:
        nonlocal position
        value = parse_factor()
        while position < len(tokens) and tokens[position] in {"∩", "\\"}:
            operator = tokens[position]
            position += 1
            right = parse_factor()
            value = value & right if operator == "∩" else value - right
        return value

    def parse_union() -> frozenset[str]:
        nonlocal position
        value = parse_intersection_difference()
        while position < len(tokens) and tokens[position] == "∪":
            position += 1
            value = value | parse_intersection_difference()
        return value

    try:
        result = parse_union()
        return result if position == len(tokens) else None
    except ValueError:
        return None


def _set_result(stem: str) -> frozenset[str] | None:
    sets = {
        name: _finite_set(elements)
        for name, elements in re.findall(r"\b([A-Z])\s*=\s*(\{[^{}]*\})", stem)
    }
    request = re.search(
        r"\b(?:find|what is|determine|calculate)\s+(.+?)\s*[?.!]?\s*$",
        stem,
        flags=re.IGNORECASE,
    )
    if not request:
        return None
    return _evaluate_set_expression(request.group(1), sets)


def repair_deterministic_answer(question: dict[str, Any]) -> bool:
    """Repair only an answer key that an exact local solver can prove is wrong.

    The stem and all options remain unchanged. A repair is allowed only when a
    deterministic school-math check identifies exactly one displayed option.
    The independent semantic reviewer still runs afterwards and replaces the
    placeholder explanation with its concise justification before acceptance.
    """
    options = question.get("options")
    if not isinstance(options, list) or len(options) != 4:
        return False

    stem = str(question.get("question", ""))
    matches: list[int] | None = None

    independent_indices, _kind = _independent_expected_indices(stem, options)
    if independent_indices is not None:
        matches = list(independent_indices)
    else:
        arithmetic = _arithmetic_result(stem)
        if arithmetic is not None:
            matches = [
                index
                for index, option in enumerate(options)
                if numeric_answer(option) == (arithmetic, "")
            ]
        else:
            set_result = _set_result(stem)
            if set_result is not None:
                matches = [
                    index
                    for index, option in enumerate(options)
                    if _finite_set(option) == set_result
                ]

    if matches is None or len(matches) != 1:
        return False

    correct_index = matches[0]
    expected_letter = LETTERS[correct_index]
    expected_text = str(options[correct_index]).strip()
    current_letter = str(question.get("answer", "")).strip().upper()

    explicit_matches = []
    if "correct_answer" in question:
        explicit_matches = matching_answer_indices(question.get("correct_answer"), options)

    if current_letter == expected_letter and explicit_matches == [correct_index]:
        return False

    question["answer"] = expected_letter
    question["correct_answer"] = expected_text
    question["explanation"] = f"Correct result: {expected_text}."
    question.pop("quality_review", None)
    return True



_ORDINALS = {
    "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5,
    "sixth": 6, "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10,
    "eleventh": 11, "twelfth": 12, "thirteenth": 13, "fourteenth": 14,
    "fifteenth": 15, "sixteenth": 16, "seventeenth": 17, "eighteenth": 18,
    "nineteenth": 19, "twentieth": 20,
}


def _ordinal_number(value: str) -> int | None:
    token = str(value or "").strip().lower()
    if token in _ORDINALS:
        return _ORDINALS[token]
    match = re.fullmatch(r"(\d+)(?:st|nd|rd|th)?", token)
    if not match:
        return None
    number = int(match.group(1))
    return number if 1 <= number <= 1000 else None


def _ap_expected_value(stem: str) -> Fraction | None:
    if "arithmetic progression" not in stem.lower():
        return None

    target_match = re.search(
        r"(?:what\s+is|find)\s+(?:the\s+)?(?:value\s+of\s+)?(?:the\s+)?([a-z]+|\d+(?:st|nd|rd|th)?)\s+term",
        stem,
        flags=re.IGNORECASE,
    )
    if not target_match:
        return None
    target = _ordinal_number(target_match.group(1))
    if target is None:
        return None

    givens: list[tuple[int, Fraction]] = []
    for ordinal, raw_value in re.findall(
        r"(?:the\s+)?([a-z]+|\d+(?:st|nd|rd|th)?)\s+term\s+is\s+([+-]?\d+(?:\.\d+)?)",
        stem,
        flags=re.IGNORECASE,
    ):
        index = _ordinal_number(ordinal)
        if index is not None:
            givens.append((index, Fraction(raw_value)))
    if len(givens) >= 2 and givens[0][0] != givens[1][0]:
        (n1, v1), (n2, v2) = givens[:2]
        difference = (v2 - v1) / (n2 - n1)
        first = v1 - (n1 - 1) * difference
        return first + (target - 1) * difference

    sequence_match = re.search(
        r"arithmetic\s+progression\s+([+-]?\d+(?:\.\d+)?)\s*,\s*([+-]?\d+(?:\.\d+)?)\s*,",
        stem,
        flags=re.IGNORECASE,
    )
    if sequence_match:
        first = Fraction(sequence_match.group(1))
        second = Fraction(sequence_match.group(2))
        return first + (target - 1) * (second - first)
    return None


def _rounding_expected_value(stem: str) -> Fraction | None:
    if "nearest multiple of" not in stem.lower():
        return None
    patterns = (
        r"(?:value\s+of\s+)?([+-]?\d+(?:\.\d+)?)\s+when\s+rounded\s+to\s+the\s+nearest\s+multiple\s+of\s+(\d+(?:\.\d+)?)",
        r"rounded\s+value\s+of\s+([+-]?\d+(?:\.\d+)?)\s+to\s+the\s+nearest\s+multiple\s+of\s+(\d+(?:\.\d+)?)",
    )
    match = next(
        (candidate for pattern in patterns if (candidate := re.search(pattern, stem, flags=re.IGNORECASE))),
        None,
    )
    if not match:
        return None
    try:
        value = Decimal(match.group(1))
        multiple = Decimal(match.group(2))
        if multiple <= 0:
            return None
        rounded = (value / multiple).quantize(Decimal("1"), rounding=ROUND_HALF_UP) * multiple
        return Fraction(rounded)
    except (InvalidOperation, ValueError, ZeroDivisionError):
        return None


def _rounding_option_value(option: Any) -> Fraction | None:
    text = strip_option_prefix(option)
    target = re.search(r"\bto\s+([+-]?\d+(?:\.\d+)?)\s*$", text, flags=re.IGNORECASE)
    raw = target.group(1) if target else text
    number = numeric_answer(raw)
    return number[0] if number is not None and number[1] == "" else None


def _quadratic_vertex(stem: str) -> tuple[Fraction, Fraction] | None:
    if "turning point" not in stem.lower() and "vertex" not in stem.lower():
        return None
    compact = stem.replace("−", "-").replace("²", "^2")
    # Conservative form: y = ax^2 + c, with no x-term. This covers the common
    # school-level translation questions without trying to parse arbitrary algebra.
    match = re.search(
        r"y\s*=\s*([+-]?(?:\d+(?:\.\d+)?)?)\s*x\s*\^\s*2\s*([+-]\s*\d+(?:\.\d+)?)?",
        compact,
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    coefficient = match.group(1).replace(" ", "")
    if coefficient in {"", "+", "-"}:
        coefficient = f"{coefficient}1" if coefficient else "1"
    try:
        if Fraction(coefficient) == 0:
            return None
        constant = Fraction((match.group(2) or "0").replace(" ", ""))
    except (ValueError, ZeroDivisionError):
        return None
    return Fraction(0), constant


def _coordinate_value(option: Any) -> tuple[Fraction, Fraction] | None:
    text = strip_option_prefix(option).replace("−", "-")
    match = re.fullmatch(
        r"\s*\(\s*([+-]?\d+(?:\.\d+)?(?:/\d+)?)\s*,\s*([+-]?\d+(?:\.\d+)?(?:/\d+)?)\s*\)\s*",
        text,
    )
    if not match:
        return None
    try:
        return Fraction(match.group(1)), Fraction(match.group(2))
    except (ValueError, ZeroDivisionError):
        return None


def _repeated_binomial_expected(stem: str) -> str | None:
    lower = stem.lower()
    if "cube" not in lower or "binomial" not in lower:
        return None
    factors = re.findall(r"\([^()]+\)", stem)
    if len(factors) < 3:
        return None
    first_three = [re.sub(r"\s+", "", item) for item in factors[:3]]
    if len(set(first_three)) != 1:
        return None
    return first_three[0] + "^3"


def _symbolic_compact(value: Any) -> str:
    text = strip_option_prefix(value).replace("³", "^3").replace("²", "^2")
    return re.sub(r"\s+", "", text).casefold()


def _log_expected_value(stem: str) -> Fraction | None:
    if "log" not in stem.lower():
        return None
    match = re.search(
        r"\blog\s*\(\s*([0-9]+(?:\.\d+)?)\s*/\s*([0-9]+(?:\.\d+)?)\s*\)",
        stem,
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    numerator, denominator = float(match.group(1)), float(match.group(2))
    if numerator <= 0 or denominator <= 0:
        return None
    value = math.log10(numerator / denominator)
    nearest = round(value)
    if abs(value - nearest) < 1e-10:
        return Fraction(nearest)
    return None


def _neither_expected_value(stem: str) -> Fraction | None:
    if "neither" not in stem.lower():
        return None
    total_match = re.search(r"class\s+of\s+(\d+)\s+students", stem, flags=re.IGNORECASE)
    both_match = re.search(r"(\d+)\s+students?\s+play\s+both", stem, flags=re.IGNORECASE)
    activity_counts = re.findall(
        r"(?<!students\s)(\d+)\s+play\s+(?!both\b)[a-z][a-z -]*",
        stem,
        flags=re.IGNORECASE,
    )
    if not total_match or not both_match or len(activity_counts) < 2:
        return None
    total = int(total_match.group(1))
    first, second = int(activity_counts[0]), int(activity_counts[1])
    both = int(both_match.group(1))
    result = total - (first + second - both)
    return Fraction(result) if result >= 0 else None


def _stream_probability_expected(stem: str) -> Fraction | None:
    lower = stem.lower()
    if "probability" not in lower or "stream" not in lower or "selected at random" not in lower:
        return None
    rows = re.findall(
        r"(\d+)\s+girls\s*,\s*(\d+)\s+boys\s+in\s+(biological|mathematics)\s+stream",
        stem,
        flags=re.IGNORECASE,
    )
    if len(rows) < 2:
        return None
    target = re.search(
        r"probability\s+that\s+the\s+student\s+is\s+a\s+(girl|boy)\s+following\s+the\s+(biological|mathematics)\s+stream",
        stem,
        flags=re.IGNORECASE,
    )
    if not target:
        return None
    gender, stream = target.group(1).lower(), target.group(2).lower()
    total = sum(int(girls) + int(boys) for girls, boys, _stream in rows)
    numerator = sum(
        int(girls if gender == "girl" else boys)
        for girls, boys, row_stream in rows
        if row_stream.lower() == stream
    )
    if total <= 0:
        return None
    return Fraction(numerator, total)


def _equality_pair(text: str) -> frozenset[str] | None:
    match = re.fullmatch(r"(AB|BC|CD|DA|AD)=(AB|BC|CD|DA|AD)", text.upper())
    if not match:
        return None
    left = "AD" if match.group(1) == "DA" else match.group(1)
    right = "AD" if match.group(2) == "DA" else match.group(2)
    return frozenset({left, right})


def _parallelogram_expected_indices(stem: str, options: list[Any]) -> list[int] | None:
    lower = stem.lower()
    if "quadrilateral abcd" not in lower or "parallelogram" not in lower:
        return None
    if not any(word in lower for word in ("side length", "condition", "ensures", "confirms")):
        return None
    required = {frozenset({"AB", "CD"}), frozenset({"AD", "BC"})}
    matches: list[int] = []
    for index, option in enumerate(options):
        clean = strip_option_prefix(option)
        parts = re.split(r"\s+(?:and|&)\s+|,|;", clean, flags=re.IGNORECASE)
        pairs = {
            pair
            for part in parts
            if (pair := _equality_pair(re.sub(r"\s+", "", part))) is not None
        }
        if required.issubset(pairs):
            matches.append(index)
    return matches


def _independent_expected_indices(stem: str, options: list[Any]) -> tuple[list[int] | None, str]:
    ap_value = _ap_expected_value(stem)
    if ap_value is not None:
        return matching_answer_indices(str(ap_value), options), "arithmetic progression"

    rounding_value = _rounding_expected_value(stem)
    if rounding_value is not None:
        return [
            index for index, option in enumerate(options)
            if _rounding_option_value(option) == rounding_value
        ], "rounding"

    vertex = _quadratic_vertex(stem)
    if vertex is not None:
        return [
            index for index, option in enumerate(options)
            if _coordinate_value(option) == vertex
        ], "quadratic turning point"

    cube = _repeated_binomial_expected(stem)
    if cube is not None:
        return [
            index for index, option in enumerate(options)
            if _symbolic_compact(option) == _symbolic_compact(cube)
        ], "binomial cube"

    log_value = _log_expected_value(stem)
    if log_value is not None:
        return matching_answer_indices(str(log_value), options), "logarithm"

    neither_value = _neither_expected_value(stem)
    if neither_value is not None:
        return matching_answer_indices(str(neither_value), options), "set counting"

    probability = _stream_probability_expected(stem)
    if probability is not None:
        return matching_answer_indices(str(probability), options), "probability"

    parallelogram = _parallelogram_expected_indices(stem, options)
    if parallelogram is not None:
        return parallelogram, "parallelogram condition"

    return None, ""

def answer_explanation_consistency_errors(question: dict[str, Any]) -> list[str]:
    answer = str(question.get("answer", "")).strip().upper()
    options = question.get("options")
    explanation = remap_explanation_options(
        str(question.get("explanation", "")).strip(), {letter: letter for letter in LETTERS}
    )
    if answer not in LETTERS or len(answer) != 1 or not isinstance(options, list) or len(options) != 4:
        return []
    errors: list[str] = []
    if re.search(
        r"\b(?:wait\b|correction\b|re-?evaluate\b|rechecking\b|error in reasoning\b|previous option was wrong\b)",
        explanation,
        re.IGNORECASE,
    ):
        errors.append("explanation contains self-correction or unresolved reasoning")
    if "correct_answer" in question:
        try:
            expected = answer_letter_from_value(question["correct_answer"], options)
            if expected != answer:
                errors.append("answer key contradicts the explicit correct_answer value")
        except ValueError as exc:
            errors.append(str(exc))
    claimed = {
        match.upper()
        for pattern in (_CORRECT_REFERENCE, _LETTER_CORRECT, _OPTION_CORRECT)
        for match in pattern.findall(explanation)
    }
    if claimed and claimed != {answer}:
        errors.append("answer key contradicts explanation, which identifies a different option as correct")
    concluded = _conclusion_indices(explanation, options)
    if len(concluded) > 1:
        errors.append("explanation states multiple different final answers")
    elif len(concluded) == 1 and LETTERS.index(answer) not in concluded:
        errors.append("answer key contradicts the sole option value stated as the explanation's conclusion")
    stem = str(question.get("question", ""))
    independent_indices, independent_kind = _independent_expected_indices(stem, options)
    if independent_indices is not None:
        keyed_index = LETTERS.index(answer)
        if len(independent_indices) != 1:
            errors.append(
                f"{independent_kind} check found that the displayed options do not contain exactly one correct answer"
            )
        elif independent_indices[0] != keyed_index:
            errors.append(f"answer key does not match the independently verified {independent_kind} result")

    result = _arithmetic_result(stem)
    if result is not None:
        matches = [i for i, option in enumerate(options) if numeric_answer(option) == (result, "")]
        if len(matches) != 1 or matches[0] != LETTERS.index(answer):
            errors.append("answer key does not match the independently calculated arithmetic result")
    set_result = _set_result(str(question.get("question", "")))
    if set_result is not None:
        matches = [index for index, option in enumerate(options) if _finite_set(option) == set_result]
        if len(matches) != 1 or matches[0] != LETTERS.index(answer):
            errors.append("answer key does not match the independently calculated set result")
    return errors
