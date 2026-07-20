"""Conservative MRZ reconstruction, TD3 parsing, and ICAO check digits."""

from __future__ import annotations

import re
from dataclasses import dataclass

MRZ_ALLOWED = re.compile(r"[A-Z0-9<]")
FORMAT_LENGTHS = {"TD1": (3, 30), "TD2": (2, 36), "TD3": (2, 44)}


@dataclass(frozen=True)
class CheckDigitResult:
    field: str
    value: str
    expected: str | None
    valid: bool

    def as_dict(self) -> dict:
        return {
            "field": self.field,
            "value": self.value,
            "expected": self.expected,
            "valid": self.valid,
        }


def parse_mrz_row_results(row_results: list[dict]) -> dict:
    """Parse only structurally valid TD3 rows; retain diagnostics for other inputs."""
    rows, reconstruction = reconstruct_rows(row_results)
    base = {
        "status": "incomplete",
        "format": None,
        "rows": rows,
        "reconstruction": reconstruction,
        "fields": {},
        "validation": {
            "length_valid": False,
            "allowed_characters_valid": False,
            "check_digits": [],
            "all_check_digits_valid": False,
        },
        "reasons": [],
    }
    if not rows:
        base["reasons"] = ["no_mrz_rows"]
        return base

    detected_format = detect_format(rows)
    base["format"] = detected_format
    expected_shape = FORMAT_LENGTHS.get(detected_format or "")
    if expected_shape is None:
        base["status"] = "unsupported_format" if len(rows) in (2, 3) else "incomplete"
        base["reasons"] = ["unsupported_or_incomplete_row_lengths"]
        return base
    expected_rows, expected_length = expected_shape
    length_valid = len(rows) == expected_rows and all(len(row) == expected_length for row in rows)
    allowed_valid = all(_allowed_characters(row) for row in rows)
    base["validation"]["length_valid"] = length_valid
    base["validation"]["allowed_characters_valid"] = allowed_valid
    if detected_format != "TD3":
        base["status"] = "unsupported_format"
        base["reasons"] = ["td3_parser_only"]
        return base
    if not length_valid:
        base["reasons"].append("td3_requires_two_44_character_rows")
    if not allowed_valid:
        base["reasons"].append("invalid_mrz_character")
    if not length_valid or not allowed_valid:
        base["status"] = "invalid"
        return base

    fields = _parse_td3_fields(rows[0], rows[1])
    checks = _td3_checks(rows[1])
    checks_valid = all(item.valid for item in checks)
    base["fields"] = fields
    base["validation"]["check_digits"] = [item.as_dict() for item in checks]
    base["validation"]["all_check_digits_valid"] = checks_valid
    if not checks_valid:
        base["reasons"].append("check_digit_failed")
    base["status"] = "valid" if checks_valid else "invalid"
    return base


def reconstruct_rows(row_results: list[dict]) -> tuple[list[str], dict]:
    """Recover two rows from separate OCR rows or one concatenated 88-char row."""
    texts = []
    for item in sorted(row_results, key=lambda value: int(value.get("row_index", 0))):
        text = _sanitize(str(item.get("normalized_text") or item.get("raw_text") or ""))
        if text:
            texts.append(text)
    if len(texts) == 1:
        for row_count, row_length, format_name in (
            (2, 44, "TD3"),
            (2, 36, "TD2"),
            (3, 30, "TD1"),
        ):
            if len(texts[0]) == row_count * row_length:
                return (
                    [texts[0][offset : offset + row_length] for offset in range(0, len(texts[0]), row_length)],
                    {"method": "split_concatenated_row", "source_row_count": 1, "format": format_name},
                )
    return texts, {"method": "row_results", "source_row_count": len(texts)}


def detect_format(rows: list[str]) -> str | None:
    for format_name, (row_count, row_length) in FORMAT_LENGTHS.items():
        if len(rows) == row_count and all(len(row) == row_length for row in rows):
            return format_name
    return None


def _parse_td3_fields(first: str, second: str) -> dict:
    name_field = first[5:44].rstrip("<")
    name_parts = name_field.split("<<", 1)
    surname = name_parts[0].replace("<", " ").strip()
    given_names = name_parts[1].replace("<", " ").strip() if len(name_parts) > 1 else ""
    return {
        "document_code": first[0:2],
        "issuing_state": first[2:5],
        "surname": surname,
        "given_names": given_names,
        "passport_number": second[0:9].rstrip("<"),
        "nationality": second[10:13],
        "date_of_birth": second[13:19],
        "sex": {"F": "female", "M": "male", "<": "unspecified"}.get(second[20], "unknown"),
        "sex_code": second[20],
        "date_of_expiry": second[21:27],
        "optional_data": second[28:42].rstrip("<"),
    }


def _td3_checks(second: str) -> list[CheckDigitResult]:
    fields = [
        ("passport_number", second[0:9], second[9]),
        ("date_of_birth", second[13:19], second[19]),
        ("date_of_expiry", second[21:27], second[27]),
        ("optional_data", second[28:42], second[42]),
        ("composite", second[0:10] + second[13:20] + second[21:43], second[43]),
    ]
    return [
        CheckDigitResult(name, actual, _compute_check_digit(value), actual == _compute_check_digit(value))
        for name, value, actual in fields
    ]


def _compute_check_digit(value: str) -> str | None:
    weights = (7, 3, 1)
    total = 0
    for index, character in enumerate(value):
        numeric = _character_value(character)
        if numeric is None:
            return None
        total += numeric * weights[index % 3]
    return str(total % 10)


def _character_value(character: str) -> int | None:
    if character == "<":
        return 0
    if character.isdigit():
        return int(character)
    if "A" <= character <= "Z":
        return ord(character) - ord("A") + 10
    return None


def _allowed_characters(value: str) -> bool:
    return bool(value) and all(MRZ_ALLOWED.fullmatch(character) for character in value)


def _sanitize(value: str) -> str:
    return "".join(character for character in value.upper() if MRZ_ALLOWED.fullmatch(character))
