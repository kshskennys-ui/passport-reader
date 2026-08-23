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
            "essential_check_digits_valid": False,
        },
        "reasons": [],
    }
    if not rows:
        base["reasons"] = ["no_mrz_rows"]
        return base

    recovered = _recover_valid_td3(rows)
    if recovered is not None:
        recovered_rows, recovery = recovered
        fields, checks = _parse_valid_td3_candidate(recovered_rows)
        base.update(
            {
                "status": "valid",
                "format": "TD3",
                "rows": recovered_rows,
                "reconstruction": {**reconstruction, **recovery},
                "fields": fields,
                "validation": {
                    "length_valid": True,
                    "allowed_characters_valid": True,
                    "check_digits": [item.as_dict() for item in checks],
                    "all_check_digits_valid": True,
                    "essential_check_digits_valid": True,
                },
            }
        )
        return base

    padded_pm = _recover_pm_with_missing_fillers(rows)
    if padded_pm is not None:
        recovered_rows, recovery = padded_pm
        fields = _parse_td3_fields(recovered_rows[0], recovered_rows[1])
        checks = _td3_checks(recovered_rows[1])
        essential_valid = _essential_td3_checks_valid(checks)
        checks_valid = all(item.valid for item in checks)
        if essential_valid:
            base.update(
                {
                    "status": "valid" if checks_valid else "partial",
                    "format": "TD3",
                    "rows": recovered_rows,
                    "reconstruction": {**reconstruction, **recovery},
                    "fields": fields,
                    "validation": {
                        "length_valid": True,
                        "allowed_characters_valid": True,
                        "check_digits": [item.as_dict() for item in checks],
                        "all_check_digits_valid": checks_valid,
                        "essential_check_digits_valid": essential_valid,
                    },
                    "reasons": [] if checks_valid else [
                        "check_digit_failed",
                        "optional_or_composite_check_failed",
                    ],
                }
            )
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
    essential_valid = _essential_td3_checks_valid(checks)
    base["validation"]["essential_check_digits_valid"] = essential_valid
    if not checks_valid:
        base["reasons"].append("check_digit_failed")
    if first_document_code(rows) == "PM" and essential_valid and not checks_valid:
        base["status"] = "partial"
        base["reasons"].append("optional_or_composite_check_failed")
        return base
    base["status"] = "valid" if checks_valid else "invalid"
    return base


def _recover_pm_with_missing_fillers(rows: list[str]) -> tuple[list[str], dict] | None:
    """Pad omitted trailing name fillers when a PM data row validates the identity fields."""
    if (
        len(rows) != 2
        or not rows[0].startswith("PM")
        or not 15 <= len(rows[0]) < 44
        or len(rows[1]) != 44
        or not all(_allowed_characters(row) for row in rows)
    ):
        return None
    return [rows[0].ljust(44, "<"), rows[1]], {"recovery": "pm_filler_padding"}


def _recover_valid_td3(rows: list[str]) -> tuple[list[str], dict] | None:
    """Try only structural TD3 repairs that are proven by all ICAO checks."""
    candidates: list[tuple[list[str], list[str]]] = []
    ordered = rows
    if len(rows) == 2 and rows[1].startswith("P") and not rows[0].startswith("P"):
        ordered = [rows[1], rows[0]]

    if len(ordered) == 2:
        first, second = ordered
        first_candidates: list[tuple[str, list[str]]] = [(first, [])]
        if 15 <= len(first) < 44 and first.startswith("P"):
            padded = first.ljust(44, "<")
            first_candidates.append((padded, ["validated_filler_padding"]))
        for marker_candidate, marker_methods in list(first_candidates):
            if (
                marker_candidate.startswith("P")
                and len(marker_candidate) == 44
                and marker_candidate[1] not in {"<", "M"}
            ):
                first_candidates.append(
                    (
                        marker_candidate[0] + "<" + marker_candidate[2:],
                        marker_methods + ["validated_structural_marker"],
                    )
                )

        second_candidates: list[tuple[str, list[str]]] = [(second, [])]
        if _optional_data_check_is_missing(second):
            second_candidates.append(
                (second[:42] + "0" + second[43:], ["validated_optional_data_check"])
            )
        second_candidates.extend(_passport_character_candidates(second))
        for first_candidate, first_methods in first_candidates:
            for second_candidate, second_methods in second_candidates:
                methods = first_methods + second_methods
                if methods:
                    candidates.append(([first_candidate, second_candidate], methods))

    if len(rows) == 1 and 80 <= len(rows[0]) <= 88 and rows[0].startswith("P"):
        text = rows[0]
        for split in range(40, min(44, len(text) - 1) + 1):
            first, second = text[:split], text[split:]
            if len(first) <= 44 and len(second) <= 44:
                candidates.append(
                    (
                        [first.ljust(44, "<"), second.ljust(44, "<")],
                        ["validated_concatenated_split"],
                    )
                )

    for candidate, methods in candidates:
        if _parse_valid_td3_candidate(candidate) is not None:
            return candidate, {"recovery": "+".join(methods)}
    return None


def _optional_data_check_is_missing(second: str) -> bool:
    """Recognize a blank optional-data field whose check digit was read as filler."""
    return (
        len(second) == 44
        and second[28:42] == "<" * 14
        and second[42] == "<"
        and second[43].isdigit()
    )


def _passport_character_candidates(second: str) -> list[tuple[str, list[str]]]:
    """Try a small OCR-confusion set only when passport and composite checks fail."""
    if len(second) != 44:
        return []
    checks = _td3_checks(second)
    failed = {item.field for item in checks if not item.valid}
    if failed != {"passport_number", "composite"}:
        return []
    confusions = {
        "H": "W",
        "W": "H",
        "O": "0",
        "0": "O",
        "I": "1",
        "1": "I",
        "B": "8",
        "8": "B",
        "Z": "2",
        "2": "Z",
    }
    candidates: list[tuple[str, list[str]]] = []
    for index, character in enumerate(second[:9]):
        replacement = confusions.get(character)
        if replacement is None:
            continue
        candidates.append(
            (
                second[:index] + replacement + second[index + 1 :],
                ["validated_passport_character_correction"],
            )
        )
    return candidates


def _parse_valid_td3_candidate(rows: list[str]) -> tuple[dict, list[CheckDigitResult]] | None:
    if (
        len(rows) != 2
        or len(rows[0]) != 44
        or not rows[0].startswith("P")
        or rows[0][1] not in "<M"
        or any(len(row) != 44 or not _allowed_characters(row) for row in rows)
    ):
        return None
    checks = _td3_checks(rows[1])
    if not all(item.valid for item in checks):
        return None
    return _parse_td3_fields(rows[0], rows[1]), checks


def _essential_td3_checks_valid(checks: list[CheckDigitResult]) -> bool:
    essential = {"passport_number", "date_of_birth", "date_of_expiry"}
    return all(item.valid for item in checks if item.field in essential) and any(
        item.field == "passport_number" for item in checks
    )


def first_document_code(rows: list[str]) -> str:
    return rows[0][0:2] if rows and len(rows[0]) >= 2 else ""


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
                rows = [texts[0][offset : offset + row_length] for offset in range(0, len(texts[0]), row_length)]
                return _normalize_td3_order(
                    rows,
                    {"method": "split_concatenated_row", "source_row_count": 1, "format": format_name},
                )
    return _normalize_td3_order(texts, {"method": "row_results", "source_row_count": len(texts)})


def _normalize_td3_order(rows: list[str], reconstruction: dict) -> tuple[list[str], dict]:
    """Put the TD3 document/name row before the personal-data row."""
    if len(rows) == 2 and all(len(row) == 44 for row in rows):
        first_is_name = rows[0].startswith("P<")
        second_is_name = rows[1].startswith("P<")
        if second_is_name and not first_is_name:
            return [rows[1], rows[0]], {**reconstruction, "row_order": "swapped"}
    return rows, reconstruction


def detect_format(rows: list[str]) -> str | None:
    for format_name, (row_count, row_length) in FORMAT_LENGTHS.items():
        if len(rows) == row_count and all(len(row) == row_length for row in rows):
            return format_name
    return None


def _parse_td3_fields(first: str, second: str) -> dict:
    name_field = _clean_name_field(first[5:44]).rstrip("<")
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


def _clean_name_field(value: str) -> str:
    """Drop OCR noise that appears after the final filler run in the name field."""
    match = re.search(r"<{3}[A-Z0-9]", value)
    return value[: match.start()] if match else value


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
