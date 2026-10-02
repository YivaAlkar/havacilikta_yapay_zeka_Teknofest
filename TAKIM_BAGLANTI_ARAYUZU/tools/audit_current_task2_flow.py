from __future__ import annotations

import ast
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]

TARGET_FILES = [
    PROJECT_ROOT / "main.py",
    PROJECT_ROOT / "src" / "object_detection_model.py",
    PROJECT_ROOT / "src" / "connection_handler.py",
    PROJECT_ROOT / "src" / "constants.py",
    PROJECT_ROOT / "src" / "detected_translation.py",
    PROJECT_ROOT / "src" / "reference_prediction.py",
]

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "task2_code_audit"

TASK2_KEYWORDS = {
    "translation",
    "health_status",
    "last_translation",
    "gt_translation",
    "csv",
    "frame_number",
    "frame_id",
    "fallback",
    "offset",
    "scale",
    "rotation",
    "calibration",
    "camera",
    "interpolation",
    "smooth",
    "kalman",
}

CALL_KEYWORDS = {
    "add_translation",
    "add_translation_object",
    "_add_task_2_translation",
    "translation",
    "health_status",
}

SENSITIVE_PATTERNS = [
    re.compile(r"password\s*=", re.IGNORECASE),
    re.compile(r"team_name\s*=", re.IGNORECASE),
    re.compile(r"evaluation_server_url\s*=", re.IGNORECASE),
]


def read_text_safely(path: Path) -> str:
    encodings = (
        "utf-8",
        "utf-8-sig",
        "cp1254",
        "latin-1",
    )

    for encoding in encodings:
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
        except OSError:
            return ""

    return ""


def is_sensitive_line(line: str) -> bool:
    return any(pattern.search(line) for pattern in SENSITIVE_PATTERNS)


def sanitize_line(line: str) -> str:
    if is_sensitive_line(line):
        return "[SENSITIVE LINE REDACTED]"

    return line.rstrip()


def node_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id

    if isinstance(node, ast.Attribute):
        parent = node_name(node.value)

        if parent:
            return f"{parent}.{node.attr}"

        return node.attr

    if isinstance(node, ast.Call):
        return node_name(node.func)

    return ""


def get_source_segment(
    source_lines: list[str],
    start_line: int,
    end_line: int,
) -> list[str]:
    result: list[str] = []

    for line_number in range(start_line, end_line + 1):
        if line_number < 1 or line_number > len(source_lines):
            continue

        line = source_lines[line_number - 1]
        result.append(
            f"{line_number:04d}: {sanitize_line(line)}"
        )

    return result


def function_is_task2_related(
    function_name: str,
    source_segment: str,
) -> bool:
    lowered_name = function_name.lower()
    lowered_source = source_segment.lower()

    if any(keyword in lowered_name for keyword in TASK2_KEYWORDS):
        return True

    return any(keyword in lowered_source for keyword in TASK2_KEYWORDS)


def extract_function_calls(node: ast.AST) -> list[str]:
    calls: list[str] = []

    for child in ast.walk(node):
        if isinstance(child, ast.Call):
            call_name = node_name(child.func)

            if call_name:
                calls.append(call_name)

    return sorted(set(calls))


def extract_assignments(node: ast.AST) -> list[str]:
    assignments: list[str] = []

    for child in ast.walk(node):
        if isinstance(child, ast.Assign):
            target_names = [
                node_name(target)
                for target in child.targets
            ]

            target_names = [
                name for name in target_names if name
            ]

            if target_names:
                assignments.append(", ".join(target_names))

        elif isinstance(child, ast.AnnAssign):
            target_name = node_name(child.target)

            if target_name:
                assignments.append(target_name)

        elif isinstance(child, ast.AugAssign):
            target_name = node_name(child.target)

            if target_name:
                assignments.append(target_name)

    return sorted(set(assignments))


def extract_conditions(node: ast.AST) -> list[dict[str, Any]]:
    conditions: list[dict[str, Any]] = []

    for child in ast.walk(node):
        if isinstance(child, ast.If):
            conditions.append(
                {
                    "line": child.lineno,
                    "condition": ast.unparse(child.test),
                }
            )

    conditions.sort(key=lambda item: item["line"])
    return conditions


def extract_returns(node: ast.AST) -> list[dict[str, Any]]:
    returns: list[dict[str, Any]] = []

    for child in ast.walk(node):
        if isinstance(child, ast.Return):
            value = (
                ast.unparse(child.value)
                if child.value is not None
                else "None"
            )

            returns.append(
                {
                    "line": child.lineno,
                    "value": value,
                }
            )

    returns.sort(key=lambda item: item["line"])
    return returns


def analyze_python_file(path: Path) -> dict[str, Any]:
    report: dict[str, Any] = {
        "path": str(path.relative_to(PROJECT_ROOT)),
        "exists": path.is_file(),
        "parse_ok": False,
        "classes": [],
        "task2_functions": [],
        "task2_lines": [],
        "imports": [],
        "error": None,
    }

    if not path.is_file():
        report["error"] = "file_not_found"
        return report

    source = read_text_safely(path)

    if not source:
        report["error"] = "file_empty_or_unreadable"
        return report

    source_lines = source.splitlines()

    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError as exc:
        report["error"] = f"SyntaxError: {exc}"
        return report

    report["parse_ok"] = True

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                report["imports"].append(alias.name)

        elif isinstance(node, ast.ImportFrom):
            report["imports"].append(node.module or "")

        elif isinstance(node, ast.ClassDef):
            report["classes"].append(
                {
                    "name": node.name,
                    "line_start": node.lineno,
                    "line_end": getattr(
                        node,
                        "end_lineno",
                        node.lineno,
                    ),
                }
            )

        elif isinstance(
            node,
            (ast.FunctionDef, ast.AsyncFunctionDef),
        ):
            line_start = node.lineno
            line_end = getattr(
                node,
                "end_lineno",
                node.lineno,
            )

            raw_segment = "\n".join(
                source_lines[line_start - 1:line_end]
            )

            if not function_is_task2_related(
                node.name,
                raw_segment,
            ):
                continue

            report["task2_functions"].append(
                {
                    "name": node.name,
                    "line_start": line_start,
                    "line_end": line_end,
                    "arguments": [
                        argument.arg
                        for argument in node.args.args
                    ],
                    "calls": extract_function_calls(node),
                    "assignments": extract_assignments(node),
                    "conditions": extract_conditions(node),
                    "returns": extract_returns(node),
                    "source": get_source_segment(
                        source_lines,
                        line_start,
                        line_end,
                    ),
                }
            )

    keyword_regex = re.compile(
        "|".join(
            re.escape(keyword)
            for keyword in sorted(
                TASK2_KEYWORDS,
                key=len,
                reverse=True,
            )
        ),
        re.IGNORECASE,
    )

    for line_number, line in enumerate(
        source_lines,
        start=1,
    ):
        if keyword_regex.search(line):
            report["task2_lines"].append(
                {
                    "line": line_number,
                    "text": sanitize_line(line.strip()),
                }
            )

    report["imports"] = sorted(
        set(report["imports"])
    )
    report["classes"].sort(
        key=lambda item: item["line_start"]
    )
    report["task2_functions"].sort(
        key=lambda item: item["line_start"]
    )

    return report


def detect_task2_features(
    file_reports: list[dict[str, Any]],
) -> dict[str, Any]:
    all_text_parts: list[str] = []

    for report in file_reports:
        for function in report["task2_functions"]:
            all_text_parts.append(
                "\n".join(function["source"])
            )

        for line in report["task2_lines"]:
            all_text_parts.append(line["text"])

    combined = "\n".join(all_text_parts).lower()

    feature_checks = {
        "uses_health_status": (
            "health_status" in combined
        ),
        "uses_last_translation": (
            "last_translation" in combined
        ),
        "uses_gt_translation": (
            "gt_translation" in combined
        ),
        "uses_csv": (
            "csv" in combined
        ),
        "uses_scale": (
            "scale" in combined
        ),
        "uses_offset": (
            "offset" in combined
        ),
        "uses_rotation": (
            "rotation" in combined
            or "rotate" in combined
        ),
        "uses_calibration": (
            "calibration" in combined
            or "kalibrasyon" in combined
            or "intrinsic" in combined
        ),
        "uses_interpolation": (
            "interpolation" in combined
            or "interpolate" in combined
        ),
        "uses_smoothing": (
            "smooth" in combined
            or "kalman" in combined
        ),
        "adds_translation_object": (
            "add_translation_object" in combined
        ),
        "explicit_zero_fallback": bool(
            re.search(
                r"\(\s*0(?:\.0)?\s*,\s*0(?:\.0)?\s*,\s*0(?:\.0)?\s*\)",
                combined,
            )
        ),
    }

    return feature_checks


def build_text_report(
    report: dict[str, Any],
) -> str:
    lines: list[str] = []

    lines.append("=" * 100)
    lines.append("MEVCUT TASK 2 KOD AKIŞI AUDIT RAPORU")
    lines.append("=" * 100)
    lines.append(f"Tarih: {report['audit_time']}")
    lines.append(f"Proje: {report['project_root']}")
    lines.append("")

    lines.append("[ÖZELLİK TESPİTİ]")

    for name, value in report[
        "detected_features"
    ].items():
        lines.append(f"- {name}: {value}")

    lines.append("")
    lines.append("[DOSYA BAZLI TASK 2 FONKSİYONLARI]")

    for file_report in report["files"]:
        lines.append("")
        lines.append("-" * 100)
        lines.append(f"Dosya: {file_report['path']}")
        lines.append(f"Var: {file_report['exists']}")
        lines.append(
            f"Parse başarılı: "
            f"{file_report['parse_ok']}"
        )
        lines.append(f"Hata: {file_report['error']}")

        if not file_report["task2_functions"]:
            lines.append(
                "Task 2 ile ilişkili fonksiyon bulunamadı."
            )
            continue

        for function in file_report["task2_functions"]:
            lines.append("")
            lines.append(
                f"Fonksiyon: {function['name']} "
                f"({function['line_start']}-"
                f"{function['line_end']})"
            )
            lines.append(
                f"Argümanlar: {function['arguments']}"
            )
            lines.append(
                f"Çağrılar: {function['calls']}"
            )
            lines.append(
                f"Atamalar: {function['assignments']}"
            )

            lines.append("Koşullar:")

            if function["conditions"]:
                for condition in function["conditions"]:
                    lines.append(
                        f"  Satır {condition['line']}: "
                        f"{condition['condition']}"
                    )
            else:
                lines.append("  Yok")

            lines.append("Return ifadeleri:")

            if function["returns"]:
                for return_item in function["returns"]:
                    lines.append(
                        f"  Satır {return_item['line']}: "
                        f"{return_item['value']}"
                    )
            else:
                lines.append("  Yok")

            lines.append("Kaynak kod:")
            lines.extend(
                f"  {source_line}"
                for source_line in function["source"]
            )

    lines.append("")
    lines.append("=" * 100)
    lines.append("[GÜVENLİK]")
    lines.append("- Üretim kodu değiştirilmedi.")
    lines.append("- Model dosyaları değiştirilmedi.")
    lines.append("- main.py çalıştırılmadı.")
    lines.append("- Sunucuya bağlanılmadı.")
    lines.append("- .env okunmadı.")
    lines.append(
        "- Kullanıcı adı ve şifre rapora alınmadı."
    )

    return "\n".join(lines)


def main() -> None:
    print("=" * 80)
    print("MEVCUT TASK 2 KOD AKIŞI AUDIT")
    print("=" * 80)
    print()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    file_reports: list[dict[str, Any]] = []

    for target_file in TARGET_FILES:
        print(
            f"[OKUNUYOR] "
            f"{target_file.relative_to(PROJECT_ROOT)}"
        )

        file_reports.append(
            analyze_python_file(target_file)
        )

    report: dict[str, Any] = {
        "audit_time": datetime.now().isoformat(
            timespec="seconds"
        ),
        "project_root": str(PROJECT_ROOT),
        "files": file_reports,
        "detected_features": detect_task2_features(
            file_reports
        ),
        "safety": {
            "production_code_modified": False,
            "model_files_modified": False,
            "main_executed": False,
            "server_connection_attempted": False,
            "env_read": False,
        },
    }

    json_path = (
        OUTPUT_DIR / "current_task2_flow_report.json"
    )
    text_path = (
        OUTPUT_DIR / "current_task2_flow_report.txt"
    )

    json_path.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    text_path.write_text(
        build_text_report(report),
        encoding="utf-8",
    )

    total_functions = sum(
        len(file_report["task2_functions"])
        for file_report in file_reports
    )

    parse_errors = sum(
        1
        for file_report in file_reports
        if file_report["error"] is not None
    )

    print()
    print("=" * 80)
    print("AUDIT TAMAMLANDI")
    print("=" * 80)
    print(
        f"İncelenen dosya       : "
        f"{len(file_reports)}"
    )
    print(
        f"Task 2 fonksiyonu     : "
        f"{total_functions}"
    )
    print(
        f"Dosya/parse hatası    : "
        f"{parse_errors}"
    )
    print()
    print(f"Metin raporu: {text_path}")
    print(f"JSON raporu : {json_path}")
    print()
    print("Üretim kodu değiştirilmedi.")
    print("main.py çalıştırılmadı.")
    print("Sunucu bağlantısı kurulmadı.")


if __name__ == "__main__":
    main()