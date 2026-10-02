from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]

TESTS = [
    "test_local_model.py",
    "test_motion_status.py",
    "test_health_status.py",
    "test_reference_matcher.py",
    "test_task3_refs.py",
    "tools/task1_detection_stats.py",
    "tools/test_task2_csv_integration.py",
    "tools/check_model_loader.py",
    "tools/check_custom_model_mapping.py",
    "tools/offline_crash_test.py",
]

def run_test(test_path: str):
    full_path = ROOT / test_path

    print("\n" + "=" * 90)
    print(f"[RUN] {test_path}")
    print("=" * 90)

    if not full_path.exists():
        print(f"[SKIP] Dosya bulunamadı: {full_path}")
        return True

    start = time.time()

    result = subprocess.run(
        [sys.executable, str(full_path)],
        cwd=str(ROOT),
        text=True,
        capture_output=True,
    )

    elapsed = time.time() - start

    if result.stdout:
        print(result.stdout)

    if result.stderr:
        print("[STDERR]")
        print(result.stderr)

    if result.returncode == 0:
        print(f"[OK] {test_path} ({elapsed:.1f}s)")
        return True

    print(f"[FAIL] {test_path} returncode={result.returncode} ({elapsed:.1f}s)")
    return False


def main():
    print("=" * 90)
    print("[START] Tüm testler çalıştırılıyor")
    print(f"[ROOT] {ROOT}")
    print("=" * 90)

    passed = 0
    failed = 0
    skipped_or_ok = 0

    for test in TESTS:
        ok = run_test(test)

        if ok:
            passed += 1
        else:
            failed += 1
            print("\n[STOP] İlk hata burada. Önce bunu düzeltmek daha mantıklı.")
            break

    print("\n" + "=" * 90)
    print("[SUMMARY]")
    print(f"passed: {passed}")
    print(f"failed: {failed}")
    print(f"total attempted: {passed + failed}")
    print("=" * 90)

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()