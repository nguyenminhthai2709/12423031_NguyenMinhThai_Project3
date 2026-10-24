"""Tự kiểm tra tiến độ của thư mục đề tài (SV chạy hằng tuần; GV chạy hàng loạt bằng _admin/check_all.py).

Kiểm tra: dữ liệu đã chuẩn bị chưa, lõi chung có bị sửa không, đã có run nào hoàn tất, đã có bảng kết quả
nào trong results/, và trạng thái từng test (test lõi + test hợp đồng của đề tài):
    ĐẠT       : test qua
    CHƯA LÀM  : hàm của đề tài còn để TODO (NotImplementedError) hoặc chưa có tệp kết quả cần kiểm tra
    SAI       : đã làm nhưng chưa đúng tiêu chí → đọc thông báo lỗi
    KHÔNG ÁP DỤNG: test lõi của thành phần đề tài không dùng (ví dụ mô hình nền không có trong đề tài)

    python -m tools.progress.check
Kết quả ghi vào results/progress_status.json.
"""

import hashlib
import io
import json
import sys
from contextlib import redirect_stdout
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config_path.config_path import LOG_DIR, RESULT_DIR, resolve_dataset  # noqa: E402

LABEL = {"passed": "ĐẠT", "skipped": "CHƯA LÀM", "failed": "SAI"}


class _Collector:
    def __init__(self):
        self.outcomes = {}
        self.docs = {}

    def pytest_collection_modifyitems(self, items):
        for item in items:
            doc = getattr(getattr(item, "function", None), "__doc__", None) or item.name
            self.docs[item.nodeid] = doc.strip().splitlines()[0]

    def pytest_runtest_logreport(self, report):
        if report.when == "call" or (report.when == "setup" and report.outcome != "passed"):
            message = ""
            if report.outcome == "skipped" and isinstance(report.longrepr, tuple):
                message = str(report.longrepr[2]).replace("Skipped: ", "")
            elif report.outcome == "failed":
                message = str(getattr(report, "longreprtext", "") or report.longrepr).strip().splitlines()[-1][:300]
            self.outcomes[report.nodeid] = (report.outcome, message)


def core_integrity():
    manifest_path = ROOT / "core_manifest.json"
    if not manifest_path.is_file():
        return {"ok": False, "changed": ["(thiếu core_manifest.json)"]}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    changed = []
    for relative, expected in manifest["files"].items():
        path = ROOT / relative
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            changed.append(relative)
    watched = [relative for relative, expected in manifest.get("watch", {}).items()
               if not (ROOT / relative).is_file() or hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() != expected]
    return {"ok": not changed, "version": manifest.get("core_version"), "changed": changed,
            "topic_config_changed": watched}


def run_tests():
    import pytest

    collector = _Collector()
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        pytest.main(["-q", "-p", "no:cacheprovider", "--tb=line", str(ROOT / "tests")], plugins=[collector])
    rows = []
    for nodeid, (outcome, message) in sorted(collector.outcomes.items()):
        kind = "hợp đồng" if "test_contract_" in nodeid else "lõi"
        label = LABEL.get(outcome, outcome)
        if outcome == "skipped" and message.startswith("KHÔNG ÁP DỤNG"):
            label = "KHÔNG ÁP DỤNG"
        rows.append({"test": nodeid.split("::")[-1], "file": nodeid.split("::")[0], "kind": kind,
                     "description": collector.docs.get(nodeid, ""), "status": label,
                     "message": message})
    return rows


def completed_runs(datasets=None):
    """Run nền đã hoàn tất, chỉ tính run thuộc dataset của đề tài (đã quy về dataset MovieLens đang chọn)."""
    wanted = None if datasets is None else {resolve_dataset(name) for name in datasets}
    runs = {}
    for manifest in LOG_DIR.glob("*/*/*/seed*/run_manifest.json"):
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        if wanted is not None and manifest.parents[2].name not in wanted:
            continue
        if payload.get("status") == "completed":
            runs.setdefault(payload["model"], []).append("{}/{}".format(payload.get("tag"), manifest.parent.name))
    return runs


def data_ready(datasets):
    """Split đã dựng và giao thức nhóm đã sinh cho từng dữ liệu đề tài cần."""
    names = [resolve_dataset(name) for name in datasets]
    return {name: (ROOT / "dataset_verify" / name / "split_manifest.json").is_file()
            and (ROOT / "preprocessed" / "evaluation_protocol" / name / "manifest.json").is_file()
            for name in names}


def main():
    topic = json.loads((ROOT / "topic.json").read_text(encoding="utf-8")) if (ROOT / "topic.json").is_file() else {}
    status = {"topic": topic.get("code"), "checked_at": datetime.now().isoformat(timespec="seconds"),
              "data_ready": data_ready(topic.get("datasets", ["ml-100k"])),
              "core": core_integrity(), "runs": completed_runs(topic.get("datasets", ["ml-100k"])),
              "results_files": sorted(p.name for p in RESULT_DIR.glob("*") if p.is_file() and p.name != "progress_status.json"),
              "tests": run_tests()}
    counts = {}
    for row in status["tests"]:
        key = (row["kind"], row["status"])
        counts[key] = counts.get(key, 0) + 1
    status["summary"] = {"{} {}".format(kind, label): value for (kind, label), value in sorted(counts.items())}
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    (RESULT_DIR / "progress_status.json").write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Đề tài:", topic.get("code"), "-", topic.get("title", ""))
    for name, ready in status["data_ready"].items():
        print("Dữ liệu {} đã chuẩn bị:".format(name), "có" if ready else "CHƯA (chạy các lệnh chuẩn bị dữ liệu trong README)")
    print("Lõi chung nguyên vẹn:", "có" if status["core"]["ok"] else "KHÔNG – đã sửa: " + ", ".join(status["core"]["changed"]))
    if status["core"].get("topic_config_changed"):
        print("Cấu hình đề tài khác bản phát (ghi lý do vào nhật ký):", ", ".join(status["core"]["topic_config_changed"]))
    print("Run hoàn tất:", {model: len(runs) for model, runs in status["runs"].items()} or "chưa có")
    print("Tệp trong results/:", ", ".join(status["results_files"]) or "chưa có")
    print()
    for row in status["tests"]:
        print("[{:8s}] ({}) {} — {}".format(row["status"], row["kind"], row["description"], row["message"]))
    print()
    print("Tổng hợp:", status["summary"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
