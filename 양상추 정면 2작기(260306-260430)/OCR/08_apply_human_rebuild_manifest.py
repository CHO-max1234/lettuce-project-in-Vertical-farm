from __future__ import annotations

import argparse
import csv
import os
import uuid


TOP_ROOT = r"/content/drive/Othercomputers/내 컴퓨터/새 폴더/양상추사진/양상추_테라웨이브/양상추 날짜/RGB_윗면/0. 원본/2작기"
FRONT_ROOT = r"/content/drive/Othercomputers/내 컴퓨터/새 폴더/양상추사진/양상추_테라웨이브/양상추 날짜/RGB_정면/0. 원본/2작기"

MANIFEST_CSV = r"/content/drive/MyDrive/양상추 분류모델/practice code/양상추 정면 2작기(260306-260430)/OCR/outputs_identity_260424/C. rename_history_manifest_human_rebuild.csv"
LOG_CSV = r"/content/drive/MyDrive/양상추 분류모델/practice code/양상추 정면 2작기(260306-260430)/OCR/outputs_identity_260424/rename_apply_log_human_rebuild_v4.csv"

DRY_RUN = True
ONLY_READY_ROWS = True
READY_FLAG_COLUMN = "rename_ready_flag"
STRICT_VALIDATE = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply-rename", action="store_true")
    return parser.parse_known_args()[0]


def load_manifest_rows(csv_path: str) -> list[dict]:
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as file_obj:
        return list(csv.DictReader(file_obj))


def str_to_bool(value: object) -> bool:
    return str(value).strip().lower() == "true"


def has_path_separator(file_name: str) -> bool:
    text = str(file_name or "")
    return ("/" in text) or ("\\" in text)


def date_to_folder_name(date_text: str) -> str:
    text = str(date_text or "").strip()
    if len(text) != 8 or not text.isdigit():
        raise ValueError(f"invalid date: {date_text}")
    return text[2:]


def pick_root_by_cam(cam_name: str) -> str:
    if cam_name in ("cam0", "cam1"):
        return TOP_ROOT
    if cam_name == "cam2":
        return FRONT_ROOT
    raise ValueError(f"unknown cam: {cam_name}")


def build_actual_path(root_dir: str, date_text: str, file_name: str) -> str:
    return os.path.join(root_dir, date_to_folder_name(date_text), file_name)


def should_use_row(row: dict, has_ready_flag: bool) -> bool:
    if not has_ready_flag or not ONLY_READY_ROWS:
        return True
    return str_to_bool(row.get(READY_FLAG_COLUMN, ""))


def build_jobs(rows: list[dict]) -> list[dict]:
    jobs = []
    has_ready_flag = bool(rows) and READY_FLAG_COLUMN in rows[0]

    for row in rows:
        if not should_use_row(row, has_ready_flag):
            continue

        cam = str(row.get("cam", "") or "").strip()
        date_text = str(row.get("date", "") or "").strip()
        current_file_name = str(row.get("current_file_name", "") or "").strip()
        new_file_name = str(row.get("new_file_name", "") or "").strip()

        if not cam or not date_text or not current_file_name or not new_file_name:
            continue

        root_dir = pick_root_by_cam(cam)
        current_file_path = build_actual_path(root_dir, date_text, current_file_name)
        new_file_path = build_actual_path(root_dir, date_text, new_file_name)

        jobs.append(
            {
                "triplet_key_old": str(row.get("triplet_key_old", "") or "").strip(),
                "cam": cam,
                "date": date_text,
                "time": str(row.get("time", "") or "").strip(),
                "saved_bed": str(row.get("saved_bed", "") or "").strip(),
                "saved_num": str(row.get("saved_num", "") or "").strip(),
                "human_visible_bed_num": str(row.get("human_visible_bed_num", "") or "").strip(),
                "ocr_detected_num": str(row.get("ocr_detected_num", "") or "").strip(),
                "rule_inferred_bed": str(row.get("rule_inferred_bed", "") or "").strip(),
                "decision_source": str(row.get("decision_source", "") or "").strip(),
                "current_state": str(row.get("current_state", "") or "").strip(),
                "root_dir": root_dir,
                "current_file_name": current_file_name,
                "new_file_name": new_file_name,
                "current_file_path": current_file_path,
                "new_file_path": new_file_path,
                "manifest_current_file_path": str(row.get("current_file_path", "") or "").strip(),
                "manifest_new_file_path": str(row.get("new_file_path", "") or "").strip(),
            }
        )

    return jobs


def classify_jobs(
    jobs: list[dict],
) -> tuple[list[dict], list[dict], list[str], list[str], list[str], list[str]]:
    actionable_jobs = []
    preview_logs = []
    duplicate_new = []
    bad_current_names = []
    bad_new_names = []
    conflicts = []
    missing_current = []

    new_path_counts: dict[str, int] = {}
    for job in jobs:
        new_path_counts[job["new_file_path"]] = new_path_counts.get(job["new_file_path"], 0) + 1
    duplicate_new = [path for path, count in new_path_counts.items() if count > 1]
    duplicate_new_set = set(duplicate_new)

    for job in jobs:
        current_file_name = job["current_file_name"]
        new_file_name = job["new_file_name"]
        current_file_path = job["current_file_path"]
        new_file_path = job["new_file_path"]

        current_exists = os.path.exists(current_file_path)
        new_exists = os.path.exists(new_file_path)

        if has_path_separator(current_file_name):
            bad_current_names.append(current_file_name)
            status = "invalid_current_file_name"
        elif has_path_separator(new_file_name):
            bad_new_names.append(new_file_name)
            status = "invalid_new_file_name"
        elif new_file_path in duplicate_new_set:
            status = "duplicate_new_path"
        elif current_file_name == new_file_name:
            status = "pass_same_name" if new_exists or current_exists else "missing_same_name"
            if not (new_exists or current_exists):
                missing_current.append(current_file_path)
        elif new_exists and not current_exists:
            status = "pass_new_exists"
        elif current_exists and not new_exists:
            status = "rename_needed"
            actionable_jobs.append(job)
        elif current_exists and new_exists:
            status = "conflict_both_exist"
            conflicts.append(new_file_path)
        else:
            status = "missing_current"
            missing_current.append(current_file_path)

        preview_logs.append({**job, "status": status, "temp_path": ""})

    return (
        actionable_jobs,
        preview_logs,
        list(dict.fromkeys(duplicate_new)),
        list(dict.fromkeys(bad_current_names)),
        list(dict.fromkeys(bad_new_names)),
        list(dict.fromkeys(conflicts + missing_current)),
    )


def write_log(log_path: str, rows: list[dict]) -> None:
    if not rows:
        return
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "w", encoding="utf-8-sig", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def apply_two_phase_rename(jobs: list[dict]) -> list[dict]:
    logs = []
    temp_rows = []

    for job in jobs:
        current_file_path = job["current_file_path"]
        new_file_path = job["new_file_path"]

        temp_path = f"{current_file_path}.tmp_rename_{uuid.uuid4().hex}"
        os.rename(current_file_path, temp_path)
        temp_rows.append((job, temp_path))
        logs.append({**job, "status": "temp_renamed", "temp_path": temp_path})

    for job, temp_path in temp_rows:
        os.rename(temp_path, job["new_file_path"])
        logs.append({**job, "status": "final_renamed", "temp_path": temp_path})

    return logs


def main() -> None:
    args = parse_args()
    dry_run = DRY_RUN and not args.apply_rename

    rows = load_manifest_rows(MANIFEST_CSV)
    jobs = build_jobs(rows)

    (
        actionable_jobs,
        preview_logs,
        duplicate_new,
        bad_current_names,
        bad_new_names,
        blocking_items,
    ) = classify_jobs(jobs)

    pass_count = sum(1 for row in preview_logs if row["status"] in ("pass_new_exists", "pass_same_name"))
    rename_needed_count = sum(1 for row in preview_logs if row["status"] == "rename_needed")
    conflict_count = sum(1 for row in preview_logs if row["status"] == "conflict_both_exist")
    missing_count = sum(1 for row in preview_logs if row["status"] in ("missing_current", "missing_same_name"))

    print(f"manifest rows: {len(rows)}")
    print(f"jobs: {len(jobs)}")
    print(f"DRY_RUN={dry_run}")
    print("=" * 60)
    print(f"pass already matched: {pass_count}")
    print(f"rename needed: {rename_needed_count}")
    print(f"conflict both exist: {conflict_count}")
    print(f"missing current: {missing_count}")
    print(f"duplicate new paths: {len(duplicate_new)}")
    print(f"bad current file names: {len(bad_current_names)}")
    print(f"bad new file names: {len(bad_new_names)}")
    print("=" * 60)

    if dry_run:
        write_log(LOG_CSV, preview_logs)
        print(f"[DRY RUN] log saved: {LOG_CSV}")
        return

    if STRICT_VALIDATE and (duplicate_new or bad_current_names or bad_new_names or blocking_items):
        write_log(LOG_CSV, preview_logs)
        print("Validation failed. Rename stopped.")
        print(f"log saved: {LOG_CSV}")
        return

    logs = apply_two_phase_rename(actionable_jobs)
    write_log(LOG_CSV, logs)
    print("rename complete")
    print(f"log saved: {LOG_CSV}")


if __name__ == "__main__":
    main()
