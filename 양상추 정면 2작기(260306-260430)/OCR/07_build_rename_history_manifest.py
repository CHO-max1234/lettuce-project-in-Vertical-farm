from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_IDENTITY_DIR = Path(__file__).resolve().parent / "outputs_identity_260424"
DEFAULT_OUTPUTS_DIR = Path(
    r"G:\다른 컴퓨터\내 컴퓨터\새 폴더\양상추사진\양상추_테라웨이브\양상추 날짜"
    r"\RGB_정면\4. 결과 출력 시각화\2작기 error\outputs"
)
DEFAULT_OCR_XLSX = Path(
    r"G:\다른 컴퓨터\내 컴퓨터\새 폴더\양상추사진\양상추_테라웨이브\양상추 날짜"
    r"\OCR_Piece\2작기\260422_ocr_cam1_.xlsx"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--identity-dir", type=Path, default=DEFAULT_IDENTITY_DIR)
    parser.add_argument("--outputs-dir", type=Path, default=DEFAULT_OUTPUTS_DIR)
    parser.add_argument("--ocr-xlsx", type=Path, default=DEFAULT_OCR_XLSX)
    parser.add_argument("--out-csv", type=Path, default=DEFAULT_IDENTITY_DIR / "C. rename_history_manifest.csv")
    return parser.parse_args()


def build_new_full_path(old_full_path: str, new_file_name: str) -> str:
    if not old_full_path or not new_file_name:
        return ""
    return str(Path(old_full_path).with_name(new_file_name))


def main() -> None:
    args = parse_args()
    b_path = args.identity_dir / "B. old_new_name_key.csv"
    triplet_path = args.outputs_dir / "02_triplet_manifest_complete_only.xlsx"

    b_df = pd.read_csv(b_path)
    triplet_df = pd.read_excel(triplet_path)
    ocr_df = pd.read_excel(args.ocr_xlsx)

    triplet_df["triplet_key"] = triplet_df["triplet_key"].astype(str)
    ocr_df.columns = [str(c) for c in ocr_df.columns]
    ocr_df["saved_bed"] = ocr_df["saved_bed"].astype(str)
    ocr_df["date"] = pd.to_numeric(ocr_df["date"], errors="coerce").astype("Int64").astype(str)
    ocr_df["time"] = ocr_df["time"].astype(str).str.replace(".0", "", regex=False).str.zfill(6)
    ocr_df["triplet_key"] = ocr_df["saved_bed"] + "_" + ocr_df["date"] + "_" + ocr_df["time"]

    b_df["triplet_key_old"] = b_df["triplet_key_old"].astype(str)
    b_df["cam"] = b_df["cam"].astype(str)
    b_df["date"] = b_df["date"].astype(str)
    b_df["time"] = b_df["time"].astype(str).str.zfill(6)

    pivot_keep = [
        "triplet_key_old",
        "triplet_key_new",
        "old_saved_bed",
        "saved_num",
        "real_bed_id",
        "date",
        "time",
        "rule_id",
        "rule_applied",
        "qc_status",
        "source_reason",
        "human_real_bed_id",
        "rule_vs_human_match",
        "rename_ready_flag",
        "review_required",
    ]
    triplet_level = b_df[pivot_keep].drop_duplicates(subset=["triplet_key_old"]).copy()

    cam_pivot = (
        b_df.pivot_table(
            index="triplet_key_old",
            columns="cam",
            values=["old_file_name", "new_file_name"],
            aggfunc="first",
        )
        .sort_index(axis=1)
    )
    cam_pivot.columns = [f"{a}_{b}" for a, b in cam_pivot.columns]
    cam_pivot = cam_pivot.reset_index()

    merged = triplet_level.merge(cam_pivot, on="triplet_key_old", how="left")
    merged = merged.merge(
        triplet_df[
            [
                "triplet_key",
                "cam0_path",
                "cam1_path",
                "cam2_path",
            ]
        ].rename(columns={"triplet_key": "triplet_key_old"}),
        on="triplet_key_old",
        how="left",
    )

    merged = merged.merge(
        ocr_df[
            [
                "triplet_key",
                "detected_num",
                "manual_visible_bed_num",
                "human",
                "confidence",
                "ocr_best_conf",
                "ocr_text_len",
                "note",
            ]
        ].rename(columns={"triplet_key": "triplet_key_old", "human": "ocr_human_raw"}),
        on="triplet_key_old",
        how="left",
    )

    merged["cam0_old_path"] = merged["cam0_path"]
    merged["cam1_old_path"] = merged["cam1_path"]
    merged["cam2_old_path"] = merged["cam2_path"]

    merged["cam0_new_path"] = merged.apply(lambda r: build_new_full_path(r["cam0_old_path"], r.get("new_file_name_cam0", "")), axis=1)
    merged["cam1_new_path"] = merged.apply(lambda r: build_new_full_path(r["cam1_old_path"], r.get("new_file_name_cam1", "")), axis=1)
    merged["cam2_new_path"] = merged.apply(lambda r: build_new_full_path(r["cam2_old_path"], r.get("new_file_name_cam2", "")), axis=1)

    merged["old_full_path"] = merged["cam1_old_path"]
    merged["new_full_path"] = merged["cam1_new_path"]
    merged["old_file_name"] = merged["old_file_name_cam1"]
    merged["new_file_name"] = merged["new_file_name_cam1"]
    merged["saved_bed"] = merged["old_saved_bed"]
    merged["manual_visible_bed_num"] = merged["human_real_bed_id"]

    out = merged[
        [
            "old_full_path",
            "new_full_path",
            "old_file_name",
            "new_file_name",
            "triplet_key_old",
            "triplet_key_new",
            "cam0_old_path",
            "cam1_old_path",
            "cam2_old_path",
            "cam0_new_path",
            "cam1_new_path",
            "cam2_new_path",
            "saved_bed",
            "saved_num",
            "real_bed_id",
            "detected_num",
            "manual_visible_bed_num",
            "rule_id",
            "rule_applied",
            "qc_status",
            "rename_ready_flag",
            "review_required",
            "source_reason",
            "rule_vs_human_match",
            "confidence",
            "ocr_best_conf",
            "ocr_text_len",
            "note",
        ]
    ].rename(columns={"detected_num": "ocr_detected_num"})

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False, encoding="utf-8-sig")

    print(f"[DONE] wrote {args.out_csv}")
    print(f"[INFO] rows = {len(out)}")
    print("[INFO] rename_ready_flag")
    print(out["rename_ready_flag"].value_counts(dropna=False).to_string())
    print("[INFO] review_required")
    print(out["review_required"].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()
