from __future__ import annotations

"""
Colab 실행 순서
1) !pip install -q ipywidgets openpyxl pillow pandas
2) from google.colab import drive; drive.mount('/content/drive')
3) 이 파일의 TOP_ROOT / FRONT_ROOT / SAVE_PATH만 수정
4) %run /content/drive/MyDrive/.../colab_triplet_reviewer.py

결과
- 같은 촬영 시점의 cam00, cam01, cam02를 한 화면에 표시
- BedNum / 날짜 검색 후 가장 빠른 시간부터 탐색
- Front / Top 입력값을 xlsx 또는 csv로 누적 저장
"""

import base64
import html
import io
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import ipywidgets as widgets
import pandas as pd
from IPython.display import display
from PIL import Image

try:
    from google.colab import output as colab_output
except ImportError:  # pragma: no cover - local Jupyter fallback
    colab_output = None


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
FILENAME_RE = re.compile(
    r"^(?P<bed>bed\d{2})_(?P<date>\d{8})_(?P<time>\d{6})_cam(?P<cam>0|1|2|00|01|02)$",
    re.IGNORECASE,
)
BASE_COLUMNS = ["image_info", "bednum", "date", "hhmm", "front_view", "top_view", "front=top?"]
CAM_COLUMNS = ["cam0_path", "cam1_path", "cam2_path"]
MATCH_TOLERANCE_SEC = 2
DATE_FOLDER_RE = re.compile(r"^\d{6}$|^\d{8}$")


@dataclass(frozen=True)
class CaptureMatch:
    image_info: str
    bednum: str
    date: str
    hhmm: str
    cam: str
    path: str
    time_seconds: int


def _enable_colab_widgets() -> None:
    if colab_output is not None:
        colab_output.enable_custom_widget_manager()


def _normalize_bed(value: str) -> str:
    token = str(value).strip().lower().replace("bed", "")
    if not token:
        return ""
    if token.isdigit():
        return token.zfill(2)
    return token


def _normalize_date(value: str) -> str:
    token = re.sub(r"[^0-9]", "", str(value).strip())
    if len(token) == 6:
        return f"20{token}"
    return token


def _iter_image_files(root: str | Path) -> Iterable[Path]:
    root_path = Path(root)
    if not root_path.exists():
        return []

    image_paths = [
        path
        for path in root_path.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    image_paths.sort(key=lambda path: (_folder_sort_key(root_path, path), path.name.lower()))
    return image_paths


def _folder_sort_key(root: Path, path: Path) -> tuple[str, ...]:
    try:
        relative_parts = path.relative_to(root).parts[:-1]
    except ValueError:
        relative_parts = path.parts[:-1]

    normalized = []
    for part in relative_parts:
        token = str(part).strip()
        normalized.append(token)
        if DATE_FOLDER_RE.match(token):
            normalized.append("0")
            normalized.append(token if len(token) == 8 else f"20{token}")
        else:
            normalized.append("1")
            normalized.append(token.lower())
    return tuple(normalized)


def _hhmmss_to_seconds(value: str) -> int:
    token = str(value).strip()
    return int(token[:2]) * 3600 + int(token[2:4]) * 60 + int(token[4:6])


def _seconds_to_hhmmss(value: int) -> str:
    seconds = max(0, int(value))
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    return f"{hours:02d}{minutes:02d}{secs:02d}"


def _parse_capture(path: Path) -> CaptureMatch | None:
    matched = FILENAME_RE.match(path.stem)
    if not matched:
        return None

    bed_token = matched.group("bed").lower()
    date_token = matched.group("date")
    time_token = matched.group("time")
    cam_token = matched.group("cam").zfill(2)
    image_info = f"{bed_token}_{date_token}_{time_token}"

    return CaptureMatch(
        image_info=image_info,
        bednum=bed_token.replace("bed", ""),
        date=date_token,
        hhmm=time_token,
        cam=f"cam{cam_token}",
        path=str(path),
        time_seconds=_hhmmss_to_seconds(time_token),
    )


def _pair_by_time(
    left_items: list[CaptureMatch],
    right_items: list[CaptureMatch],
    tolerance_sec: int,
) -> list[tuple[CaptureMatch, CaptureMatch]]:
    if not left_items or not right_items:
        return []

    right_used: set[int] = set()
    paired: list[tuple[CaptureMatch, CaptureMatch]] = []

    for left in sorted(left_items, key=lambda item: item.time_seconds):
        best_index = None
        best_delta = None

        for idx, right in enumerate(right_items):
            if idx in right_used:
                continue

            delta = abs(left.time_seconds - right.time_seconds)
            if delta > tolerance_sec:
                continue

            if best_delta is None or delta < best_delta:
                best_index = idx
                best_delta = delta

            if delta == 0:
                break

        if best_index is None:
            continue

        right_used.add(best_index)
        paired.append((left, right_items[best_index]))

    paired.sort(key=lambda pair: min(pair[0].time_seconds, pair[1].time_seconds))
    return paired


def _build_top_events(grouped: dict[tuple[str, str], dict[str, list[CaptureMatch]]]) -> list[dict[str, str]]:
    top_events: list[dict[str, str]] = []

    for (bednum, date), cam_map in sorted(grouped.items(), key=lambda item: (item[0][1], item[0][0])):
        cam0_list = sorted(cam_map.get("cam00", []), key=lambda item: item.time_seconds)
        cam1_list = sorted(cam_map.get("cam01", []), key=lambda item: item.time_seconds)
        cam2_list = sorted(cam_map.get("cam02", []), key=lambda item: item.time_seconds)

        top_pairs = _pair_by_time(cam0_list, cam1_list, tolerance_sec=MATCH_TOLERANCE_SEC)
        top_pair_rows = []
        for cam0, cam1 in top_pairs:
            base_seconds = min(cam0.time_seconds, cam1.time_seconds)
            top_pair_rows.append(
                {
                    "bednum": bednum,
                    "date": date,
                    "hhmm": _seconds_to_hhmmss(base_seconds),
                    "time_seconds": base_seconds,
                    "cam0_path": cam0.path,
                    "cam1_path": cam1.path,
                }
            )

        if not top_pair_rows or not cam2_list:
            continue

        cam2_used: set[int] = set()
        for top_row in top_pair_rows:
            best_index = None
            best_delta = None

            for idx, cam2 in enumerate(cam2_list):
                if idx in cam2_used:
                    continue

                delta = abs(top_row["time_seconds"] - cam2.time_seconds)
                if delta > MATCH_TOLERANCE_SEC:
                    continue

                if best_delta is None or delta < best_delta:
                    best_index = idx
                    best_delta = delta

                if delta == 0:
                    break

            if best_index is None:
                continue

            cam2_used.add(best_index)
            cam2 = cam2_list[best_index]
            base_seconds = min(int(top_row["time_seconds"]), cam2.time_seconds)
            hhmm = _seconds_to_hhmmss(base_seconds)
            top_events.append(
                {
                    "image_info": f"bed{bednum}_{date}_{hhmm}",
                    "bednum": bednum,
                    "date": date,
                    "hhmm": hhmm,
                    "cam0_path": top_row["cam0_path"],
                    "cam1_path": top_row["cam1_path"],
                    "cam2_path": cam2.path,
                }
            )

    return top_events


def build_capture_table(top_root: str | Path, front_root: str | Path) -> pd.DataFrame:
    grouped: dict[tuple[str, str], dict[str, list[CaptureMatch]]] = {}

    for root in [top_root, front_root]:
        for file_path in _iter_image_files(root):
            capture = _parse_capture(file_path)
            if capture is None:
                continue

            row = grouped.setdefault((capture.bednum, capture.date), {})
            row.setdefault(capture.cam, []).append(capture)

    complete_rows = _build_top_events(grouped)

    if not complete_rows:
        return pd.DataFrame(columns=BASE_COLUMNS + CAM_COLUMNS)

    frame = pd.DataFrame(complete_rows)
    frame = frame.sort_values(["date", "hhmm", "bednum"], ascending=[True, True, True]).reset_index(drop=True)

    for column in ["front_view", "top_view", "front=top?"]:
        frame[column] = ""

    return frame[BASE_COLUMNS + CAM_COLUMNS]


def load_existing_labels(save_path: str | Path) -> pd.DataFrame:
    target = Path(save_path)
    if not target.exists():
        return pd.DataFrame(columns=BASE_COLUMNS)

    if target.suffix.lower() == ".csv":
        frame = pd.read_csv(target, dtype=str).fillna("")
    else:
        frame = pd.read_excel(target, dtype=str).fillna("")

    missing = [column for column in BASE_COLUMNS if column not in frame.columns]
    for column in missing:
        frame[column] = ""

    return frame[BASE_COLUMNS]


def merge_labels(index_frame: pd.DataFrame, save_path: str | Path) -> pd.DataFrame:
    merged = index_frame.copy()
    existing = load_existing_labels(save_path)
    if existing.empty:
        return merged

    existing = existing.drop_duplicates(subset=["image_info"], keep="last").set_index("image_info")
    for column in ["front_view", "top_view", "front=top?"]:
        merged[column] = merged["image_info"].map(existing[column]).fillna("")

    return merged


def write_annotation_table(frame: pd.DataFrame, save_path: str | Path) -> None:
    export_frame = frame[BASE_COLUMNS].copy()
    completed_mask = (
        export_frame["front_view"].astype(str).str.strip().ne("")
        & export_frame["top_view"].astype(str).str.strip().ne("")
    )
    export_frame = export_frame.loc[completed_mask].reset_index(drop=True)
    target = Path(save_path)
    target.parent.mkdir(parents=True, exist_ok=True)

    if target.suffix.lower() == ".csv":
        export_frame.to_csv(target, index=False, encoding="utf-8-sig")
    else:
        try:
            export_frame.to_excel(target, index=False, sheet_name="review")
        except ImportError as exc:  # pragma: no cover - depends on Colab environment
            raise ImportError("xlsx 저장을 위해서는 openpyxl 설치가 필요합니다. `!pip install openpyxl` 후 다시 실행해 주세요.") from exc


@lru_cache(maxsize=256)
def build_image_data_uri(path: str, max_px: int = 900) -> str:
    with Image.open(path) as image:
        image = image.convert("RGB")
        image.thumbnail((max_px, max_px))
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=92)

    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded}"


def make_triplet_html(record: pd.Series) -> str:
    cards = []
    for cam_label, path_key in [("cam00", "cam0_path"), ("cam01", "cam1_path"), ("cam02", "cam2_path")]:
        image_uri = build_image_data_uri(str(record[path_key]))
        cards.append(
            f"""
            <div style="flex:1 1 30%; min-width:280px;">
              <div style="font-size:20px; font-weight:700; margin-bottom:8px; text-align:center;">{cam_label}</div>
              <div style="height:420px; border:1px solid #cfd8e3; background:#f7fafc; border-radius:8px; overflow:hidden;">
                <img src="{image_uri}" style="width:100%; height:100%; object-fit:contain; display:block;" />
              </div>
            </div>
            """
        )

    info_text = html.escape(
        f"{record['image_info']} | bed {record['bednum']} | {record['date']} | {record['hhmm']}"
    )
    return f"""
    <div style="margin:8px 0 10px 0; font-size:16px; font-weight:600;">{info_text}</div>
    <div style="display:flex; gap:16px; flex-wrap:wrap; align-items:flex-start;">
      {''.join(cards)}
    </div>
    """


class TripletReviewer:
    def __init__(
        self,
        top_root: str,
        front_root: str,
        save_path: str,
        label_guide: str = "0=꽉 참, 1=띄엄띄엄, 2=비어있음, 3=이전 날짜와 다름",
    ) -> None:
        _enable_colab_widgets()

        indexed = build_capture_table(top_root=top_root, front_root=front_root)
        if indexed.empty:
            raise ValueError("cam0, cam1, cam2가 모두 있는 이미지 묶음을 찾지 못했습니다. 경로와 파일명을 확인해 주세요.")

        self.save_path = save_path
        self.data = merge_labels(indexed, save_path).reset_index(drop=True)
        self.filtered_indices = list(self.data.index)
        self.current_position = 0

        self.prev_button = widgets.Button(description="<", button_style="warning", layout=widgets.Layout(width="70px"))
        self.next_button = widgets.Button(description=">", button_style="warning", layout=widgets.Layout(width="70px"))
        self.search_button = widgets.Button(description="Search", button_style="info", layout=widgets.Layout(width="100px"))
        self.save_button = widgets.Button(description="Enter", button_style="success", layout=widgets.Layout(width="120px"))
        self.export_button = widgets.Button(description="Export", button_style="", layout=widgets.Layout(width="100px"))

        self.bed_input = widgets.Text(value="", description="BedNum", placeholder="00 or bed00")
        self.date_input = widgets.Text(value="", description="Yeardate", placeholder="yyyymmdd")
        self.front_input = widgets.Text(value="", description="Front", placeholder="0")
        self.top_input = widgets.Text(value="", description="Top", placeholder="1")

        self.counter_html = widgets.HTML()
        self.status_html = widgets.HTML()
        self.images_html = widgets.HTML()
        self.guide_html = widgets.HTML(
            value=f"<div style='font-size:18px; margin-top:8px;'>{html.escape(label_guide)}</div>"
        )

        self.prev_button.on_click(self._go_prev)
        self.next_button.on_click(self._go_next)
        self.search_button.on_click(self._search)
        self.save_button.on_click(self._save_current)
        self.export_button.on_click(self._export_now)
        self.bed_input.on_submit(self._search)
        self.date_input.on_submit(self._search)
        self.front_input.on_submit(self._save_current)
        self.top_input.on_submit(self._save_current)

        self._write_file()
        self._build_ui()
        self._render_current()
        self.status_html.value = (
            f"<div style='color:#0f766e;'>"
            f"{len(self.data)}개 이미지 묶음을 불러왔고, 라벨 파일을 준비했습니다: {html.escape(self.save_path)}"
            f"</div>"
        )

    def _build_ui(self) -> None:
        top_row = widgets.HBox(
            [self.prev_button, self.bed_input, self.date_input, self.search_button, self.next_button],
            layout=widgets.Layout(gap="10px", align_items="center"),
        )
        bottom_row = widgets.HBox(
            [self.front_input, self.top_input, self.save_button, self.export_button],
            layout=widgets.Layout(gap="10px", align_items="center"),
        )
        container = widgets.VBox(
            [top_row, self.counter_html, self.images_html, bottom_row, self.guide_html, self.status_html],
            layout=widgets.Layout(gap="12px"),
        )
        display(container)

    def _current_record(self) -> pd.Series:
        return self.data.loc[self.filtered_indices[self.current_position]]

    def _render_current(self) -> None:
        if not self.filtered_indices:
            self.images_html.value = "<div style='font-size:18px; color:#b42318;'>검색 결과가 없습니다.</div>"
            self.counter_html.value = ""
            return

        record = self._current_record()
        self.images_html.value = make_triplet_html(record)
        self.counter_html.value = (
            f"<div style='font-size:16px; font-weight:600;'>"
            f"{self.current_position + 1} / {len(self.filtered_indices)}"
            f"</div>"
        )
        self.front_input.value = str(record["front_view"])
        self.top_input.value = str(record["top_view"])

    def _search(self, _=None) -> None:
        bed = _normalize_bed(self.bed_input.value)
        date = _normalize_date(self.date_input.value)

        filtered = self.data.copy()
        if bed:
            filtered = filtered[filtered["bednum"] == bed]
        if date:
            filtered = filtered[filtered["date"] == date]

        self.filtered_indices = filtered.index.tolist()
        self.current_position = 0

        if not self.filtered_indices:
            self.images_html.value = "<div style='font-size:18px; color:#b42318;'>조건에 맞는 이미지가 없습니다.</div>"
            self.counter_html.value = ""
            self.status_html.value = "<div style='color:#b42318;'>BedNum 또는 날짜를 다시 확인해 주세요.</div>"
            return

        first = self.data.loc[self.filtered_indices[0]]
        self.status_html.value = (
            f"<div style='color:#0f766e;'>검색 완료: 가장 빠른 촬영 시점 "
            f"{html.escape(first['image_info'])} 를 표시합니다.</div>"
        )
        self._render_current()

    def _go_prev(self, _=None) -> None:
        if not self.filtered_indices:
            return
        self.current_position = max(0, self.current_position - 1)
        self._render_current()
        self.status_html.value = ""

    def _go_next(self, _=None) -> None:
        if not self.filtered_indices:
            return
        self.current_position = min(len(self.filtered_indices) - 1, self.current_position + 1)
        self._render_current()
        self.status_html.value = ""

    def _save_current(self, _=None) -> None:
        if not self.filtered_indices:
            return

        front_value = self.front_input.value.strip()
        top_value = self.top_input.value.strip()
        if not front_value or not top_value:
            self.status_html.value = "<div style='color:#b42318;'>Front와 Top 값을 모두 입력해 주세요.</div>"
            return

        row_index = self.filtered_indices[self.current_position]
        self.data.at[row_index, "front_view"] = front_value
        self.data.at[row_index, "top_view"] = top_value
        self.data.at[row_index, "front=top?"] = str(front_value == top_value).upper()
        self._write_file()
        saved_image_info = str(self.data.at[row_index, "image_info"])

        if self.current_position < len(self.filtered_indices) - 1:
            self.current_position += 1
        self._render_current()
        self.status_html.value = (
            f"<div style='color:#166534;'>저장 완료: {html.escape(saved_image_info)}</div>"
        )

    def _export_now(self, _=None) -> None:
        self._write_file()
        self.status_html.value = (
            f"<div style='color:#166534;'>엑셀 파일을 다시 저장했습니다: {html.escape(self.save_path)}</div>"
        )

    def _write_file(self) -> None:
        target = Path(self.save_path)
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            empty_frame = pd.DataFrame(columns=BASE_COLUMNS)
            if target.suffix.lower() == ".csv":
                empty_frame.to_csv(target, index=False, encoding="utf-8-sig")
            else:
                try:
                    empty_frame.to_excel(target, index=False, sheet_name="review")
                except ImportError as exc:  # pragma: no cover - depends on Colab environment
                    raise ImportError("xlsx 저장을 위해서는 openpyxl 설치가 필요합니다. `!pip install openpyxl` 후 다시 실행해 주세요.") from exc

        write_annotation_table(self.data, self.save_path)


def launch_colab_triplet_reviewer(
    top_root: str,
    front_root: str,
    save_path: str,
    label_guide: str = "0=꽉 참, 1=띄엄띄엄, 2=비어있음, 3=이전 날짜와 다름",
) -> TripletReviewer:
    """
    top_root: cam0/cam1이 들어 있는 윗면 루트 경로
    front_root: cam2가 들어 있는 정면 루트 경로
    save_path: 라벨이 저장될 xlsx 또는 csv 경로
    """
    return TripletReviewer(
        top_root=top_root,
        front_root=front_root,
        save_path=save_path,
        label_guide=label_guide,
    )


# ============================================================================
# Colab에서 아래 3개 경로만 바꾼 뒤, 이 파일을 실행하면 됩니다.
# 예시:
#   !pip install -q ipywidgets openpyxl pillow pandas
#   from google.colab import drive
#   drive.mount("/content/drive")
#   %run /content/drive/MyDrive/.../colab_triplet_reviewer.py
# ============================================================================
TOP_ROOT = "/content/drive/MyDrive/RGB_윗면/0. 원본/2작기"
FRONT_ROOT = "/content/drive/MyDrive/RGB_정면/0. 원본/2작기"
SAVE_PATH = "/content/drive/MyDrive/lettuce_triplet_review.xlsx"


if __name__ == "__main__":
    launch_colab_triplet_reviewer(
        top_root=TOP_ROOT,
        front_root=FRONT_ROOT,
        save_path=SAVE_PATH,
    )
