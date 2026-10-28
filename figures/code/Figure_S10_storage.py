"""3작기 실증의 저장 루트와 이전 실행 기록의 경로 이관 규칙.

데이터 파일이나 SQLite를 수정하지 않는다. 옛 절대경로는 읽을 때만 E:로
해석한다. 저장 파일이 실제로 없는 경우에는 호출자가 그대로 실패한다.
"""
from pathlib import Path, PureWindowsPath

OLD_STORAGE_ROOT = PureWindowsPath(r'D:\양상추연구_local\newfolder')
STORAGE_ROOT = Path(r'E:\newfolder_양상추')
ROI_ROOT = STORAGE_ROOT / '3작기 실증 적용' / '2. 전처리'
CROP_FOLDER = '5. BBOX to crop,feature'
OLD_CROP_FOLDER = '5. BBOX to crop'


def moved_path(value):
    """옛 저장 루트와 옛 crop 폴더 이름만 새 위치에 대응한다."""
    if not isinstance(value, (str, Path, PureWindowsPath)):
        return value
    raw = str(value)
    if not raw:
        return raw
    path = PureWindowsPath(raw)
    parts = path.parts
    old_parts = OLD_STORAGE_ROOT.parts
    new_parts = PureWindowsPath(STORAGE_ROOT).parts
    if len(parts) >= len(old_parts) and tuple(x.casefold() for x in parts[:len(old_parts)]) == tuple(x.casefold() for x in old_parts):
        suffix = parts[len(old_parts):]
    elif len(parts) >= len(new_parts) and tuple(x.casefold() for x in parts[:len(new_parts)]) == tuple(x.casefold() for x in new_parts):
        suffix = parts[len(new_parts):]
    else:
        return value
    if len(suffix) >= 2 and suffix[:2] == ('3작기 실증 적용', '2. ROI_BOX'):
        suffix = (suffix[0], '2. 전처리') + suffix[2:]
    if (len(suffix) >= 3 and suffix[:2] == ('3작기 실증 적용', '2. 전처리')
            and suffix[2] == OLD_CROP_FOLDER):
        suffix = suffix[:2] + (CROP_FOLDER,) + suffix[3:]
    return str(PureWindowsPath(STORAGE_ROOT, *suffix))


def moved_tree(value):
    """CSV/JSON/SQLite의 경로 문자열만 메모리에서 변환한다."""
    if isinstance(value, dict):
        return {moved_path(k): moved_tree(v) for k, v in value.items()}
    if isinstance(value, list):
        return [moved_tree(v) for v in value]
    if isinstance(value, tuple):
        return tuple(moved_tree(v) for v in value)
    return moved_path(value)


def legacy_identity_path(value):
    """이전 absolute-path 기반 frame_id 계산에 쓸 저장 위치만 되돌린다."""
    path = PureWindowsPath(str(value))
    parts = path.parts
    new_parts = PureWindowsPath(STORAGE_ROOT).parts
    if len(parts) >= len(new_parts) and tuple(x.casefold() for x in parts[:len(new_parts)]) == tuple(x.casefold() for x in new_parts):
        suffix = parts[len(new_parts):]
        if len(suffix) >= 2 and suffix[:2] == ('3작기 실증 적용', '2. 전처리'):
            suffix = (suffix[0], '2. ROI_BOX') + suffix[2:]
        if (len(suffix) >= 3 and suffix[:2] == ('3작기 실증 적용', '2. ROI_BOX')
                and suffix[2] == CROP_FOLDER):
            suffix = suffix[:2] + (OLD_CROP_FOLDER,) + suffix[3:]
        return str(PureWindowsPath(OLD_STORAGE_ROOT, *suffix))
    return str(value)
