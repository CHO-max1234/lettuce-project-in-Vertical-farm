import pandas as pd
from pathlib import Path

# ============================================================
# 경로 설정 (여기만 수정)
# ============================================================
P1P4_CSV_PATH = Path('/home/ubuntu/upload/per_image_results_v3.1.csv')
MANUAL_CALIB_CSV_PATH = Path('/home/ubuntu/work_brightness/manual_dynamic_y_calibration_template.csv')
OUT_DIR = Path('/home/ubuntu/work_brightness/dynamic_y_ratio_helper_output')

# 보고 싶은 대표 이미지들
TARGET_IMAGES = [
    'bed00_20260313_124253_cam2',
    'bed00_20260318_113024_cam2',
    'bed00_20260410_080135_cam2',
    'bed01_20260312_080929_cam2',
]

# 슬롯 x 위치 (기존 코드와 동일)
X_POS = [0.10, 0.26, 0.42, 0.58, 0.74, 0.90]

# 비교할 후보값
TOP_ROW_CANDIDATES = [0.22, 0.25, 0.28, 0.30]
BOT_ROW_CANDIDATES = [0.90, 0.93, 0.95, 0.97]


# ============================================================
# 공용 함수
# ============================================================
def ensure_dir(path):
    path.mkdir(parents=True, exist_ok=True)


def lerp(a, b, r):
    return a + (b - a) * r


def load_p1p4(csv_path):
    df = pd.read_csv(csv_path)
    num_cols = ['p1_x', 'p1_y', 'p2_x', 'p2_y', 'p3_x', 'p3_y', 'p4_x', 'p4_y', 'matched_count']
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    df['base_key'] = df['image_name'].astype(str).str.replace('.jpg', '', regex=False)
    return df


def get_row_line_y(row, x_ratio):
    tx = lerp(row['p1_x'], row['p2_x'], x_ratio)
    ty = lerp(row['p1_y'], row['p2_y'], x_ratio)
    bx = lerp(row['p4_x'], row['p3_x'], x_ratio)
    by = lerp(row['p4_y'], row['p3_y'], x_ratio)
    return tx, ty, bx, by


def anchor_y_from_ratio(row, x_ratio, ratio):
    _, ty, _, by = get_row_line_y(row, x_ratio)
    return lerp(ty, by, ratio)


def ratio_from_target_y(row, x_ratio, target_y):
    _, ty, _, by = get_row_line_y(row, x_ratio)
    gap = by - ty
    if pd.isna(gap) or abs(gap) < 1e-6:
        return None
    return (target_y - ty) / gap


# ============================================================
# 1) 현재 후보값들이 실제 픽셀 y로 어디에 놓이는지 출력
# ============================================================
def export_candidate_anchor_table(df):
    rows = []
    use_df = df[df['base_key'].isin(TARGET_IMAGES)].copy()

    for _, row in use_df.iterrows():
        for x_ratio in X_POS:
            _, top_edge_y, _, bot_edge_y = get_row_line_y(row, x_ratio)
            item = {
                'image_name': row['base_key'],
                'x_ratio': x_ratio,
                'top_edge_y': round(top_edge_y, 2),
                'bot_edge_y': round(bot_edge_y, 2),
                'edge_gap': round(bot_edge_y - top_edge_y, 2),
            }
            for top_r in TOP_ROW_CANDIDATES:
                item[f'top_y_r_{str(top_r).replace(".", "p")}'] = round(anchor_y_from_ratio(row, x_ratio, top_r), 2)
            for bot_r in BOT_ROW_CANDIDATES:
                item[f'bot_y_r_{str(bot_r).replace(".", "p")}'] = round(anchor_y_from_ratio(row, x_ratio, bot_r), 2)
            rows.append(item)

    out_df = pd.DataFrame(rows)
    out_path = OUT_DIR / 'candidate_anchor_positions.csv'
    out_df.to_csv(out_path, index=False, encoding='utf-8-sig')
    return out_path, out_df


# ============================================================
# 2) 사용자가 직접 다음 작기 값을 정할 수 있게 하는 보정 도구
# manual_dynamic_y_calibration_template.csv 에
# image_name,x_ratio,target_top_y,target_bot_y 를 적으면
# 이미지별 ratio를 역산해서 추천 median을 구함
# ============================================================
def export_manual_template(df):
    rows = []
    use_df = df[df['base_key'].isin(TARGET_IMAGES)].copy()

    for _, row in use_df.iterrows():
        for x_ratio in [0.42, 0.58, 0.74]:
            _, top_edge_y, _, bot_edge_y = get_row_line_y(row, x_ratio)
            rows.append({
                'image_name': row['base_key'],
                'x_ratio': x_ratio,
                'top_edge_y_reference': round(top_edge_y, 2),
                'bot_edge_y_reference': round(bot_edge_y, 2),
                'target_top_y': '',
                'target_bot_y': '',
                'memo': '이미지 overlay를 보고 사용자가 원하는 top/bottom 기준 y를 직접 입력',
            })

    out_df = pd.DataFrame(rows)
    out_df.to_csv(MANUAL_CALIB_CSV_PATH, index=False, encoding='utf-8-sig')
    return MANUAL_CALIB_CSV_PATH


def run_manual_calibration(df, calib_csv_path):
    if not calib_csv_path.exists():
        return None, 'manual calibration csv not found'

    calib = pd.read_csv(calib_csv_path)
    need_cols = ['image_name', 'x_ratio', 'target_top_y', 'target_bot_y']
    for col in need_cols:
        if col not in calib.columns:
            return None, f'missing column: {col}'

    merged = calib.merge(df, left_on='image_name', right_on='base_key', how='left')
    merged['target_top_y'] = pd.to_numeric(merged['target_top_y'], errors='coerce')
    merged['target_bot_y'] = pd.to_numeric(merged['target_bot_y'], errors='coerce')
    merged['x_ratio'] = pd.to_numeric(merged['x_ratio'], errors='coerce')

    merged['derived_top_r'] = merged.apply(
        lambda r: ratio_from_target_y(r, r['x_ratio'], r['target_top_y']) if pd.notna(r['target_top_y']) else None,
        axis=1
    )
    merged['derived_bot_r'] = merged.apply(
        lambda r: ratio_from_target_y(r, r['x_ratio'], r['target_bot_y']) if pd.notna(r['target_bot_y']) else None,
        axis=1
    )

    valid_top = merged['derived_top_r'].dropna()
    valid_bot = merged['derived_bot_r'].dropna()

    summary = {
        'top_count': int(valid_top.shape[0]),
        'bot_count': int(valid_bot.shape[0]),
        'top_median': round(valid_top.median(), 4) if not valid_top.empty else None,
        'top_mean': round(valid_top.mean(), 4) if not valid_top.empty else None,
        'bot_median': round(valid_bot.median(), 4) if not valid_bot.empty else None,
        'bot_mean': round(valid_bot.mean(), 4) if not valid_bot.empty else None,
    }

    detail_path = OUT_DIR / 'manual_calibration_derived_ratios.csv'
    merged.to_csv(detail_path, index=False, encoding='utf-8-sig')

    summary_path = OUT_DIR / 'manual_calibration_summary.txt'
    lines = []
    lines.append('[dynamic_y_manual_calibration_summary]')
    lines.append(f"top_count={summary['top_count']}")
    lines.append(f"bot_count={summary['bot_count']}")
    lines.append(f"top_median={summary['top_median']}")
    lines.append(f"top_mean={summary['top_mean']}")
    lines.append(f"bot_median={summary['bot_median']}")
    lines.append(f"bot_mean={summary['bot_mean']}")
    lines.append('')
    lines.append('권장 사용 방식: median 값을 TOP_ROW_R, BOT_ROW_R 후보로 먼저 사용')
    summary_path.write_text('\n'.join(lines), encoding='utf-8')

    return (detail_path, summary_path), None


# ============================================================
# 3) 현재 0.28/0.95가 실제로 무엇을 뜻하는지 한 줄 요약 파일 생성
# ============================================================
def export_current_ratio_explanation(df):
    use_df = df[df['base_key'].isin(TARGET_IMAGES)].copy()
    rows = []

    for _, row in use_df.iterrows():
        for x_ratio in [0.58]:
            _, top_edge_y, _, bot_edge_y = get_row_line_y(row, x_ratio)
            top_y = anchor_y_from_ratio(row, x_ratio, 0.28)
            bot_y = anchor_y_from_ratio(row, x_ratio, 0.95)
            rows.append({
                'image_name': row['base_key'],
                'x_ratio': x_ratio,
                'top_edge_y': round(top_edge_y, 2),
                'bot_edge_y': round(bot_edge_y, 2),
                'top_anchor_y_at_0p28': round(top_y, 2),
                'bot_anchor_y_at_0p95': round(bot_y, 2),
                'top_anchor_relative_desc': '상단 edge에서 28% 내려온 위치',
                'bot_anchor_relative_desc': '하단 edge에 거의 붙는 95% 위치',
            })

    out_df = pd.DataFrame(rows)
    out_path = OUT_DIR / 'current_ratio_meaning_examples.csv'
    out_df.to_csv(out_path, index=False, encoding='utf-8-sig')
    return out_path


# ============================================================
# main
# ============================================================
def main():
    ensure_dir(OUT_DIR)
    df = load_p1p4(P1P4_CSV_PATH)

    candidate_path, _ = export_candidate_anchor_table(df)
    template_path = export_manual_template(df)
    meaning_path = export_current_ratio_explanation(df)
    calib_result, calib_err = run_manual_calibration(df, MANUAL_CALIB_CSV_PATH)

    print('[saved] candidate anchor table:', candidate_path)
    print('[saved] manual template:', template_path)
    print('[saved] current ratio meaning examples:', meaning_path)

    if calib_err is not None:
        print('[info]', calib_err)
        print('[info] template file을 먼저 채운 뒤 다시 실행하면 추천 ratio가 계산됩니다.')
    else:
        detail_path, summary_path = calib_result
        print('[saved] manual calibration detail:', detail_path)
        print('[saved] manual calibration summary:', summary_path)

    print('\n[how_to_use]')
    print('1) candidate_anchor_positions.csv 로 현재 후보값이 실제 y 픽셀 어디에 놓이는지 확인')
    print('2) manual_dynamic_y_calibration_template.csv 를 열어서 사용자가 원하는 top/bottom y를 입력')
    print('3) 이 스크립트를 다시 실행')
    print('4) manual_calibration_summary.txt 의 median 값을 다음 작기 TOP_ROW_R, BOT_ROW_R 후보로 사용')


if __name__ == '__main__':
    main()
