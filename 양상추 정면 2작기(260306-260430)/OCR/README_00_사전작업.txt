사전작업 코드 설명
====================

위치:
G:\내 드라이브\양상추 분류모델\practice code\양상추 정면 2작기(260306-260430)\OCR\00_build_triplet_manifest.py

판단:
처음부터 최초수집 2200개만 잡고 가기보다, 전체 이미지를 먼저 manifest로 만든 뒤
그 안에서 날짜+saved_bed별 최초 complete triplet을 따로 뽑는 방식이 안전하다.
최초수집 시간이 흔들리거나 누락되면 anchor를 놓칠 수 있기 때문이다.

이 스크립트가 하는 일:
1. RGB_윗면/0. 원본/2작기에서 cam0, cam1 파일을 읽는다.
2. RGB_정면/0. 원본/2작기에서 cam2 파일을 읽는다.
3. 파일명 bed02_20260329_153922_cam2.jpg 에서
   saved_bed, date, time, cam 정보를 뽑는다.
4. bed02_20260329_153922 를 triplet_key로 삼는다.
5. cam0/cam1/cam2가 모두 있으면 complete triplet으로 표시한다.
6. 전체 manifest와 complete-only manifest를 CSV/XLSX로 저장한다.
7. 날짜+saved_bed별 최초 complete triplet도 따로 저장한다.

실행:
"C:\Users\jiniy\anaconda3\python.exe" "G:\내 드라이브\양상추 분류모델\practice code\양상추 정면 2작기(260306-260430)\OCR\00_build_triplet_manifest.py"

하루치 테스트 실행:
"C:\Users\jiniy\anaconda3\python.exe" "G:\내 드라이브\양상추 분류모델\practice code\양상추 정면 2작기(260306-260430)\OCR\00_build_triplet_manifest.py" --only-date 20260329 --no-excel --output-dir "G:\내 드라이브\양상추 분류모델\practice code\양상추 정면 2작기(260306-260430)\OCR\outputs_test_20260329"

전체 실행이 너무 느리거나 Google Drive 리소스 오류가 나면:
1. 날짜별로 --only-date YYYYMMDD를 주고 나누어 실행한다.
2. 먼저 --no-excel로 CSV만 만든다.
3. CSV가 안정적으로 만들어진 뒤 XLSX는 나중에 따로 변환한다.

기본 출력 폴더:
G:\내 드라이브\양상추 분류모델\practice code\양상추 정면 2작기(260306-260430)\OCR\outputs

주요 출력:
00_file_inventory_all.csv/xlsx
01_triplet_manifest_all.csv/xlsx
02_triplet_manifest_complete_only.csv/xlsx
03_first_triplet_per_date_saved_bed.csv/xlsx
04_summary_counts.csv/xlsx

앞으로의 파트 연결:
1st OCR 파트는 02_triplet_manifest_complete_only 또는
03_first_triplet_per_date_saved_bed를 입력으로 받아 cam1_path 오른쪽 번호표를 crop한다.
그 결과를 cam1_tag_crop_path, ocr_visible_bed_num, ocr_confidence 열에 채운다.

주의:
saved_bed는 실제 물리 베드 번호가 아니라 가상/임시 이름으로 취급한다.
최종 분석 키는 나중에 cam1_visible_number 기반 real_bed_id로 다시 부여한다.

20260329 하루치 테스트 결과:
parsed_image_files = 1106
cam0 = 369
cam1 = 369
cam2 = 368
triplet_rows_all = 369
triplet_rows_complete = 368
즉 bedXX_YYYYMMDD_HHMMSS 키로 cam0/cam1/cam2를 묶는 방식은 정상 동작한다.


1ST crop 테스트 코드
====================

위치:
G:\내 드라이브\양상추 분류모델\practice code\양상추 정면 2작기(260306-260430)\OCR\01_test_piece_crop_50.py

역할:
RGB_윗면/0. 원본/2작기 하위 폴더를 모두 탐색해서 cam1 중
bed01, bed30, bed50, bed70, bed90에서 10장씩 랜덤 추출한다.
각 이미지에서 오른쪽 큰 흰 번호표 위치를 A-style 고정비율 crop으로 자르고
원본이름_piece.jpg로 저장한다.

Colab 실행 예:
python "/content/drive/MyDrive/경로/OCR/01_test_piece_crop_50.py"

출력:
260421 piece crop_ex/원본이름_piece.jpg
260421 piece crop_ex/piece_crop_test_manifest.csv
260421 piece crop_ex/piece_crop_test_montage.jpg
