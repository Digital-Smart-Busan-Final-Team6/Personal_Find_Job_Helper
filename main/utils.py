# utils.py (최종 완성 버전 - 정규식 대신 수동 파싱)

import re

def parse_markdown_table_to_json(markdown_text: str) -> list:
    """
    정규식 대신 수동으로 한 줄씩 분석하여 Markdown 테이블을 추출합니다.
    Agent의 응답 형식이 약간 바뀌어도 더 안정적으로 작동합니다.
    """
    if not markdown_text or not markdown_text.strip():
        print("파싱 경고: 입력된 Markdown 텍스트가 비어 있습니다.")
        return []

    lines = markdown_text.strip().split('\n')
    
    table_started = False
    header_line = None
    separator_line = None
    data_lines = []

    for line in lines:
        line = line.strip()
        if not line.startswith('|') or not line.endswith('|'):
            if table_started: # 테이블이 시작된 후에 테이블 형식이 아닌 줄이 나오면 종료
                break
            continue

        # 테이블 헤더 찾기
        if not table_started and '---' not in line:
            header_line = line
            continue

        # 구분선 찾기 (테이블 시작의 강력한 증거)
        if header_line and '---' in line:
            separator_line = line
            table_started = True
            continue
        
        # 데이터 라인 수집
        if table_started:
            data_lines.append(line)

    if not table_started or not data_lines:
        print("파싱 실패: 유효한 Markdown 테이블 블록을 찾을 수 없습니다. (입력 텍스트:", markdown_text, ")")
        return []

    # --- 테이블 파싱 시작 ---
    headers_raw = [h.strip() for h in header_line.strip('|').split('|')]
    header_map = {
        '순위': 'rank',
        '공고 ID': '공고_ID',
        '제목': 'full_title',
        '적합도': '적합도',
    }
    headers = [header_map.get(h, h.lower().replace(' ', '_')) for h in headers_raw]

    results = []
    for line in data_lines:
        values = [v.strip() for v in line.strip('|').split('|')]
        if len(values) != len(headers):
            continue

        job_dict = dict(zip(headers, values))
        
        # 데이터 가공
        full_title = job_dict.get('full_title', '')
        if ' - ' in full_title:
            parts = full_title.split(' - ', 1)
            job_dict['회사명'] = parts[0]
            job_dict['제목'] = parts[1]
        else:
            # Agent가 '회사명'과 '제목'을 별도 컬럼으로 줄 경우도 대비
            job_dict['회사명'] = job_dict.get('회사명', "회사 정보 없음")
            job_dict['제목'] = job_dict.get('제목', full_title)
        
        try:
            job_dict['적합도'] = float(job_dict.get('적합도', 0))
        except (ValueError, TypeError):
            job_dict['적합도'] = 0.0

        job_dict['기술_스택'] = []
        job_dict.pop('full_title', None)
        job_dict.pop('rank', None)

        results.append(job_dict)
    
    print(f"파싱 성공: {len(results)}개의 공고를 JSON으로 변환했습니다.")
    return results