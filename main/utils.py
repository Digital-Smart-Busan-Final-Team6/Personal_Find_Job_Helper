# utils.py

import re

def parse_markdown_table_to_json(markdown_text: str) -> list:
    """
    설명문이 섞인 Markdown 텍스트에서도 테이블 부분만 정확히 추출하고,
    '회사명 - 제목' 형식을 분리하여 JSON으로 변환합니다.
    """
    if not markdown_text or not markdown_text.strip():
        return []

    table_match = re.search(r'((?:\|.*\|\s*\n){3,})', markdown_text)
    if not table_match:
        print("파싱 실패: Markdown 테이블을 찾을 수 없습니다.")
        return []

    table_text = table_match.group(1)
    lines = table_text.strip().split('\n')
    
    header_line = lines[0]
    data_lines = lines[2:]

    headers_raw = [h.strip() for h in header_line.split('|') if h.strip()]
    header_map = {
        '순위': 'rank',
        '공고 ID': 'id',
        '제목': 'full_title',  # '제목' 컬럼을 'full_title'로 받음
        '적합도': 'score',
    }
    headers = [header_map.get(h, h.lower().replace(' ', '_')) for h in headers_raw]

    results = []
    for line in data_lines:
        values = [v.strip() for v in line.split('|') if v.strip()]
        if len(values) != len(headers):
            continue

        job_dict = dict(zip(headers, values))
        
        full_title = job_dict.get('full_title', '')
        if ' - ' in full_title:
            parts = full_title.split(' - ', 1)
            job_dict['company'] = parts[0]
            job_dict['title'] = parts[1]
        else:
            # 분리 실패 시, 원본 제목을 제목으로 사용하고 회사명은 비워둠
            job_dict['company'] = None 
            job_dict['title'] = full_title

        try:
            # 점수(문자열)를 부동소수점 숫자로 변환 후, 백분율 정수로 만듦
            score_float = float(job_dict.get('score', 0))
            job_dict['score_percent'] = int(score_float * 100)
        except (ValueError, TypeError):
            job_dict['score_percent'] = 0

        # 불필요한 full_title 키는 제거
        job_dict.pop('full_title', None)
        
        # 지역 정보는 이 단계에서 알 수 없으므로, 기본값을 설정
        job_dict['location'] = job_dict.get('location', '지역 정보 없음')

        results.append(job_dict)

    return results