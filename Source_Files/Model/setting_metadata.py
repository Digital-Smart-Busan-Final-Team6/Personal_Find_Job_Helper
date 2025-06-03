'''
경로의 json 파일 읽어오기
json 리스트 형태여야 함.
'''
import json




'''
평문으로 바꾸기 json->평문
'''
def to_plaintext(data: dict) -> str:
    """
    · 완전 숫자(dict→숫자) → 통계형
    · 중첩 숫자(dict→dict→숫자) → 중첩 통계형
    · 그 외 dict → 공고형
    """
    lines = []

    def is_simple_stat(d):
        return isinstance(d, dict) and all(isinstance(v, (int, float)) for v in d.values())

    def is_nested_stat(d):
        return (
                isinstance(d, dict)
                and all(isinstance(v, dict) and is_simple_stat(v) for v in d.values())
        )

    for key, val in data.items():
        # 1) 완전 숫자 통계
        if is_simple_stat(val):
            parts = [f"{k} {int(v)}개" for k, v in val.items()]
            lines.append(f"{key}는 {', '.join(parts)}입니다.")

        # 2) 중첩 숫자 통계
        elif is_nested_stat(val):
            # ex) “직군별 직무 개수:”
            lines.append(f"{key}:")
            for subkey, subdict in val.items():
                parts = [f"{k} {int(v)}개" for k, v in subdict.items()]
                lines.append(f"  • {subkey}: {', '.join(parts)}입니다.")

        # 3) 공고형
        elif isinstance(val, dict):
            # ID번 공고: 필드1은…, 필드2는…입니다.
            parts = []
            for field, v in val.items():
                if isinstance(v, list):
                    items = [str(x).strip() for x in v if x]
                    txt = ", ".join(items) if items else "없음"
                else:
                    txt = str(v).strip() or "없음"
                parts.append(f"{field}은 {txt}")
            lines.append(f"{key}번 공고: " + " , ".join(parts) + "입니다.")

        # 4) 기타 단일값
        else:
            lines.append(f"{key}는 {val}입니다.")

    return "".join(lines)


'''
input : document(Document 객체 list), data : 읽어들인 파일, index : 리스트 인덱스 번호
output : None
'''


from langchain_core.documents import Document

def make_post_docs(post_dict: dict) -> list[Document]:
    """
    post_dict: {
        "278301": { ...공고 상세 dict... },
        "278604": { ... },
         …
    }
    형태로, 최상위 키가 post_id일 때 사용합니다.
    """
    docs = []

    for post_id, post in post_dict.items():
        # 1) post_id는 문자열이라면 그대로, 숫자형이 필요하면 int(post_id)
        pid = post_id  # 또는 pid = int(post_id)

        # 2) 평문 컨텐츠 생성
        content = to_plaintext({pid: post})

        # 3) metadata에 post_id와 출처 저장
        meta = {
            "id": pid,
            "source": "채용 공고"
        }

        docs.append(Document(page_content=content, metadata=meta))

    return docs


def make_analysis_docs(analysis_stats: dict) -> list[Document]:
    """
    analysis_stats: {"직군별 개수": {...}, "직군별 직무 개수": {...}, …}
    각 키별로 to_plaintext({key: stats}) 호출 → Document로 감싸 반환
    """
    docs = []
    for category, stats in analysis_stats.items():
        content = to_plaintext({category: stats})
        meta    = {"category": category, "source": "통계 분석"}
        docs.append(Document(page_content=content, metadata=meta))
    return docs
