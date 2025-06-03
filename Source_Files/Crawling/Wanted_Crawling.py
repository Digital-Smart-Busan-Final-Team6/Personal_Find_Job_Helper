# 임포트 해야하는 것
import time
import requests
from tqdm import tqdm
import pandas as pd
import json
from bs4 import BeautifulSoup
import re
from Data_Analysis import *

# 점핏 메인 페이지의 url입니다
MAIN_URL = "https://www.wanted.co.kr/wdlist?country=kr&job_sort=job.popularity_order&years=-1&locations=all/offset="
DETAIL_URL = 'https://jumpit.saramin.co.kr/position/'

'''
원티드 메인 페이지를 가져오는 함수입니다
json 형식으로 되어있는 페이지를 가져옵니다
페이지를 넘기면서 가져오고, 빈 페이지가 나올 때까지 반복합니다
'''


def get_wanted_main_crawling():
    offset, limit = 0, 20
    posts = []  # ← 리스트
    pbar = tqdm(desc="Fetching job posts", unit="post")

    while True:
        url = (f"https://www.wanted.co.kr/api/chaos/navigation/v1/results"
               f"?country=kr&job_sort=job.popularity_order&years=-1"
               f"&locations=all&limit={limit}&offset={offset}")
        data = requests.get(url, timeout=10).json()
        page_posts = data.get("data", [])

        if not page_posts:  # 더 없으면 탈출
            break

        posts.extend(page_posts)  # ⭐ list → list (중복 허용)
        offset += limit
        pbar.update(len(page_posts))

    pbar.close()
    return posts


# def get_wanted_main_crawling():
#     offset = 0
#     limit = 20
#     posts = {}                       # key: 공고 id, value: 공고 json
#
#     pbar = tqdm(desc="Fetching job posts", unit="post")
#
#     while True:
#         url = (
#             "https://www.wanted.co.kr/api/chaos/navigation/v1/results?"
#             "1747219507653=&country=kr&job_sort=job.popularity_order&"
#             f"years=-1&locations=all&limit={limit}&offset={offset}"
#         )
#         response = requests.get(url, timeout=10)
#         response.raise_for_status()                     # 네트워크 오류 감지
#         data = response.json()
#
#         # ────────────────────────────────────────────────
#         # ① API는 `data` 키 아래에 공고 리스트를 준다
#         # ────────────────────────────────────────────────
#         one_page_posts = data.get("data", [])
#
#         # one_page_posts(=list)를 {id: post} 형태로 변환
#         page_dict = {post["id"]: post for post in one_page_posts}
#
#         # ② 딕셔너리끼리 병합  (동일 id가 있으면 최신 정보로 덮어씀)
#         posts.update(page_dict)
#
#         if not one_page_posts:       # 더 이상 데이터가 없으면 종료
#             break
#
#         offset += limit
#         pbar.update(len(one_page_posts))
#
#     pbar.close()
#     return posts


import time, random, logging, requests, pandas as pd
from tqdm import tqdm
from requests.exceptions import JSONDecodeError as ReqJSONDecodeError

HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"),
    "Accept": "application/json, text/plain, */*",
}

def get_wanted_detail_crawling(
        ids,
        max_retry: int = 3,
        backoff: float = 2.0,
        sleep: float = 0.1):

    rows = []                                    # ← all_df들을 모아둘 리스트
    session = requests.Session()
    pbar = tqdm(total=len(ids), desc="Fetching Detail Contents", unit="post")

    for jid in ids:
        url = f"https://www.wanted.co.kr/api/chaos/jobs/v4/{jid}/details"

        # ───── HTTP 호출 + 재시도 ───────────────────────────────
        for attempt in range(max_retry):
            try:
                r = session.get(url, headers=HEADERS, timeout=10)

                if (r.status_code == 200 and
                        "application/json" in r.headers.get("content-type", "")):
                    data_detail = r.json()
                    break                       # ▶ 성공 → retry 루프 탈출

                if r.status_code in (404, 410):  # 마감/삭제 공고
                    logging.info(f"skip {jid} (status {r.status_code})")
                    data_detail = None
                    break

                raise ValueError(f"status {r.status_code}")

            except (requests.RequestException, ReqJSONDecodeError, ValueError) as e:
                if attempt < max_retry - 1:
                    wait = backoff ** attempt + random.random()
                    logging.warning(f"{jid=} {e} → retry in {wait:.1f}s")
                    time.sleep(wait)
                    continue
                logging.error(f"give-up {jid=} after {max_retry} retries")
                data_detail = None

        # ───── 파싱 & DF 조립 ──────────────────────────────────
        if data_detail:
            job_tag = data_detail["data"]["job"]

            detail   = get_detail_tags(job_tag)
            attr     = get_attribute_tags(job_tag)
            company  = get_company_tags(job_tag)
            address  = get_address_tags(job_tag)
            category = get_category_tags(job_tag)
            skill    = get_skill_tags(job_tag)
            another  = get_another_tags(job_tag)

            all_df = pd.concat(
                [detail, attr, company, address, category, skill, another],
                axis=1
            )
            all_df["job_id"] = jid            # id 컬럼 추가
            rows.append(all_df)               # 리스트에 모음

        time.sleep(sleep)                     # 서버 부담 완화
        pbar.update(1)

    pbar.close()
    # ───── 리스트 → 최종 DataFrame ─────────────────────────────
    posts_detail = (
        pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    )
    return posts_detail


def get_tags_json(type):
    url = "https://static.wanted.co.kr/tags?tags="
    if type == "skill":
        url = url + type
    elif type == "category":
        url = url + type

    response = requests.get(url)
    tags = response.json()
    return tags


def get_ids(posts):
    ids = []
    for post in posts:
        id = post['id']
        ids.append(id)
    return ids


'''
Getter 함수들입니다
각각의 정보를 가져오는 함수들입니다
'''


def get_detail_data_tag(detail_json):
    # 일단 Data 가져오기
    df_data = pd.json_normalize(detail_json['data'],
                                sep='_',
                                errors='ignore',
                                max_level=1)
    return df_data


def get_detail_job_tag(detail_json):
    # job키 안에 모든게 있음 job 불러오기
    df_job = pd.json_normalize(detail_json['data']['job'],
                               sep='_',
                               errors='ignore',
                               max_level=1)
    return df_job


def get_detail_tags(job_tag):
    # detail부터 긁어오기
    df_detail = pd.json_normalize(job_tag['detail'],
                                  sep='_',
                                  meta=['id'],
                                  meta_prefix=['job_'], )

    df_detail.drop(columns=['id'], inplace=True)
    return df_detail


def get_attribute_tags(job_tag):
    # 1️⃣ 태그가 없으면 공백으로 반환
    if not job_tag.get("attraction_tags"):      # None, [], 키 자체 없음 모두 처리
        return pd.DataFrame({"attraction": [""]})

    # 2️⃣ 태그가 있을 때는 기존 방식 그대로
    df_attr = (
        pd.json_normalize(
            job_tag,
            record_path=["attraction_tags"],
            meta=["id"],
            meta_prefix="job_",
            sep="_"
        )
        .rename(columns={"title": "attraction"})
    )

    attr_lst = df_attr["attraction"].tolist()

    df_attr = pd.DataFrame({"attraction": [attr_lst]})
    return df_attr


def get_company_tags(job_tag):
    # 1️⃣ 태그가 없으면 공백으로 반환
    if not job_tag.get("company"):
        return pd.DataFrame({"company": [""]})
    # 2️⃣ 태그가 있을 때는 기존 방식 그대로
    df_company = pd.json_normalize(job_tag['company'],
                                   sep='_',
                                   meta=['id'],
                                   meta_prefix=['job_'], )
    df_company = df_company.rename(columns={'name': 'company'})
    df_company = df_company[['company']]
    return df_company


def get_address_tags(job_tag):
    # 1️⃣ 태그가 없으면 공백으로 반환
    if not job_tag.get("address"):
        return pd.DataFrame({"address": [""]})
    # 2️⃣ 태그가 있을 때는 기존 방식 그대로

    df_address = pd.json_normalize(job_tag['address'],
                                   sep='_',
                                   meta=['id'],
                                   meta_prefix=['job_'], )
    df_address = df_address[['country', 'location', 'district']]
    return df_address


def get_category_tags(job_tag):


    cat = job_tag.get("category_tag", {})  # 없으면 빈 dict

    # ────────────── parent ──────────────
    parent_tag = cat.get("parent_tag")  # dict 또는 None
    parent_txt = (parent_tag or {}).get("text")  # 없으면 None
    # 필요하다면 ''(빈 문자열) 등으로:
    # parent_txt = (parent_tag or {}).get("text", "")

    # ────────────── children ─────────────
    child_tags = cat.get("child_tags") or []  # 리스트 또는 빈 리스트
    children_txt = [c.get("text") for c in child_tags if c.get("text")]

    # ────────────── DataFrame ────────────
    df_category = pd.DataFrame({
        "category_parent": [parent_txt],  # None 또는 문자열
        "category_child": [children_txt]  # 빈 리스트 OK
    })

    return df_category


def get_skill_tags(job_tag):


    # 1️⃣  skill 문자열 목록 만들기
    skill_tags = job_tag.get("skill_tags") or []  # 리스트 or []
    skills = [s.get("text", "") for s in skill_tags if s.get("text")]

    # “완전히 비어 있으면 공백 한 칸이라도 넣고 싶다”면
    if not skills:
        skills = [""]  # → ['']

    # 2️⃣  리스트를 한 셀에 넣는 DataFrame
    df_skill = pd.DataFrame({
        "skill": [skills]  # ← 리스트 그대로
    })

    return df_skill


def get_another_tags(job_tag):
    # job에서 annual_from, annual_to, is_newbie을 데이터 프레임으로 변환
    df_another = pd.json_normalize(job_tag,
                                   sep='_',
                                   errors='ignore',
                                   max_level=1)
    df_another = df_another[['annual_from', 'annual_to', 'is_newbie','due_time']]
    return df_another


# 날짜 입력 후 실행하면 'Crawling_DataFile_MainPage_20250503' 형식의 파일이 저장됨
# 저장 폴더는 Data_Files
# 해당 폴더는 gitignore에 추가되어있으니 올라가지 않을것임
def toCsvFile(dataFrame, date, type):
    dataFrame['date'] = pd.to_datetime(dataFrame['date'], format='%Y%m%d')
    dataFrame['date'] = dataFrame['date'].dt.strftime('%Y년-%m월-%d일')

    if type == 'Main':
        dataFrame.T.to_csv(f"../../Data_Files/Crawling_DataFile_MainPage_Csv_{date}.txt",
                           sep='\t')
    elif type == 'Detail':
        dataFrame.T.to_csv(f"../../Data_Files/Crawling_DataFile_DetailPage_Csv_{date}.txt",
                           sep='\t')
    elif type == 'Merge':
        dataFrame.T.to_csv(f"../../Data_Files/Crawling_DataFile_MergePage_Csv_{date}.txt",
                           sep='\t')


def toJsonFile(dataFrame, date, type):
    dataFrame['date'] = date
    dataFrame['date'] = pd.to_datetime(dataFrame['date'], format='%Y%m%d')
    dataFrame['date'] = dataFrame['date'].dt.strftime('%Y년%m월%d일')
    if type == 'Main':
        dataFrame.T.to_json(f"../../Data_Files/Crawling_DataFile_MainPage_json_{date}.txt",
                            force_ascii=False)
    elif type == 'Detail':
        dataFrame.T.to_json(f"../../Data_Files/Crawling_DataFile_DetailPage_json_{date}.txt",
                            force_ascii=False)
    elif type == 'Merge':
        dataFrame.T.to_json(f"../../Data_Files/Crawling_DataFile_MergePage_json_{date}.txt",
                            force_ascii=False)


'''
텍스트 전처리 함수
'''


import re

def toImproveText(text):
    """
    문자열·리스트·숫자·None 전부 처리하는 안전 버전
    - 리스트  : 내부 요소를 재귀적으로 클린
    - 숫자형  : 그대로 반환
    - None    : 빈 문자열("") 반환
    - 문자열  : 특수문자 제거 후 반환
    - 기타    : str()로 바꾼 뒤 클린
    """
    # 1) 리스트인 경우 ─ 재귀적으로 각 원소 처리
    if isinstance(text, list):
        return [toImproveText(item) for item in text]

    # 2) 숫자형이면 그대로
    if isinstance(text, (int, float)):
        return text

    # 3) None → 빈 문자열
    if text is None:
        return ""

    # 4) 문자열 아닌 다른 타입(dict 등)은 일단 str로 변환
    if not isinstance(text, str):
        text = str(text)

    # 5) 실제 클린 작업
    cleaned = (
        text.replace("\r", "").replace("\t", "")
    )
    cleaned = re.sub(r"[^a-zA-Z0-9가-힣\s+#,]", "", cleaned).strip()
    return cleaned


def toImproveDataFrame(data):
    columns = data.columns
    for column in columns:
        data[column] = data[column].apply(toImproveText)
    return data
