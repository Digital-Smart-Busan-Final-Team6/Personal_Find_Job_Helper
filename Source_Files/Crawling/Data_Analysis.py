import pandas as pd
import json
import os

from debugpy.common.util import force_ascii


def getNewPost(DataFrameBefore, DataFrameNew):
    newPost = DataFrameNew[~DataFrameNew.index.isin(DataFrameBefore.index)]

    return newPost

def getClosedPost(DataFrameBefore, DataFrameNew):
    closedPost = DataFrameBefore[~DataFrameBefore.index.isin(DataFrameNew.index)]

    return closedPost

# main, detail DataFrame을 합치는 함수입니다
def mergeDataFrame(mainDataFrame, detailDataFrame):
    combinedDataFrame = pd.merge(detailDataFrame, mainDataFrame,
                                 left_index=True,
                                 right_index=True)

    return combinedDataFrame




# 평문 생성 함수 정의
def generatePlainText(entry):
    title = entry.get("title", "")
    company = entry.get("company", "")
    location = ", ".join(entry.get("jobLocation", []))
    date = entry.get("date", "")
    closed_at = entry.get("closedAt", "")
    category = entry.get("jobCategory", "")
    min_career = entry.get("minCareer", 0)
    max_career = entry.get("maxCareer", 0)

    # 주요 업무
    main_work = " ".join([w.strip() for w in entry.get("mainWork", []) if w.strip()])

    # 자격 요건
    requirements = " ".join([r.strip() for r in entry.get("requirements", []) if r.strip()])

    # 우대 사항
    preferential = " ".join([p.strip() for p in entry.get("preferential", []) if p.strip()])

    # 복지 및 혜택
    welfare = " ".join([w.strip() for w in entry.get("welfare", []) if w.strip()])

    # 지원 절차
    procedures = " ".join([p.strip() for p in entry.get("procedures", []) if p.strip()])

    # 학력
    education = entry.get("education", "")

    # 기술 스택
    techstack = ", ".join(entry.get("techStack", []))

    # 특이사항
    pride = ", ".join(entry.get("pride", []))

    # 평문 구성
    text = (
        f"{company}는 {title} 포지션을 모집하고 있습니다. 이 포지션은 {location}에서 근무하게 되며, "
        f"최소 {min_career}년에서 최대 {max_career}년까지의 경력을 요구합니다. 해당 공고는 {date}에 게시되었으며, 마감일은 {closed_at}입니다. "
        f"주요 직무는 {category}입니다. "
        f"주요 업무로는 {main_work} 등이 있습니다. "
        f"자격 요건은 다음과 같습니다. {requirements} "
        f"우대 사항으로는 {preferential} 등이 있으며, "
        f"복지 및 혜택으로는 {welfare} 등이 제공됩니다. "
        f"지원 절차는 다음과 같습니다. {procedures} "
        f"학력 요건은 {education}이며, "
        f"요구 기술 스택은 {techstack}입니다. "
    )
    if pride:
        text += f"특이사항으로는 {pride} 등이 있습니다. "
    return text.strip()



def checkFormat(data):
    """주어진 문자열이 JSON 형식인지 텍스트 형식인지 확인합니다."""
    if isinstance(data,dict):
        return "JSON"
    elif isinstance(data, str):
        return "Text"

def getOriginalJson(file_path):
    """
    기존 JSON 파일을 읽어옵니다.
    """
    existingData = {}
    if os.path.exists(file_path):
        with open(file_path, encoding='utf-8') as f:
            try:
                existingData = json.load(f)  # JSON 객체 형태로 읽음
            except json.JSONDecodeError:
                existingData = {}
    return existingData

def mergeJsonDicts(originalJson, newJson):
    merged = originalJson.copy()
    for key, value in newJson.items():
        if key in merged:
            if isinstance(value, list) and isinstance(merged[key], list):
                merged[key].extend(value)  # 리스트면 누적
            else:
                merged[key] = value  # 그 외는 덮어씀
        else:
            merged[key] = value
    return merged

def saveData(data, file_path):
    dataType = checkFormat(data)
    file_exists = os.path.exists(file_path)
    mode = 'w'
    encoding = 'utf-8'

    def saveJson():
        with open(file_path, mode, encoding=encoding) as f:
            json.dump(data, f, ensure_ascii=False, indent=4)


    def saveText():
        with open(file_path, mode, encoding=encoding) as f:
            f.write(data)

    if dataType == "JSON":
        saveJson()
        msg = "기존 파일에 JSON을 덮어쓰기 성공했습니다." if file_exists else "새로운 JSON 파일을 생성했습니다."
    elif dataType == "Text":
        saveText()
        msg = "기존 파일에 텍스트를 덮어쓰기 성공했습니다." if file_exists else "새로운 텍스트 파일을 생성했습니다."
    else:
        msg = f"지원되지 않는 데이터 타입입니다: {dataType}"

    print(msg)


