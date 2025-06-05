# 임포트 해야하는 것
import time
import requests
from tqdm import tqdm
import pandas as pd
import json
from bs4 import BeautifulSoup
import re
# 점핏 메인 페이지의 url입니다
MAIN_URL = "https://jumpit-api.saramin.co.kr/api/positions?sort=rsp_rate&highlight=false&page="
DETAIL_URL = 'https://jumpit.saramin.co.kr/position/'

TITLE_CLASS = "sc-923c3317-0 jIWuEG"
MAIN_CLASS = "position_info"
BLIND_CLASS = "sc-b12ae455-1 hvXrQd"

'''
점핏 메인 페이지를 가져오는 함수입니다
xml 형식으로 되어있는 페이지의 Positions를 가져옵니다
페이지를 넘기면서 가져오고, 빈 페이지가 나올 때까지 반복합니다
'''
def getJumpitPositions():
    isEmpty = False
    positions = []
    i = 1
    while(isEmpty == False):
        pbar = tqdm(total=None, desc="Loading", unit="page")  # 진행률 표시줄 초기화
        page = requests.get(MAIN_URL + str(i))
        soup = BeautifulSoup(page.content, "lxml")  # lxml Parser 사용
        json_text = soup.find('p').text
        data = json.loads(json_text)  # json 형식으로 변환 

        isEmpty = data['result']['emptyPosition']  # 빈 페이지인지 체크

        result = data['result']  
        positions.extend(result['positions'])  # result 안의 positions 리스트를 데이터 딕셔너리 형태로 모두 가져옴
        i += 1
        pbar.update(1)  # 진행률 표시줄 업데이트
    print('모든 페이지 정보를 가져 왔습니다')
    return positions


'''
Getter 함수들입니다
각각의 정보를 가져오는 함수들입니다
'''
def getTitles(positions):
    titles = []
    for position in positions:
        title = position['title']
        titles.append(title)
    return titles

def getCompanyNames(positions):
    companyNames = []
    for position in positions:
        companyName = position['companyName']
        companyNames.append(companyName)
    return companyNames

def getJobCategories(positions):
    jobCategories = []
    for position in positions:
        jobCategory = position['jobCategory']
        jobCategories.append(jobCategory)
    return jobCategories

def getTechStacks(positions):
    techStacks = []
    for position in positions:
        techStack = position['techStacks']
        techStacks.append(techStack)
    return techStacks


def getJobLocations(positions):
    jobLocations = []
    for position in positions:
        jobLocation = position['locations']
        jobLocations.append(jobLocation)
    return jobLocations

def getMinCareer(positions):
    minCareers = []
    for position in positions:
        minCareer = position['minCareer']
        minCareers.append(minCareer)
    return minCareers

def getMaxCareer(positions):
    maxCareers = []
    for position in positions:
        maxCareer = position['maxCareer']
        maxCareers.append(maxCareer)
    return maxCareers

def getIDs(positions):
    ids = []
    for position in positions:
        id = position['id']
        ids.append(id)
    return ids

def getClosedAt(positions):
    closedAts = []
    for position in positions:
        closedAt = position['closedAt']
        closedAts.append(closedAt)
    return closedAt


# 날짜 입력 후 실행하면 'Crawling_DataFile_MainPage_20250503' 형식의 파일이 저장됨
# 저장 폴더는 Data_Files
# 해당 폴더는 gitignore에 추가되어있으니 올라가지 않을것임
def toCsvFile(dataFrame, date, type):
    dataFrame['date'] = pd.to_datetime(dataFrame['date'], format='%Y%m%d')
    dataFrame['date'] = dataFrame['date'].dt.strftime('%Y년-%m월-%d일')

    if type == 'Main':
        dataFrame.T.to_csv(f"../Data_Files/Crawling_DataFile_MainPage_Csv_{date}.txt",
                         sep='\t')
    elif type == 'Detail':
        dataFrame.T.to_csv(f"../Data_Files/Crawling_DataFile_DetailPage_Csv_{date}.txt",
                         sep='\t')
    elif type == 'Merge':
        dataFrame.T.to_csv(f"../Data_Files/Crawling_DataFile_MergePage_Csv_{date}.txt",
                         sep='\t')

def toJsonFile(dataFrame, date, type):
    dataFrame['date'] = date
    dataFrame['date'] = pd.to_datetime(dataFrame['date'], format='%Y%m%d')
    dataFrame['date'] = dataFrame['date'].dt.strftime('%Y년%m월%d일')
    if type == 'Main':
        dataFrame.T.to_json(f"../Data_Files/Crawling_DataFile_MainPage_json_{date}.txt",
                        force_ascii=False)
    elif type == 'Detail':
        dataFrame.T.to_json(f"../Data_Files/Crawling_DataFile_DetailPage_json_{date}.txt",
                        force_ascii=False)
    elif type == 'Merge':
        dataFrame.T.to_json(f"../Data_Files/Crawling_DataFile_MergePage_json_{date}.txt",
                        force_ascii=False)

'''
점핏 상세 페이지를 가져오는 함수입니다
각각의 상세 페이지 정보를 가져옵니다
'''

'''
텍스트 전처리 함수
'''
def toImproveText(text):
    if isinstance(text, list):  # 입력이 리스트인지 확인
        cleaned_list = []
        for item in text:
            cleaned_item = item.replace('\r', '').replace('\t', '')
            # +, #, ,는 유지하면서 나머지 특수문자 제거
            cleaned_item = re.sub(r'[^a-zA-Z0-9가-힣\s+#,]', '', cleaned_item).strip()
            cleaned_list.append(cleaned_item)
        return cleaned_list
    elif isinstance(text, int):
        return text
    else:
        cleaned_text = text.replace('\r', '').replace('\t', '')
        cleaned_text = re.sub(r'[^a-zA-Z0-9가-힣\s+#,]', '', cleaned_text).strip()
        return cleaned_text

def toImproveDataFrame(data):
    columns = data.columns
    for column in columns:
        data[column] = data[column].apply(toImproveText)
    return data

# 상세 페이지 body 부분만 추출하는 함수
def getDetailPage(url):
    page = requests.get(url)
    soup = BeautifulSoup(page.content, "html.parser")  # lxml Parser 사용
    body = soup.find('body')
    return body

# 상세 페이지 맨 윗부분의 제목, 회사명, 자랑거리? 부분을 리턴
def getTitleInfo(body):
    titleInfo = body.find_all('div', class_ = TITLE_CLASS)
    titles = ''
    company = ''
    pride = []
    for info in titleInfo:
        h1_tag = info.find('h1')
        company_tag = info.find('a')
        if h1_tag:  # <h1> 태그가 존재하는 경우에만 추가
            titles = h1_tag.text

        if company_tag:
            company = company_tag.text

    for info in titleInfo:
        pride_tags = info.find_all('li')
        for pride_tag in pride_tags:
            pride.append(pride_tag.text)
    
    return titles, company, pride


# 상세 페이지의 본문 부분을 리턴
# 기술 스택, 주요 업무, 자격 요건, 우대 사항, 복지 혜택, 전형 절차를 리턴
# 기술 스택은 리스트로 리턴
# 나머지는 \n 기준으로 (즉 줄바꿈) 배열로 리턴시킴
def getMainInfo(body):
    mainWork = ''
    requirements = ''
    preferential = ''
    welfare = ''
    procedures = ''
    mainInfo = body.find_all('div', class_ = MAIN_CLASS)
    techStack = mainInfo[0].find_all('div', class_ = "sc-d9de2de1-0 hMgXwE")
    techStack = [i.text for i in techStack] # 기술 스택

    allPre = mainInfo[0].find_all('pre')  
    allPre = [i.text for i in allPre]

    for i in range(len(allPre)):
        if i == 1:
            mainWork = allPre[i]
        elif i == 2:
            requirements = allPre[i]
        elif i == 3:
            preferential = allPre[i]
        elif i == 4:
            welfare = allPre[i]
        elif i == 5:
            procedures = allPre[i]
            
    mainWork = mainWork.split('\n')
    requirements = requirements.split('\n')
    preferential = preferential.split('\n')
    welfare = welfare.split('\n')
    procedures = procedures.split('\n')

    return techStack, mainWork, requirements, preferential, welfare, procedures

# 상세 페이지 맨 아래 부분 의 블라인드 정보 부분을 리턴
# 경력, 학력, 마감일, 근무지
# 마감일은 숫자만 남기고 제거
# 년 월 일 형식으로 변환
def getBlindInfo(body):
    blindInfo = body.find_all('dl', class_ = "sc-b12ae455-1 hvXrQd")
    career = blindInfo[0].find('dd').text
    education = blindInfo[1].find('dd').text
    closedAt = blindInfo[2].find('dd').text
    jobLocation = blindInfo[3].find('li').text
    
    # 숫자만 남기고 제거    
    closedAt = re.sub(r'[^0-9]', '', closedAt)  
    # 년 월 일 형식으로 변환
    closedAt = f"{closedAt[:4]}년 {closedAt[4:6]}월 {closedAt[6:]}일"

    return career, education, closedAt, jobLocation

# 상세 페이지의 정보를 가져오는 함수
# 각 상세 페이지의 정보를 가져와서 딕셔너리 형태로 리턴
# 위에 있는 함수 다 합쳐서 진짜 상세 페이지 가져와서 딕셔너리 형태로 반환 시켜주는 함수임

def crawlingDetailPage():
    positions = getJumpitPositions()
    ids = getIDs(positions)
    data = []
    for id in ids:
        url = DETAIL_URL + str(id)
        print(f"상세 페이지 {id+1} : {url}")
        time.sleep(0.2)

        body = getDetailPage(url)
        title, company, pride = getTitleInfo(body)
        techStack, mainWork, requirements, preferential, welfare, procedures = getMainInfo(body)
        career, education, closedAt, jobLocation = getBlindInfo(body)

        data.append({
            'id' : id,
            'title': title,
            'company': company,
            'pride': pride,
            'techStack': techStack,
            'mainWork': mainWork,
            'requirements': requirements,
            'preferential': preferential,
            'welfare': welfare,
            'procedures': procedures,
            'career': career,
            'education': education,
            'closedAt': closedAt,
            'jobLocation': jobLocation
        })
    print('상세 페이지 정보 반환 완료')
    return data