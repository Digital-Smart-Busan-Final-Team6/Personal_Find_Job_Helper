# 🧑🏻‍💻 Personal Find Job Helper

> “취업 준비, 더 똑똑하게!” – 이력서 기반 맞춤형 공고 추천 & AI Q\&A 시스템

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12-blue?logo=python" />
  <img src="https://img.shields.io/badge/Django-5.2.2-092E20?logo=django&logoColor=white" />
  <img src="https://img.shields.io/badge/LangChain-⚡-orange" />
  <img src="https://img.shields.io/badge/License-MIT-lightgrey" />
  <img src="https://img.shields.io/badge/PRs-welcome-brightgreen" />
</p>

## 📚 프로젝트 개요

“Personal Find Job Helper”(PFJH)는 **이력서와 선호 조건**을 입력하면 자연어 처리·LLM(대규모 언어 모델)을 통해

1. **채용 공고 크롤링 → 필터링 → 랭킹**
2. **AI 챗봇**을 통한 공고 분석 & 커리어 상담

을 한 번에 제공하는 **취업 지원 플랫폼**입니다.

> 디지털스마트부산 6기 **팀 6** 최종 프로젝트 ✨

---

## ✨ 핵심 기능

| 기능               | 설명                                                                 |
| ---------------- | ------------------------------------------------------------------ |
| 🔍 **맞춤형 공고 추천** | 이력서의 학력·스킬·지역·희망 직무를 태깅해 Web 크롤링 공고와 **SBERT + BM25**로 유사도 계산 & 랭크 |
| 🤖 **LLM Q\&A**  | GPT‑4o(OpenAI) 기반 LangChain Agent로 “해당 공고 합격 전략 알려줘” 등 자유 질의 응답    |
| 📝 **이력서 CRUD**  | Django Admin & 전용 폼으로 이력서 등록·수정·삭제·내보내기(JSON)                      |
| 📊 **리포트 뷰**     | 선택 이력서 × 검색어 조합에 대한 종합 분석 리포트(JSON + UI)                           |


---

## 🏗️ 폴더 구조

```text
├─ Crawling/            # 공고 크롤러 & 스케줄러
├─ Data_Files/          # 전처리된 JSON·CSV·모델 파일
├─ LLM_Judge/           # LLM‑기반 평가 스크립트
├─ Model_Predict/       # SBERT 파인튜닝 & 추론 모듈
├─ Run_Pipeline/        # 전체 ETL·에이전트 파이프라인
├─ main/                # Django 앱 (View·Template·Static)
├─ accounts/            # 사용자/이력서 모델
├─ config/              # Django 설정
└─ ...
```

---

## ⚙️ 설치 & 실행

### 1) 환경 세팅

```bash
# 가상환경(Windows)
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python -m venv venv
source venv/bin/activate

# 공통 – 의존성 설치
pip install -r requirements.txt
```

### 2) 환경 변수

프로젝트 루트에 **`.env`** 파일을 만들고 다음을 설정하세요.

```env
OPENAI_API_KEY=YOUR_KEY_HERE
SERPAPI_API_KEY=YOUR_KEY_HERE
```

### 3) 데이터 준비 (미완)

```bash
python Run_Pipeline/run_all.py   # 크롤링 → 전처리 → 임베딩
```

### 4) 서버 실행

```bash
# Django (관리 페이지 & Web UI)
python manage.py migrate
python manage.py runserver

# FastAPI (LLM Inference, 선택)
uvicorn server:app --reload
```

서버가 실행되면 **[http://127.0.0.1:8000](http://127.0.0.1:8000)** 에서 서비스를 확인할 수 있습니다.

---

## 🛠 기술 스택

* **Backend** : Django 5, FastAPI, Uvicorn
* **LLM / NLP** : OpenAI GPT‑4o-mini, LangChain, Sentence‑Transformers(SBERT), Rank‑BM25
* **DB & Vector Store** : SQLite(개발), ChromaDB
* **Frontend** : HTML · CSS · JavaScript(Bootstrap 5), Django Template
* **ETL & 크롤링** : Selenium, BeautifulSoup4, Pandas
* **MLOps** : HuggingFace Hub, LangServe, Docker(예정)

---


## 👥 팀원

| 이름      | 역할                        | GitHub                                       |
| ------- | ------------------------- | -------------------------------------------- |
| **심수훈** | 프로젝트 리더 · 백엔드 · NLP 파이프라인 | [@suhoon1020](https://github.com/suhoon1020) |
| **경빈**  | 프론트엔드 · UX/UI · 데이터 시각화   | [@Lapisbin](https://github.com/Lapisbin)     |

---
## 📝 라이선스

본 프로젝트는 **MIT License**를 따릅니다. 라이선스 전문은 `LICENSE` 파일을 확인하세요.

---

## 🙌 Reference & Thanks

* **Digital Smart Busan** 교육과정에 도움 주신 모든 강사님·멘토님께 감사드립니다.
* OpenAI, LangChain, Sentence‑Transformers, ChromaDB 등 훌륭한 오픈소스 커뮤니티에 감사드립니다.

> 🐣 프로젝트가 도움이 되셨다면 ⭐Star 로 응원해 주세요!
