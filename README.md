# UCPhraseEvMine-KR

한국어 뉴스 데이터를 기반으로 데이터 내 주요 이벤트를 자동 탐지하는 이벤트 마이닝 프레임워크입니다.  
UCPhrase와 EvMine 프레임워크의 한국어 확장 구현이며, 특정 주제를 기준으로 수집된 뉴스 데이터에서 이벤트 탐지 및 분석을 목표로 합니다.

---

## 프로젝트 개요

- 본 프로젝트는 뉴스 데이터를 분석하여 **시장에 큰 영향을 미치는 주요 이벤트를 자동 탐지**하고, 이를 활용한 **리스크 분석 및 시장 모니터링**을 지원합니다.
- 기존 영어 기반의 [UCPhrase](https://github.com/xgeric/UCPhrase-exp)와 [EvMine](https://github.com/yzhan238/EvMine) 기법을 한국어 뉴스에 맞게 확장하여 적용하였습니다.

---

## 주요 기능

- 뉴스 기사 전처리 및 **품질 문구(Quality Phrase)** 추출
- **피크 문구(Peak Phrase)** 자동 탐지 및 시계열 분석
- 문구 간 의미적/시간적 관계 기반 **이벤트 클러스터링**
- 클러스터 속 뉴스들의 제목을 바탕으로 **GPT-4o로 사건(클러스터) 자동 명명**

---

## 탐지된 이벤트 예시

### Topic 1: 한일 무역 분쟁
- **기간:** 2018-12-07 ~ 2022-02-28
- **관련 뉴스 수:** 1,905건

| 날짜       | 기사 제목                                                |
|------------|-----------------------------------------------------------|
| 2019-07-01 | "수출 규제 강화, 징용판결 보복 아냐"                     |
| 2019-08-21 | "수출허가에 기류 변화 판단 일러…불확실성 여전"         |
| 2018-12-07 | "\"화웨이 쓰지말라\" 동맹국에 요청…한국 및 LGU+는?"  |
| ...        | ...                                        |

---

## 🛠️ 기술 스택

- **언어 및 환경**
  - Python 3.x

- **자연어처리 (NLP)**
  - `Transformers` (Huggingface)
  - `KLUE-BERT-base` 모델 활용
  - `nltk`, `spacy`, `scispacy`, `inflect`, `stop-words`, `Unidecode`

- **머신러닝 및 딥러닝**
  - `torch` (PyTorch)
  - `scikit-learn`
  - `openai` API (GPT-4o 사용)

- **데이터 처리 및 분석**
  - `pandas`, `numpy`, `orjson`, `Cython`, `datefinder`

- **시각화 및 기타**
  - `matplotlib`, `termcolor`, `ipdb`
