# 🚧 건설 공사 사고 정보 기반 RAG-LLM 경진대회 프로젝트

[![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![Environment](https://img.shields.io/badge/Environment-Jupyter/Colab-orange.svg)](https://colab.research.google.com/)
[![LLM Engine](https://img.shields.io/badge/LLM-Ollama-lightgrey.svg)](https://ollama.com)
[![Vector Search](https://img.shields.io/badge/Search-FAISS-blueviolet.svg)](https://github.com/facebookresearch/faiss)
[![Langchain](https://img.shields.io/badge/Framework-Langchain-yellowgreen.svg)](https://python.langchain.com/)

**건설 현장 사고 데이터를 활용하여 RAG (Retrieval-Augmented Generation) 및 LLM(Ollama) 기반으로 사고 예방/조치 계획을 생성하는 프로젝트입니다. 각 단계는 Jupyter Notebook으로 구현되었으며, 최종 결과는 코사인 유사도를 통해 평가됩니다.**

---

**목차**
1.  [프로젝트 개요](#1-프로젝트-개요)
2.  [데이터 구성](#2-데이터-구성)
3.  [개발 워크플로우 (Notebooks)](#3-개발-워크플로우-notebooks)
4.  [주요 실험 및 결과](#4-주요-실험-및-결과)
5.  [핵심 교훈 및 개선 방향](#5-핵심-교훈-및-개선-방향)
6.  [파일 구조](#6-파일-구조)
7.  [사용된 주요 기술](#7-사용된-주요-기술)

---

## 1. 프로젝트 개요

본 프로젝트는 "건설 공사 사고 정보 기반 RAG-LLM 경진대회"의 일환으로 진행되었습니다. 목표는 주어진 건설 사고 정보(텍스트)를 입력받아, 관련된 과거 사례 및 정보를 검색(Retrieval)하고, 이를 바탕으로 LLM이 최적의 사고 방지 또는 조치 계획(정답 문장)을 생성(Generation)하는 RAG 시스템을 구축하는 것입니다. 최종 평가는 생성된 계획과 실제 정답 계획 간의 유사도를 **유사도 점수 (코사인 유사도 * 0.7 + 자카드 유사도 * 0.3)** 로 측정하여 이루어집니다.

*   **코사인 유사도 (Cosine Similarity):** 두 텍스트를 벡터 공간에 표현했을 때, 두 벡터 간의 각도의 코사인 값을 측정합니다. 방향성에 초점을 맞춰 문맥적, 의미적 유사성을 파악하는 데 유용합니다. (예: 임베딩 벡터 간 유사도)
*   **자카드 유사도 (Jaccard Similarity):** 두 텍스트에 사용된 단어(또는 토큰) 집합 간의 중복 정도를 측정합니다. 두 집합의 교집합 크기를 합집합 크기로 나눈 값으로, 공유하는 단어의 비율을 통해 유사성을 평가합니다.

---

## 2. 데이터 구성

*   **훈련 데이터:** 각 사고 사례에 대한 상세 정보와 그에 해당하는 재발 방지 대책/향후 조치 계획(정답 문장) 텍스트가 포함되어 있습니다.

    *   **사고 설명문:** 사고 정보 컬럼을 조합하여 생성됩니다:
        *   `작업프로세스`: 사고가 발생한 작업 과정
        *   `사고객체(중분류)`: 사고와 관련된 시설물 또는 장비
        *   `인적사고`: 발생한 인적 사고 유형 (떨어짐, 넘어짐 등)
        *   `물적사고`: 발생한 물적 사고 유형
        *   `사고원인`: 사고의 직접적인 원인

    *   **정답 문장:** `재발방지대책 및 향후조치계획` 컬럼에 기록된 예방/조치 계획을 사용합니다. 이는 RAG 시스템이 생성해야 하는 목표 문장이며, 최종 평가에서 생성된 문장과의 유사도를 측정하는 기준이 됩니다.

*   **테스트 데이터:** 사고 정보만 제공되며, 동일한 방식으로 구성된 사고 설명문을 기반으로 예방/조치 계획을 생성하고 이를 실제 정답과 비교하여 성능을 평가합니다.

---

## 3. 개발 워크플로우 (Notebooks)

본 프로젝트의 전체 개발 과정은 아래의 Jupyter Notebook들을 순차적으로 실행하며 진행됩니다.

### 3.1. 전처리 (`preprocess/`)

1.  **`10-local-formatting.ipynb`**: 초기 데이터 로딩 및 기본 형식 정리 작업을 수행합니다.
2.  **`20-colab-spelling_correction.ipynb`**: 텍스트 데이터의 품질 향상을 위해 오탈자를 수정합니다.
3.  **`30-colab-structured_description.ipynb`**: 사고 정보 텍스트에서 '사고 배경', '사고 결과', '발생 원인' 등 구조화된 정보를 LLM을 이용해 추출합니다. 이는 후속 RAG 단계에서 검색 효율성을 높이기 위함입니다.
4.  **`40-local-reason_extract.ipynb`**: 구조화된 정보 중 특히 RAG 검색의 핵심 키가 될 '발생 원인' 정보를 추출하고 정제합니다.

### 3.2. RAG 및 생성 (`rag/`)

5.  **`50-local-rag_prompt.ipynb`**: FAISS를 활용한 벡터 검색 시스템을 구축/로드하고, '발생 원인'을 쿼리로 사용하여 유사 사고 사례의 정답 문장을 검색하는 RAG 파이프라인을 구현합니다. 검색된 결과 중 가장 유사한 답변 외 중간, 낮은 유사도 답변을 함께 제공하여 LLM이 다양한 케이스를 참고하도록 개선했습니다. 초기 검색(Retrieve)은 상위 25개를 가져오며, 이 중 유사도가 높은 상위, 중간, 하위를 1개씩 선택하여 총 3개로 재랭킹(Reranking)합니다.
6.  **`60-colab-llm_summary.ipynb`**: 검색된 정보를 바탕으로 Ollama LLM을 호출하여 예방/조치 계획을 N개 생성합니다.
7.  **`70-local-answer_extract.ipynb`**: LLM이 생성한 JSON 텍스트에서 'answer'을 추출합니다. LLM이 지정된 답변 형식대로 응답하지 않았을 경우, 올바른 형식으로 답변할 때까지 반복적으로 재시도합니다.
8.  **`80-local-answers_rerank.ipynb`**: LLM이 생성한 N개의 답변과 25개의 Retrieve한 답변과의 코사인 유사도를 계산하여, 가장 높은 유사도를 가진 답변을 최종 결과로 선정합니다.

---

## 4. 주요 실험 및 결과

*   **초기 접근:** 훈련 데이터의 정답 문장 임베딩 후 코사인 유사도가 가장 높은 대표 문장을 사용하는 방식 (평균 0.7 유사도 달성).
*   **군집화 시도:** 사고 정보 임베딩 기반 군집화는 벡터 경계 모호성 및 균일 분포로 인해 유의미한 군집 형성에 실패 (PCA 분석 결과 참고).
*   **RAG Query 개선:** 단순 사고 정보 대신 LLM으로 추출한 '발생 원인'을 RAG 쿼리로 사용하여 검색 정확도 향상.
*   **Reranking 도입:** 검색된 결과 중 대표 문장과 높은, 중간, 낮은 유사도 답변을 함께 제공하여 LLM이 다양한 케이스를 참고하도록 개선.
*   **추론 속도 최적화:** 대량 테스트 데이터 추론 시 Ollama 병렬 처리(subprocess 활용)로 약 2배 속도 향상.

---

## 5. 핵심 교훈 및 개선 방향

*   **프롬프트 엔지니어링:** 명확하고 간결한 프롬프트, 특히 JSON 형식 출력 지시는 sLLM 활용 및 후속 처리 효율화에 중요. 핵심 정보 추출이 모델 성능에 큰 영향.
*   **임베딩/군집화:** 데이터 특성에 맞는 임베딩 전략과 군집화 기법 선택의 중요성 확인. 단순 적용의 한계 인지.
*   **RAG 전략:** 단순 유사도 검색을 넘어, 검색 결과의 품질과 다양성을 높이는 재랭킹(Reranking) 등 후처리 전략이 효과적.
*   **성능 최적화:** 대규모 데이터 처리 시 I/O 병목 현상 등 성능 저하 요인을 파악하고 병렬 처리 등 최적화 기법 적용 필요.
*   **향후 방향:**
    *   임베딩 벡터의 특성에 맞는 군집화 방법 탐색 및 적용.
    *   지속적인 RAG 검색/재랭킹 전략 개선.
    *   CPU-GPU 간 데이터 전송 등 추론 병목 현상 추가 분석 및 최적화.

---

## 6. 파일 구조

```
colab/
├── data/
│   ├── train.csv
│   └── test.csv
│
├── preprocess/
│   ├── 10-local-formatting.ipynb
│   ├── 20-colab-spelling_correction.ipynb
│   ├── 30-colab-structured_description.ipynb
│   ├── 40-local-reason_extract.ipynb
│   └── utils.py
│
├── rag/
│   ├── vector_stores/    # 벡터 저장소 디렉토리
│   │   └── faiss_index/  # FAISS 인덱스
│   ├── 50-local-rag_prompt.ipynb
│   ├── 60-colab-llm_summary.ipynb
│   ├── 70-local-answer_extract.ipynb
│   ├── 80-local-answers_rerank.ipynb
│   ├── faiss_utils.py
│   └── utils.py
│
├── settings.py         # Colab 환경 설정 스크립트 (Ollama, 라이브러리 설치 등)
└── README.md
```

---

## 7. 사용된 주요 기술

*   **언어:** Python 3.8+
*   **환경:** Jupyter Notebook / Google Colab
*   **LLM:** Ollama (로컬 구동 LLM)
*   **RAG 프레임워크:** Langchain
*   **벡터 검색:** FAISS (CPU/GPU)
*   **데이터 처리:** Pandas, Numpy
*   **기타 라이브러리:** Scikit-learn (PCA 등), HuggingFace Transformers/Sentence-Transformers (임베딩 모델 로딩 등)

(참고: `settings.py` 및 각 Notebook의 import 구문을 통해 더 상세한 라이브러리 확인 가능)