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
3.  [최종 파이프라인 (Notebooks)](#3-최종-파이프라인-notebooks)
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

## 3. 최종 파이프라인 (Notebooks)

본 프로젝트의 전체 파이프라인은 아래의 Jupyter Notebook들을 순차적으로 실행하며 진행됩니다.

### 3.1. 전처리

1.  **[10-local-formatting.ipynb](https://github.com/j8n17/Dacon_HanSolDeco/blob/main/preprocess/10-local-formatting.ipynb)**: 초기 데이터 로딩 및 기본 형식 정리 작업을 수행합니다.
2.  **[20-colab-spelling_correction.ipynb](https://github.com/j8n17/Dacon_HanSolDeco/blob/main/preprocess/20-colab-spelling_correction.ipynb)**: 텍스트 데이터의 품질 향상을 위해 오탈자를 수정합니다.
    <details>
    <summary><b>프롬프트 예시</b></summary>

    ```
    당신은 맞춤법 수정 전문가입니다. 제공될 문장의 맞춤법을 수정하되, 형식을 유지하고 추가적인 정보를 생성하지 마세요.
    문장: "넘어짐 사고 (사고자가 보행로가 아닌 콘크리트 L형 측구 양생을 위해 덮어둔 천막위를 걷다가 집수정(맨홀) 개구부를 밝아 실족하여 상해 발생)"
    수정: 
    ```

    **LLM 답변**

    ```
    넘어짐 사고(사고자는 보행로가 아닌 콘크리트 L형 측구 양생을 위해 덮어둔 천막 위를 걷다가 집수정(맨홀) 개구부를 밟아 실족하여 상해 발생)
    ```
    </details>

3.  **[30-colab-structured_description.ipynb](https://github.com/j8n17/Dacon_HanSolDeco/blob/main/preprocess/30-colab-structured_description.ipynb)**: LLM을 이용해 사고 정보 텍스트에서 '발생 배경', '사고 종류', '사고 원인' 등 구조화된 정보를 추출합니다. 이는 후속 RAG 단계에서 검색 효율성을 높이기 위함입니다.
    <details>
    <summary><b>프롬프트 예시</b></summary>
    
    ```
    한국 건설 공사 안전 사고 관련 문장을 언어 모델에 사용하기 위해 전처리하려 합니다.
    사고가 발생한 배경과 핵심적인 피해 내용, 사고 원인을 도출하고, "발생 배경, 사고 종류, 사고 원인"을 json 형식으로 핵심적인 내용만 간결히 정리해주세요.
    추가적인 정보를 임의로 추론하거나 생성하지 말고, 원문에 주어진 정보만을 반영하세요.
    제공될 문장은 [문장: "사고 종류 (사고 설명 또는 원인)"] 형식으로 제공됩니다.
    문장: "설치작업 중 넘어짐 사고 (크레인 이용 작업 중 줄걸이에 작업자 생명줄이 걸려 중심을 잃고 인접 시스템 동바리 자재에 부딪혀 부상)"
    ```

    **LLM 답변**

    ````
    ```json
    {   
        "발생 배경": "크레인 이용 작업 중",
        "사고 종류": "넘어짐 사고 (줄걸이)",
        "사고 원인": "줄걸이에 작업자 생명줄이 걸려 중심을 잃고 인접 시스템 동바리 자재에 부딪힘"
    }
    ```
    ````
    </details>
4.  **[40-local-reason_extract.ipynb](https://github.com/j8n17/Dacon_HanSolDeco/blob/main/preprocess/40-local-reason_extract.ipynb)**: LLM의 JSON 출력에서 '발생 배경', '사고 종류', 특히 RAG 검색의 핵심 키가 될 '발생 원인' 정보를 추출합니다.

### 3.2. RAG

5.  **[50-local-rag_prompt.ipynb](https://github.com/j8n17/Dacon_HanSolDeco/blob/main/rag/50-local-rag_prompt.ipynb)**: ‘발생 원인’을 쿼리로, '재발방지대책 및 향후조치계획'를 답변으로 사용해 유사 사고 사례의 QA를 검색하는 FAISS RAG 파이프라인을 구현합니다. 초기 검색(Retrieve)에서는 쿼리와 유사한 상위 25개의 결과를 검색하고, 이후 검색된 결과들의 답변과 대표 문장과의 유사도를 기준으로 결과를 재정렬(Reranking)하여 가장 높은 유사도, 중간, 그리고 낮은 유사도 답변를 가진 3개의 QA를 제공합니다. 이를 통해 LLM이 다양한 사례를 참고할 수 있도록 개선하였습니다.

6.  **[60-colab-llm_summary.ipynb](https://github.com/j8n17/Dacon_HanSolDeco/blob/main/rag/60-colab-llm_summary.ipynb)**: 검색된 QA정보를 바탕으로 Ollama LLM을 호출하여 답변을 N개 생성합니다.
    <details>
    <summary><b>프롬프트 예시</b></summary>

    ````
    아래의 모범 답안 예시를 참고해 answer을 작성해주세요. 각각의 answer은 모두 question과 base_answer을 반영해 만든 결과입니다. 결과는 ""user_question""만 json으로 출력해주세요.
    ```json
    {
    ""base_answer"": ""작업전 안전교육 강화 및 작업장 위험요소 점검을 통한 재발 방지와 안전관리 교육 철저를 통한 향후 조치 계획.""
    ""examples"": [
        {
        ""question"": ""발생 배경: 절단 작업 중, 사고 종류: 기계 사용 부주의로 인한 절단, 베임 사고, 사고 원인: 기계 사용 부주의"",
        ""answer"": ""장비 점검 및 작업자 안전교육 실시.""
        },
        {
        ""question"": ""발생 배경: 설치작업 중, 사고 종류: 보강토 옹벽 관련 절단, 베임 사고, 사고 원인: 자재 절단 작업 시 절단부로부터 작업자 이격 미흡, 안전 장비(베임 방지 장갑) 미착용"",
        ""answer"": ""절단작업 시 안전장갑 지급과 근로자 안전교육 실시를 통한 재발 방지 대책 마련.""
        },
        {
        ""question"": ""발생 배경: 절단 작업 중, 사고 종류: 공구류 관련 절단, 베임 사고, 사고 원인: 공구 사용 중 작업자 부주의"",
        ""answer"": ""공구사용 전 주의사항 고지 및 공구사용 숙련자 배치와 함께 건설현장 재해예방을 위한 안전관리 철저 지시.""
        }
    ]
    }
    ```
    ```json
    {
    ""user_question"": {
        ""question"": ""발생 배경: 절단 작업 중 각도 절단기 이용, 사고 종류: 각도 절단기 관련 베임 사고, 사고 원인: 방호덮개 하락 지연으로 인한 재해자 손상"",
        ""answer"": """"
    }
    }
    ```
    ````

    **LLM 답변**

    ````
    ```json
    {
    ""user_question"": {
        ""question"": ""발생 배경: 절단 작업 중 각도 절단기 이용, 사고 종류: 각도 절단기 관련 베임 사고, 사고 원인: 방호덮개 하락 지연으로 인한 재해자 손상"",
        ""answer"": ""각도절단기 안전 점검 강화 및 작업자 교육 실시, 방호덮개 작동 상태 확인 철저, 재해 발생 시 응급처치 교육 실시를 통한 향후 조치 계획.""
        }
    }
    ```
    ````
7.  **[70-local-answer_extract.ipynb](https://github.com/j8n17/Dacon_HanSolDeco/blob/main/rag/70-local-answer_extract.ipynb)**: LLM이 생성한 JSON 텍스트에서 'answer'을 추출합니다. LLM이 지정된 답변 형식대로 응답하지 않았을 경우, 올바른 형식으로 답변할 때까지 반복적으로 재생성합니다.
8.  **[80-local-answers_rerank.ipynb](https://github.com/j8n17/Dacon_HanSolDeco/blob/main/rag/80-local-answers_rerank.ipynb)**: LLM이 생성한 N개의 답변과 25개의 Retrieve된 답변과의 코사인 유사도를 계산하여, N개의 답변 중 가장 높은 유사도를 가진 답변을 최종 결과로 선정합니다.

---

## 4. 주요 실험 및 결과

*   **초기 접근:** 훈련 데이터의 정답 문장 임베딩 후 코사인 유사도가 가장 높은 대표 문장을 사용하는 방식 (평균 0.7 유사도 달성).
*   **군집화 시도:** 사고 정보 임베딩 기반 군집화는 벡터 경계 모호성 및 균일 분포로 인해 유의미한 군집 형성에 실패 (PCA 분석 결과 참고).
*   **RAG Query 개선:** 단순 사고 정보 대신 LLM으로 추출한 '발생 원인'을 RAG 쿼리로 사용하여 검색 정확도 향상.
*   **Reranking 도입:** 검색된 결과 중 대표 문장과 높은, 중간, 낮은 유사도 답변을 함께 제공하여 LLM이 다양한 케이스를 참고하도록 개선.
*   **추론 속도 최적화:** 대량 테스트 데이터 추론 시 Ollama 병렬 처리(subprocess 활용)로 약 2배 속도 향상.

자세한 실험 및 분석 내용은 [DETAILS.md](https://github.com/j8n17/Dacon_HanSolDeco/blob/main/DETAILS.md) 참고.

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
    *   Jupyter Notebook으로 구성된 파이프라인을 통합하여, 단일 스크립트로 실행 가능한 형태로 구축.

---

## 6. 파일 구조

```
Dacon_HanSolDeco/
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

자세한 내용은 [settings.py](https://github.com/j8n17/Dacon_HanSolDeco/blob/main/settings.py), [requirements.txt](https://github.com/j8n17/Dacon_HanSolDeco/blob/main/requirements.txt) 참고.