# 👋 안녕하세요! 데이터 분석을 공부하고 있는 김효준입니다.

이 저장소는 제가 직접 수행한 **데이터 분석 및 머신러닝 프로젝트**, 그리고 **핵심 이론 정리**를 모아둔 공간입니다.  
단순한 이론 학습을 넘어, **실무에 활용 가능한 분석 역량**을 기르기 위해 다양한 실습을 진행해왔습니다.

---

## 🚀 주요 프로젝트 소개

### 👔 2026년 남성 가을 패션 트렌드 예측 (NLP)
- **네이버 블로그 API** 기반 비정형 텍스트 데이터 수집 및 정제
- **KoNLPy(Okt) & TF-IDF**를 활용한 형태소 분석 및 키워드 벡터화
- **LDA(Latent Dirichlet Allocation)** 토픽 모델링을 통한 잠재 트렌드 추출
- **XGBoost & Random Forest** 기반의 다중 분류 모델링 및 하이퍼파라미터 최적화
- **주요 성과:** 최적화를 통해 XGBoost 모델의 Accuracy를 **24.7% → 29.0%**로 개선 (수백 개의 키워드 분류 태스크)

---

### 🚢 Titanic 생존자 예측 프로젝트
- Titanic 데이터셋 기반으로 생존 여부를 예측하는 분류 모델 개발
- **EDA → Feature Engineering → 모델링 → 결과 해석(SHAP)** 전 과정을 수행
- **Random Forest, XGBoost** 등 다양한 모델 비교 및 성능 분석  
- **최종 Accuracy: 0.78 (Kaggle 기준)**

---

### 🌊 LSTM 기반 수위 예측 시스템
- **초음파 센서 실험 데이터 + 딥러닝 모델(LSTM)** 을 활용한 수위 예측 프로젝트
- 실시간 거리 데이터를 수집하여 LSTM 모델로 **잔여 시간 예측**
- **실시간 재난 대응 시스템**의 가능성을 실험적으로 확인한 연구형 프로젝트
- **MAE(Mean Absolute Error): 0.097**

---

### 🏠 House Price Prediction
- Kaggle의 주택 가격 예측 데이터셋을 활용한 **회귀 분석 프로젝트**
- **로그 변환, 결측치 처리, 이상치 제거, 변수 인코딩, 모델 튜닝** 등 데이터 전처리 수행
- **XGBoost 모델 활용 → RMSLE 기준 성능 개선**
  - Train RMSLE: **0.1116**
  - Kaggle 점수: **0.13741**

---

### 📘 데이터 분석 기초 정리
- ADsP 시험 및 실무 기초 학습을 위한 **데이터 분석 이론 요약**
- 통계 개념, 분석 기법, 시각화 방법 등 핵심 개념을 정리한 문서

---

## 🛠 사용 도구 및 기술 스택

| 분야 | 도구 및 라이브러리 |
|------|------------------|
| 언어 및 환경 | Python, Jupyter Notebook |
| 데이터 처리 | Pandas, NumPy, BeautifulSoup |
| 자연어 처리 | **KoNLPy (Okt), TF-IDF, LDA Topic Modeling** |
| 시각화 | Matplotlib, Seaborn, **WordCloud** |
| 머신러닝 | Scikit-learn, **XGBoost**, Random Forest |
| 딥러닝 | TensorFlow, Keras (LSTM) |
| 모델 최적화/해석 | GridSearchCV, SHAP |
| 협업 및 문서화 | Git, GitHub, Markdown |

---

## 📁 폴더 구성

| 폴더명 | 내용 |
|----------------------------------|--------------------------------------|
| `01._theory` | 데이터 분석 기초 이론 정리 |
| `02_python-coding` | 기본 실습용 코드 정리 |
| `03._project-titanic` | Titanic 생존자 예측 프로젝트 (V1, V2) |
| `04._project_titanic2` | Titanic 예측 프로젝트 확장 버전 |
| `05._LSTM 기반 수위 예측 모델` | 센서 기반 시계열 데이터 + LSTM 예측 프로젝트 |
| `06._project_house_prices` | House Price 회귀 예측 프로젝트 |
| **`07._2026년 남자 가을 패션 트렌드`** | **네이버 API 데이터 기반 트렌드 예측 프로젝트 (NLP)** |

---

## 🎯 이런 점에 집중했습니다

- **데이터 기반 문제 해결**을 위한 전 과정(End-to-End) 실습 경험  
- **비정형 데이터 처리:** API 수집부터 형태소 분석, 토픽 모델링까지의 NLP 파이프라인 구축  
- 단순한 모델 적용이 아닌, **데이터 이해 → 전처리 → 모델링 → 해석 → 개선안 제시**까지 전 과정 수행  
- **SHAP, RMSLE, LDA 등** 다양한 분석 기법과 도메인에 맞는 평가지표 활용  
- 결과의 **정량적 근거를 바탕으로 한 인사이트 도출 능력** 강화  

---

데이터를 좋아하고, 문제 해결을 즐기는 저는  
**늘 배우고 성장하는 자세**로 새로운 도전을 이어가고 있습니다. 😊  
📩 협업이나 문의는 언제든 환영입니다!# 👋 안녕하세요! AI 엔지니어를 꿈꾸는 김효준입니다.

이 저장소는 **머신러닝 및 딥러닝 모델 설계**, **모델 최적화 파이프라인**, 그리고 **데이터 기반 AI 솔루션**을 구현한 프로젝트들을 모아둔 공간입니다.  
단순한 데이터 핸들링을 넘어, **알고리즘의 성능을 극대화**하고 실무적인 AI 가치를 창출하기 위해 다양한 연구와 실습을 지속하고 있습니다.

---

## 🚀 주요 AI 프로젝트

### 👔 NLP 기반 패션 트렌드 예측 모델 (Hybrid ML Model)
- **Problem:** 정답이 고정되지 않은 비정형 텍스트 데이터의 잠재적 패턴 분류
- **Architecture:** - **LDA(Latent Dirichlet Allocation)** 기반 비지도 학습 피처 추출
  - **TF-IDF + 토픽 분포 피처**를 결합한 하이브리드 입력 레이어 설계
- **Optimization:** GridSearchCV를 활용한 하이퍼파라미터 튜닝으로 **XGBoost 성능 4.3%p 향상**
- **Insight:** 고차원 텍스트 데이터에서 실무적인 트렌드 분류 체계 자동화 가능성 확인

---

### 🌊 LSTM 기반 시계열 수위 예측 시스템
- **Model:** 초음파 센서 데이터 처리를 위한 **LSTM(Long Short-Term Memory)** 아키텍처 설계
- **Core:** 시계열 데이터의 장기 의존성(Long-term dependency) 학습을 통한 실시간 잔여 시간 예측
- **Performance:** 센서 오차를 최소화한 학습을 통해 **MAE(Mean Absolute Error): 0.097** 달성
- **Goal:** 실시간 재난 대응 시스템을 위한 딥러닝 모델의 실제 적용성 검증

---

### 🚢 타이타닉 생존 예측 모델
- **Objective:** 정교한 **Feature Engineering**을 통한 이진 분류 모델의 일반화 성능 확보
- **Algorithm:** Random Forest, XGBoost 등 다양한 앙상블 모델의 성능 비교 분석
- **Explainability:** **SHAP**을 활용한 모델의 의사결정 근거 분석 및 설명 가능성(XAI) 확보
- **Result:** Kaggle 기준 최종 **Accuracy 0.78** 달성

---

### 🏠 House Price Prediction (Regression Optimization)
- **Objective:** 다양한 회귀 알고리즘 적용 및 손실 함수(Loss Function) 최적화
- **Pipeline:** 변수 스케일링, 로그 변환, 이상치 제거 등 모델 학습 효율 극대화 파이프라인 구축
- **Result:** **RMSLE 0.1116** 기록 및 하이퍼파라미터 튜닝을 통한 모델 안정성 확보

---

## 🛠 AI 기술 & 도구

| 분야 | 기술 스택 |
|------|------------------|
| **Languages** | Python |
| **Deep Learning** | **TensorFlow, Keras, LSTM** |
| **Machine Learning** | **Scikit-learn, XGBoost, Random Forest, LDA** |
| **NLP** | **KoNLPy (Okt), TF-IDF, Topic Modeling, BeautifulSoup** |
| **Model Optimization** | **GridSearchCV, Hyperparameter Tuning** |
| **XAI & Visualization** | **SHAP**, Matplotlib, Seaborn, WordCloud |
| **DevOps & Docs** | Git, GitHub, Markdown |

---

## 📁 파일 구조

| 폴더명 | 내용 |
|----------------------------------|--------------------------------------|
| `01._공부` | AI/ML 핵심 알고리즘 및 통계 이론 정리 |
| `02_python-coding` | 알고리즘 구현 및 AI 모델링 스크립트 |
| `03._타이타닉 생존예측 모델` | Classification: 이진 분류 모델링 파이프라인 |
| `05._LSTM-기반 수위예측 모델` | **RNN/LSTM: 시계열 센서 데이터 예측 엔진** |
| `06._project_house_prices` | Regression: 수치형 데이터 회귀 예측 모델 |
| `07._26년 남자가을 패션 트렌드 예측` | **NLP: 비정형 텍스트 트렌드 분류 시스템** |

---

## 🎯 이런점에 집중했어요

- **Model Performance Optimization:** 모델의 수치적 정확도와 일반화 성능 최적화에 집중합니다.
- **End-to-End Pipeline:** 데이터 크롤링부터 전처리, 아키텍처 설계, 평가까지의 전체 AI 워크플로우를 구축합니다.
- **Explainable AI (XAI):** 단순히 결과만 도출하는 것이 아니라, 모델이 왜 그런 예측을 했는지 논리적으로 해석합니다.
- **Continuous Learning:** 새로운 알고리즘과 딥러닝 논문을 학습하며 실질적인 문제 해결에 적용합니다.

---

데이터를 통해 세상을 학습시키고 문제를 해결하는 과정 자체를 즐깁니다. 😊  
📩 협업이나 기술적 논의는 언제든 환영입니다!

감사합니다. 🙌
