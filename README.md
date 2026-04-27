# 👋 안녕하세요! 데이터와 AI에 대해서 공부하고 있는 김효준입니다.

이 저장소는 **머신러닝 및 딥러닝 모델 설계**, **모델 최적화 파이프라인**, 그리고 **데이터 기반 AI 솔루션**을 구현한 프로젝트들을 모아둔 공간입니다.  
단순한 데이터 핸들링을 넘어, **알고리즘의 성능을 극대화**하고 실무적인 AI 가치를 창출하기 위해 다양한 연구와 실습을 지속하고 있습니다.

---

## 🚀 주요 AI 프로젝트

### 👔 NLP 기반 패션 트렌드 예측 시스템 (Hybrid ML Model)
- **Problem:** 정답이 고정되지 않은 비정형 텍스트 데이터의 잠재적 패턴 분류
- **Architecture:** - **LDA(Latent Dirichlet Allocation)** 기반 비지도 학습 피처 추출
  - **TF-IDF + 토픽 분포 피처**를 결합한 하이브리드 입력 레이어 설계
- **Optimization:** GridSearchCV를 활용한 하이퍼파라미터 튜닝으로 **XGBoost 성능 4.3%p 향상**
- **Insight:** 고차원 텍스트 데이터에서 실무적인 트렌드 분류 체계 자동화 가능성 확인

---

### 🌊 LSTM 기반 시계열 수위 예측 엔진
- **Model:** 초음파 센서 데이터 처리를 위한 **LSTM(Long Short-Term Memory)** 아키텍처 설계
- **Core:** 시계열 데이터의 장기 의존성(Long-term dependency) 학습을 통한 실시간 잔여 시간 예측
- **Performance:** 센서 오차를 최소화한 학습을 통해 **MAE(Mean Absolute Error): 0.097** 달성
- **Goal:** 실시간 재난 대응 시스템을 위한 딥러닝 모델의 실제 적용성 검증

---

### 🚢 Classification Pipeline (Titanic)
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

## 🛠 AI 기술 스택 & 도구

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

## 📁 Repository Structure

| 폴더명 | 내용 |
|----------------------------------|--------------------------------------|
| `01._공부` | AI/ML 핵심 알고리즘 및 통계 이론 정리 |
| `02_python-coding` | 알고리즘 구현 및 AI 모델링 스크립트 |
| `03._타이타닉 생존 예측 모델` | Classification: 이진 분류 모델링 파이프라인 |
| `05._LSTM 기반 수위 예측 모델l` | **RNN/LSTM: 시계열 센서 데이터 예측 엔진** |
| `06._project_house_prices` | Regression: 수치형 데이터 회귀 예측 모델 |
| `07._26년 남자 가을 패션 트렌드 예측` | **NLP: 비정형 텍스트 트렌드 분류 시스템** |

---

## 🎯 AI Engineering Focus

- **Model Performance Optimization:** 모델의 수치적 정확도와 일반화 성능 최적화에 집중합니다.
- **End-to-End Pipeline:** 데이터 크롤링부터 전처리, 아키텍처 설계, 평가까지의 전체 AI 워크플로우를 구축합니다.
- **Explainable AI (XAI):** 단순히 결과만 도출하는 것이 아니라, 모델이 왜 그런 예측을 했는지 논리적으로 해석합니다.
- **Continuous Learning:** 새로운 알고리즘과 딥러닝 논문을 학습하며 실질적인 문제 해결에 적용합니다.

---

데이터를 통해 세상을 학습시키고 문제를 해결하는 과정 자체를 즐깁니다. 😊  
📩 협업이나 기술적 논의는 언제든 환영입니다!
