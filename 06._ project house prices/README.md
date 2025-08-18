# 🏠 House Price 예측 프로젝트

## 📌 프로젝트 개요  
본 프로젝트는 Kaggle의 "House Prices - Advanced Regression Techniques" 데이터셋을 활용하여  
주택의 다양한 특성을 바탕으로 **SalePrice**(주택 가격)을 예측하는 회귀 모델을 개발한 프로젝트입니다.

---

## 🛠 사용한 주요 기술 및 라이브러리  
- Python (pandas, numpy, matplotlib, seaborn 등)  
- scikit-learn (LinearRegression, Ridge, Lasso, RandomForest 등)  
- XGBoost Regressor  
- Feature Engineering 및 One-Hot Encoding  
- 결측치 처리 및 정규화 (StandardScaler 등)  
- 교차검증, 하이퍼파라미터 튜닝 (GridSearchCV)

---

## 📂 프로젝트 구성  

| 파일명                         | 설명                                        |
|------------------------------|-------------------------------------------|
| `01. EDA.md`                 | 탐색적 데이터 분석 및 시각화                      |
| `02. 결측치 처리.md`         | 컬럼별 결측치 파악 및 적절한 처리 방법 적용         |
| `03. Feature Engineering.md` | 파생 변수 생성, 범주형 처리, 이상치 탐지 등           |
| `04. 인코딩 & 정규화.md`     | 범주형 변수 인코딩 및 수치형 변수 정규화             |
| `05. 모델링 및 튜닝.md`      | 다양한 회귀 모델 비교, 교차검증, 하이퍼파라미터 튜닝 |
| `06. 평가 및 예측.md`        | MAE, RMSE 계산 및 submission 파일 생성            |
| `README.md`                  | 전체 프로젝트 설명 및 실행 가이드 문서               |

---

## 📊 주요 결과

- 사용 모델: **XGBoost Regressor**
- 교차검증 평균 RMSE (Log 기준): **0.1270**
- 최종 Kaggle 점수: **0.12345**  
  *(예시는 점수이며, 실제 결과를 반영해 수정하세요)*

---

## ▶ 실행 순서

1. `01. EDA.md` - 데이터 탐색 및 분포 확인  
2. `02. 결측치 처리.md` - 결측값 전략 수립 및 처리  
3. `03~04.md` - 피처 엔지니어링, 인코딩 및 정규화  
4. `05. 모델링.md` - 다양한 모델 학습 및 성능 비교  
5. `06. 평가 및 예측.md` - 예측값 생성 및 submission 저장  

---

## 🔍 핵심 인사이트  

- **GrLivArea, OverallQual, TotalBsmtSF** 등은 주택 가격과 가장 높은 상관관계를 가짐  
- 이상치(Outliers)를 제거함으로써 모델의 안정성 향상  
- 범주형 변수 중 일부는 One-Hot 인코딩 대신 Label Encoding이 더 효과적  
- XGBoost가 다른 모델보다 가장 낮은 오류율을 기록함

---

## 💬 프로젝트 요약  

이 프로젝트는 단순한 회귀 문제 해결에 그치지 않고,  
**피처 엔지니어링, 모델 비교, 성능 향상, 시각적 해석**까지 머신러닝 전 과정의 실전 경험을 목적으로 진행되었습니다.  
실제 서비스에서의 적용을 고려하여, **과적합 방지** 및 **데이터 분포 이해**에 초점을 맞췄습니다.

---

감사합니다!
