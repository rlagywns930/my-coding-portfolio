# Titanic 생존자 예측 프로젝트

## 프로젝트 개요
Titanic 데이터셋을 활용하여 승객의 생존 여부를 예측하는 머신러닝 모델을 개발했습니다.  
데이터 전처리, 교차검증, 하이퍼파라미터 튜닝, SHAP 분석 등을 포함한 전반적인 머신러닝 파이프라인을 구성했습니다.

---

## 사용한 주요 기술 및 라이브러리
- Python (pandas, numpy, scikit-learn, shap, matplotlib, seaborn 등)
- Random Forest Classifier
- 교차검증 및 하이퍼파라미터 튜닝 (GridSearchCV)
- SHAP (모델 해석)

---

## 프로젝트 구성

| 파일명                     | 설명                                  |
|----------------------------|-------------------------------------|
| `data_preprocessing.py`    | 데이터 전처리 및 결측치 처리 코드       |
| `model_training.py`        | 모델 학습, 교차검증, 하이퍼파라미터 튜닝  |
| `shap_analysis.ipynb`      | SHAP 분석 및 시각화                   |
| `submission.csv`           | 제출용 예측 결과 파일                   |
| `README.md`                | 프로젝트 개요 및 설명                  |

---

## 주요 결과 및 점수

- 교차검증 평균 정확도: **0.82** (예시)
- Kaggle 제출 점수: **0.77990** (최종 제출 기준)

---

## 실행 방법

1. `data_preprocessing.py`를 실행하여 데이터 전처리를 수행합니다.  
2. `model_training.py`를 실행하여 모델을 학습하고 최적 하이퍼파라미터를 찾습니다.  
3. `shap_analysis.ipynb`에서 SHAP 분석 및 시각화를 확인할 수 있습니다.  
4. `submission.csv` 파일은 최종 예측 결과이며, Kaggle 제출용입니다.

---

감사합니다!
