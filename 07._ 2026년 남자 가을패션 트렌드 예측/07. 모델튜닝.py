1. 랜덤포레스트 튜닝

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle

# 데이터 불러오기
df = pd.read_csv("C:/Users/user/Desktop/men_fashion_2026naver_blog/6_모델링/men_fashion_2025_naver_blog_lda_topics.csv")

# 불필요한 컬럼 삭제
df.drop(['text', 'postdate'], axis=1, inplace=True, errors='ignore')

# X, y 분리
X = df.drop('keyword', axis=1)
y = df['keyword']

# 레이블 인코더 불러오기 (한글 키워드 복원용)
with open("C:/Users/user/Desktop/men_fashion_2026naver_blog/5_피처엔지니어링/le_keyword.pkl", "rb") as f:
    le_keyword = pickle.load(f)

# train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 랜덤포레스트 기본 모델
rf = RandomForestClassifier(random_state=42)

# 하이퍼파라미터 그리드 설정 (작게 시작)
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'max_features': ['sqrt', 'log2']
}

# GridSearchCV 설정
grid_rf = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=3,
    scoring='accuracy',
    verbose=1,
    n_jobs=1  # 병렬 줄이기
)

# 탐색 실행
grid_rf.fit(X_train, y_train)

# 최적 파라미터, 모델 선택
best_rf = grid_rf.best_estimator_

# 예측 및 평가
y_pred_rf = best_rf.predict(X_test)
print("RF 최적 Accuracy:", accuracy_score(y_test, y_pred_rf))
print("RF 최적 Classification Report:\n", classification_report(y_test, y_pred_rf))

# 한글 키워드 변환
y_pred_rf_labels = le_keyword.inverse_transform(y_pred_rf)
print("RF 예측 결과 (한글 키워드):", y_pred_rf_labels[:10])
---

2. XGBoost 튜닝

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier
import pickle

# 1. 데이터 불러오기
df = pd.read_csv("C:/Users/user/Desktop/men_fashion_2026naver_blog/6_모델링/men_fashion_2025_naver_blog_lda_topics.csv")

# 2. 필요 없는 컬럼 제거
df.drop(['text', 'postdate'], axis=1, inplace=True, errors='ignore')

# 3. 피처/타겟 분리
X = df.drop('keyword', axis=1)
y = df['keyword']

# 4. LabelEncoder 불러오기 (한글 키워드 복원용)
with open("C:/Users/user/Desktop/men_fashion_2026naver_blog/5_피처엔지니어링/le_keyword.pkl", "rb") as f:
    le_keyword = pickle.load(f)

# 5. train / test 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. XGBoost 분류기 정의
xgb_model = XGBClassifier(eval_metric='mlogloss', random_state=42)

# 7. 파라미터 그리드 축소
param_grid = {
    'n_estimators': [100, 150],
    'max_depth': [3, 5],
    'learning_rate': [0.1, 0.2],
    # 'subsample': [0.8],  # 생략하거나 하나만
    # 'colsample_bytree': [0.8]  # 생략하거나 하나만
}

grid_search = GridSearchCV(
    estimator=xgb_model, 
    param_grid=param_grid,
    cv=2,                 # 폴드 수 줄이기
    scoring='accuracy',
    verbose=1,
    n_jobs=1              # 병렬처리 줄임
)

# 8. 탐색 실행
grid_search.fit(X_train, y_train)

# 9. 최적 모델 평가
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

print("최적 XGBoost Accuracy:", accuracy_score(y_test, y_pred))
print("최적 XGBoost Classification Report:\n", classification_report(y_test, y_pred))

# 10. 한글 키워드 예측 결과 출력
y_pred_labels = le_keyword.inverse_transform(y_pred)
print("예측 결과 (한글 키워드):", y_pred_labels[:10])
