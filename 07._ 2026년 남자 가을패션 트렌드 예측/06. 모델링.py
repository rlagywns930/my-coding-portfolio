1. LDA 모델링

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# 1. 데이터 불러오기
df = pd.read_csv("C:/Users/user/Desktop/men_fashion_2026naver_blog/5_피처엔지니어링/men_fashion_2025_naver_blog_featured.csv")

# 3. CountVectorizer로 단어 빈도 행렬 생성
vectorizer = CountVectorizer(max_features=1000, stop_words='english')  # 필요에 따라 한국어 불용어 처리 추가
count_data = vectorizer.fit_transform(df['text'])

# 4. LDA 모델 학습 (예: 토픽 10개)
lda = LatentDirichletAllocation(n_components=10, random_state=42)
lda.fit(count_data)

# 5. 토픽별 주요 단어 출력
def display_topics(model, feature_names, no_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print(f"토픽 {topic_idx+1}: ", end='')
        print(" | ".join([feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]))

no_top_words = 10
tf_feature_names = vectorizer.get_feature_names_out()
display_topics(lda, tf_feature_names, no_top_words)

# 6. 문서별 토픽 분포 저장
doc_topic_dist = lda.transform(count_data)  # (문서수 x 토픽수) 배열

# 7. 문서별 토픽 분포를 데이터프레임에 붙이기
topic_df = pd.DataFrame(doc_topic_dist, columns=[f'topic_{i+1}' for i in range(doc_topic_dist.shape[1])])
df = pd.concat([df.reset_index(drop=True), topic_df.reset_index(drop=True)], axis=1)

# 8. 결과 저장
df.to_csv("men_fashion_2025_naver_blog_lda_topics.csv", index=False, encoding='utf-8-sig')
print("✅ LDA 토픽 모델링 완료 및 저장됨")
----

2. 랜덤 포레스트

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle

# 1. 데이터 불러오기 (피처엔지니어링 후 저장된 파일)
df = pd.read_csv("C:/Users/user/Desktop/men_fashion_2026naver_blog/6_모델링/men_fashion_2025_naver_blog_lda_topics.csv")

# 2. 'text'와 'postdate' 컬럼 삭제 (만약 존재한다면)
df.drop(['text', 'postdate'], axis=1, inplace=True, errors='ignore')

# 3. 타겟 변수와 설명 변수 분리
X = df.drop('keyword', axis=1)
y = df['keyword']

# 4. LabelEncoder 불러오기 (한글 키워드 복원을 위해)
with open("C:/Users/user/Desktop/men_fashion_2026naver_blog/5_피처엔지니어링/le_keyword.pkl", "rb") as f:
    le_keyword = pickle.load(f)

# 5. train/test 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. 랜덤포레스트 모델 학습
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)

# 7. 예측 및 평가
y_pred = rf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# 8. 예측 결과를 원래 한글 키워드로 변환 (LabelEncoder 역변환)
y_pred_labels = le_keyword.inverse_transform(y_pred)
print("예측 결과 (한글 키워드):", y_pred_labels)
---

3. XGBoost

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pickle
import xgboost as xgb

# 1. 데이터 불러오기 (LDA 토픽 모델링 후 저장된 파일)
df = pd.read_csv("C:/Users/user/Desktop/men_fashion_2026naver_blog/6_모델링/men_fashion_2025_naver_blog_lda_topics.csv")

# 2. 'text'와 'postdate' 컬럼 삭제 (만약 존재한다면)
df.drop(['text', 'postdate'], axis=1, inplace=True, errors='ignore')

# 3. 타겟 변수와 설명 변수 분리
X = df.drop('keyword', axis=1)
y = df['keyword']

# 4. LabelEncoder 불러오기 (한글 키워드 복원을 위해)
with open("C:/Users/user/Desktop/men_fashion_2026naver_blog/5_피처엔지니어링/le_keyword.pkl", "rb") as f:
    le_keyword = pickle.load(f)

# 5. train/test 데이터 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. XGBoost 모델 학습
xgb_model = xgb.XGBClassifier(eval_metric='mlogloss', random_state=42)
xgb_model.fit(X_train, y_train)

# 7. XGBoost 예측 및 평가
y_pred_xgb = xgb_model.predict(X_test)
print("XGBoost Accuracy:", accuracy_score(y_test, y_pred_xgb))
print("XGBoost Classification Report:\n", classification_report(y_test, y_pred_xgb))

# 8. XGBoost 예측 결과를 한글 키워드로 변환
y_pred_xgb_labels = le_keyword.inverse_transform(y_pred_xgb)
print("XGBoost 예측 결과 (한글 키워드):", y_pred_xgb_labels)
