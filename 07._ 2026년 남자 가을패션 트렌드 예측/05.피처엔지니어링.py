import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

# 1. CSV 불러오기
df = pd.read_csv("C:/Users/user/Desktop/men_fashion_2026naver_blog/3_전처리/men_fashion_2025_naver_blog_preprocessed.csv")

# 2. 기존 그룹값을 새로운 3+1 체계로 매핑
group_map = {
    '아이템': '아이템 그룹',
    '스타일': '스타일링 그룹',
    '라이프스타일': '스타일링 그룹',
    '악세사리': '스타일링 그룹',
    '트렌드 컬러': '트렌드 컬러 그룹'
}
df['group'] = df['group'].map(group_map)

# 3. clean_title과 clean_description 합쳐서 text 컬럼 생성
df['text'] = df['clean_title'].fillna('') + ' ' + df['clean_description'].fillna('')

# 4. 필요없는 컬럼 삭제
df.drop(['clean_title', 'clean_description'], axis=1, inplace=True)

# 5. 범주형 변수 인코딩 (group, keyword)
le_group = LabelEncoder()
le_keyword = LabelEncoder()

df['group'] = le_group.fit_transform(df['group'].astype(str))
df['keyword'] = le_keyword.fit_transform(df['keyword'].astype(str))

# 6. 텍스트 벡터화 (TF-IDF)
vectorizer = TfidfVectorizer(max_features=300)
text_vectors = vectorizer.fit_transform(df['text'])

# 7. 벡터 결과 DataFrame 변환
text_vectors_df = pd.DataFrame(text_vectors.toarray(), columns=[f'tf_{i}' for i in range(text_vectors.shape[1])])

# 8. 벡터화 결과와 기존 데이터 병합
df = pd.concat([df.reset_index(drop=True), text_vectors_df.reset_index(drop=True)], axis=1)

# 9. 필요시 저장
output_path = "C:/Users/user/Desktop/men_fashion_2026naver_blog/5_피처엔지니어링/men_fashion_2025_naver_blog_featured.csv"
df.to_csv(output_path, index=False, encoding='utf-8-sig')
print(f"✅ 저장 완료: {output_path}")
