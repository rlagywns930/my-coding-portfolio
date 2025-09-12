# 1. [전처리 후 데이터 구조, 결측치, 중복치 확인]

import pandas as pd

# 정확한 전체 경로를 사용해 CSV 불러오기
df = pd.read_csv("C:/Users/user/Desktop/men_fashion_2026naver_blog/3_전처리/men_fashion_2025_naver_blog_preprocessed.csv")

# 1. 데이터프레임 정보 확인
print("🔎 [1] DataFrame 정보 (info):")
print(df.info())
print("\n" + "="*50 + "\n")

# 2. 결측치 확인
print("🧹 [2] 결측치 확인:")
print(df.isnull().sum())
print("\n" + "="*50 + "\n")

# 3. 중복 데이터 확인
print("🔁 [3] 중복 데이터 확인:")
print(f"중복된 행 수: {df.duplicated().sum()}")
---

# 2. [그룹별 워드클라우드]

import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import math

# 데이터 불러오기
df = pd.read_csv("C:/Users/user/Desktop/men_fashion_2026naver_blog/3_전처리/men_fashion_2025_naver_blog_preprocessed.csv")

# 한글 폰트 설정
font_path = "C:/Windows/Fonts/malgun.ttf"
font_prop = fm.FontProperties(fname=font_path)

def draw_group_wordcloud(df, group_col='group', text_col='clean_description'):
    groups = df[group_col].unique()
    n_groups = len(groups)

    cols = 3
    rows = math.ceil(n_groups / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.5, rows * 3.5), facecolor='white')
    axes = axes.flatten()

    for idx, group in enumerate(groups):
        text_data = " ".join(df[df[group_col] == group][text_col].dropna())

        wordcloud = WordCloud(
            font_path=font_path,
            background_color='white',
            max_words=100,
            width=350,
            height=250,
            relative_scaling=0.4,
            colormap='viridis',
            regexp=r'\b\w+\b'
        ).generate(text_data)

        axes[idx].imshow(wordcloud, interpolation='bilinear')
        axes[idx].axis('off')
        axes[idx].set_title(
            f"{group} 그룹",
            fontproperties=font_prop,
            fontsize=13,
            y=1.03  # 제목 위치 위로
        )

    # 남는 subplot 제거
    for i in range(n_groups, len(axes)):
        axes[i].axis('off')

    # subplot 간 여백 조절 (tight_layout 대체)
    fig.subplots_adjust(
        top=0.90,   # 전체 상단 여백
        hspace=0.35,  # 수직 간격
        wspace=0.25   # 수평 간격
    )

    plt.show()

# 실행
draw_group_wordcloud(df)
---

# 3. [그룹간 상관관계 분석]

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from konlpy.tag import Okt

# 한글 폰트 설정 (윈도우)
import matplotlib.font_manager as fm
font_path = "C:/Windows/Fonts/malgun.ttf"
plt.rc('font', family='Malgun Gothic')

# 데이터 불러오기
df = pd.read_csv("C:/Users/user/Desktop/men_fashion_2026naver_blog/3_전처리/men_fashion_2025_naver_blog_preprocessed.csv")

# 1. title + description 결합
df["text"] = df["clean_title"].fillna('') + " " + df["clean_description"].fillna('')

# 2. 형태소 분석기 초기화
okt = Okt()

# 3. 명사만 추출하는 함수 정의
def extract_nouns(text):
    return " ".join(okt.nouns(text))

# 4. 명사만 추출한 컬럼 생성
df["nouns"] = df["text"].apply(extract_nouns)

# 5. 그룹별 텍스트 묶기
group_texts = df.groupby("group")["nouns"].apply(lambda x: " ".join(x)).reset_index()

# 6. CountVectorizer로 키워드 등장 횟수 추출
vectorizer = CountVectorizer(max_features=100)  # 상위 100개 키워드 사용 (필요 시 조정)
X = vectorizer.fit_transform(group_texts["nouns"])

# 7. 키워드 출현 행렬 생성
keywords_df = pd.DataFrame(X.toarray(), index=group_texts["group"], columns=vectorizer.get_feature_names_out())

# 8. 상관관계 분석
correlation_matrix = keywords_df.T.corr()

# 9. 시각화
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', square=True, linewidths=0.5, fmt=".2f")
plt.title("그룹 간 키워드 기반 상관관계")
plt.tight_layout()
plt.show()
---

# 4. [그룹별 키워드 분포 비교]

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import matplotlib.font_manager as fm

# 한글 폰트 설정
font_path = "C:/Windows/Fonts/malgun.ttf"
font_prop = fm.FontProperties(fname=font_path).get_name()
plt.rc('font', family=font_prop)

# CSV 불러오기
df = pd.read_csv("C:/Users/user/Desktop/men_fashion_2026naver_blog/3_전처리/men_fashion_2025_naver_blog_preprocessed.csv")

# clean_title + clean_description 합치기
df['text'] = df['clean_title'].fillna('') + ' ' + df['clean_description'].fillna('')

# 분석할 그룹 리스트
groups = df['group'].unique()

# 결과 저장용
top_keywords_per_group = {}

# 토큰화 및 키워드 추출
for group in groups:
    text = " ".join(df[df['group'] == group]['text'])
    tokens = text.split()
    counter = Counter(tokens)
    top_keywords = counter.most_common(10)
    top_keywords_per_group[group] = dict(top_keywords)

# 데이터프레임 변환
keyword_df = pd.DataFrame(top_keywords_per_group).fillna(0).astype(int)

# 전치해서 그룹 기준으로 보기 좋게 정리
keyword_df = keyword_df.T

# 시각화 (막대그래프)
fig, ax = plt.subplots(figsize=(12, 6))  # 크기 조금 줄임
keyword_df.plot(kind='bar', ax=ax)

plt.title("그룹별 Top 10 키워드 분포", fontsize=15)
plt.xlabel("그룹")
plt.ylabel("빈도수")
plt.xticks(rotation=45)

# 범례 위치 조정 + 글자 크기 줄이기
plt.legend(title="키워드", bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8, title_fontsize=10)

plt.tight_layout()
plt.show()
