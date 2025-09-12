import requests
import pandas as pd
import urllib.parse
import time
import html
from datetime import datetime

CLIENT_ID = 'kNxRdK1NoBr6GRWiyhgH'
CLIENT_SECRET = '5d5Pt93ALu'

# 키워드별 그룹 매핑
keyword_to_group = {
    # 아이템 그룹
    "남자 가을 코트": "아이템",
    "남자 니트 가디건": "아이템",
    "남자 셔츠": "아이템",
    "남자 트랙셋업": "아이템",
    "남자 윈드브레이커": "아이템",
    "남자 레더 자켓": "아이템",
    "남자 데님 셋업": "아이템",
    "남자 가을 신발": "아이템",
    "남자 헨리넥": "아이템",
    "남자 블루종 자켓": "아이템",
    "남자 스웨이드 자켓": "아이템",
    "남자 후드집업": "아이템",
    "남자 카고 팬츠 가을": "아이템",

    # 악세사리 그룹
    "남자 가을 악세사리": "악세사리",
    "남자 가을 안경": "악세사리",
    "남자 가을 모자": "악세사리",
    "남자 가을 시계": "악세사리",
    "남자 가을 가방": "악세사리",
    "남자 스카프 가을 스타일": "악세사리",

    # 스타일 그룹
    "남자 스트릿 패션 가을": "스타일",
    "남자 미니멀룩 가을": "스타일",
    "남자 캐주얼 가을 코디": "스타일",
    "남자 레트로 패션": "스타일",
    "남자 비즈니스 캐주얼": "스타일",
    "남자 가을 데일리룩": "스타일",
    "남자 룩북 가을 스타일": "스타일",
    "꾸안꾸 남자 가을 코디": "스타일",
    "느좋남 남자 가을 스타일": "스타일",
    "남자 가을 뉴트로 패션": "스타일",  
    "남자 가을 젠더리스 패션": "스타일",  

    # 트렌드 컬러 그룹
    "가을 트렌드 컬러 모카 브라운": "트렌드 컬러",
    "가을 트렌드 컬러 버건디": "트렌드 컬러",
    "가을 트렌드 컬러 그레이": "트렌드 컬러",
    "가을 트렌드 컬러 베이지": "트렌드 컬러",
    "가을 트렌드 컬러 올리브": "트렌드 컬러",
    "남자 브라운 코디 가을": "트렌드 컬러",
    "남자 버건디 니트 코디": "트렌드 컬러",
    "가을 남자 컬러 매치": "트렌드 컬러",
    "라이트 블루 컬러 가을": "트렌드 컬러", 
    "다크 그린 컬러 가을": "트렌드 컬러",  

    # 라이프스타일 그룹
    "남자 출근룩 가을": "라이프스타일",
    "남자 데이트 코디 가을": "라이프스타일",
    "남자 캠퍼스룩 가을": "라이프스타일",
    "남자 하객룩 코디 가을": "라이프스타일",
    "남자 아웃도어룩 가을": "라이프스타일" 
}

keywords = list(keyword_to_group.keys())

all_blog_data = []

pages_per_keyword = 21  # 약 210개 수집 목표

for keyword in keywords:
    print(f"🔍 [키워드 시작] {keyword}")
    query = f"{keyword} 2025 남자 가을 패션"
    encoded = urllib.parse.quote(query)

    for start in range(1, pages_per_keyword * 10, 10):
        print(f"  - 페이지 시작 번호: {start}")
        url = f"https://openapi.naver.com/v1/search/blog.json?query={encoded}&display=10&start={start}&sort=date"
        headers = {
            "X-Naver-Client-Id": CLIENT_ID,
            "X-Naver-Client-Secret": CLIENT_SECRET
        }

        res = requests.get(url, headers=headers)
        if res.status_code == 200:
            items = res.json().get("items", [])
            for item in items:
                postdate_str = f"{item['postdate'][:4]}-{item['postdate'][4:6]}-{item['postdate'][6:]}"
                date_obj = datetime.strptime(postdate_str, "%Y-%m-%d")
                all_blog_data.append({
                    "group": keyword_to_group[keyword],
                    "keyword": keyword,
                    "title": html.unescape(item["title"]),
                    "description": html.unescape(item["description"]),
                    "postdate": postdate_str,
                    "year": date_obj.year,
                    "month": date_obj.month
                })
        else:
            print(f"API 호출 실패: {res.status_code} / 키워드: {keyword}")

        time.sleep(1)  # API 제한 회피 딜레이

df = pd.DataFrame(all_blog_data)
df.to_csv("naver_blog_men_fashion_2025_grouped.csv", index=False, encoding='utf-8-sig')

print("블로그 데이터 수집 및 그룹화 완료!")

