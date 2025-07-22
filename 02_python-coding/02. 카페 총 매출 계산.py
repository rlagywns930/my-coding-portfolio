## 2. 카페 매출 계산하기

# 각 음료들의 가격 설정
americano_price = 2000
cafelatte_price = 3000
cafemocha_price = 3500

# 판매된 음료들의 개수를 사용자로부터 입력받음
americanos = int(input("아메리카노 판매 개수: "))
cafelattes = int(input("카페라떼 판매 개수: "))
cafemochas = int(input("카페모카 판매 개수: "))

# 총 매출 계산
sales = americanos * americano_price
sales += cafelattes * cafelatte_price
sales += cafemochas * cafemocha_price

# 총 매출 출력
print("총 매출은", sales, "입니다.")
