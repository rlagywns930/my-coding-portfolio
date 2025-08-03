## 랜덤 산수 문제 발생기

===
import random

answer = "yes"  # 사용자가 문제를 풀지 계속 결정하는 변수

while answer == "yes":
    answer = input("문제를 푸시겠습니까? (yes/no)")  # 문제 출제 여부 입력받기
        
    if answer == "no":  # 사용자가 'no' 입력하면 프로그램 종료
        print("문제 발생기를 종료합니다.")
        break

    # 두 개의 랜덤 정수 생성 (1부터 100까지)
    x = random.randint(1,100)
    y = random.randint(1,100)

    # 사칙연산 중 하나를 무작위로 선택
    op = random.choice(["+", "-", "x", "/"])

    # 더하기 문제
    if op == "+":
        print(x, "+", y, "= ?")  # 문제 출력
        q = int(input("정답은?: "))  # 사용자 답 입력
        if q == x+y:  # 정답 확인
            print("정답입니다!")
        else:
            print("틀렸어요 정답은..")
            print(x+y)
            
    # 빼기 문제
    elif op == "-":
        print(x, "-", y, "= ?")
        q = int(input("정답은?: "))
        if q == x-y:
            print("정답입니다!")
        else:
            print("틀렸어요 정답은..")
            print(x-y)
            
    # 곱하기 문제
    elif op == "x":
        print(x, "x", y, "= ?")
        q = int(input("정답은?: "))
        if q == x*y:
            print("정답입니다!")
        else:
            print("틀렸어요 정답은..")
            print(x*y)

    # 나누기 문제 (나눗셈은 y 범위를 1~10으로 제한하여 소수점 답 계산)
    elif op == "/":
        x = random.randint(1,100)  # 나누기 문제용 새로운 x 재설정
        y = random.randint(1,10)   # y는 1부터 10 사이로 제한
        print(x, "/", y, "= ?")
        q = float(input("정답은?: "))  # 소수점 포함 답 입력
        # 사용자가 입력한 답과 실제 몫 차이가 0.1 미만이면 정답으로 인정
        if abs(q - (x/y)) < 0.1:
            print("정답입니다!")
        else:
            print("틀렸어요 정답은..")
            print(x/y)


[출력결과]  

![image](https://github.com/user-attachments/assets/02fc74b6-4191-4406-8c43-8891e45cc5ae)

