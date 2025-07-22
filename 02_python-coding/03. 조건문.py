## 동전 던지기 게임

# random 모듈을 불러와서 무작위 수를 생성
import random

print("동전 던지기 게임을 지금 시작합니다.")

# 0 또는 1 중 무작위 값을 생성하여 coin 변수에 저장
coin = random.randrange(2)

# 동전의 결과에 따라 앞면 또는 뒷면 출력
if coin == 0 :
    print("앞면입니다.")
else :
    print("뒷면입니다.")
print("게임이 종료되었습니다.")
