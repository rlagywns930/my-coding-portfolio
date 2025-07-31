#define TRIG 9
#define ECHO 10

unsigned long tStart;
unsigned long t_1_10 = 0, t_1_8 = 0, t_1_6 = 0, t_1_4 = 0, t_1_2 = 0, t_full = 0;
bool flag_1_10 = false, flag_1_8 = false, flag_1_6 = false, flag_1_4 = false, flag_1_2 = false, flag_full = false;

const int stableCountThreshold = 3;
int stableCount_1_10 = 0, stableCount_1_8 = 0, stableCount_1_6 = 0, stableCount_1_4 = 0, stableCount_1_2 = 0, stableCount_full = 0;

void setup() {
  Serial.begin(9600);
  pinMode(TRIG, OUTPUT);
  pinMode(ECHO, INPUT);
  tStart = millis();
}

void loop() {
  long duration;
  float distance;

  // 초음파 측정
  digitalWrite(TRIG, LOW);
  delayMicroseconds(2);
  digitalWrite(TRIG, HIGH);
  delayMicroseconds(10);
  digitalWrite(TRIG, LOW);

  duration = pulseIn(ECHO, HIGH, 30000);

  if (duration == 0 || duration < 100) {
    delay(500);
    return;
  }

  distance = duration * 0.034 / 2;

  if (distance < 0.5 || distance > 25.0) {
    delay(500);
    return;
  }

  unsigned long currentTime = millis();

  if (!flag_1_10 && distance <= 22.5) {
    stableCount_1_10++;
    if (stableCount_1_10 >= stableCountThreshold) {
      t_1_10 = currentTime - tStart;
      flag_1_10 = true;
      Serial.print("✅ 1/10 지점 시간(ms): ");
      Serial.println(t_1_10);
    }
  }

  if (!flag_1_8 && distance <= 21.5) {
    stableCount_1_8++;
    if (stableCount_1_8 >= stableCountThreshold) {
      t_1_8 = currentTime - tStart;
      flag_1_8 = true;
      Serial.print("✅ 1/8 지점 시간(ms): ");
      Serial.println(t_1_8);
    }
  }

  if (!flag_1_6 && distance <= 20.5) {
    stableCount_1_6++;
    if (stableCount_1_6 >= stableCountThreshold) {
      t_1_6 = currentTime - tStart;
      flag_1_6 = true;
      Serial.print("✅ 1/6 지점 시간(ms): ");
      Serial.println(t_1_6);
    }
  }

  if (!flag_1_4 && distance <= 18.5) {
    stableCount_1_4++;
    if (stableCount_1_4 >= stableCountThreshold) {
      t_1_4 = currentTime - tStart;
      flag_1_4 = true;
      Serial.print("✅ 1/4 지점 시간(ms): ");
      Serial.println(t_1_4);
    }
  }

  if (!flag_1_2 && distance <= 12.0) {
    stableCount_1_2++;
    if (stableCount_1_2 >= stableCountThreshold) {
      t_1_2 = currentTime - tStart;
      flag_1_2 = true;
      Serial.print("✅ 1/2 지점 시간(ms): ");
      Serial.println(t_1_2);
    }
  }

  // ✅ 전체 물이 찼을 때 거리 조건 (예: 1cm 이하)
  if (!flag_full && distance <= 2.0) {
    stableCount_full++;
    if (stableCount_full >= stableCountThreshold) {
      t_full = currentTime - tStart;
      flag_full = true;
      Serial.print("💧 물 가득 찼을 때 시간(ms): ");
      Serial.println(t_full);
    }
  }

  delay(500);
}
