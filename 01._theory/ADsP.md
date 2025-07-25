# 📊 ADsP 요약 정리

## 📌 목차
1. [데이터의 이해](#1-데이터의-이해)
2. [데이터 분석 기획](#2-데이터-분석-기획)
3. [데이터 분석 기법](#3-데이터-분석-기법)
4. [데이터의 활용](#4-데이터의-활용)

---

## 1. 데이터의 이해

# ADsP 요약 정리

데이터 분석 준전문가(ADsP) 시험 대비를 위해 학습한 내용을 정리한 내용입니다. 핵심 개념 위주로 구성되어 있으며, 데이터의 이해부터 분석 기술까지 다룹니다.

---

## 1. 데이터의 이해

### ✅ 1-1. 정형 / 비정형 데이터란?

- **정형 데이터**:  
  구조화되어 있으며 연산이 가능함. 주로 관계형 데이터베이스(RDBMS)에 저장됨.

- **비정형 데이터**:  
  구조가 없거나 일정하지 않아 연산이 어려움. 주로 NoSQL에 저장됨.

---

### ✅ 1-2. 암묵지와 형식지의 상호작용 관계 (SECI 모델)

1. **공통화(Socialization)**: 경험 공유 → 암묵지 전달  
2. **표출화(Externalization)**: 암묵지 → 형식지  
3. **연결화(Combination)**: 형식지 → 형식지 결합  
4. **내면화(Internalization)**: 형식지 → 암묵지로 내면화

---

### ✅ 1-3. DIKW 피라미드의 정의

- **Data(데이터)**: 가공되지 않은 원시 사실  
- **Information(정보)**: 데이터를 조직화해 의미를 부여한 것  
- **Knowledge(지식)**: 정보를 기반으로 한 원리, 경험  
- **Wisdom(지혜)**: 지식을 기반으로 한 최적의 판단 능력

---

## 2. 데이터베이스 정의와 특징

### ✅ 2-1. 데이터베이스란?

- 통합된 데이터로 설계됨  
- 데이터의 중복 저장 최소화  
- 일관성과 신뢰성 유지 목적

**주요 구성요소**:  
외부 스키마, 개념 스키마, 내부 스키마, 메타데이터, 테이블, 인덱스, 뷰, 데이터 사전 등

---

## 3. 데이터베이스의 활용

### ✅ 3-1. DB 설계 절차

1. 요구사항 분석  
2. 개념적 설계  
3. 논리적 설계  
4. 물리적 설계  
5. 구현

---

### ✅ 3-2. DBMS의 종류

- 관계형 DBMS  
- 객체지향 DBMS  
- 네트워크 DBMS  
- 계층형 DBMS

---

### ✅ 3-3. 데이터 웨어하우스

- 기업 전사 차원의 데이터를 통합  
- 분석과 의사결정에 최적화된 저장소

---

### ✅ 3-4. SQL이란?

- 데이터베이스와 상호작용하기 위한 하부 언어

**SQL의 분류**

| 분류 | 설명 | 예시 |
|------|------|------|
| DDL | 데이터 정의어 | CREATE, ALTER, DROP, TRUNCATE |
| DML | 데이터 조작어 | SELECT, INSERT, UPDATE, DELETE |
| DCL | 데이터 제어어 | GRANT, REVOKE, COMMIT, ROLLBACK |

**SQL의 집계 함수**

- `AVG`, `SUM`, `STDDEV`: 수치형 데이터에 사용  
- `COUNT`: 모든 데이터 타입에서 사용 가능

---

### ✅ 3-5. NoSQL이란?

- 기존 RDBMS의 한계를 보완하는 비관계형 데이터베이스  
- 예시: MongoDB, HBase, Redis, Cassandra

---

### ✅ 3-6. 기업 내부 데이터베이스 시스템

- **OLTP (Online Transaction Processing)**:  
  실시간 트랜잭션 처리 중심 시스템

- **OLAP (Online Analytical Processing)**:  
  정보 분석 중심의 처리 시스템

**기타 시스템**

- CRM (Customer Relationship Management): 고객 관계 관리  
- SCM (Supply Chain Management): 공급망 관리

---

## 4. 빅데이터의 이해

### ✅ 4-1. 빅데이터의 3V

1. **Volume (양)**  
2. **Variety (다양성)**  
3. **Velocity (속도)**

---

### ✅ 4-2. 빅데이터 출현 배경

- **산업계**: 고객 데이터 축적  
- **학계**: 거대 데이터 기반 분석 확대  
- **기술 발전**: 저장, 분석 기술 발달

---

### ✅ 4-3. 빅데이터의 기능 (비유적 표현)

- 21세기의 **원유**  
- 산업혁명의 **석탄과 철**  
- 데이터를 보는 **렌즈**  
- 서비스 연결의 **플랫폼**

---

### ✅ 4-4. 데이터 분석의 패러다임 변화

| 기존 방식 | 빅데이터 방식 |
|-----------|----------------|
| 사전 처리 | 사후 처리 |
| 표본 조사 | 전수 조사 |
| 질 중심 분석 | 양 중심 분석 |
| 인과관계 분석 | 상관관계 분석 |

---

## 5. 비즈니스 모델 및 분석 기술

### ✅ 5-1. 빅데이터 분석 기술

- **Hadoop**: 분산 파일 시스템 기반 대용량 데이터 처리 기술  
- **Apache Spark**: 실시간 분산형 컴퓨팅 플랫폼 (Scala, Java, R, Python 지원)  
- **Smart Factory**: IoT 기반 실시간 공장 운영 및 의사결정  
- **Machine Learning & Deep Learning**: 데이터 기반 학습 및 추론 기술

---

## 6. 전략 인사이트 도출을 위한 필요 역량

### ✅ 6-1. 데이터 사이언스란?

데이터 공학, 통계학, 컴퓨터공학, 시각화, 도메인 지식, 해커 사고방식을 융합한 통합 학문

---

### ✅ 6-2. 데이터 사이언스의 구성요소

1. **IT 역량**: 데이터 처리 및 엔지니어링  
2. **분석 역량**: 수학, 통계, 모델링  
3. **비즈니스 역량**: 문제 해결, 컨설팅 능력

---

### ✅ 6-3. 인문학과의 융합

- **컨버전스 → 디버전스**: 단순 세계화에서 복잡 세계화로  
- **생산 → 서비스**: 제품 생산에서 서비스 중심으로  
- **공급자 중심 → 시장 창조**: 기술 경쟁에서 무형자산 경쟁으로

---

## 2. 데이터 분석 기획

---

## 1. 분석 기획의 주요 개념

### ✅ 분석 기획의 주요 과제
- 분석 목표 정의 및 문제 인식  
- 분석 대상 및 분석 방법 설계  
- 데이터 확보 및 전처리 계획  
- 인력, 일정, 예산 등 프로젝트 운영계획  
- 결과 활용 전략 및 정책 반영  

---

### ✅ 분석 유형 (4가지 핵심 관점)

| 유형 | 설명 | 예시 |
|------|------|------|
| **최적화(Optimization)** | 자원, 비용, 성능 등을 최적으로 조정 | 가격 최적화, 캠페인 예산 배분 |
| **통찰(Insight)** | 숨겨진 패턴과 의미 있는 정보 발견 | 고객 세분화, 트렌드 분석 |
| **솔루션(Solution)** | 문제 해결을 위한 실행 가능한 방안 제시 | 전략 수립, 업무 프로세스 개선 |
| **발견(Discovery)** | 기존에 알지 못했던 사실의 발견 | 이상탐지, 잠재 수요 탐색 |

---

### ✅ 목표 시점별 분석 기획 방안

| 구분 | 내용 |
|------|------|
| 과제 단위 | 단기 분석 (실행 중심, 실무 과제) |
| 마스터 플랜 단위 | 중·장기 전략 중심 분석 계획 수립 (조직 전체 관점) |

---

### ✅ 분석 기획 시 고려사항

- **가용 데이터**  
- **적절한 유즈케이스**  
- **장애 요소에 대한 사전 계획 수립**  
  - 고정관념  
  - 편향된 사고  
  - 프레이밍 효과  

---

## 2. 분석 방법론 개요

### ✅ 경험과 감에 따른 의사결정 → 데이터 기반 의사결정
- 주관적/직관적 판단 탈피  
- 정량적 분석과 예측 기반 전략 도출

---

### ✅ 분석 방법론의 적용: 업무의 특성에 따른 모델 선택

| 모델 | 설명 |
|------|------|
| **폭포수 모델** | 계획-분석-설계-구현-시험-유지보수 순으로 단계적 수행 |
| **프로토타입 모델** | 반복적인 프로토타입 제작과 피드백을 통해 완성도 향상 |
| **나선형 모델** | 점진적 계획 수립과 리스크 중심 반복 개발 진행 |

---

### ✅ KDD 분석 절차
1. 데이터 선택  
2. 전처리  
3. 변환  
4. 데이터 마이닝  
5. 해석 및 평가  

---

### ✅ CRISP-DM 프로세스
1. **Business Understanding**  
2. **Data Understanding**  
3. **Data Preparation**  
4. **Modeling**  
5. **Evaluation**  
6. **Deployment**

---

### ✅ KDD vs CRISP-DM

| 항목 | KDD | CRISP-DM |
|------|-----|-----------|
| 중심 목적 | 지식 발견 | 실무 적용 중심 |
| 적용 분야 | 데이터 마이닝, 이론 연구 | 산업 전반, 실무 프로젝트 |
| 단계 수 | 5단계 | 6단계 |
| 특징 | 모델 생성 중심 | 비즈니스 문제 해결 강조 |

---

## 3. 빅데이터 분석 방법론 (5단계)

| 단계 | 설명 |
|------|------|
| **1. 분석 기획** | 분석 목표, 대상, 방법 정의 및 추진 전략 수립 |
| **2. 데이터 준비** | 데이터 수집, 정제, 통합 및 저장체계 마련 |
| **3. 데이터 분석** | 모델링, 패턴 탐색, 시각화 등을 통한 인사이트 도출 |
| **4. 시스템 구현** | 분석 결과를 시스템에 적용 및 서비스화 |
| **5. 평가 및 전개** | 분석 성과 평가 및 조직 내 확산 전략 마련 |

---

## 4. 분석 과제 발굴 방법론

### ✅ 분석 유형 기반 접근
- **Optimization**
- **Insight**
- **Solution**
- **Discovery**

---

### ✅ 하향식 접근법 (Top-down)
- 상위 전략 → KPI → 문제 정의 → 분석 과제 도출

---

### ✅ 상향식 접근법 (Bottom-up)
- 데이터 → 패턴 탐색 → 분석 과제 제안

---

### ✅ 디자인 사고 및 프로토타이핑
- 사용자 중심 문제 접근 → 반복적 아이디어 검증 → 분석 과제 도출

---

### ✅ 분석 기회 발굴의 범위 확장

| 범위 | 세부 내용 |
|------|-----------|
| **경쟁자 확대** | 대체재, 경쟁자, 신규진입자 분석 |
| **거시적 관점** | 사회, 기술, 경제, 환경, 정치적 변화 고려 |
| **시장 니즈 탐색** | 고객, 채널, 영향자 중심 트렌드 반영 |
| **역량의 재해석** | 내부역량 재정의 및 파트너 네트워크 활용 |

---

## 5. 분석 과제 및 프로젝트 관리

### ✅ 분석 프로젝트 영역별 주요 관리 항목

- 범위  
- 시간  
- 원가  
- 품질  
- 통합  
- 조달  
- 자원  
- 리스크  
- 의사소통  
- 이해관계자  

---

## 6. 분석 마스터 플랜 수립

### ✅ ROI 관점에서 빅데이터 핵심 특징

- **3V (크기, 다양성, 속도)** → 투자 비용 요소  
- **4V (가치)** → 비즈니스 효과 요소  

---

### ✅ 우선순위 평가 기준 (사분면 분석)

- **X축**: 시급성  
- **Y축**: 난이도  

          ↑ 난이도
          |
     (I)  |   (II)
          |
──────────┼──────────→ 시급성
          |
    (III) |   (IV)
          |
  
- 시급성 기준 우선순위: **III → IV → II**  
- 난이도 기준 우선순위: **III → I → II**

---

## 7. 데이터 분석 수준 진단

### ✅ 분석 준비도 구성 항목
- 분석 업무  
- 분석 인력 및 조직  
- 분석 기법  
- 분석 데이터  
- 분석 문화  
- 분석 인프라  

---

### ✅ 분석 성숙도 구성 항목
- **도입 단계**  
- **활용 단계**  
- **확산 단계**  
- **최적화 단계**

- 진단 분류:
  - **비즈니스 부문**
  - **조직 및 역량 부문**
  - **IT 부문**

---

### ✅ 분석 성숙도 모델

- **CMMI 모델**을 활용한 조직 성숙도 평가 도구
- 성숙도 수준: **도입 → 활용 → 확산 → 최적화**
- 진단 영역: **비즈니스, 조직·역량, IT**

#### 성숙도 유형 설명
| 유형 | 설명 |
|------|------|
| **도입형** | 일부 부서 중심의 실험적 분석 적용 |
| **준비형** | 분석 기반 구축 준비 단계 |
| **확산형** | 전사 확산 및 문화 정착 단계 |
| **정착형** | 분석이 일상 업무에 통합되어 정착 |

---

## 8. 데이터 거버넌스

### ✅ 정의
- 전사 차원의 모든 데이터에 대해 **정책 및 지침, 표준화, 운영 절차 및 책임 체계**를 수립하고 **운영 프레임워크 및 저장소**를 구축하는 것

### ✅ 중요 관리 대상
- 마스터 데이터  
- 메타 데이터  
- 데이터 사전

### ✅ 데이터 거버넌스 효과
- 데이터의 가용성, 유용성, 통합성, 보안성, 안전성 확보

### ✅ 구성 요소
- 원칙 (Principles)  
- 조직 (Organization)  
- 프로세스 (Process)

### ✅ 데이터 거버넌스 체계
- 데이터 표준화  
- 데이터 관리체계  
- 데이터 저장소 관리  
- 표준화 활동

---

### ✅ 조직 구조 유형

| 구조 | 설명 |
|------|------|
| **집중 구조** | 중앙 조직에서 모든 데이터 관리 담당 |
| **기능 구조** | 부서별 데이터 관리 책임 구분 |
| **분산 구조** | 각 조직 또는 현업 부서에 데이터 관리 권한 분산 |

---

## 🔚 과제 관리 프로세스

### 가. 과제 발굴
- 다양한 채널을 통해 **과제 후보를 POOL**로 수집

### 나. 과제 수행
- **확정된 과제는 프로젝트 또는 포트폴리오** 단위로 관리
- 수행 결과 또한 **POOL**로 관리




---

## 3. 데이터 분석 기법

---

## 4. 데이터의 활용


