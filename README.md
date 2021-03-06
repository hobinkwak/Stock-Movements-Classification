# Stock Movements Classification

### 기본 컨셉
- WICS 기준 10개의 업종 중 2개씩 뽑아 총 45개의 페어에 대해 월별 수익률 우위 업종 예측을 위한 지도학습
- 학습한 모델을 기반으로 Target date에서 월별로 우위인 업종을 뽑아 1점씩 부여 (스포츠 경기 리그 형식)
- 업종 스코어와 ETF의 업종별 구성 비중을 활용하여 월별 ETF 스코어 부여

### 유니버스
- 유니버스
  - 제공된 데이터 pdf data monthly 상의 국내 주식형 ETF 59종과 이들의 움직임을 추종하는 레버리지/인버스 ETF 7종
- 페어
  - WICS 대분류10개 업종
에너지 , 소재, 산업재, 경기관련소비재, 필수소비재, 건강관리, 금융, IT, 커뮤니케이션서비스, 유틸리티

### 모델 구성

- Adjusted **Soft Voting Ensemble**
  - Light GBM
  - XGBoost
  - ExtraRandomForest
  - Logistic Regression
  - SVM

- Post-Hoc Feature Selection
  - Permutation Importance

#### 모델 초모수 튜닝

- Grid, Bayesian, Genetic Algorithm
