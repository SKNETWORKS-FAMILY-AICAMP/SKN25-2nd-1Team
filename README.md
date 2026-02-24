# 🎧 SKN25-2nd-1Team

### KKBOX Churn Prediction & Targeting Dashboard

![Python](https://img.shields.io/badge/Python-3.10-blue)
![XGBoost](https://img.shields.io/badge/Model-XGBoost-orange)
![ResNet](https://img.shields.io/badge/Model-ResNet-blue)
![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-red)
![SHAP](https://img.shields.io/badge/Explainability-SHAP-purple)

---

# 📌 1. 팀 소개

본 프로젝트는 5명의 팀원이 참여하여 개발되었습니다:
- 김나연
- 박범수
- 양예승
- 이근혁
- 최현우

---

# 📅 2. 프로젝트 기간

2026.02.19 ~ 2026.02.24

---

# 📖 3. 프로젝트 개요

## 📕 프로젝트명

 **KeepTune**

---

## ✅ 프로젝트 배경 및 목적

* 음악 스트리밍 서비스의 사용자 이탈은 수익 감소로 직결
* 단순 예측을 넘어 **이탈 위험 사용자에 대한 전략 수립 자동화** 필요

---

## 🖐️ 프로젝트 소개

KKBox 음악 스트리밍 서비스의 구독자 이탈(Churn)을 예측하는 머신러닝 및 딥러닝 기반 웹 대시보드입니다.  
XGBoost와 ResNet 모델을 활용하여 이탈 예측을 수행하며, Streamlit을 통해 직관적인 사용자 인터페이스를 제공합니다.  
데이터 탐색, 예측, 전략 추천까지 원스톱으로 지원합니다.

---

## ❤️ 기대 효과

* 이탈 예측 정확도 향상으로 마케팅 비용 절감
* 자동화된 전략 추천으로 의사결정 시간 단축
* 고객 유지율 증가 및 수익성 개선

---

## 👤 대상 사용자

* 마케팅 전략 담당자
* CRM 팀
* 데이터 분석가
* 비즈니스 의사결정자

---

# 🛠 4. 기술 스택

### 📊 Data

* Pandas
* NumPy
* PyArrow

### 🤖 Modeling

* XGBoost
* ResNet
* SHAP

### 📈 Dashboard

* Streamlit

### 🧠 ML Pipeline

* sklearn
* Imbalanced-learn (SMOTE)
* Optuna

---

# 📂 5. Repository Structure

```
SKN25-2nd-1Team/
│
├── app/                         # 📊 Streamlit 대시보드
│   ├── model_engine/            # 모델 추론 및 전략 엔진
│   ├── app_eda.py               # EDA 페이지
│   ├── app_home.py              # 홈 화면
│   ├── app_predict.py           # 예측 결과 페이지
│   ├── app_strategy.py          # 전략 추천 페이지
│   └── main.py                  # 🚀 대시보드 실행 파일
│
├── data/
│   ├── raw/                     # 원본 데이터
│   └── preprocessed/            # 전처리 데이터
│
├── ml_pipeline/
│   └── train.py                 # 🤖 전처리 및 모델 학습 실행
│
├── models/                      # 💾 학습된 모델 저장
│   ├── xgb_model.pkl
│   └── resnet_model.pkl
│
├── notebooks/                   # 📓 실험 노트북
│   ├── EDA.ipynb
│   └── modeling_shap.ipynb
│
├── src/                         # ⚙️ 공통 모듈
│   ├── analysis_engine/
│   └── model_loader.py
│
├── requirements.txt
└── README.md
```

---

# 🚀 6. 실행 방법

## 1️⃣ 모델 학습

```bash
python ml_pipeline/train.py
```

---

## 2️⃣ 대시보드 실행

```bash
streamlit run app.py
```

---

# 📊 7. 수행 결과

## 모델 성능 비교

| 모델 | AP Score | Precision | Recall | F1-Score |
|------|----------|-----------|--------|----------|
| XGBoost | 0.9522 | 0.8319 | 0.9452 | 0.8847 |
| ResNet | 0.9450 | 0.9514 | 0.7011 | 0.8073 |

## 주요 성과
- 2개 모델 앙상블로 예측 정확도 향상
- SHAP을 통한 모델 해석 가능성 제공
- Streamlit 대시보드로 실시간 예측 및 전략 추천

---

# 📝 8. 한 줄 회고

팀원 각자의 한 줄 회고를 여기에 작성하세요.

- 김나연: 협업의 중요성을 배웠습니다.
- 박범수: 머신러닝 모델링 실력이 향상되었습니다.
- 양예승: 대시보드 개발이 재미있었습니다.
- 이근혁: 데이터 분석 능력이 성장했습니다.
- 최현우: 프로젝트 관리 경험을 쌓았습니다.

---

*SKN 25기 2차 프로젝트 - 1팀*
