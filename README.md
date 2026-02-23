

# 🎧 SKN25-2nd-1Team

### KKBOX Churn Prediction & Targeting Dashboard

![Python](https://img.shields.io/badge/Python-3.10-blue)
![LightGBM](https://img.shields.io/badge/Model-LightGBM-green)
![XGBoost](https://img.shields.io/badge/Model-XGBoost-orange)
![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-red)
![SHAP](https://img.shields.io/badge/Explainability-SHAP-purple)

---

# 📌 1. 팀 소개


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


---

## ❤️ 기대 효과



---

## 👤 대상 사용자

* 마케팅 전략 담당자
* CRM 팀
* 데이터 분석가

---

# 🛠 4. 기술 스택

### 📊 Data

* Pandas
* NumPy
* polar

### 🤖 Modeling

* LightGBM
* XGBoost
* RandomForest
* MLP
* SHAP

### 📈 Dashboard

* Streamlit

### 🧠 ML Pipeline

* sklearn
* Imbalanced-learn (SMOTE)

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
│   ├── gbm_model.pkl
│   ├── lgbm_model.pkl
│   ├── mlp_model.pkl
│   ├── rf_model.pkl
│   └── xgb_model.pkl
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
python app.py
```

---

# 📊 7. 수행 결과


---

# 📝 8. 한 줄 회고

