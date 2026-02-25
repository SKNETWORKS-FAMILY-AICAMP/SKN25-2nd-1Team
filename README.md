# 🎧 SKN25-2nd-1Team

### KKBOX Churn Prediction & Targeting Dashboard

![Python](https://img.shields.io/badge/Python-3.10-blue)
![XGBoost](https://img.shields.io/badge/Model-XGBoost-orange)
![ResNet](https://img.shields.io/badge/Model-ResNet-blue)
![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-red)
![SHAP](https://img.shields.io/badge/Explainability-SHAP-purple)

---

# 📌 1. 팀 소개

본 프로젝트는 5명의 팀원이 참여하여 개발되었습니다.

| 이름 | 역할 | 담당 업무 |
|------|------|----------|
| 김나연 | 팀장 | 프로젝트 총괄 및 조율 |
| 박범수 | 개발자 | Streamlit 대시보드 개발 |
| 양예승 | 발표자 | 프로젝트 발표 및 홍보 |
| 이근혁 | 개발자 | Streamlit 대시보드 개발 |
| 최현우 | 모델 개발자 | XGBoost, ResNet 모델 개발 |

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
* 데이터 분석가, 비즈니스 의사결정자

---

# 🛠 4. 기술 스택

### 📊 Data
![Pandas](https://img.shields.io/badge/Pandas-2.3.3-blue)
![NumPy](https://img.shields.io/badge/NumPy-2.4.2-lightblue)
![PyArrow](https://img.shields.io/badge/PyArrow-23.0.1-green)

### 🤖 Modeling
![XGBoost](https://img.shields.io/badge/XGBoost-3.2.0-orange)
![PyTorch](https://img.shields.io/badge/PyTorch-2.10.0-red)
![SHAP](https://img.shields.io/badge/SHAP-0.50.0-purple)

### 📈 Dashboard
![Streamlit](https://img.shields.io/badge/Streamlit-1.54.0-red)

### 🧠 ML Pipeline
![Scikit--learn](https://img.shields.io/badge/Scikit--learn-1.8.0-blue)
![Imbalanced--learn](https://img.shields.io/badge/Imbalanced--learn-0.12.0-yellow)
![Optuna](https://img.shields.io/badge/Optuna-4.7.0-green)

---

# 📂 5. Repository Structure

```
SKN25-2nd-1Team/
│
├── app/                         # 📊 Streamlit 대시보드 화면 구성
│   ├── app_eda.py               # EDA 및 데이터 시각화 페이지
│   ├── app_home.py              # 대시보드 홈 화면
│   ├── app_predict.py           # 머신러닝/딥러닝 예측 결과 페이지
│   ├── app_strategy.py          # 마케팅 타겟팅 및 전략 추천 페이지
│   └── main.py                  # Streamlit 앱 내비게이션 컨트롤러
│
├── data/                        # 🗂️ 데이터 파일 보관소 (GitHub 미포함)
│   ├── preprocessed/            # 🚨 [필수 다운로드!!] Streamlit 속도 최적화를 위한 사전 집계 데이터(.pkl) 반드시 데이터 다운받고 이곳에 배치!
│   └── kkbox_v3.parquet         # 전처리 및 병합 완료된 주요 데이터
│
├── results/                     # 💾 학습된 AI 모델 저장소 (GitHub 미포함)
│   ├── resnet_model.pth         # ResNet 딥러닝 모델 가중치
│   ├── resnet_scaler.pkl        # 🚨 [필수 다운로드!!] 딥러닝용 정규화 스케일러 반드시 데이터 다운받고 이곳에 배치!
│   └── xgboost_model.pkl        # 🚨 [필수 다운로드!!] XGBoost 머신러닝 모델 반드시 데이터 다운받고 이곳에 배치!
│
├── scripts/                     # ⚙️ 보조 및 전처리 스크립트 모음
│   ├── save_eda_data.py         # Streamlit용 사전 요약 데이터 생성 스크립트
│   ├── eda_interactive.py       # EDA 차트 생성 관련 함수 스크립트
│   └── run_shap.py              # SHAP 분석 결과 저장 시뮬레이션 코드
│
├── src/                         # ⚙️ 공통 로직 모듈 (백엔드 성격)
│   ├── data_loader.py           # 데이터 로드 처리
│   ├── model_loader.py          # 저장된 모델 불러오기
│   ├── predict.py               # 예측 수행 로직
│   └── ...                      # 평가(metrics) 등 각종 모듈
│
├── app.py                       # 🚀 Streamlit 대시보드 최상위 실행 파일
├── EDA.ipynb                    # 📓 초기 탐색적 데이터 분석(EDA) 결과
├── requirements.txt             # 📦 프로젝트 필수 라이브러리 목록
└── README.md                    # 📖 본 프로젝트 안내 문서
```

---

# 🚀 6. 실행 방법

## 0️⃣ 환경 설정

### Python 버전
- Python 3.12 이상 권장

### 가상환경 생성
```bash
conda create -n keeptune python=3.12
conda activate keeptune
```

### 의존성 설치
```bash
pip install -r requirements.txt
```

### 데이터 & 모델 파일 준비 (필수)
프로젝트 실행에 필요한 주요 데이터(`.parquet`)와 딥러닝 모델 가중치(`.pth`)는 이미 GitHub에 포함되어 있습니다. 단, **용량이 큰 `.pkl` (전처리 요약 및 머신러닝 모델) 파일과 `.csv` (원본 데이터) 파일은 GitHub에 포함되어 있지 않습니다.**
반드시 아래 Google Drive 링크에서 **해당 `.pkl` 파일들을 다운로드 받아 정확한 폴더 위치에 배치해야 Streamlit 대시보드가 정상 작동합니다!** (참고로 `.csv` 파일은 모델 재학습 등 파이프라인 전체를 다시 실행할 때만 필요하며, 단순히 대시보드를 실행하는 데는 필수가 아닙니다.)

* 🔗 **[원본데이터](https://drive.google.com/drive/folders/1jQzwvlpdP1cnwhTBjDg9Pp1raBIhHjWx)** (선택)
* 🔗 **[preprocessed](https://drive.google.com/drive/folders/1h2GBcQeztjQ8DdyqpW2mRgxEtDanf54c)** (필수)
* 🔗 **[모델](https://drive.google.com/drive/folders/1kune4FUe0NHCm51zQL5jBQOdLup2VO0T)** (필수)

```text
다운로드 받은 후 반드시 5. Repository Structure 구조와 똑같이 파일을 배치해 주세요.

```

```
data/kkbox_v3.parquet     ← 전처리 완료 데이터 (필수)
data/raw/*.csv            ← 원본 CSV 파일들
```

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

## 👤 김나연
> 

## 👤 박범수
> 제가 몰랐던 부분들을 팀원분들이 알려주어서 새로운 것을 많이 배우게 되는 프로젝트 기간이었습니다. 또 맡은 역할들을 모두 열심히 해주셔서 좋은 경험이었습니다.

## 👤 양예승
> 수업에서 배울땐 웬만하면 다 알고 있다고 생각했지만 막상 프로젝트를 시작해보니, 부족한 부분을 많이 느꼈습니다. 하지만, 팀원들이 협조해줘서 잘 마무리 할 수 있었습니다. 팀원들에게 정말 감사하다고 말하고 싶습니다.  

## 👤 이근혁
> 쉽지 않은 주제였지만, 각자 주어진 역할 이상을 하는 팀원들 덕분에 비전공자로서 어렵지 않게 프로젝트를 진행할 수 있었습니다. 짧은 시간이었지만 많을 것을 배울 수 있었습니다.

## 👤 최현우
> 좋은 조원들과 좋은 프로젝트 했습니다 ^^

---

*SKN 25기 2차 프로젝트 - 1팀*
