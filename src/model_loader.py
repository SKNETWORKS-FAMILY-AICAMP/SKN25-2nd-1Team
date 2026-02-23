import joblib
import pandas as pd
import numpy as np
import streamlit as st
import os
## app_predict.py에서 사용
# 0. root 경로 선언
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]

MODELS_DIR = ROOT_DIR / "models" # 모델 저장 경로
SAVE_DIR =ROOT_DIR / "data" / "preprocessed" # 전처리 된 데이터 저장 경로

@st.cache_resource
def get_resources():
    print("🚀 [System] Loading models into memory...")

    try:
        lgbm = joblib.load(MODELS_DIR / "lgbm_model.pkl")
        rf = joblib.load(MODELS_DIR / "rf_model.pkl")
        mlp = joblib.load(MODELS_DIR / "mlp_model.pkl")
        scaler = joblib.load(SAVE_DIR / "scaler.pkl")
        feature_names = joblib.load(SAVE_DIR / "feature_names.pkl")

        print("✅ [System] All models loaded successfully.")

    except Exception as e:
        print(f"❌ [Error] Failed to load models: {e}")
        return None

    return lgbm, rf, mlp, scaler, feature_names

def predict_churn(data_dict):
    resources = get_resources()
    if not resources: return 0, 0, 0, 0
    
    lgbm, rf, mlp, scaler, feature_names = resources
    
    # 1 & 2 & 3. 데이터프레임 생성 및 정렬 최적화
    # np.zeros로 만들고 루프를 돌리는 것보다 dict를 바로 넣는 것이 더 빠릅니다.
    df = pd.DataFrame([data_dict])
    
    # 누락된 컬럼은 0으로 채우고, 순서 강제 정렬
    df = df.reindex(columns=feature_names, fill_value=0)
    
    # 4. 예측 수행 (MLP를 위한 스케일링은 한 번만 실행)
    # N100의 부하를 줄이기 위해 순차적으로 실행
    p1 = lgbm.predict_proba(df)[0][1]
    p2 = rf.predict_proba(df)[0][1]
    
    scaled_df = scaler.transform(df)
    p3 = mlp.predict_proba(scaled_df)[0][1]
    
    avg_p = (p1 + p2 + p3) / 3
    
    return p1, p2, p3, avg_p