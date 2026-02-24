import pandas as pd
from pathlib import Path
import numpy as np

TARGET = 'is_churn'

def prepare_eda_data():
    ROOT_DIR = Path(__file__).resolve().parents[1]
    DATA_PATH = ROOT_DIR / "data" / "kkbox_v3.parquet"
    SAVE_PATH = ROOT_DIR / "data" / "preprocessed"

    # 폴더가 없으면 생성
    SAVE_PATH.mkdir(parents=True, exist_ok=True)
    
    print("데이터 로딩 중...")
    df = pd.read_parquet(DATA_PATH)
    

    # 1. 기초 통계 값 저장
    summary_stats = {
        'total_users': len(df),
        'churn_rate': df[TARGET].mean() * 100,
        'avg_secs': df['total_secs_mean'].mean()
    }
    pd.to_pickle(summary_stats, SAVE_PATH / "eda_summary.pkl")
 
    # -------------------------------------------------------------------------------------------
    # 2. Tab 3용 : 시각화 데이터 생성
    # -------------------------------------------------------------------------------------------
    print("Tab 3 시각화용 집계 데이터 생성 중...")
    
    # [A] 카테고리형 변수 사전 집계
    cat_candidates = ['gender', 'age_group', 'registered_via']
    cat_summary = {}
    
    for col in cat_candidates:
        if col in df.columns:
            stats = df.groupby(col)[TARGET].agg(['mean', 'count']).reset_index()
            stats.columns = [col, 'churn_rate', 'n']
            cat_summary[col] = stats
            
    pd.to_pickle(cat_summary, SAVE_PATH / "eda_cat_summary.pkl")

    # [B] 수치형 변수 사전 집계 
    num_candidates = ['total_paid', 'total_secs_sum']
    num_df_light = df[num_candidates + [TARGET]].copy()
    
    # 메모리 절약을 위해 float64 -> float32 등 형변환 고려 가능
    pd.to_pickle(num_df_light, SAVE_PATH / "eda_num_light.pkl")

    print(f"EDA 전용 요약 데이터 완료 (위치: {SAVE_PATH})")

if __name__ == "__main__":
    prepare_eda_data()