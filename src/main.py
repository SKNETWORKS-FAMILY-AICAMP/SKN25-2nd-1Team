import os
from pathlib import Path
import pickle
from data_loader import load_data
from preprocessing import preprocess_for_modeling
from model_train import train_model
from model_eval import evaluate_model, plot_shap_values

def main():
    print("="*50)
    print("KKBox 이탈 예측 파이프라인")
    print("="*50)

    # 🔹 1. 프로젝트 루트 (현재 파일 기준 부모 폴더로 한 번 이동)
    ROOT_DIR = Path(__file__).resolve().parent.parent
    
    # 🔹 2. data 디렉토리 설정
    DATA_DIR = ROOT_DIR / "data"

    # 🔹 3. parquet 파일 경로
    data_path = DATA_DIR / "kkbox_v3.parquet"

    if not data_path.exists():
        print("⚠️ Parquet 파일을 찾을 수 없습니다. pkl로 대체 시도합니다.")
        data_path = DATA_DIR / "kkbox_v3.pkl"


    # 2. 데이터 로드
    try:
        df = load_data(data_path)
    except Exception as e:
        print(f"데이터 로드 중 오류 발생: {e}")
        return

    # 3. 전처리
    print("\n[Step 1] 데이터 전처리 중...")
    X, y = preprocess_for_modeling(df)
    
    # 4. 모델 학습 (Optuna 튜닝 적용)
    print("\n[Step 2] XGBoost 모델 학습 및 하이퍼파라미터 튜닝 중...")
    model, X_va, va_proba, y_va = train_model(X, y, use_tuning=True)
    
    # 5. 결과 저장
    results_dir = "results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
        
    print(f"\n[Step 3] 모델을 {results_dir}에 저장 중...")
    with open(os.path.join(results_dir, "xgboost_model.pkl"), "wb") as f:
        pickle.dump(model, f)
    
    # 피처 이름 저장
    feature_names = model.get_booster().feature_names
    with open(os.path.join(results_dir, "feature_names.pkl"), "wb") as f:
        pickle.dump(feature_names, f)
        
    # 6. 평가 및 SHAP 분석
    print("\n[Step 4] 모델 평가 및 시각화 생성 중...")
    evaluate_model(model, X_va, y_va, results_dir=results_dir, va_proba=va_proba)
    plot_shap_values(model, X_va, X_va, results_dir=results_dir)
    
    print("\n" + "="*50)
    print("파이프라인 실행 완료.")
    print("="*50)

if __name__ == "__main__":
    main()
