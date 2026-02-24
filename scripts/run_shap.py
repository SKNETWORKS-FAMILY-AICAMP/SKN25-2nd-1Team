import pandas as pd
import numpy as np
import shap
import pickle
import sys
from pathlib import Path

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.inspection import PartialDependenceDisplay


# =========================
# ROOT 경로 (scripts/run_shap.py 기준)
# =========================
ROOT_DIR = Path(__file__).resolve().parents[1]  # .../SKN25-2nd-1Team

# ✅ src import 안정화 (가장 중요)
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

DATA_DIR = ROOT_DIR / "data"
SAVE_DIR = ROOT_DIR / "data" / "preprocessed"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# ✅ 경로 검증 (안되면 여기서 바로 터뜨림)
data_file = DATA_DIR / "kkbox_v3.parquet"
if not data_file.exists():
    raise FileNotFoundError(f"데이터 파일 없음: {data_file}")

from src.preprocessing import preprocess_for_modeling
from src.data_loader import load_data

print("데이터 로딩 중...")
df = load_data(DATA_DIR/'kkbox_v3.parquet')
X, y = preprocess_for_modeling(df)

# =========================
# 원핫 인코딩 추가 (XGBoost용)
# =========================
cat_cols = X.select_dtypes(include=["object", "category"]).columns
if len(cat_cols) > 0:
    print(f"원핫 인코딩 적용 컬럼: {list(cat_cols)}")
    X = pd.get_dummies(X, columns=cat_cols, drop_first=True)

# =========================
# Train / Valid Split
# =========================
X_tr, X_va, y_tr, y_va = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# =========================
# 불균형 가중치 계산
# =========================
neg = (y_tr == 0).sum()
pos = (y_tr == 1).sum()
scale_pos_weight = float(neg) / float(pos) if pos > 0 else 1.0

print("모델 학습 중...(XGBoost)")
model = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    eval_metric="logloss",
    scale_pos_weight=scale_pos_weight
)

model.fit(X_tr, y_tr)

# =========================
# SHAP 계산
# =========================
print("SHAP 값 계산 중...")
explainer = shap.TreeExplainer(model)

X_sample = X_va.sample(1000, random_state=42)
shap_vals = explainer.shap_values(X_sample)

if isinstance(shap_vals, list):
    shap_vals = shap_vals[1]

# =========================
# SHAP Top 8
# =========================
importance = np.abs(shap_vals).mean(axis=0)
feat_importance = (
    pd.DataFrame({"feature": X.columns, "importance": importance})
    .sort_values("importance", ascending=False)
    .reset_index(drop=True)
)

top8 = feat_importance.head(8)["feature"].tolist()
top8_idx = [X.columns.get_loc(f) for f in top8]

print("\nSHAP 기반 상위 8개 피처:")
print(feat_importance.head(8))

shap_top8_viz = {
    "top_features": top8,
    "importance_df": feat_importance.head(50),
    "X_sample_top8": X_sample[top8].copy(),
    "shap_values_top8": shap_vals[:, top8_idx],
}

with open(SAVE_DIR / "shap_top8_viz.pkl", "wb") as f:
    pickle.dump(shap_top8_viz, f)

print("✅ SHAP 저장 완료")


print("PDP 계산 중...")

# PDP는 데이터가 너무 크면 느릴 수 있어 샘플링 권장
X_pdp = X_va.sample(min(20000, len(X_va)), random_state=42)

pdp_results = {}

for feat in top8:
    # 1) 최신 sklearn: class_idx 사용 가능
    try:
        disp = PartialDependenceDisplay.from_estimator(
            model,
            X_pdp,
            features=[feat],
            kind="average",
            grid_resolution=20,
            response_method="predict_proba",
            class_idx=1
        )

    except TypeError:
        # 2) 중간 버전: response_method는 있는데 class_idx가 없을 수 있음
        try:
            disp = PartialDependenceDisplay.from_estimator(
                model,
                X_pdp,
                features=[feat],
                kind="average",
                grid_resolution=20,
                response_method="predict_proba"
            )
        except TypeError:
            # 3) 구버전: response_method/kind도 없을 수 있음
            disp = PartialDependenceDisplay.from_estimator(
                model,
                X_pdp,
                features=[feat],
                grid_resolution=20
            )


    res0 = disp.pd_results[0]

    # grid 꺼내기 (버전별 키 대비)
    if "grid_values" in res0:
        grid = res0["grid_values"][0]
    elif "values" in res0:
        grid = res0["values"][0]
    else:
        raise KeyError(f"PDP grid key not found. keys={list(res0.keys())}")

    avg = res0["average"][0]

    # avg가 (2, n_grid) 형태로 나오면 class=1로 정리
    avg = np.array(avg)
    if avg.ndim == 2 and avg.shape[0] >= 2:
        avg = avg[1]

    pdp_df = pd.DataFrame({feat: grid, "pdp": avg})

    pdp_results[feat] = {
        "grid": grid,
        "pdp": avg,
        "pdp_df": pdp_df
    }

with open(SAVE_DIR / "pdp_top8.pkl", "wb") as f:
    pickle.dump(
        {"top_features": top8, "pdp_results": pdp_results},
        f
    )

print(f"✅ PDP(top8) pkl 저장 완료: {SAVE_DIR / 'pdp_top8.pkl'}")