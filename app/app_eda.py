import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
import platform
from matplotlib import font_manager, rc
from scripts.eda_interactive import plot_churn_style_st, set_korean_font
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go


# 전역 스타일 설정
plt.rcParams["axes.grid"] = True
TARGET = "is_churn"

# 경로 설정
ROOT_DIR = Path(__file__).resolve().parents[1]
EDA_DATA_DIR = ROOT_DIR / "data" / "preprocessed"



# 탭 실행 전 호출
set_korean_font()


@st.cache_data 
def load_tab_data(file_name):
    # 각 탭에 필요한 요약 데이터 로드
    return pd.read_pickle(EDA_DATA_DIR / file_name)

def run_eda():
    # 0. 요약 데이터
    summary = load_tab_data("eda_summary.pkl")
    st.title("📊 데이터 심층 인사이트 (EDA)")
    st.markdown("미리 계산된 데이터로 인사이트를 빠르게 확인하세요.")
    st.markdown("---")

    # 1. 상단 요약 지표
    c1, c2, c3 = st.columns(3)
    c1.metric("분석 대상 유저", f"{summary['total_users']:,} 명")
    c2.metric("평균 이탈률", f"{summary['churn_rate']:.1f}%")
    c3.metric("평균 청취 시간", f"{summary['avg_secs'] / 60:,.1f}분")

    # 2. 탭 구성
    tab1, tab2, tab3 = st.tabs(["🔍 핵심 변수 영향력", "🎧이탈 핵심 요인 탐색", "💳데이터 시각화"])

    # ==========================
    # Tab1 : SHAP 기반 탐구
    # ==========================
    with tab1:
        st.markdown("### 🔍 **모델이 주목한 이탈 핵심 요인 (SHAP)**")
        st.info("XGBoost 모델이 이탈 예측에 어떤 변수를 얼마나 강하게 반영했는지, 그리고 각 변수 값이 이탈 방향(↑/↓)에 어떤 영향을 주는지 탐구합니다.")

        try:
            shap_pack = load_tab_data("shap_top8_viz.pkl")  # dict
            top_features = shap_pack["top_features"]
            imp_df = shap_pack["importance_df"].copy()
            Xs = shap_pack["X_sample_top8"].copy()
            shap_top = shap_pack["shap_values_top8"]  # numpy (n, 8)

            # (1) 중요도 Bar
            imp_top8 = imp_df[imp_df["feature"].isin(top_features)].copy()
            imp_top8 = imp_top8.sort_values("importance", ascending=True)

            col_l, col_r = st.columns([1.2, 1])
            with col_l:
                fig_imp = px.bar(
                    imp_top8,
                    x="importance",
                    y="feature",
                    orientation="h",
                    title="SHAP 평균 영향력 (Top8)",
                    labels={"importance": "Mean(|SHAP|)", "feature": "Feature"},
                )
                fig_imp.update_layout(height=420, margin=dict(l=10, r=10, t=60, b=10))
                st.plotly_chart(fig_imp, use_container_width=True)

            with col_r:
                st.markdown("#### 탐구 옵션")
                feat_sel = st.selectbox("상세 분석할 변수 선택", top_features, index=0)
                q_cut = st.select_slider(
                    "값 구간(분위수) 비교",
                    options=[0.1, 0.25, 0.5, 0.75, 0.9],
                    value=0.75
                )
                st.caption("선택한 변수의 값이 낮은 구간 vs 높은 구간에서 SHAP 평균(방향)을 비교합니다.")

            # (2) 선택 피처의 SHAP 분포/방향 탐구
            j = top_features.index(feat_sel)
            df_sc = pd.DataFrame({
                "feature_value": Xs[feat_sel].values,
                "shap_value": shap_top[:, j],
            })

            # 상/하 분위수 구간 요약
            lo = df_sc["feature_value"].quantile(1 - q_cut)
            hi = df_sc["feature_value"].quantile(q_cut)

            low_grp = df_sc[df_sc["feature_value"] <= lo]["shap_value"]
            high_grp = df_sc[df_sc["feature_value"] >= hi]["shap_value"]

            low_mean = float(low_grp.mean()) if len(low_grp) else np.nan
            high_mean = float(high_grp.mean()) if len(high_grp) else np.nan

            # (3) Scatter: feature value vs shap
            fig_sc = px.scatter(
                df_sc,
                x="feature_value",
                y="shap_value",
                opacity=0.65,
                title=f"{feat_sel} 값에 따른 SHAP 영향(방향)",
                labels={"feature_value": feat_sel, "shap_value": "SHAP value (churn 방향)"},
            )
            fig_sc.update_layout(height=420, margin=dict(l=10, r=10, t=60, b=10))
            st.plotly_chart(fig_sc, use_container_width=True)

            # (4) 인사이트 카드
            cA, cB, cC = st.columns(3)
            with cA:
                st.metric("낮은 구간 평균 SHAP", f"{low_mean:+.4f}")
            with cB:
                st.metric("높은 구간 평균 SHAP", f"{high_mean:+.4f}")
            with cC:
                if np.isfinite(low_mean) and np.isfinite(high_mean):
                    direction = "높을수록 이탈↑" if high_mean > low_mean else "높을수록 이탈↓"
                    st.metric("방향성(구간 비교)", direction)
                else:
                    st.metric("방향성(구간 비교)", "N/A")

            st.markdown("#### 📌 자동 해석")
            if np.isfinite(low_mean) and np.isfinite(high_mean):
                if high_mean - low_mean > 0:
                    st.success(f"**{feat_sel} 값이 높아질수록 평균적으로 이탈 예측(+) 방향으로 기여**하는 경향이 있습니다. (상위구간 SHAP {high_mean:+.4f} > 하위구간 {low_mean:+.4f})")
                else:
                    st.success(f"**{feat_sel} 값이 높아질수록 평균적으로 이탈 예측(-) 방향으로 기여**하는 경향이 있습니다. (상위구간 SHAP {high_mean:+.4f} < 하위구간 {low_mean:+.4f})")
            else:
                st.warning("구간 비교에 필요한 표본이 부족합니다. 분위수 설정을 조정해보세요.")

            # (5) 샘플(유저) 하나를 골라 “이 샘플을 이탈로 본 이유” 탐구
            st.markdown("---")
            st.markdown("### 🧩 **개별 샘플(유저) 설명 탐구**")
            st.caption("샘플 인덱스를 골라, Top8 변수가 해당 샘플의 이탈 방향에 어떻게 기여했는지 확인합니다.")

            row_idx = st.slider("샘플 선택 (X_sample 내부 행)", 0, len(Xs) - 1, 0)
            row_vals = Xs.iloc[row_idx]
            row_shap = shap_top[row_idx, :]

            df_row = pd.DataFrame({
                "feature": top_features,
                "value": [row_vals[f] for f in top_features],
                "shap": row_shap
            }).sort_values("shap", ascending=True)

            fig_row = px.bar(
                df_row,
                x="shap",
                y="feature",
                orientation="h",
                title="선택 샘플의 변수별 기여도(SHAP)",
                labels={"shap": "SHAP ( + 이탈↑ / - 이탈↓ )", "feature": "Feature"},
            )
            fig_row.update_layout(height=420, margin=dict(l=10, r=10, t=60, b=10))
            st.plotly_chart(fig_row, use_container_width=True)

            with st.expander("선택 샘플 상세 값 보기"):
                st.dataframe(df_row.sort_values("shap", ascending=False))

        except FileNotFoundError:
            st.warning("SHAP 결과 파일(shap_top8_viz.pkl)을 찾을 수 없습니다. run_shap.py를 먼저 실행해주세요.")
    
    ### ===========================================================================================
    ### Tab2 : PDP 기반 이탈 요인 확인하기.
    ### =========================================================================================== 
    with tab2:
        st.markdown("### 🎧 **변수별 이탈 민감도 탐색 (PDP)**")
        st.info("선택한 변수 값이 바뀔 때, 모델의 평균 이탈 확률 예측이 어떻게 변하는지를 확인합니다.")

        try:
            pdp_pack = load_tab_data("pdp_top8.pkl")
            top_features_pdp = pdp_pack["top_features"]
            pdp_results = pdp_pack["pdp_results"]

            feat_sel = st.selectbox("PDP로 볼 변수 선택", top_features_pdp, index=0, key="pdp_feat")
            res = pdp_results[feat_sel]

            # pdp_df가 있으면 쓰고 없으면 만들기
            if "pdp_df" in res:
                pdp_df = res["pdp_df"].copy()
                # 컬럼명이 feature 이름일 수 있어 정규화
                if feat_sel not in pdp_df.columns:
                    # 첫 컬럼을 feature로 간주
                    first_col = [c for c in pdp_df.columns if c != "pdp"][0]
                    pdp_df = pdp_df.rename(columns={first_col: feat_sel})
            else:
                pdp_df = pd.DataFrame({feat_sel: res["grid"], "pdp": res["pdp"]})

            # 라인차트
            fig_pdp = px.line(
                pdp_df,
                x=feat_sel,
                y="pdp",
                markers=True,
                title=f"PDP: {feat_sel} 변화에 따른 평균 이탈확률",
                labels={feat_sel: feat_sel, "pdp": "Predicted churn probability (avg)"},
            )
            fig_pdp.update_layout(height=450, margin=dict(l=10, r=10, t=60, b=10))
            st.plotly_chart(fig_pdp, use_container_width=True)

            # 간단 인사이트: min/max 구간
            min_p = float(np.nanmin(pdp_df["pdp"]))
            max_p = float(np.nanmax(pdp_df["pdp"]))
            delta = max_p - min_p

            c1, c2, c3 = st.columns(3)
            c1.metric("최소 평균 이탈확률", f"{min_p:.4f}")
            c2.metric("최대 평균 이탈확률", f"{max_p:.4f}")
            c3.metric("변동 폭(민감도)", f"{delta:.4f}")

            st.markdown("#### 📌 해석 힌트")
            if delta >= 0.05:
                st.success("이 변수는 **모델 예측을 크게 흔드는 민감 변수**일 가능성이 큽니다. 특정 구간에서 이탈 확률이 급상승/급하락하는지 확인해보세요.")
            elif delta >= 0.02:
                st.info("이 변수는 **중간 정도의 민감도**를 보입니다. 다른 변수와 함께 볼 때 설명력이 좋아질 수 있어요.")
            else:
                st.warning("이 변수는 PDP 기준으로 **평균 예측에 미치는 영향 변화가 작습니다**. (개별 유저 수준에서는 달라질 수 있음)")

            with st.expander("PDP 데이터 보기"):
                st.dataframe(pdp_df)

        except FileNotFoundError:
            st.warning("PDP 결과 파일(pdp_top8.pkl)을 찾을 수 없습니다. run_shap.py에서 PDP 저장을 먼저 완료해주세요.")
        except KeyError as e:
            st.error(f"PDP pkl 구조가 예상과 달라요: {e}")

    ### ===========================================================================================
    ### Tab3 : 카테고리별 & 수치별 데이터 시각화 제공 탭.
    ### =========================================================================================== 
    ### ===========================================================================================
    ### Tab3 : 카테고리별 & 수치별 데이터 시각화 제공 탭.
    ### =========================================================================================== 
    with tab3:
        st.markdown("### **데이터 시각화**")
        st.caption("사전 가공된 집계 데이터를 활용하여 변수별 이탈률을 시각화합니다.")
        
        # 전역 스타일 설정
        plt.rcParams["axes.grid"] = True
        TARGET = "is_churn"
        
        # ---------------------------------------------------------
        # 1) 카테고리 변수별 이탈률 (사전 집계 데이터 사용)
        # ---------------------------------------------------------
        st.subheader("1) 카테고리 변수별 이탈률")
        try:
            cat_summary = pd.read_pickle(EDA_DATA_DIR / "eda_cat_summary.pkl")
            cat_candidates = list(cat_summary.keys())

            col1, col2 = st.columns(2)
            with col1:
                cat_col = st.selectbox("컬럼 선택", cat_candidates, index=0, key="cat_select")
                top_n = st.slider("상위 N개만 표시", 5, 50, 20, step=5)
            with col2:
                min_n = st.number_input("최소 표본수 필터", min_value=1, value=100, step=50)
                sort_by = st.radio("정렬 기준", ["churn", "n"], horizontal=True,
                                   format_func=lambda x: "이탈률 높은 순" if x == "churn" else "표본 많은 순")

            run_cat = st.button("📊 카테고리 그래프 생성", use_container_width=True)

            if run_cat:
                g = cat_summary[cat_col].copy()
                g = g[g['n'] >= min_n] # 필터링 로직 유지
                
                if sort_by == "churn":
                    g = g.sort_values("churn_rate", ascending=False)
                else:
                    g = g.sort_values("n", ascending=False)
                
                g = g.head(top_n)
                
                if g.empty:
                    st.warning("조건에 맞는 데이터가 없습니다 (최소 표본수 필터를 확인하세요).")
                else:
                    # 표본 수(Line)가 제거된 새로운 스타일의 함수 호출
                    fig = plot_churn_style_st(g, cat_col, f"카테고리별 이탈률: {cat_col}", palette="magma")
                    st.pyplot(fig, clear_figure=True)
                    
                    # (선택 사항) 표본 수를 확인하고 싶을 사용자를 위해 데이터프레임 하단 출력
                    with st.expander("상세 데이터 보기"):
                        st.dataframe(g.style.format({'churn_rate': '{:.2%}', 'n': '{:,}'}))

        except FileNotFoundError:
            st.error("사전 가공된 카테고리 데이터(eda_cat_summary.pkl)를 찾을 수 없습니다.")

        st.markdown("---")
        # ---------------------------------------------------------
        # 2) 수치형 변수 bin별 이탈률 
        # ---------------------------------------------------------
        st.subheader("2) 수치형 변수 구간별 이탈률")

        try:
            # 경량화된 수치 데이터 로드
            df_num_light = pd.read_pickle(EDA_DATA_DIR / "eda_num_light.pkl")
            num_candidates = [c for c in df_num_light.columns if c != TARGET]

            col1, col2 = st.columns(2)
            with col1:
                num_col = st.selectbox("수치형 컬럼 선택", num_candidates, index=0, key="num_select")
            with col2:
                q_val = st.slider("구간 수", 4, 20, 10)

            run_num = st.button("📈 수치형 구간 그래프 생성", use_container_width=True)

            if run_num:
                # [경량 데이터를 실시간 구간화] - 전체 로드보다 훨씬 빠름
                df_tmp = df_num_light[[num_col, TARGET]].copy()
                df_tmp['bin'] = pd.qcut(df_tmp[num_col], q=q_val, duplicates='drop').astype(str)
                
                g = df_tmp.groupby('bin')[TARGET].agg(['mean', 'count']).reset_index()
                g.columns = ['bin', 'churn_rate', 'n']
                g = g.sort_values('bin')

                title = f"Churn Rate by {num_col} {q_val} Bins)"
                fig = plot_churn_style_st(g, 'bin', title, palette="viridis")
                st.pyplot(fig, clear_figure=True)
                
        except FileNotFoundError:
            st.error("사전 가공된 수치형 데이터(eda_num_light.pkl)를 찾을 수 없습니다.")