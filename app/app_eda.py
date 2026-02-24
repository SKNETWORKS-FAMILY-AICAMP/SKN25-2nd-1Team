import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
import platform
from matplotlib import font_manager, rc
from src.eda_interactive import plot_churn_style_st, set_korean_font

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

    ### ===========================================================================================
    ### Tab1 : Shap 기반 이탈 요인 확인하기.
    ### =========================================================================================== 
    with tab1:
        st.markdown("### 🔍 **모델이 주목한 이탈 핵심 요인**")
        st.info("XGBoost 모델을 활용하여 사용자의 이탈을 예측할 때 어떤 변수에 가장 큰 비중을 두었는지 보여줍니다.")

        # 1. SHAP 중요도 데이터 로드
        try:
            df_shap = load_tab_data("top_5_shap_features.pkl")
            
            # 시각화를 위해 데이터 정렬 (중요도 높은 순)
            df_shap = df_shap.sort_values(by='importance', ascending=True)

            # 2. Plotly 수평 바 차트 생성
            fig_shap = px.bar(
                df_shap,
                x='importance',
                y='feature',
                orientation='h',
                title="이탈 핵심 변수 영향력",
                labels={'importance': '평균 영향력', 'feature': '변수명'},
                color='importance',
                color_continuous_scale='Reds'
            )

            # 레이아웃 미세 조정
            fig_shap.update_layout(
                showlegend=False,
                height=500,
                margin=dict(l=20, r=20, t=50, b=20),
                yaxis={'categoryorder': 'total ascending'}
            )

            # 차트 출력
            st.plotly_chart(fig_shap, use_container_width=True)

            # 3. 인사이트 요약
            st.markdown("#### **📌 분석 결과 해석**")
            top_1 = df_shap.iloc[-1]['feature']
            top_2 = df_shap.iloc[-2]['feature']
            
            col_a, col_b = st.columns(2)
            with col_a:
                st.success(f"**1순위 핵심 지표: {top_1}**\n\n이 지표의 변화가 유저 이탈 예측에 가장 결정적인 역할을 합니다.")
            with col_b:
                st.success(f"**2순위 핵심 지표: {top_2}**\n\n해당 수치가 특정 임계치를 넘을 경우 이탈 위험군으로 분류될 가능성이 높습니다.")

        except FileNotFoundError:
            st.warning("SHAP 결과 파일(top_5_shap_features.pkl)을 찾을 수 없습니다. 분석 스크립트를 먼저 실행해주세요.")
    
    ### ===========================================================================================
    ### Tab2 : PDP 기반 이탈 요인 확인하기.
    ### =========================================================================================== 
    with tab2:
        st.markdown("### **이탈 핵심 요인 탐색**")
        # df_sample = load_tab_data("eda_box_plot.pkl")
        
        # fig_box = px.box(df_sample, x='is_churn', y='total_secs_mean', color='is_churn',
        #                  labels={'is_churn': '이탈 여부', 'total_secs_mean': '평균 청취 시간(초)'})
        # st.plotly_chart(fig_box, use_container_width=True)

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