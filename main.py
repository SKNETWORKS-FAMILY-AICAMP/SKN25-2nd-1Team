import streamlit as st
from app_home import run_home
from app_eda import run_eda
from app_predict import run_predict
from app_strategy import run_strategy

# 1. 페이지 설정
st.set_page_config(page_title="KeepTune Dashboard", layout="wide", page_icon="🎧")

# 2. 사이드바 구성
st.sidebar.title("🎧 KeepTune")
st.sidebar.markdown("---")

# 페이지 상태 관리 (기본값 설정)
if 'page' not in st.session_state: 
    st.session_state.page = '대시보드'

st.sidebar.subheader("분석 리포트")

# 버튼형 메뉴
if st.sidebar.button("🏠 대시보드", use_container_width=True): 
    st.session_state.page = '대시보드'
if st.sidebar.button("🔍 유저 행동 인사이트", use_container_width=True): 
    st.session_state.page = '유저 행동 인사이트'
if st.sidebar.button("🔮 이탈 위험도 시뮬레이터", use_container_width=True): 
    st.session_state.page = '이탈 위험도 시뮬레이터'
if st.sidebar.button("🚀 비즈니스 전략", use_container_width=True): 
    st.session_state.page = '비즈니스 전략'

st.sidebar.markdown("---")
st.sidebar.caption("© 2026 KeepTune. All rights reserved.")

# 3. 페이지 전환 로직
if st.session_state.page == '대시보드': 
    run_home()
elif st.session_state.page == '유저 행동 인사이트': 
    run_eda()
elif st.session_state.page == '이탈 위험도 시뮬레이터': 
    run_predict()
elif st.session_state.page == '비즈니스 전략': 
    run_strategy()