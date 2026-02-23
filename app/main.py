import streamlit as st
import sys
from pathlib import Path

# 1. 파일 경로 설정 (상대 경로 임포트 에러 방지)
# 메인 파일이 app 폴더 안에 있으므로 부모 폴더(루트)를 경로에 추가합니다.
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

# 2. 페이지 모듈 임포트 (app.을 붙여서 루트 기준 임포트)
try:
    from app.app_home import run_home
    from app.app_eda import run_eda
    from app.app_predict import run_predict
    from app.app_strategy import run_strategy
except ImportError:
    # 만약 위 방식이 안되면 직접 파일명으로 임포트 시도
    from app_home import run_home
    from app_eda import run_eda
    from app_predict import run_predict
    from app_strategy import run_strategy

def main():
    # --- [페이지 설정] ---
    st.set_page_config(page_title="KeepTune Dashboard", layout="wide", page_icon="🎧")

    # --- [사이드바 구성] ---
    st.sidebar.title("🎧 KeepTune")
    st.sidebar.markdown("---")

    # 페이지 상태 관리
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

    # --- [페이지 전환 로직] ---
    if st.session_state.page == '대시보드': 
        run_home()
    elif st.session_state.page == '유저 행동 인사이트': 
        run_eda()
    elif st.session_state.page == '이탈 위험도 시뮬레이터': 
        run_predict()
    elif st.session_state.page == '비즈니스 전략': 
        run_strategy()

if __name__ == "__main__":
    main()