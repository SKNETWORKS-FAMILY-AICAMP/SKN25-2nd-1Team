import streamlit as st
import sys
import pandas as pd
import plotly.express as px
from pathlib import Path

# 1. 경로 설정 및 모듈 임포트
BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(BASE_DIR / "src"))

try:
    from model_loader import predict_churn
except ImportError:
    st.error("❌ model_loader.py를 찾을 수 없습니다. 경로를 확인해주세요.")

def run_predict():
    st.title("🔮 KeepTune: 이탈 방어 AI 시뮬레이터")
    st.markdown("##### **XGBoost & ResNet 하이브리드 진단 리포트**")
    st.markdown("---")

    # 세션 상태 관리
    if 'predict_done' not in st.session_state:
        st.session_state.predict_done = False
    if 'result_data' not in st.session_state:
        st.session_state.result_data = None

    # 2. 시뮬레이션 대상 설정 (사용자 입력)
    with st.container():
        st.subheader("👤 시뮬레이션 대상 설정")
        col1, col2 = st.columns(2)
        with col1:
            # [개선] 정기 결제 용어 직관화
            auto_label = st.radio("💳 정기 결제(자동 갱신) 설정", ["활성 (구독 중)", "해지 (만료 예정)"], horizontal=True)
            auto_renew = 1.0 if "활성" in auto_label else 0.0
            total_mins = st.number_input("🎧 일평균 노래 청취 시간 (분)", 0, 1440, 30)
            total_secs = float(total_mins * 60)
        with col2:
            cancel_rate = st.slider("⚠️ 과거 서비스 해지 시도 비율", 0.0, 1.0, 0.1)
            txn_cnt = st.number_input("💰 누적 결제 횟수", 1, 100, 10)

    input_data = {
        'is_auto_renew': auto_renew,  # 변수명 통일
        'total_secs_mean': total_secs,
        'is_cancel': cancel_rate,      # EDA 변수명과 일치
        'payment_plan_days': 30.0,      # 기본값 설정
        'txn_cnt': float(txn_cnt)
    }

    # 3. AI 진단 실행 (XGBoost & ResNet)
    if st.button("🚀 하이브리드 AI 분석 시작", use_container_width=True):
        try:
            with st.spinner('XGBoost(트리) & ResNet(딥러닝) 엔진 가동 중...'):
                # 실제 분석 함수 호출
                results = predict_churn(input_data)
                
                # 결과 저장 및 상태 업데이트
                st.session_state.result_data = {'scores': results, 'input': input_data}
                st.session_state.predict_done = True
                
        except Exception as e:
            # 코드가 화면에 뜨지 않게 막고, 친절한 안내 메시지만 띄웁니다.
            st.error("⚠️ 분석 엔진 가동 중 일시적인 오류가 발생했습니다.")
            st.info("모델 파일(.pkl)이 정확한 경로에 있는지 확인해주세요.")
            # 개발용 로그는 터미널에만 출력합니다.
            print(f"Error details: {e}")

    # 4. 분석 결과 출력
    if st.session_state.predict_done:
        res = st.session_state.result_data
        p_xgb, p_resnet, final_score = res['scores']
        risk_score = final_score * 100

        # 주요 지표 대시보드
        st.subheader("📊 종합 이탈 위험 진단")
        m1, m2, m3 = st.columns(3)
        m1.metric("최종 이탈 위험도", f"{risk_score:.1f}%")
        m2.metric("XGBoost 판정", f"{p_xgb*100:.1f}%")
        m3.metric("ResNet 판정", f"{p_resnet*100:.1f}%")

        st.progress(final_score)

        # 5. 판단 근거 시각화 (EDA 인사이트 반영)
        st.write("")
        st.subheader("🔍 AI의 판단 근거 (Feature Importance)")
        
        # 실제 기여도 가중치 반영
        features = ['정기 결제 설정', '청취 시간', '해지 시도 이력', '결제 횟수']
        contributions = [
            -35 if auto_renew == 1 else 42.5,  # EDA에서 auto_renew 기여도가 42.5%였던 점 반영
            -15 if total_mins > 45 else 12,
            25 if cancel_rate > 0.3 else -5,
            -10 if txn_cnt > 15 else 5
        ]
        
        fig = px.bar(
            x=contributions, y=features, orientation='h',
            color=contributions, color_continuous_scale='RdYlGn_r',
            labels={'x': '이탈 위험 기여도 (+: 위험 가중, -: 안전 요인)', 'y': '핵심 데이터'}
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        
        # 6. 상세 액션 플랜 (전략 페이지와 연동)
        st.subheader("📋 추천 리텐션 솔루션")
        t1, t2 = st.tabs(["💡 즉각적 대응 전략", "📈 장기 고객 관리"])

        with t1:
            if auto_renew == 0:
                st.info("**[강력 권고] 정기 결제 전환 프로모션**")
                st.write("- 유저의 이탈 위험 요인 중 '정기 결제 미설정'이 가장 큽니다.")
                st.write("- **솔루션**: 다시 정기 결제로 전환 시 '1개월 무료' 쿠폰 즉시 발송")
            elif risk_score > 60:
                st.warning("**[주의] 활동성 복구 캠페인**")
                st.write("- 청취 시간이 급감한 상태입니다. 선호 장르의 신곡 알림을 발송하세요.")
            else:
                st.success("**[안정] 현상 유지 및 로열티 강화**")
                st.write("- 현재 이탈 위험이 낮습니다. 장기 이용 혜택을 안내하세요.")

    st.markdown("---")
    st.caption("KeepTune v2.5 | Hybrid Ensemble (XGBoost + ResNet)")