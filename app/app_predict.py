import streamlit as st
import pandas as pd
import plotly.express as px
from src.predict import predict_churn

def run_predict():
    st.title("🔮 KeepTune AI : 이탈 방어 시뮬레이터")
    st.markdown("##### **하이브리드 AI 모델 기반 기업 맞춤형 전략 진단**")
    st.markdown("---")

    # 1. 세션 상태 관리
    if 'predict_done' not in st.session_state:
        st.session_state.predict_done = False
    if 'result_data' not in st.session_state:
        st.session_state.result_data = None

    # 2. 시뮬레이션 설정 섹션
    with st.container():
        st.subheader("👤 시뮬레이션 대상 설정")
        col1, col2 = st.columns(2)
        
        with col1:
            auto_label = st.radio("💳 정기 결제(자동 갱신) 설정", ["활성 (구독 중)", "해지 (만료 예정)"], horizontal=True)
            auto_renew = 1.0 if "활성" in auto_label else 0.0
            
            total_mins = st.slider("🎧 일평균 노래 청취 시간 (분)", 0, 720, 30, step=1)
            total_secs = float(total_mins * 60)
            
        with col2:
            cancel_rate = st.slider("⚠️ 과거 서비스 해지 시도 비율", 0.0, 1.0, 0.1, step=0.01)
            txn_cnt = st.slider(
                "💰 누적 결제 횟수 (회)", 1, 100, 10, step=1,
                help="결제 횟수는 유저의 '서비스 숙련도'와 '권태기 진입 여부'를 판단하는 핵심 지표입니다."
            )

    # 3. 데이터 조립 및 AI 진단
    input_data = {
        'is_auto_renew': auto_renew, 'total_secs_mean': total_secs, 'is_cancel': cancel_rate,
        'payment_plan_days': 30.0, 'txn_cnt': float(txn_cnt),
        'total_paid': float(txn_cnt) * 30.0, 'total_secs_sum': total_secs * float(txn_cnt),
        'auto_renew_rate': auto_renew, 'cancel_rate': cancel_rate,
    }

    if st.button("🚀 AI 하이브리드 전략 진단 시작", use_container_width=True, type="primary"):
        try:
            with st.spinner('AI 분석 엔진이 최적의 대응 전략을 도출 중입니다...'):
                p_xgb, p_resnet, _ = predict_churn(input_data)
                
                # 유저 숙련도(txn_cnt)에 따른 가중치 로직
                w_xgb, w_resnet = (0.7, 0.3) if txn_cnt >= 5 else (0.3, 0.7)
                final_score = (p_xgb * w_xgb) + (p_resnet * w_resnet)
                
                st.session_state.result_data = {
                    'p_xgb': float(p_xgb), 'p_resnet': float(p_resnet),
                    'final_score': float(final_score), 'w_xgb': w_xgb, 'w_resnet': w_resnet,
                }
                st.session_state.predict_done = True
                st.toast("✅ 분석 및 맞춤 전략 도출이 완료되었습니다!")
        except Exception as e:
            st.error(f"⚠️ 진단 중 오류가 발생했습니다: {e}")

    # 4. 분석 리포트 출력
    if st.session_state.predict_done:
        res = st.session_state.result_data
        risk_score = res['final_score'] * 100

        st.markdown("---")
        st.subheader("📊 AI 하이브리드 진단 리포트")
        
        # 위험 등급 분류
        if risk_score > 80:
            status, color = "초고위험 (Critical)", "red"
        elif risk_score > 40:
            status, color = "주의군 (Warning)", "orange"
        else:
            status, color = "안정권 (Stable)", "green"

        st.markdown(f"**진단 결과:** :{color}[**{status}**]")

        m1, m2, m3 = st.columns(3)
        m1.metric("최종 이탈 확률", f"{risk_score:.1f}%", delta=f"{res['w_resnet']*100}% 패턴 반영", delta_color="inverse")
        m2.metric("통계 기반 위험도", f"{res['p_xgb']*100:.1f}%")
        m3.metric("패턴 기반 위험도", f"{res['p_resnet']*100:.1f}%")

        st.progress(res['final_score'])

        # 5. 시각화 차트 (판단 근거)
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            st.write("**모델별 분석 가중치**")
            fig_pie = px.pie(values=[res['w_xgb'], res['w_resnet']], names=['통계(XGBoost)', '패턴(ResNet)'], 
                             color_discrete_sequence=['#31ed8c', '#00d4ff'], hole=0.4)
            fig_pie.update_layout(height=280, margin=dict(l=0, r=0, t=20, b=0))
            st.plotly_chart(fig_pie, use_container_width=True)
            
        with col_c2:
            st.write("**데이터별 위험 기여도**")
            features = ['정기결제', '활동성', '이용기간(횟수)', '이탈이력']
            impact = [
                35 if auto_renew == 0 else -20, 
                25 if total_mins < 30 else -15,
                15 if txn_cnt > 40 else -5,
                25 if cancel_rate > 0.5 else -10
            ]
            fig_bar = px.bar(x=impact, y=features, orientation='h', color=impact, color_continuous_scale='RdYlGn_r')
            fig_bar.update_layout(height=280, margin=dict(l=0, r=0, t=20, b=0), coloraxis_showscale=False)
            st.plotly_chart(fig_bar, use_container_width=True)

        # --- 🛠️ 6. AI 전략 실행 계획 (한글판) ---
        st.markdown("---")
        st.subheader("🛠️ AI 전략 실행 계획 (기업 맞춤형 제안)")
        st.caption("유저 세그먼트별 데이터 분석 결과를 바탕으로 도출된 비즈니스 액션 플랜입니다.")

        t1, t2, t3 = st.tabs(["💰 수익성 및 성장 전략", "🎨 제품 경험 및 활성화", "🤝 고객 소통 및 케어"])

        with t1:
            st.markdown("#### **1. 매출 최적화 및 결제 유지 전략**")
            if auto_renew == 0:
                st.markdown("> **[핵심 과제] 자동 결제 전환 락인(Lock-in) 캠페인**")
                st.write("- **현황**: 수동 결제 유저로서 매 결제 주기마다 이탈을 결정할 위험이 높습니다.")
                st.write("- **전략**: 자동 결제 전환 시 '6개월간 결제액 20% 페이백'을 제공하여 결제 지속성을 확보하세요.")
            else:
                st.markdown("> **[핵심 과제] 구독 모델 업셀링(Up-selling)**")
                st.write("- **전략**: 월 단위 구독을 연간 구독권으로 전환 유도하여 고객생애가치(LTV)를 극대화하세요.")
            
            if txn_cnt > 30:
                st.markdown("> **[장기 고객] 이용 권태기(Maturity) 방어 전략**")
                st.write("- **전략**: 누적 결제 50회 기념 '프리미엄 한정판 굿즈' 증정. 가격 할인보다 브랜드 애착 형성이 효과적입니다.")

        with t2:
            st.markdown("#### **2. 사용자 활동성 및 경험 강화**")
            if total_mins < 30:
                st.markdown("> **[활성화] 실시간 맥락(Context) 푸시 전략**")
                st.write("- **현황**: 청취 활동이 임계점 이하로 서비스 이탈 초기 단계입니다.")
                st.write("- **전략**: 과거 선호 아티스트의 신곡 프리뷰를 활동 가능 시간대에 맞춰 타겟팅 발송하세요.")
            else:
                st.markdown("> **[심화] 프리미엄 콘텐츠 경험 확대**")
                st.write("- **전략**: 고음질(Hi-Fi) 스트리밍 무료 체험권 제공을 통해 고관여 유저 세그먼트로 이동을 유도하세요.")

        with t3:
            st.markdown("#### **3. 예방적 관리 및 브랜드 소통**")
            if cancel_rate > 0.5:
                st.error("**[긴급 대응] 해지 시도 고빈도 유저 집중 케어**")
                st.write("- **전략**: 해지 버튼 클릭 시 '1:1 실시간 상담 연결' 또는 즉각적인 '구독 일시정지' 옵션을 최우선 제안하세요.")
            else:
                st.markdown("> **[브랜드] 앰배서더 및 커뮤니티 전략**")
                st.write("- **전략**: 서비스 개선 설문 참여 시 포인트 지급 및 브랜드 커뮤니티 '서포터즈' 권한을 부여하세요.")

    st.markdown("---")
    st.caption("KeepTune v2.6 | 기업용 AI 의사결정 지원 시스템 가동 중")