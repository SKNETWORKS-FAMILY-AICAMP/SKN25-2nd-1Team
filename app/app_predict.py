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

    # 2. 시뮬레이션 대상 설정 섹션
    with st.container():
        st.subheader("👤 시뮬레이션 대상 설정")
        col1, col2 = st.columns(2)
        
        with col1:
            # 정기 결제 설정
            auto_label = st.radio("💳 정기 결제(자동 갱신) 설정", ["활성 (구독 중)", "해지 (만료 예정)"], horizontal=True)
            auto_renew = 1.0 if "활성" in auto_label else 0.0
            
            # [직관적 설명 추가] 청취 시간 슬라이더
            total_mins = st.slider("🎧 일평균 노래 청취 시간 (분)", 0, 720, 30, step=1)
            if total_mins == 0:
                st.caption("👻 **Status: Inactive** - 접속 기록이 없는 이탈 고위험군입니다.")
            elif total_mins < 30:
                st.caption("📉 **Status: Light User** - 서비스 안착을 위한 활성화 전략이 필요합니다.")
            elif total_mins < 180:
                st.caption("✅ **Status: Active** - 안정적인 이용 패턴을 유지하고 있습니다.")
            else:
                st.caption("🔥 **Status: Heavy User** - 우리 서비스의 핵심 팬층(Loyal)입니다.")
            
            total_secs = float(total_mins * 60)
            
        with col2:
            # [직관적 설명 추가] 해지 시도 비율 슬라이더
            cancel_rate = st.slider(
                "⚠️ 서비스 이탈 징후 (과거 해지 시도 확률)", 
                0.0, 1.0, 0.1, step=0.01,
                help="과거 해지 페이지 방문 흔적 등을 수치화한 지표입니다."
            )
            if cancel_rate < 0.2:
                st.write("😇 유저 심리: **평온 (만족하며 이용 중)**")
            elif cancel_rate < 0.6:
                st.write("🤔 유저 심리: **번민 (타사 서비스와 저울질 중)**")
            else:
                st.write("🚨 유저 심리: **위험 (해지 버튼 클릭 직전 상태)**")

            # [직관적 설명 추가] 누적 결제 횟수 슬라이더
            txn_cnt = st.slider(
                "💰 누적 결제 횟수 (회)", 1, 100, 10, step=1,
                help="결제 횟수는 유저의 '서비스 숙련도'와 '브랜드 로열티'를 상징합니다."
            )
            if txn_cnt <= 3:
                st.caption("🌱 **Stage: Early** - 서비스 탐색기이며 초기 이탈 위험이 큽니다.")
            elif txn_cnt <= 12:
                st.caption("🏃 **Stage: Settled** - 1년 내외 이용자로 안정적인 구독 단계입니다.")
            elif txn_cnt <= 36:
                st.caption("💎 **Stage: Loyal** - 3년 이상 이용한 핵심 고객입니다.")
            else:
                st.caption("👑 **Stage: VIP** - 강력한 팬덤을 가진 최상위 등급 유저입니다.")

    # 3. 데이터 조립 및 AI 진단
    input_data = {
        'is_auto_renew': auto_renew, 
        'total_secs_mean': total_secs, 
        'is_cancel': 1.0 if cancel_rate > 0.5 else 0.0,
        'payment_plan_days': 30.0, 
        'txn_cnt': float(txn_cnt),
        'total_paid': float(txn_cnt) * 30.0, 
        'total_secs_sum': total_secs * float(txn_cnt),
        'auto_renew_rate': auto_renew, 
        'cancel_rate': cancel_rate,
    }

    if st.button("🚀 AI 하이브리드 전략 진단 시작", use_container_width=True, type="primary"):
        try:
            with st.spinner('AI 분석 엔진이 최적의 대응 전략을 도출 중입니다...'):
                p_xgb, p_resnet, _ = predict_churn(input_data)
                
                # 가중치 로직
                w_xgb, w_resnet = (0.7, 0.3) if txn_cnt >= 5 else (0.3, 0.7)
                final_score = (p_xgb * w_xgb) + (p_resnet * w_resnet)
                
                # [🚨 KeyError 해결 핵심] app_strategy.py에서 요구하는 'scores' 리스트를 생성합니다.
                st.session_state.result_data = {
                    'p_xgb': float(p_xgb), 
                    'p_resnet': float(p_resnet),
                    'final_score': float(final_score), 
                    'scores': [float(final_score)], # 비즈니스 전략 페이지 필수 키값
                    'w_xgb': w_xgb, 
                    'w_resnet': w_resnet,
                }
                st.session_state.predict_done = True
                st.toast("✅ 분석 및 맞춤 전략 도출이 완료되었습니다!")
        except Exception as e:
            st.error(f"⚠️ 진단 중 오류가 발생했습니다: {e}")

    # 4. 분석 리포트 출력
    if st.session_state.predict_done:
        res = st.session_state.result_data
        # scores 리스트에서 값을 가져오도록 안전하게 변경
        risk_score = res['scores'][-1] * 100 

        st.markdown("---")
        st.subheader("📊 AI 하이브리드 진단 리포트")
        
        if risk_score > 80:
            status, color = "초고위험 (Critical)", "red"
        elif risk_score > 40:
            status, color = "주의군 (Warning)", "orange"
        else:
            status, color = "안정권 (Stable)", "green"

        st.markdown(f"**진단 결과:** :{color}[**{status}**]")

        m1, m2, m3 = st.columns(3)
        m1.metric("최종 이탈 위험도", f"{risk_score:.1f}%", delta="Critical" if risk_score > 60 else "Safe", delta_color="inverse")
        m2.metric("통계 기반 점수 (XGB)", f"{res['p_xgb']*100:.1f}%")
        m3.metric("패턴 기반 점수 (ResNet)", f"{res['p_resnet']*100:.1f}%")

        st.progress(res['scores'][-1])

        # 5. 시각화 차트
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            st.write("**모델별 분석 가중치**")
            fig_pie = px.pie(values=[res['w_xgb'], res['w_resnet']], names=['통계(XGBoost)', '패턴(ResNet)'], 
                             color_discrete_sequence=['#31ed8c', '#00d4ff'], hole=0.4)
            fig_pie.update_layout(height=280, margin=dict(l=0, r=0, t=20, b=0))
            st.plotly_chart(fig_pie, use_container_width=True)
            
        with col_c2:
            st.write("**데이터별 위험 기여도**")
            features = ['정기결제', '활동성', '숙련도', '이탈이력']
            impact = [
                35 if auto_renew == 0 else -20, 
                25 if total_mins < 30 else -15,
                15 if txn_cnt > 40 else -5,
                25 if cancel_rate > 0.5 else -10
            ]
            fig_bar = px.bar(x=impact, y=features, orientation='h', color=impact, color_continuous_scale='RdYlGn_r')
            fig_bar.update_layout(height=280, margin=dict(l=0, r=0, t=20, b=0), coloraxis_showscale=False)
            st.plotly_chart(fig_bar, use_container_width=True)

        # 6. AI 전략 실행 계획 (탭 분리)
        st.markdown("---")
        st.subheader("🛠️ AI 전략 실행 계획 (기업 맞춤형 제안)")
        st.caption("유저 세그먼트별 데이터 분석 결과를 바탕으로 도출된 비즈니스 액션 플랜입니다.")

        t1, t2, t3 = st.tabs(["💰 수익성 전략", "🎨 경험 강화", "🤝 고객 케어"])

        with t1:
            st.markdown("#### **1. 매출 최적화 및 결제 유지 전략**")
            if txn_cnt > 36:
                st.markdown("> **[VIP 고객] 최상위 유지(Retention) 전략**")
                st.write("- **전략**: 장기 결제 감사 캠페인 및 프리미엄 전용 혜택 제공.")
            elif auto_renew == 0:
                st.markdown("> **[핵심 과제] 자동 결제 전환 캠페인**")
                st.write("- **전략**: 자동 결제 전환 시 '6개월간 결제액 20% 페이백' 제공.")
            else:
                st.markdown("> **[핵심 과제] 연간 구독 모델 제안**")
                st.write("- **전략**: 월 단위 구독을 연간 구독권으로 전환 유도하여 LTV 극대화.")

        with t2:
            st.markdown("#### **2. 사용자 활동성 및 경험 강화**")
            if total_mins < 30:
                st.markdown("> **[활성화] 맥락 기반 푸시 전략**")
                st.write("- **전략**: 선호 아티스트 신곡 정보를 활동 예상 시간대에 맞춰 타겟팅 발송.")
            else:
                st.markdown("> **[심화] 프리미엄 경험 확대**")
                st.write("- **전략**: 고음질 체험권 등을 통해 고관여 유저 세그먼트로 이동 유도.")

        with t3:
            st.markdown("#### **3. 예방적 관리 및 브랜드 소통**")
            if cancel_rate > 0.5:
                st.error("**[긴급 대응] 이탈 징후 유저 집중 케어**")
                st.write("- **전략**: 해지 시도 시 '1:1 실시간 상담' 혹은 '구독 일시정지' 옵션 즉시 제안.")
            else:
                st.markdown("> **[브랜드] 커뮤니티 전략**")
                st.write("- **전략**: 브랜드 서포터즈 권한 부여 및 서비스 개선 설문 참여 유도.")

    st.markdown("---")
    st.caption("KeepTune v2.6 | Enterprise AI Decision Support System Operating")