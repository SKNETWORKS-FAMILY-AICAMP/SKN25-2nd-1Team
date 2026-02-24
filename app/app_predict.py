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
            
            # 청취 시간 슬라이더 및 Status (은은한 캡션)
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
            # 서비스 이탈 징후 슬라이더 및 유저 심리 (은은한 캡션)
            cancel_rate = st.slider(
                "⚠️ 서비스 이탈 징후 (과거 해지 시도 확률)", 
                0.0, 1.0, 0.1, step=0.01,
                help="과거 해지 페이지 방문 흔적 등을 수치화한 지표입니다."
            )
            if cancel_rate < 0.2:
                st.caption("😇 **유저 심리: 평온** - 현재 서비스에 만족하며 안정적으로 이용 중입니다.")
            elif cancel_rate < 0.6:
                st.caption("🤔 **유저 심리: 번민** - 타사 서비스와 혜택 및 이용 경험을 저울질 중입니다.")
            else:
                st.caption("🚨 **유저 심리: 위험** - 해지 버튼 클릭 직전의 심리적 이탈 임계점입니다.")

            # 누적 결제 횟수 슬라이더 및 Stage (은은한 캡션)
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
        'is_auto_renew': auto_renew, 'total_secs_mean': total_secs, 'is_cancel': 1.0 if cancel_rate > 0.5 else 0.0,
        'payment_plan_days': 30.0, 'txn_cnt': float(txn_cnt),
        'total_paid': float(txn_cnt) * 30.0, 'total_secs_sum': total_secs * float(txn_cnt),
        'auto_renew_rate': auto_renew, 'cancel_rate': cancel_rate,
    }

    if st.button("🚀 AI 하이브리드 전략 진단 시작", use_container_width=True, type="primary"):
        try:
            with st.spinner('하이브리드 엔진이 엔터프라이즈급 전략을 수립 중입니다...'):
                p_xgb, p_resnet, _ = predict_churn(input_data)
                
                # 데이터 숙련도에 따른 가중치 최적화
                w_xgb, w_resnet = (0.7, 0.3) if txn_cnt >= 5 else (0.3, 0.7)
                final_score = (p_xgb * w_xgb) + (p_resnet * w_resnet)
                
                st.session_state.result_data = {
                    'p_xgb': float(p_xgb), 'p_resnet': float(p_resnet),
                    'final_score': float(final_score), 'scores': [float(final_score)],
                    'w_xgb': w_xgb, 'w_resnet': w_resnet,
                }
                st.session_state.predict_done = True
                st.toast("✅ 전략적 진단 리포트 도출이 완료되었습니다!")
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
        m1.metric("최종 이탈 확률", f"{risk_score:.1f}%", delta="주의" if risk_score > 50 else "정상", delta_color="inverse")
        m2.metric("통계 기반 위험도 (XGB)", f"{res['p_xgb']*100:.1f}%")
        m3.metric("패턴 기반 위험도 (ResNet)", f"{res['p_resnet']*100:.1f}%")

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

        # 6. AI 전략 실행 계획 (기업 맞춤형 제안)
        st.markdown("---")
        st.subheader("🛠️ AI 전략 실행 계획 (기업 맞춤형 리포트)")
        st.caption("**데이터 분석 결과를 바탕으로 도출된 핵심 비즈니스 액션 플랜입니다.**")

        t1, t2, t3 = st.tabs(["💰 수익성 및 성장 전략", "🎨 제품 경험 및 활성화", "🤝 고객 소통 및 관리"])

        with t1:
            st.markdown("#### **1. 매출 최적화 및 결제 유지 전략**")
            if auto_renew == 0:
                st.markdown("> **[핵심 과제] 자동 결제 전환 락인(Lock-in) 캠페인**")
                st.write("- **현황**: 매 결제 주기마다 이탈을 결정하는 피로도가 높은 상태입니다.")
                st.write("- **전략**: '자동 결제 전환 시 첫 달 90% 페이백' 등 공격적 혜택으로 결제 지속성을 확보하세요.")
            else:
                st.markdown("> **[핵심 과제] 고부가가치 모델 업셀링(Up-selling)**")
                st.write("- **전략**: 현재 월 결제 유저를 연간 구독권으로 유도하여 고객생애가치(LTV)를 극대화하세요.")
            
            if txn_cnt > 30:
                st.markdown("> **[장기 고객] 서비스 권태기(Maturity) 방어 전략**")
                st.write("- **전략**: 누적 결제 50회 기념 한정판 굿즈 증정 등 브랜드 애착심을 고취하는 로열티 프로그램을 가동하세요.")

        with t2:
            st.markdown("#### **2. 사용자 활동성 및 경험 최적화**")
            if total_mins < 30:
                st.warning("> **[활성화] 실시간 맥락(Context) 기반 푸시 전략**")
                st.write("- **전략**: 유저의 과거 청취 선호도를 분석하여 활동 가능 시간대에 맞춰 타겟팅 알림을 발송하세요.")
            elif total_mins > 180:
                st.success("> **[심화] 프리미엄 콘텐츠 리드 생성**")
                st.write("- **전략**: 고관여 유저에게 고음질(Hi-Fi) 스트리밍 무료 체험을 제공하여 수익 구조를 다변화하세요.")
            else:
                st.markdown("> **[유지] 큐레이션 만족도 강화**")
                st.write("- **전략**: 개인화된 플레이리스트 노출 빈도를 높여 체류 시간을 점진적으로 확대하세요.")

        with t3:
            st.markdown("#### **3. 예방적 관리 및 브랜드 소통 전략**")
            if cancel_rate > 0.5:
                st.error("**[긴급 대응] 해지 시도 고빈도 유저 집중 케어**")
                st.write("- **전략**: 해지 시도 시 '1:1 실시간 상담' 또는 즉시 사용 가능한 '구독 일시정지' 옵션을 최우선 제안하세요.")
            else:
                st.success("**[브랜드] 앰배서더 및 커뮤니티 전략**")
                st.write("- **전략**: 브랜드 커뮤니티 초대권 및 서포터즈 권한을 부여하여 긍정적인 브랜드 경험을 확산시키세요.")

        # --- 7. 비즈니스 전략 페이지 연동 안내 (추가됨) ---
        st.markdown("---")
        st.info("""
        💡 **더 알아보기**
        
        회사 전체 고객 세그먼트별 전략 및 ROI 예상치는 **'비즈니스 전략' 페이지**에서 확인할 수 있습니다.
        - 어떤 고객군에 예산을 어떻게 배분할지 (Budget Allocation)
        - 회사 차원의 기대 효과 및 수익성 전망 (Profitability Projection)
        - 분기별 실행 로드맵 및 핵심 성과 지표 (KPIs)
        """)

    st.markdown("---")
    st.caption("KeepTune v2.6 | Enterprise AI Strategic Consulting Support")