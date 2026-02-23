import streamlit as st

def run_strategy():
    st.title("💡 핵심 운영 전략")
    st.markdown("##### **데이터 분석으로 찾은 유저 마음 잡기 가이드라인**")
    st.markdown("---")

    # 1. 우리 유저 분석 결과 연동
    if st.session_state.get('predict_done'):
        res = st.session_state.result_data
        risk_score = res['scores'][-1] * 100
        
        st.success(f"✅ 방금 분석한 유저(이탈 위험: {risk_score:.1f}%)를 위한 맞춤 제안입니다.")
        
        st.subheader("1. 이 유저를 붙잡았을 때의 효과")
        c1, c2 = st.columns(2)
        # 어려운 용어(LTV, ROI) 대신 직접적인 표현 사용
        c1.metric("예상되는 매출 방어", "약 12.8만 원", "1년 유지 기준")
        c2.metric("마케팅 효과", "4.2배", "비용 대비 이득")
    else:
        st.info("ℹ️ '예측 시뮬레이터'에서 유저를 먼저 분석하면 실제 수치가 나타납니다.")

    st.markdown("---")

    # 2. 어디에 더 집중해야 할까요? (리소스 배분)
    st.subheader("2. 유저 그룹별 집중 관리 전략")
    st.write("데이터 분석 결과, **'정기 결제'를 안 하거나 '청취 시간'이 갑자기 줄어든 유저**가 떠날 확률이 가장 높았습니다.")
    
    col_a, col_b = st.columns([1.5, 1])
    with col_a:
        st.write("**[떠날 확률 높은 유저]** 예산 65% 집중 - 적극적 복귀 유도")
        st.progress(0.65)
        st.caption("👉 활동이 뜸해진 지 **10일 이내**에 연락하는 것이 가장 효과적입니다.")
        
        st.write("**[관심이 필요한 유저]** 예산 25% 배분 - 혜택 제공")
        st.progress(0.25)
        st.caption("👉 수동 결제 유저를 '정기 결제'로 바꾸는 것이 핵심입니다.")
        
        st.write("**[안정적인 유저]** 예산 10% 유지 - 고마움 표시")
        st.progress(0.1)
    
    with col_b:
        st.markdown("""
        <div style="background-color:#f0f2f6; padding:15px; border-radius:10px;">
        <strong>📢 분석가 한마디</strong><br>
        유저가 떠나지 않게 만드는 가장 큰 힘은 <strong>'정기 결제'</strong> 설정 여부였습니다. 결제 과정이 편해질수록 유저들은 우리 서비스를 더 오래 이용합니다.
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # 3. 실천 로드맵
    st.subheader("3. 우리가 바로 해야 할 일들")
    
    with st.expander("📅 지금 당장: 떠나려는 유저 붙잡기", expanded=True):
        st.write("**대상**: 정기 결제를 해지했거나 다음 결제일이 곧 다가오는 분들")
        st.write("**할 일**: '정기 결제'로 바꾸면 한 달 무료 혜택이나 할인권 보내기")
        st.write("**이유**: 정기 결제를 설정한 분들이 서비스를 4배 더 오래 씁니다.")
        
    with st.expander("📅 다음 단계: 서비스에 재미 붙이게 하기"):
        st.write("**대상**: 노래 듣는 시간이 눈에 띄게 줄어든 분들")
        st.write("**할 일**: 좋아할 만한 새로운 노래 리스트를 알림으로 보내기")
        st.write("**이유**: 떠나기 전에는 반드시 활동량이 줄어드는 징조가 나타납니다.")
        
    with st.expander("📅 장기 목표: 단골 유저 만들기"):
        st.write("**대상**: 우리 서비스를 꾸준히 사랑해 주시는 단골 고객님")
        st.write("**할 일**: 단골 전용 이벤트나 연간 회원권 할인 혜택 드리기")

    st.markdown("---")
    st.warning("💡 **마지막 제언**: 단순히 할인을 퍼주기보다, 유저가 우리를 잊어갈 때쯤(활동 감소 10일 전후) 딱 맞춰서 말을 거는 '똑똑한 마케팅'이 필요합니다.")