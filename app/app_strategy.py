import streamlit as st
import plotly.graph_objects as go

def run_strategy():
    # 1. Header setup
    st.title("💡 우리 서비스 단골 고객 지키기 전략")
    st.markdown("##### **XGBoost & ResNet AI 분석으로 찾은 고객 유지(Retention) 가이드**")
    st.markdown("---")

    
    # 3. Resource optimization strategy
    st.subheader("1. 어떤 고객에게 더 집중해야 할까요?")
    st.write("데이터 분석 결과, **'정기 결제'를 안 하거나 '노래 듣는 시간'이 줄어든 고객**을 먼저 챙겨야 합니다.")
    
    col_a, col_b = st.columns([1.5, 1])
    with col_a:
        st.write("**[떠날 확률 높은 고객]** 예산 65% 집중 - 적극적으로 마음 돌리기")
        st.progress(0.65)
        st.caption("👉 노래를 안 들은 지 **10일 이내**에 연락하는 것이 가장 효과적입니다.")
        
        st.write("**[주의가 필요한 고객]** 예산 25% 배분 - 혜택으로 붙잡기")
        st.progress(0.25)
        st.caption("👉 매번 결제하는 분들을 '정기 결제'로 바꾸는 것이 가장 중요합니다.")
        
        st.write("**[안정적인 단골 고객]** 예산 10% 유지 - 고마움 표현하기")
        st.progress(0.1)
    
    with col_b:
        st.markdown("""
        <div style="background-color:#f0f2f6; padding:15px; border-radius:10px;">
        <strong>📋 분석가 총평 (핵심 요약)</strong><br>
        유저가 떠나지 않게 만드는 가장 큰 힘은 <strong>'정기 결제'</strong> 설정이었습니다. 
        결제가 편해질수록 고객들은 우리 서비스를 더 오래 사랑해 주십니다.
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # 4. Action Plan
    st.subheader("2. 고객의 마음을 잡기 위해 바로 해야 할 일")
    
    # [Fix] Removed problematic image text that caused SyntaxError.
    
    with st.expander("📅 지금 당장: 떠나려는 고객 붙잡기", expanded=True):
        st.write("**대상**: 정기 결제를 취소했거나 곧 결제일이 다가오는 분들")
        st.write("**방법**: 다시 정기 결제로 바꾸면 '한 달 무료' 혜택 보내기")
        st.write("**이유**: 정기 결제를 하는 분들이 서비스를 4배 더 오래 씁니다.")
        
    with st.expander("📅 다음 단계: 서비스에 다시 재미를 느끼게 하기"):
        st.write("**대상**: 노래 듣는 시간이 눈에 띄게 줄어든 분들")
        st.write("**방법**: 좋아할 만한 새로운 노래 리스트를 알림으로 보내기")
        st.write("**이유**: 떠나기 전에는 반드시 활동량이 줄어드는 신호가 나타납니다.")
        
    with st.expander("📅 장기 목표: 우리 서비스의 찐팬 만들기"):
        st.write("**대상**: 6개월 넘게 꾸준히 이용해 주시는 고마운 분들")
        st.write("**방법**: 단골 전용 이벤트나 연간 회원권 할인 혜택 드리기")

    # 4. Marketing strategy using models specialized for specific values by business situation
    
    st.subheader("3. 비즈니스 상황별 타겟팅 전략 시뮬레이션")
    fig = go.Figure(data=[
        go.Bar(name='🎯 Precision (정밀도)', 
               x=['🧠 모델 : ResNet', '🌲 모델 : XGBoost'], 
               y=[0.9586, 0.8434], 
               text=['<b>Precision-정밀도</b><br>95.9%', '<b>Precision-정밀도</b><br>84.3%'], # Display text above bars
               textposition='auto',
               marker_color='#5D9CE0'), # Blue color scheme
               
        go.Bar(name='🕸️ Recall (재현율)', 
               x=['🧠 모델 : ResNet', '🌲 모델 : XGBoost'], 
               y=[0.7089, 0.9593], 
               text=['<b>Recall-재현율</b><br>70.9%', '<b>Recall-재현율</b><br>95.9%'], 
               textposition='auto',
               marker_color='#E07A5F')  # Orange color scheme
    ])

    # Graph design (layout) refinement
    fig.update_layout(
        barmode='group', # Place bars side by side
        title_text="분석 모델별 핵심 지표 (Precision vs Recall)",
        height=400,
        plot_bgcolor='rgba(0,0,0,0)', # Transparent background
        yaxis=dict(range=[0, 1.1])    # y-axis range 0~1.1 (with margin)
    )

    # Render chart on Streamlit
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 🧠 모델 : ResNet (정밀도 중심)\n**수익성 극대화 및 방어 비용 최소화**")
        st.markdown("""
        * **Precision (정밀도) : 0.9586 ⭐️**
        * **Recall (재현율)** : 0.7089
        ---
        * 🎯 **성과**: 이탈 예측 64,582명 중 **61,909명이 진짜 이탈자! (타율 95.9%)**
        * 💡 **특징**: 모델이 이탈한다고 지목하면 거의 무조건 맞습니다. 혜택만 받고 떠나는 체리피커를 원천 차단하여 마케팅 예산 낭비를 막습니다. (단, 일부 이탈자는 놓칠 수 있음)
        """)

    with col2:
        st.markdown("#### 🌲 모델 : XGBoost (재현율 중심)\n**시장 점유율 방어 및 락인(Lock-in)**")
        st.markdown("""
        * **Precision (정밀도)** : 0.8434
        * **Recall (재현율) : 0.9593 ⭐️**
         ---
        * 🎯 **성과**: 실제 이탈자 87,330명 중 **83,778명 적중! (3,552명만 놓침)**
        * 💡 **특징**: 놓치는 이탈자가 거의 없습니다. 오예측(15,557명)으로 인한 예산이 조금 더 소진되더라도, 단 한 명의 고객도 뺏기지 않고 방어합니다.
        """)
    
    strategy_type = st.radio(
        "현재 회사의 최우선 목표는 무엇입니까?",
        [" 마켸팅 예산 절감 및 ROI 극대화 (수익성 우선)", "시장 점유율 방어 및 고객 lock in (성장성 우선)"],
        horizontal=False
    )

    if "수익성" in strategy_type:
        st.markdown("🎯 **선택된 전략: High Precision 모델 기반 타겟팅**\n\n이탈 확률 90% 이상의 '확실한 위험군'에게만 고비용(상담원 콜, 한 달 무료권) 혜택을 집중하여 예산 낭비를 막고 ROI를 극대화합니다.")
        col1, col2 = st.columns(2)
        with col1:
            st.info("**💡 왜 이 전략인가요?**\n예산이 한정적일 때, 떠나지 않을 고객(체리피커)에게 혜택을 주는 것은 순손실입니다. 확실한 이탈자만 골라내는 '정밀도(Precision)'를 높여 가성비를 극대화해야 합니다.")
        with col2:
            st.markdown("**🚀 구체적인 실행 액션**\n* VIP 전담 상담원의 1:1 해지 방어 콜\n* 한 달 무료 이용권 즉시 지급\n* 파격적인 반값 할인 오퍼 제공")
    else:
        st.markdown("🕸️ **선택된 전략: High Recall 모델 기반 타겟팅**\n\n이탈 확률 50% 이상의 '잠재적 위험군' 전체에게 저비용(앱 푸시, 소액 쿠폰) 알림을 넓게 배포하여 단 한 명의 고객 이탈도 선제적으로 방어합니다.")
        col1, col2 = st.columns(2)
        with col1:
            st.info("**💡 왜 이 전략인가요?**\n단 한 명의 고객도 뺏겨서는 안 되는 상황입니다. 진짜 이탈자를 놓치지 않는 '재현율(Recall)'을 높여 점유율을 무조건 사수해야 합니다.")
        with col2:
            st.markdown("**🚀 구체적인 실행 액션**\n* 타겟 전체 대상 앱 푸시 알림 발송\n* 맞춤형 플레이리스트 제안\n* 소액 쿠폰 대량 살포")
    st.markdown("---")

    
    
    st.subheader("4. 듀얼 AI 앙상블: '초고위험군' 집중 마케팅 전략")
    st.markdown("<p style='color: #D90429; font-weight: bold;'>*개별 모델의 예측을 넘어, 두 모델이 공통으로 이탈을 예측한 교집합은 절대 놓쳐서는 안 될 최우선 관리 대상입니다.</p>", unsafe_allow_html=True)

    fig_venn = go.Figure()

    # 1. XGBoost circle (left) - larger to match population size (diameter 0.6)
    fig_venn.add_shape(type="circle",
        x0=0.05, y0=0.1, x1=0.65, y1=0.7, # Large circle with width/height of 0.6
        fillcolor="rgba(0, 150, 255, 0.25)", line_color="rgba(0, 120, 220, 0.9)", line_width=2
    )
    
    # 2. ResNet circle (right) - relatively smaller (diameter 0.5)
    fig_venn.add_shape(type="circle",
        x0=0.35, y0=0.15, x1=0.85, y1=0.65, # Smaller than XGBoost circle (width/height 0.5)
        fillcolor="rgba(255, 100, 80, 0.25)", line_color="rgba(220, 80, 60, 0.9)", line_width=2
    )

    # 3. Text placement (precisely adjusted to match circle centers and intersection position)
    # XGBoost only (center of the larger left circle)
    fig_venn.add_annotation(x=0.22, y=0.4, text="<b>34,905명</b><br><span style='font-size:12px'>(XGBoost만)</span>", 
                            showarrow=False, font=dict(size=16, color='#000000'))
    
    # Intersection (exactly at x=0.5 where the two circles overlap)
    fig_venn.add_annotation(x=0.5, y=0.4, text="<b>64,430명</b><br><span style='font-size:14px'>(두 모델 동의)</span>", 
                            showarrow=False, font=dict(size=18, color="#6A1B9A")) 
    
    # ResNet only (center of the right circle)
    fig_venn.add_annotation(x=0.73, y=0.4, text="<b>152명</b><br><span style='font-size:12px'>(ResNet만)</span>", 
                            showarrow=False, font=dict(size=14, color="#000000"))
    
    # Title and info (top center)
    fig_venn.add_annotation(x=0.5, y=0.9, text="<b>이탈 예측 집합 관계 (전체 970,960명)</b>", 
                            showarrow=False, font=dict(size=20, color="#000000"))
    
    fig_venn.add_annotation(x=0.5, y=0.82, text="정상 예측: 871,473명 | 이탈 예측: 99,487명", 
                            showarrow=False, font=dict(size=13, color="#555555"))

    # [🔥 Key] Set height to 600 and yaxis range to [0, 1] to enforce perfect 1:1 aspect ratio
    fig_venn.update_layout(
        xaxis=dict(showgrid=False, zeroline=False, visible=False, range=[0, 1]),
        yaxis=dict(
            showgrid=False, zeroline=False, visible=False, range=[0, 1], # 👈 Same range as x-axis
            scaleanchor="x", # 👈 Force y-axis to follow x-axis ratio (prevents distortion)
            scaleratio=1     
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=600, # 👈 Significantly enlarge the entire diagram
        margin=dict(l=0, r=0, t=10, b=10),
        autosize=True
    )

    st.plotly_chart(fig_venn, use_container_width=True)
    st.markdown("<br>", unsafe_allow_html=True)
    
    c1, c2, c3 = st.columns(3)
    
    with c1:
        st.markdown("""
        <div style="text-align: center;">
            <p style="margin: 0; font-size: 14px; color: #555;">XGBoost 단독</p>
            <p style="margin: 0; font-size: 32px; font-weight: bold; color: #000;">34,905명</p>
            <span style="background-color: #f0f2f6; padding: 2px 8px; border-radius: 10px; font-size: 12px; color: #555;">↑ 잠재 위험군</span>
        </div>
        """, unsafe_allow_html=True)
        
    with c2:
        st.markdown("""
        <div style="text-align: center;">
            <p style="margin: 0; font-size: 14px; color: #555;">🔥두 모델 공통</p>
            <p style="margin: 0; font-size: 32px; font-weight: bold; color: #6A1B9A;">64,430명</p>
            <span style="background-color: #f3e5f5; padding: 2px 8px; border-radius: 10px; font-size: 12px; color: #6A1B9A;">↑ 최우선 타겟</span>
        </div>
        """, unsafe_allow_html=True)
        
    with c3:
        st.markdown("""
        <div style="text-align: center;">
            <p style="margin: 0; font-size: 14px; color: #555;">ResNet 단독</p>
            <p style="margin: 0; font-size: 32px; font-weight: bold; color: #000;">152명</p>
            <span style="background-color: #f0f2f6; padding: 2px 8px; border-radius: 10px; font-size: 12px; color: #555;">↑ 특수 패턴</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br><br>", unsafe_allow_html=True)
    st.warning("💡 **마지막 제언**: 무조건 할인을 해주기보다, 고객이 우리를 잊어갈 때쯤(활동 감소 10일 전후) 딱 맞춰서 말을 거는 '똑똑한 마케팅'이 필요합니다.")
