import streamlit as st

def run_home():
    # 1. 헤더 섹션 (브랜드 아이덴티티 강조)
    st.title("🎧 KeepTune: AI 구독 이탈 방지 솔루션")
    st.markdown("#### **\"데이터로 유저의 리듬을 지키다 (Keep the Tune)\"**")
    st.write("---")

    # 2. 핵심 지표 (KPI) - 디자인 가독성 향상
    st.subheader("📍 프로젝트 핵심 지표 (Key Performance Indicators)")
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.metric(label="분석 로그 규모", value="1,812만 건", delta="Big Data")
    with kpi2:
        st.metric(label="이탈 예측 정확도", value="97.7%", delta="Ensemble")
    with kpi3:
        st.metric(label="최적화 변수", value="24개", delta="Optimized")
    with kpi4:
        st.metric(label="기대 방어 효율", value="25.4%", delta="Target")

    st.write("") # 간격 조절
    
    # 3. 비즈니스 가치 제안 (카드 형태로 구성)
    col1, col2 = st.columns(2)
    with col1:
        st.error("### 📉 기존 문제점")
        st.markdown("""
        - **막대한 유치 비용**: 신규 유치 비용이 유지 비용의 5~7배 수준
        - **사각지대 발생**: 정성적 판단으로는 대규모 유저의 이탈 징후 포착 불가
        - **마케팅 매너리즘**: 타겟팅 없는 무분별한 할인권 발포로 수익성 악화
        """)
    with col2:
        st.success("### 🚀 KeepTune의 가치")
        st.markdown("""
        - **초정밀 타겟팅**: 97% 이상의 정확도로 잠재적 이탈자 선제적 파악
        - **데이터 인사이트**: 청취 패턴 분석을 통한 서비스 고도화 근거 마련
        - **맞춤형 대응**: 유저 등급별 최적의 리텐션(Retention) 시나리오 가이드
        """)

    st.write("---")

    # 4. 분석 아키텍처 (기술적 깊이)
    st.subheader("⚙️ 분석 아키텍처 및 파이프라인")
    
    tab1, tab2 = st.tabs(["📊 데이터 통합", "🧠 하이브리드 지능"])
    
    with tab1:
        st.markdown("""
        - **Log Analytics**: 일평균 청취 시간, 접속 간격 등 유저 행동 패턴 정밀 수집
        - **Transaction Tracking**: 결제 수단 변화, 자동 갱신 해지 등 금융 시그널 추적
        - **Fast Processing**: **Polars**를 활용하여 도커 서버 환경에서도 초고속 연산 처리
        """)

    with tab2:
        st.info("**5-Tier Hybrid Ensemble Model Architecture**")
        st.write("다양한 학습 알고리즘의 강점을 결합하여 예측 편차를 최소화했습니다.")
        st.code("""
        1. LightGBM: 대용량 데이터의 빠른 학습 및 편향 제어
        2. XGBoost: 비선형 패턴의 정교한 분류 성능 극대화
        3. Random Forest: 과적합 방지 및 데이터 노이즈 방어
        4. CatBoost: 범주형 변수(Categorical Features) 무손실 처리
        5. MLP: 딥러닝 기반의 다차원 유저 특징 추출
        """)

    st.write("---")

    # 5. 기대 효과 및 비전
    st.subheader("📈 기대 효과 및 비즈니스 임팩트")
    expander = st.expander("비즈니스 시뮬레이션 상세 결과 보기")
    expander.write("""
    - **이탈 고객 방어**: 고위험군 집중 관리를 통해 연간 약 15~20%의 이탈 방어 성공 기대
    - **LTV(고객생애가치) 극대화**: 우량 고객의 이탈을 막음으로써 장기적 수익 구조 안정화
    - **지능형 마케팅 체계**: 주관적 감(感)이 아닌 '데이터'에 기반한 마케팅 ROI 최적화
    """)

    st.write("---")
    st.caption("KeepTune AI Solution Project 2026")