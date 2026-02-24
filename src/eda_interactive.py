import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import platform
from matplotlib import font_manager, rc
    

plt.rcParams["figure.figsize"] = (10, 4)
plt.rcParams["axes.grid"] = True


TARGET = "is_churn"
def set_korean_font():
    """운영체제별 한글 폰트 설정"""
    try:
        if platform.system() == 'Windows':
            # 윈도우: 맑은 고딕
            rc('font', family='Malgun Gothic')
        elif platform.system() == 'Darwin':
            # 맥: 애플 고딕
            rc('font', family='AppleGothic')
        else:
            # 리눅스(Streamlit Cloud 등): 나눔바른고딕 설치 가정
            # 설치가 안 되어 있다면 시스템 기본 폰트 사용
            plt.rc('font', family='NanumBarunGothic')
        
        # 마이너스 기호 깨짐 방지
        plt.rcParams['axes.unicode_minus'] = False
    except:
        print("한글 폰트 설정에 실패했습니다. 기본 폰트를 사용합니다.")

# =========================
# Plot 함수들 (Streamlit용)
# =========================
def compute_churn_by_category(df: pd.DataFrame, col: str, min_n: int = 100):
    g = (df.groupby(col)[TARGET]
           .agg(churn_rate="mean", n="size")
           .reset_index())
    g["churn_rate"] *= 100
    g = g[g["n"] >= min_n].copy()
    return g

def plot_churn_style_st(df_plot, x_col, title, palette="viridis"):
    # 한글 폰트 재설정
    set_korean_font()
    
    # 그래프 크기 설정
    fig, ax1 = plt.subplots(figsize=(10, 5))
    
    # 1. Bar Chart: 이탈률만 표시
    sns.barplot(
        data=df_plot, 
        x=x_col, 
        y='churn_rate', 
        ax=ax1, 
        palette=palette, 
        alpha=0.8, 
        edgecolor='black'
    )
    
    # 축 라벨 및 제목 설정
    ax1.set_ylabel("이탈률 (Churn Rate)", fontsize=11, fontweight='bold')
    ax1.set_xlabel(x_col, fontsize=11, fontweight='bold')
    ax1.set_title(title, fontsize=14, pad=20, fontweight='bold')
    
    # 수치 표시 (막대 위에 이탈률 퍼센트 표시 - 선택 사항)
    for p in ax1.patches:
        ax1.annotate(f"{p.get_height():.1%}", 
                     (p.get_x() + p.get_width() / 2., p.get_height()), 
                     ha='center', va='center', 
                     fontsize=10, color='black', 
                     xytext=(0, 7), 
                     textcoords='offset points')

    # x축 라벨 회전 및 정렬
    plt.xticks(rotation=45, ha='right')
    
    # 그리드 가독성 조절
    ax1.yaxis.grid(True, linestyle='--', alpha=0.6)
    ax1.set_axisbelow(True) # 그리드를 막대 뒤로
    
    plt.tight_layout()
    return fig

def compute_churn_by_bins_equal_width(df: pd.DataFrame, col: str, width: float = 0.2):
    """
    auto_renew_rate 같은 '0~1 비율' 변수에 추천:
    0.2 단위 등폭 bin
    """
    tmp = df[[col, TARGET]].copy()
    tmp[col] = pd.to_numeric(tmp[col], errors="coerce").fillna(0)

    # 등폭 bin 만들기 (0~1 가정)
    start = 0.0
    end = 1.0
    bins = np.arange(start, end + width, width)
    # 혹시 값이 1.0을 조금 넘는 경우 포함
    tmp["_val_clip"] = tmp[col].clip(lower=start, upper=end)

    tmp["bin"] = pd.cut(tmp["_val_clip"], bins=bins, include_lowest=True, right=True)

    g = (tmp.groupby("bin")[TARGET]
            .agg(churn_rate="mean", n="size")
            .reset_index())
    g["churn_rate"] *= 100
    return g


def compute_churn_by_quantile_bins(df: pd.DataFrame, col: str, q: int = 10):
    tmp = df[[col, TARGET]].copy()
    tmp[col] = pd.to_numeric(tmp[col], errors="coerce").fillna(0)

    # 동일값 많으면 qcut이 bin 개수 줄일 수 있음
    tmp["bin"] = pd.qcut(tmp[col], q=q, duplicates="drop")

    g = (tmp.groupby("bin")[TARGET]
            .agg(churn_rate="mean", n="size")
            .reset_index())
    g["churn_rate"] *= 100
    return g


def plot_churn_by_bins_line_st(g: pd.DataFrame, title: str):
    if g.empty:
        st.warning("표시할 bin 결과가 없어요.")
        return None

    fig, ax = plt.subplots()
    ax.plot(np.arange(len(g)), g["churn_rate"], marker="o")
    ax.set_xticks(np.arange(len(g)))
    ax.set_xticklabels(g["bin"].astype(str), rotation=45, ha="right")
    ax.set_ylabel("Churn rate (%)")
    ax.set_title(title)
    plt.tight_layout()
    return fig


def plot_churn_by_bins_bar_st(g: pd.DataFrame, title: str):
    if g.empty:
        st.warning("표시할 bin 결과가 없어요.")
        return None

    fig, ax = plt.subplots()
    ax.bar(g["bin"].astype(str), g["churn_rate"])
    ax.set_ylabel("Churn rate (%)")
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    return fig
