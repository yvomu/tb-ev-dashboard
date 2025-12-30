import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
# =========================================================
# Page config
# =========================================================
st.set_page_config(
    page_title="Table Tennis Strategy EV Dashboard",
    layout="wide"
)
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK JP']
plt.rcParams['axes.unicode_minus'] = False
# 重新載入所有系統字型
font_manager.fontManager.addfont("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
# =========================================================
# Global constants (防呆核心)
# =========================================================
SERVE_ACTIONS = [15, 16, 17, 18]
NON_SERVE_ACTIONS = list(range(0, 15))

# =========================================================
# Labels
# =========================================================
action_label = {
    0:  "無(Zero)",
    1:  "拉球(Drive)",
    2:  "反拉(Counter)",
    3:  "殺球(Smash)",
    4:  "擰球(Twist)",
    5:  "快帶(Fast drive)",
    6:  "推擠(Fast push)",
    7:  "挑撥(Flip)",
    8:  "拱球(Long push)",
    9:  "磕球(Fast push)",
    10: "搓球(Long push)",
    11: "擺短(Drop shot)",
    12: "削球(Chop)",
    13: "擋球(Block)",
    14: "放高球(Lob)",
    15: "傳統(Traditional serve)",
    16: "勾手(Hook serve)",
    17: "逆旋轉(Reverse serve)",
    18: "下蹲式(Squat serve)",
}

spin_label = {
    0: "無(Zero)",
    1: "上旋(Top)",
    2: "下旋(Back)",
    3: "不旋(No spin)",
    4: "側上旋(Side top)",
    5: "側下旋(Side back)",
}

# =========================================================
# Scenario registry（整個系統的骨架）
# =========================================================
SCENARIOS = {
    "S1": {
        "name": "終局策略（後四拍・含旋轉）",
        "csv": "data/last4_action_spin.csv",
        "serve_only": False,
        "use_spin": True,
    },
    "S2": {
        "name": "終局策略（後四拍・不含旋轉）",
        "csv": "data/last4_action.csv",
        "serve_only": False,
        "use_spin": False,
    },
    "S3": {
        "name": "發球策略（前三拍・含旋轉）",
        "csv": "data/serve3_action_spin.csv",
        "serve_only": True,
        "use_spin": True,
    },
    "S4": {
        "name": "發球策略（前三拍・不含旋轉）",
        "csv": "data/serve3_action.csv",
        "serve_only": True,
        "use_spin": False,
    },
}

# =========================================================
# Utils
# =========================================================
@st.cache_data
def load_csv(path: str):
    return pd.read_csv(path)

def make_c_label(df, use_spin: bool):
    if use_spin:
        return df.apply(
            lambda r: f"{action_label[r.C_actionId]} + {spin_label[r.C_spinId]}",
            axis=1
        )
    else:
        return df["C_actionId"].map(action_label)

def plot_ev_usage(df, x_labels):
    df = df.sort_values("EV", ascending=False).reset_index(drop=True)

    x = np.arange(len(df))
    ev_vals = df["EV"].values
    usage_vals = df["usage_rate"].values

    fig, ax1 = plt.subplots(figsize=(14, 6))

    ax1.bar(x, ev_vals)
    ax1.set_ylabel("Expected Value (EV)")
    ax1.set_ylim(0, 1.05)

    for i, v in enumerate(ev_vals):
        ax1.text(i, v + 0.015, f"{v:.3f}", ha="center", fontsize=9)

    ax2 = ax1.twinx()
    ax2.plot(x, usage_vals, marker="o", color="black")
    ax2.set_ylabel("Usage Rate")

    for i, u in enumerate(usage_vals):
        ax2.text(i, u + max(usage_vals)*0.03, f"{u*100:.1f}%", ha="center", fontsize=9)

    ax1.set_xticks(x)
    ax1.set_xticklabels(x_labels, rotation=45, ha="right")

    plt.tight_layout()
    st.pyplot(fig)

# =========================================================
# Sidebar
# =========================================================
st.sidebar.header("EV 評估視角")

scenario_key = st.sidebar.radio(
    "Scenario",
    list(SCENARIOS.keys()),
    format_func=lambda k: SCENARIOS[k]["name"]
)

cfg = SCENARIOS[scenario_key]
df = load_csv(cfg["csv"])

# Action selector（防呆）
if cfg["serve_only"]:
    A_action_options = SERVE_ACTIONS
else:
    A_action_options = NON_SERVE_ACTIONS

A_action = st.sidebar.selectbox(
    "A_action（先手動作）",
    A_action_options,
    format_func=lambda x: action_label[x]
)

# Spin selector
if cfg["use_spin"]:
    A_spin_options = sorted(
        df[df["A1_actionId"] == A_action]["A1_spinId"].unique()
    )
    A_spin = st.sidebar.selectbox(
        "A_spin（旋轉）",
        A_spin_options,
        format_func=lambda x: spin_label[x]
    )
else:
    A_spin = None

top_k = st.sidebar.slider("Top-K", 3, 15, 5)

# =========================================================
# Data filtering
# =========================================================
if cfg["use_spin"]:
    df_sel = df[
        (df["A1_actionId"] == A_action) &
        (df["A1_spinId"] == A_spin)
    ].copy()
else:
    df_sel = df[
        (df["A1_actionId"] == A_action)
    ].copy()

if df_sel.empty:
    st.warning("此條件下沒有資料")
    st.stop()

df_sel["C_label"] = make_c_label(df_sel, cfg["use_spin"])

# =========================================================
# Main title
# =========================================================
st.title("桌球策略期望值（EV）Dashboard")

spin_text = spin_label[A_spin] if A_spin is not None else "未區分旋轉"

st.markdown(f"""
**Scenario：** {cfg["name"]}  
**A_action：** {action_label[A_action]}  
**A_spin：** {spin_text}
""")

# =========================================================
# Plot
# =========================================================
plot_ev_usage(df_sel, df_sel["C_label"])

# =========================================================
# Top-K table
# =========================================================
st.subheader(f"Top-{top_k} 策略")

cols = ["C_actionId", "EV", "usage_rate", "count"]
if cfg["use_spin"]:
    cols.insert(1, "C_spinId")

topk_df = df_sel.sort_values("EV", ascending=False).head(top_k)[cols].copy()

topk_df["C_action"] = topk_df["C_actionId"].map(action_label)
if cfg["use_spin"]:
    topk_df["C_spin"] = topk_df["C_spinId"].map(spin_label)

topk_df["usage_rate"] = (topk_df["usage_rate"] * 100).round(2)

display_cols = ["C_action"]
if cfg["use_spin"]:
    display_cols.append("C_spin")
display_cols += ["EV", "usage_rate", "count"]

topk_df = topk_df[display_cols].rename(columns={
    "EV": "Expected Value",
    "usage_rate": "Usage Rate (%)",
    "count": "Count"
})

st.dataframe(topk_df, use_container_width=True)
