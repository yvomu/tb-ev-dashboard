import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
import os

# =========================================================
# Page config
# =========================================================
st.set_page_config(
    page_title="Table Tennis Strategy EV Dashboard",
    layout="wide"
)

# =========================================================
# Font (cloud-safe)
# =========================================================
FONT_PATH = os.path.join("fonts", "NotoSansCJK-Regular.ttc")
if os.path.exists(FONT_PATH):
    font_manager.fontManager.addfont(FONT_PATH)
    plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP"]
else:
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# =========================================================
# Global constants
# =========================================================
SERVE_ACTIONS = [15, 16, 17, 18]
NON_SERVE_ACTIONS = list(range(0, 15))

# =========================================================
# Labels
# =========================================================
action_label = {
    0:"無(Zero)",1:"拉球(Drive)",2:"反拉(Counter)",3:"殺球(Smash)",
    4:"擰球(Twist)",5:"快帶(Fast drive)",6:"推擠(Fast push)",
    7:"挑撥(Flip)",8:"拱球(Long push)",9:"磕球(Fast push)",
    10:"搓球(Long push)",11:"擺短(Drop shot)",12:"削球(Chop)",
    13:"擋球(Block)",14:"放高球(Lob)",
    15:"傳統(Traditional serve)",16:"勾手(Hook serve)",
    17:"逆旋轉(Reverse serve)",18:"下蹲式(Squat serve)"
}

spin_label = {
    0:"無(Zero)",1:"上旋(Top)",2:"下旋(Back)",
    3:"不旋(No spin)",4:"側上旋(Side top)",5:"側下旋(Side back)"
}

# =========================================================
# Scenario registry
# =========================================================
SCENARIOS = {
    "S1": {
        "name": "終局策略（後四拍・含旋轉）",
        "csv": "data/last4_action_spin.csv",
        "serve_only": False,
        "use_spin": True,
        "player": False,
    },
    "S2": {
        "name": "終局策略（後四拍・不含旋轉）",
        "csv": "data/last4_action.csv",
        "serve_only": False,
        "use_spin": False,
        "player": False,
    },
    "S3": {
        "name": "發球策略（前三拍・含旋轉）",
        "csv": "data/serve3_action_spin.csv",
        "serve_only": True,
        "use_spin": True,
        "player": True,
        "player_csv": "data/strategy_player_share_A1C_spin.csv",
    },
    "S4": {
        "name": "發球策略（前三拍・不含旋轉）",
        "csv": "data/serve3_action.csv",
        "serve_only": True,
        "use_spin": False,
        "player": True,
        "player_csv": "data/strategy_player_share_A1C.csv",
    },
}
player_map_df = pd.read_csv("data/player_id_mapping.csv")

# =========================================================
# Utils
# =========================================================
@st.cache_data
def load_csv(path):
    return pd.read_csv(path)

def make_c_label(row, use_spin):
    if use_spin:
        return f"{action_label[row.C_actionId]} + {spin_label[row.C_spinId]}"
    else:
        return action_label[row.C_actionId]

def plot_ev_usage(df):
    df = df.sort_values("EV", ascending=False).reset_index(drop=True)
    x = np.arange(len(df))

    fig, ax1 = plt.subplots(figsize=(14,6))
    ax1.bar(x, df["EV"])
    ax1.set_ylim(0,1.05)
    ax1.set_ylabel("Expected Value (EV)")

    for i,v in enumerate(df["EV"]):
        ax1.text(i, v+0.015, f"{v:.3f}", ha="center", fontsize=9)

    ax2 = ax1.twinx()
    ax2.plot(x, df["usage_rate"], color="black", marker="o")
    ax2.set_ylabel("Usage Rate")

    for i,u in enumerate(df["usage_rate"]):
        ax2.text(i, u+max(df["usage_rate"])*0.03, f"{u*100:.1f}%", ha="center", fontsize=9)

    ax1.set_xticks(x)
    ax1.set_xticklabels(df["C_label"], rotation=45, ha="right")
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

A_action_options = SERVE_ACTIONS if cfg["serve_only"] else NON_SERVE_ACTIONS
A_action = st.sidebar.selectbox(
    "A_action（先手動作）",
    A_action_options,
    format_func=lambda x: action_label[x]
)

if cfg["use_spin"]:
    A_spin = st.sidebar.selectbox(
        "A_spin（旋轉）",
        sorted(df[df.A1_actionId==A_action]["A1_spinId"].unique()),
        format_func=lambda x: spin_label[x]
    )
else:
    A_spin = None

# =========================================================
# Filter EV data
# =========================================================
if cfg["use_spin"]:
    df_sel = df[(df.A1_actionId==A_action)&(df.A1_spinId==A_spin)].copy()
else:
    df_sel = df[df.A1_actionId==A_action].copy()

if df_sel.empty:
    st.warning("此條件下沒有資料")
    st.stop()

df_sel["C_label"] = df_sel.apply(
    lambda r: make_c_label(r, cfg["use_spin"]), axis=1
)

# =========================================================
# Main
# =========================================================
st.title("桌球策略期望值（EV）Dashboard")

spin_text = spin_label[A_spin] if A_spin is not None else "未區分旋轉"
st.markdown(f"""
**Scenario：** {cfg["name"]}  
**A_action：** {action_label[A_action]}  
**A_spin：** {spin_text}
""")

# EV Plot
plot_ev_usage(df_sel)

# =========================================================
# Select C_action for player analysis
# =========================================================
st.markdown("### 選擇欲分析的後續策略（C_action）")

df_sorted = df_sel.sort_values("EV", ascending=False).reset_index(drop=True)
c_idx = st.selectbox(
    "C_action",
    range(len(df_sorted)),
    format_func=lambda i: df_sorted.loc[i,"C_label"]
)
c_row = df_sorted.loc[c_idx]

# =========================================================
# Player Top-5 table
# =========================================================
st.subheader("此策略前 5 高使用率選手")

if not cfg.get("player", False):
    st.info("此視角未提供選手行為分析")
else:
    pdf = load_csv(cfg["player_csv"])
    pdf = pdf.merge(
    player_map_df,
    left_on="A1_playerId",
    right_on="player_id",
    how="left"    )

    if cfg["use_spin"]:
        pdf = pdf[
            (pdf.A1_actionId==A_action)&
            (pdf.A1_spinId==A_spin)&
            (pdf.C_actionId==c_row.C_actionId)&
            (pdf.C_spinId==c_row.C_spinId)
        ]
    else:
        pdf = pdf[
            (pdf.A1_actionId==A_action)&
            (pdf.C_actionId==c_row.C_actionId)
        ]

    if pdf.empty:
        st.warning("此策略在選手層級樣本不足")
    else:
        top_players = (
            pdf.sort_values("usage_share", ascending=False)
               .head(5)
               .copy()
        )
        top_players["usage_share"] = (top_players["usage_share"]*100).round(2)
        top_players["win_rate"] = (top_players["win_rate"]*100).round(1)

        st.dataframe(
            top_players.rename(columns={
                "player_name":"Player",
                "use_count":"Use Count",
                "usage_share":"Usage Rate (%)",
                "win_rate":"Win Rate (%)"
            })[
                ["Player","Use Count","Usage Rate (%)","Win Rate (%)"]
            ],
            use_container_width=True
        )
