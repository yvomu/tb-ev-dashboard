import math
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib import font_manager
from matplotlib.patches import Patch

# =========================================================
# Page config
# =========================================================
st.set_page_config(
    page_title="Table Tennis Strategy EV Dashboard",
    layout="wide"
)

# =========================================================
# Font
# =========================================================
FONT_PATH = os.path.join("fonts", "NotoSansCJK-Regular.ttc")
if os.path.exists(FONT_PATH):
    font_manager.fontManager.addfont(FONT_PATH)
    plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP"]
else:
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# =========================================================
# Constants
# =========================================================
SERVE_ACTIONS = [15, 16, 17, 18]
NON_SERVE_ACTIONS = list(range(0, 15))
legend_elements = [
    Patch(facecolor="#1f77b4", label="高信心 (CI ≤ 0.15)"),
    Patch(facecolor="#8ee6d9", label="中信心 (0.15~0.25)"),
    Patch(facecolor="#ffb703", label="低信心 (0.25~0.40)"),
    Patch(facecolor="#d62828", label="極低信心 (> 0.40)"),
]

action_label = {
    0: "無(Zero)", 1: "拉球(Drive)", 2: "反拉(Counter)", 3: "殺球(Smash)",
    4: "擰球(Twist)", 5: "快帶(Fast drive)", 6: "推擠(Fast push)",
    7: "挑撥(Flip)", 8: "拱球(Long push)", 9: "磕球(Fast push)",
    10: "搓球(Long push)", 11: "擺短(Drop shot)", 12: "削球(Chop)",
    13: "擋球(Block)", 14: "放高球(Lob)",
    15: "傳統(Traditional serve)", 16: "勾手(Hook serve)",
    17: "逆旋轉(Reverse serve)", 18: "下蹲式(Squat serve)"
}
spin_label = {
    0: "無(Zero)", 1: "上旋(Top)", 2: "下旋(Back)",
    3: "不旋(No spin)", 4: "側上旋(Side top)", 5: "側下旋(Side back)"
}

# =========================================================
# Scenario registry
# =========================================================
GLOBAL_SCENARIOS = {
    "S1": {
        "name": "發球策略（前三拍・不含旋轉）",
        "csv": "data/serve3_action.csv",
        "serve_only": True,
        "use_spin": False,
        "player": True,
        "player_csv": "data/strategy_player_share_A1C.csv",
    },
    "S2": {
        "name": "發球策略（前三拍・含旋轉）",
        "csv": "data/serve3_action_spin.csv",
        "serve_only": True,
        "use_spin": True,
        "player": True,
        "player_csv": "data/strategy_player_share_A1C_spin.csv",
    },
    "S3": {
        "name": "相持階段策略（最後三拍・不含旋轉）",
        "csv": "data/last4_action.csv",
        "serve_only": False,
        "use_spin": False,
        "player": False,
        
    },
    "S4": {
        
        "name": "相持階段策略（最後三拍・含旋轉）",
        "csv": "data/last4_action_spin.csv",
        "serve_only": False,
        "use_spin": True,
        "player": False,
    },
}

SELF_PLAYER_SCENARIOS = {
    "P1": {
        "name": "發球策略（前三拍・不含旋轉）",
        "csv": "self_front_ev_table_action.csv",
        "serve_only": True,
        "use_spin": False,
    },
    "P2": {
        "name": "發球策略（前三拍・含旋轉）",
        "csv": "self_front_ev_table_action_spin.csv",
        "serve_only": True,
        "use_spin": True,
    },
    "P3": {
        "name": "相持階段策略（最後三拍・不含旋轉）",
        "csv": "self_last_ev_table_action.csv",
        "serve_only": False,
        "use_spin": False,
    },
    "P4": {
        "name": "相持階段策略（最後三拍・含旋轉）",
        "csv": "self_last_ev_table_action_spin.csv",
        "serve_only": False,
        "use_spin": True,
    },
}


# =========================================================
# Utils
# =========================================================
@st.cache_data
def load_csv(path):
    return pd.read_csv(path)


@st.cache_data
def load_player_mapping(path="player_id_mapping.csv"):
    if not os.path.exists(path):
        return {}
    pdf = pd.read_csv(path)
    if "player_id" not in pdf.columns or "player_name" not in pdf.columns:
        return {}
    pdf = pdf.dropna(subset=["player_id", "player_name"]).copy()
    pdf["player_id"] = pdf["player_id"].astype(int)

    if "rally_count" not in pdf.columns:
        pdf["rally_count"] = np.nan
    if "match_count" not in pdf.columns:
        pdf["match_count"] = np.nan

    mapping = {}
    for _, row in pdf.iterrows():
        mapping[int(row["player_id"])] = {
            "player_name": row["player_name"],
            "rally_count": row.get("rally_count", np.nan),
            "match_count": row.get("match_count", np.nan),
        }
    return mapping


def safe_action_name(x):
    try:
        return action_label.get(int(x), str(int(x)))
    except Exception:
        return str(x)


def safe_spin_name(x):
    try:
        return spin_label.get(int(x), str(int(x)))
    except Exception:
        return str(x)


def make_c_label(row, use_spin):
    if use_spin:
        return f"{safe_action_name(row.C_actionId)} + {safe_spin_name(row.C_spinId)}"
    return safe_action_name(row.C_actionId)


def wilson_ci(p, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    margin = z * math.sqrt((p * (1 - p) / n) + (z**2 / (4 * n**2))) / denom
    return center - margin, center + margin


def ci_confidence(row):
    width = row["ci_high"] - row["ci_low"]
    if width > 0.40:
        return "極低"
    elif width > 0.25:
        return "低"
    elif width > 0.15:
        return "中"
    return "高"


def confidence_color(conf):
    return {
        "高": "#1f77b4",
        "中": "#8ee6d9",
        "低": "#ffb703",
        "極低": "#d62828",
    }[conf]


def prepare_strategy_df(df, use_spin):
    df = df.copy()
    df["C_label"] = df.apply(lambda r: make_c_label(r, use_spin), axis=1)
    df["ci_low"], df["ci_high"] = zip(*df.apply(lambda r: wilson_ci(r["EV"], r["count"]), axis=1))
    df["Strategy_Confidence"] = df.apply(ci_confidence, axis=1)
    return df


def plot_ev_usage(df):
    df = df.sort_values("EV", ascending=False).reset_index(drop=True)
    x = np.arange(len(df))
    colors = df["Strategy_Confidence"].apply(confidence_color)

    fig, ax1 = plt.subplots(figsize=(14, 6))
    ax1.bar(x, df["EV"], color=colors)
    ax1.set_ylim(0, 1.05)
    ax1.set_ylabel("Expected Value (EV)")
    ax1.legend(
        handles=legend_elements,
        title="策略估計可信度 (95% CI)",
        loc="upper right",
        frameon=True,
    )

    for i, v in enumerate(df["EV"]):
        ax1.text(i, v + 0.015, f"{v:.3f}", ha="center", fontsize=9)

    ax2 = ax1.twinx()
    ax2.plot(x, df["usage_rate"], color="black", marker="o")
    ax2.set_ylabel("Usage Rate")
    usage_max = max(df["usage_rate"].max(), 1e-6)
    for i, u in enumerate(df["usage_rate"]):
        ax2.text(i, u + usage_max * 0.03, f"{u * 100:.1f}%", ha="center", fontsize=9)

    ax1.set_xticks(x)
    ax1.set_xticklabels(df["C_label"], rotation=45, ha="right")
    plt.tight_layout()
    st.pyplot(fig)


def render_strategy_section(df_sel, use_spin, section_key):
    if df_sel.empty:
        st.warning("此條件下沒有資料")
        return None

    df_sel = prepare_strategy_df(df_sel, use_spin)

    plot_ev_usage(df_sel)
    st.caption("EV 為策略層級勝率估計值；長條顏色代表估計信心度（依 Wilson 信賴區間寬度）")
    st.markdown("#### 策略估計可信度")

    summary_df = df_sel.copy()
    summary_df["95% CI"] = summary_df.apply(lambda r: f"[{r.ci_low:.2f}, {r.ci_high:.2f}]", axis=1)
    summary_df = summary_df[["C_label", "EV", "count", "95% CI", "Strategy_Confidence"]].rename(columns={
        "C_label": "策略",
        "count": "樣本數",
        "Strategy_Confidence": "信心度",
    })
    summary_df = summary_df.sort_values("EV", ascending=False).reset_index(drop=True)
    st.dataframe(summary_df, width="stretch")

    st.markdown("### 選擇欲分析的後續策略")
    df_sorted = df_sel.sort_values("EV", ascending=False).reset_index(drop=True)
    idx = st.selectbox(
        "C_action",
        range(len(df_sorted)),
        key=f"c_action_{section_key}",
        format_func=lambda i: df_sorted.loc[i, "C_label"],
    )
    return df_sorted.loc[idx]


def phase_text_from_csv(path: str):
    lower = os.path.basename(path).lower()
    if "front" in lower:
        return "前三拍"
    if "last" in lower:
        return "最後三拍"
    return "未指定"


def format_count_value(x):
    if pd.isna(x):
        return "-"
    try:
        return f"{int(x):,}"
    except Exception:
        return str(x)


def get_player_info(player_id, player_info_map):
    player_id = int(player_id)
    default_name = f"Player {player_id}"
    info = player_info_map.get(player_id, {})
    return {
        "player_name": info.get("player_name", default_name),
        "rally_count": info.get("rally_count", np.nan),
        "match_count": info.get("match_count", np.nan),
    }


def player_display(player_id, player_info_map):
    player_id = int(player_id)
    info = get_player_info(player_id, player_info_map)
    return f"{info['player_name']}  (match：{format_count_value(info['match_count'])} ,  ID: {player_id})"


player_name_map = load_player_mapping()

# =========================================================
# Sidebar / Main
# =========================================================
st.sidebar.header("EV 評估視角")
view_mode = st.sidebar.radio("分頁", ["Global", "策略使用者視角"])

st.title("桌球策略期望值 (EV) Dashboard")

# =========================================================
# Global
# =========================================================
if view_mode == "Global":
    scenario_key = st.sidebar.radio(
        "Scenario",
        list(GLOBAL_SCENARIOS.keys()),
        format_func=lambda k: GLOBAL_SCENARIOS[k]["name"],
    )
    cfg = GLOBAL_SCENARIOS[scenario_key]
    df = load_csv(cfg["csv"])

    A_action_options = SERVE_ACTIONS if cfg["serve_only"] else NON_SERVE_ACTIONS
    A_action = st.sidebar.selectbox(
        "A_action (先手動作)",
        A_action_options,
        format_func=lambda x: safe_action_name(x),
    )

    if cfg["use_spin"]:
        spin_candidates = sorted(df[df.A1_actionId == A_action]["A1_spinId"].dropna().unique())
        A_spin = st.sidebar.selectbox(
            "A_spin (旋轉)",
            spin_candidates,
            format_func=lambda x: safe_spin_name(x),
        )
        df_sel = df[(df.A1_actionId == A_action) & (df.A1_spinId == A_spin)].copy()
    else:
        A_spin = None
        df_sel = df[df.A1_actionId == A_action].copy()

    spin_text = safe_spin_name(A_spin) if A_spin is not None else "未區分旋轉"
    st.markdown(f"""
**分頁：** Global  
**Scenario：** {cfg['name']}  
**A_action：** {safe_action_name(A_action)}  
**A_spin：** {spin_text}
""")

    c_row = render_strategy_section(df_sel, cfg["use_spin"], "global")

    if c_row is not None:
        st.markdown("#### 前 5 高使用率選手（此策略）")
        if not cfg.get("player", False):
            st.info("此視角未提供選手行為分析")
        else:
            pdf = load_csv(cfg["player_csv"])
            if cfg["use_spin"]:
                pdf = pdf[
                    (pdf.A1_actionId == A_action)
                    & (pdf.A1_spinId == A_spin)
                    & (pdf.C_actionId == c_row.C_actionId)
                    & (pdf.C_spinId == c_row.C_spinId)
                ]
            else:
                pdf = pdf[(pdf.A1_actionId == A_action) & (pdf.C_actionId == c_row.C_actionId)]

            if pdf.empty:
                st.warning("此策略在選手層級樣本不足")
            else:
                top_players = pdf.sort_values("usage_share", ascending=False).head(5).copy()
                top_players["usage_share"] = (top_players["usage_share"] * 100).round(2)
                top_players["win_rate"] = (top_players["win_rate"] * 100).round(1)
                st.dataframe(
                    top_players.rename(columns={
                        "A1_playerId": "Player",
                        "use_count": "Use Count",
                        "usage_share": "Usage Share (%)",
                        "win_rate": "Win Rate (%)",
                    })[["Player", "Use Count", "Usage Share (%)", "Win Rate (%)"]],
                    width="stretch",
                )

# =========================================================
# Self Player
# =========================================================
else:
    scenario_key = st.sidebar.radio(
        "Scenario",
        list(SELF_PLAYER_SCENARIOS.keys()),
        format_func=lambda k: SELF_PLAYER_SCENARIOS[k]["name"],
    )
    cfg = SELF_PLAYER_SCENARIOS[scenario_key]

    if not os.path.exists(cfg["csv"]):
        st.error(f"找不到檔案：{cfg['csv']}")
        st.stop()

    df = load_csv(cfg["csv"])

    if "A1_playerId" not in df.columns:
        st.error("self_player csv 缺少 A1_playerId 欄位")
        st.stop()

    player_ids = sorted(df["A1_playerId"].dropna().astype(int).unique())
    selected_player = st.sidebar.selectbox(
        "選手",
        player_ids,
        format_func=lambda x: player_display(x, player_name_map),
    )

    df_player = df[df["A1_playerId"].astype(int) == int(selected_player)].copy()
    if df_player.empty:
        st.warning("此選手沒有資料")
        st.stop()

    A_action_options = SERVE_ACTIONS if cfg["serve_only"] else NON_SERVE_ACTIONS
    available_actions = sorted(set(df_player["A1_actionId"].dropna().astype(int)).intersection(A_action_options))
    A_action = st.sidebar.selectbox(
        "A_action (先手動作)",
        available_actions,
        format_func=lambda x: safe_action_name(x),
    )

    if cfg["use_spin"]:
        spin_candidates = sorted(df_player[df_player.A1_actionId == A_action]["A1_spinId"].dropna().unique())
        A_spin = st.sidebar.selectbox(
            "A_spin (旋轉)",
            spin_candidates,
            format_func=lambda x: safe_spin_name(x),
        )
        df_sel = df_player[(df_player.A1_actionId == A_action) & (df_player.A1_spinId == A_spin)].copy()
    else:
        A_spin = None
        df_sel = df_player[df_player.A1_actionId == A_action].copy()

    spin_text = safe_spin_name(A_spin) if A_spin is not None else "未區分旋轉"
    player_info = get_player_info(selected_player, player_name_map)
    player_name = player_info["player_name"]
    #phase_text = phase_text_from_csv(cfg["csv"])

    st.markdown(f"""
**分頁：** Self Player  
**Scenario：** {cfg['name']}  
**選手：** {player_name} (ID: {int(selected_player)})  
&nbsp;&nbsp;&nbsp;&nbsp;Rally Count：{format_count_value(player_info['rally_count'])}  
&nbsp;&nbsp;&nbsp;&nbsp;比賽場次數量：{format_count_value(player_info['match_count'])}  
**A_action：** {safe_action_name(A_action)}  
**A_spin：** {spin_text}
""")

    c_row = render_strategy_section(df_sel, cfg["use_spin"], "self_player")

    if c_row is not None:
        st.markdown("#### 此選手在目前條件下的策略摘要")
        detail_cols = ["EV", "count"]
        for col in ["train_winrate", "usage_rate"]:
            if col in c_row.index:
                detail_cols.append(col)

        if cfg["use_spin"]:
            head_cols = ["A1_actionId", "A1_spinId", "C_actionId", "C_spinId"]
        else:
            head_cols = ["A1_actionId", "C_actionId"]

        detail_df = c_row[head_cols + detail_cols].to_frame().T.copy()
        detail_df["A_action"] = detail_df["A1_actionId"].apply(safe_action_name)
        detail_df["C_action"] = detail_df["C_actionId"].apply(safe_action_name)
        if cfg["use_spin"]:
            detail_df["A_spin"] = detail_df["A1_spinId"].apply(safe_spin_name)
            detail_df["C_spin"] = detail_df["C_spinId"].apply(safe_spin_name)

        ordered_cols = ["A_action"]
        if cfg["use_spin"]:
            ordered_cols.append("A_spin")
        ordered_cols.append("C_action")
        if cfg["use_spin"]:
            ordered_cols.append("C_spin")
        ordered_cols += [c for c in ["EV", "count", "train_winrate", "usage_rate"] if c in detail_df.columns]

        rename_map = {
            "A_action": "先手",
            "A_spin": "先手旋轉",
            "C_action": "後續策略",
            "C_spin": "後續旋轉",
            "count": "樣本數",
            "train_winrate": "Train Win Rate",
            "usage_rate": "Usage Rate",
        }
        st.dataframe(detail_df[ordered_cols].rename(columns=rename_map), width="stretch")
