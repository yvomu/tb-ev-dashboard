import math
import os
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib import font_manager
from matplotlib.patches import Patch

# =========================================================
# Page config
# =========================================================
st.set_page_config(page_title="Table Tennis Strategy EV Dashboard", layout="wide")

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
DATA_DIR = "data"
PLAYER_MAPPING_PATH = os.path.join(DATA_DIR, "player_id_mapping.csv") if os.path.exists(os.path.join(DATA_DIR, "player_id_mapping.csv")) else "player_id_mapping.csv"

SERVE_ACTIONS = [15, 16, 17, 18]
NON_SERVE_ACTIONS = list(range(0, 15))

legend_elements = [
    Patch(facecolor="#1f77b4", label="高信心 (CI ≤ 0.15)"),
    Patch(facecolor="#8ee6d9", label="中信心 (0.15~0.25)"),
    Patch(facecolor="#ffb703", label="低信心 (0.25~0.40)"),
    Patch(facecolor="#d62828", label="極低信心 (> 0.40)"),
    Patch(facecolor="#9e9e9e", label="沒資料"),
]

action_label = {
    0: "無(Zero)", 1: "拉球(Drive)", 2: "反拉(Counter)", 3: "殺球(Smash)",
    4: "擰球(Twist)", 5: "快帶(Fast drive)", 6: "推擠(Fast push)",
    7: "挑撥(Flip)", 8: "拱球(Long push)", 9: "磕球(Fast push)",
    10: "搓球(Long push)", 11: "擺短(Drop shot)", 12: "削球(Chop)",
    13: "擋球(Block)", 14: "放高球(Lob)",
    15: "傳統(Traditional serve)", 16: "勾手(Hook serve)",
    17: "逆旋轉(Reverse serve)", 18: "下蹲式(Squat serve)",
}
spin_label = {
    0: "無(Zero)", 1: "上旋(Top)", 2: "下旋(Back)",
    3: "不旋(No spin)", 4: "側上旋(Side top)", 5: "側下旋(Side back)",
}

VIEW_OPTIONS = {
    "global": "Global",
    "self_player": "Self Player",
    "opponent": "Opponent",
    "both": "Both",
}

PHASE_OPTIONS = {
    "front": "前三拍",
    "receive": "接發球",
    "last": "相持階段",
}

VARIANT_OPTIONS = {
    "action": {"label": "球種", "use_spin": False},
    "action_spin": {"label": "球種 + 旋轉", "use_spin": True},
}

VIEW_FILE_PREFIXES = {
    "global": ["global"],
    "self_player": ["self_player", "self"],
    "opponent": ["opponent", "oppent"],
    "both": ["both"],
}

PHASE_FILE_ALIASES = {
    "front": ["front", "serve3", "early"],
    "receive": ["receive", "recv", "receiver"],
    "last": ["last", "last4", "late", "rally"],
}

# 欄位顯示設定：依資料有什麼 player 欄位，就動態顯示什麼 filter
PLAYER_FILTER_SPECS = [
    {"column": "A1_playerId", "label": "策略使用者", "key": "self"},
    {"column": "B1_playerId", "label": "對手", "key": "opponent"},
]


# =========================================================
# Utils
# =========================================================
@st.cache_data
def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_data
def load_player_mapping(path: str = PLAYER_MAPPING_PATH) -> Dict[int, dict]:
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
        pid = int(row["player_id"])
        mapping[pid] = {
            "player_name": row["player_name"],
            "rally_count": row.get("rally_count", np.nan),
            "match_count": row.get("match_count", np.nan),
        }
    return mapping


def safe_action_name(x) -> str:
    try:
        return action_label.get(int(x), str(int(x)))
    except Exception:
        return str(x)


def safe_spin_name(x) -> str:
    try:
        return spin_label.get(int(x), str(int(x)))
    except Exception:
        return str(x)


def format_count_value(x) -> str:
    if pd.isna(x):
        return "-"
    try:
        return f"{int(x):,}"
    except Exception:
        return str(x)


def get_player_info(player_id: int, player_info_map: Dict[int, dict]) -> dict:
    player_id = int(player_id)
    info = player_info_map.get(player_id, {})
    return {
        "player_name": info.get("player_name", f"Player {player_id}"),
        "rally_count": info.get("rally_count", np.nan),
        "match_count": info.get("match_count", np.nan),
    }


def player_display(player_id: int, player_info_map: Dict[int, dict]) -> str:
    info = get_player_info(player_id, player_info_map)
    return f"{info['player_name']} (match: {format_count_value(info['match_count'])}, ID: {int(player_id)})"


def make_c_label(row: pd.Series, use_spin: bool) -> str:
    if use_spin and "C_spinId" in row.index:
        return f"{safe_action_name(row.C_actionId)} + {safe_spin_name(row.C_spinId)}"
    return safe_action_name(row.C_actionId)


def wilson_ci(p: float, n: float, z: float = 1.96) -> Tuple[float, float]:
    if pd.isna(p) or pd.isna(n) or n <= 0:
        return np.nan, np.nan
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    margin = z * math.sqrt((p * (1 - p) / n) + (z**2 / (4 * n**2))) / denom
    return center - margin, center + margin


def ci_confidence(row: pd.Series) -> str:
    n = row.get("count", np.nan)
    if pd.isna(n) or n <= 0 or pd.isna(row.get("ci_low", np.nan)) or pd.isna(row.get("ci_high", np.nan)):
        return "沒資料"

    width = row["ci_high"] - row["ci_low"]
    if width > 0.40:
        return "極低"
    if width > 0.25:
        return "低"
    if width > 0.15:
        return "中"
    return "高"


def confidence_color(conf: str) -> str:
    return {
        "高": "#1f77b4",
        "中": "#8ee6d9",
        "低": "#ffb703",
        "極低": "#d62828",
        "沒資料": "#9e9e9e",
    }.get(conf, "#9e9e9e")


def ensure_numeric_columns(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    df = df.copy()
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def normalize_ev_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    rename_map = {}
    if "n_train" in df.columns and "count" not in df.columns:
        rename_map["n_train"] = "count"
    if "use_count" in df.columns and "count" not in rename_map and "count" not in df.columns:
        rename_map["use_count"] = "count"
    if "xgb_pred_mean" in df.columns and "EV" not in df.columns:
        rename_map["xgb_pred_mean"] = "EV"
    if "train_xgb_pred_mean" in df.columns and "EV" not in rename_map and "EV" not in df.columns:
        rename_map["train_xgb_pred_mean"] = "EV"
    if rename_map:
        df = df.rename(columns=rename_map)

    for col in ["EV", "count", "usage_rate", "train_winrate", "usage_share", "win_rate"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def prepare_strategy_df(df: pd.DataFrame, use_spin: bool) -> pd.DataFrame:
    df = normalize_ev_columns(df)
    df = ensure_numeric_columns(df, ["A1_actionId", "A1_spinId", "C_actionId", "C_spinId"])
    df = df.copy()
    df["C_label"] = df.apply(lambda r: make_c_label(r, use_spin), axis=1)
    if "EV" in df.columns and "count" in df.columns:
        df["ci_low"], df["ci_high"] = zip(*df.apply(lambda r: wilson_ci(r["EV"], r["count"]), axis=1))
        df["Strategy_Confidence"] = df.apply(ci_confidence, axis=1)
    else:
        df["ci_low"] = np.nan
        df["ci_high"] = np.nan
        df["Strategy_Confidence"] = "-"
    return df


def list_data_files() -> List[str]:
    if not os.path.exists(DATA_DIR):
        return []
    return sorted(
        [os.path.join(DATA_DIR, fname) for fname in os.listdir(DATA_DIR) if fname.lower().endswith(".csv")]
    )


def find_table_file(view_key: str, phase_key: str, variant_key: str, table_kind: str = "ev_table") -> Optional[str]:
    files = list_data_files()
    if not files:
        return None

    prefixes = VIEW_FILE_PREFIXES.get(view_key, [view_key])
    phase_aliases = PHASE_FILE_ALIASES.get(phase_key, [phase_key])
    variant_tokens = [variant_key]

    kind_aliases = {
        "ev_table": ["ev_table"],
        "player_share": ["player_share", "strategy_player_share", "player_usage"],
    }.get(table_kind, [table_kind])

    candidates: List[Tuple[int, str]] = []
    for path in files:
        basename = os.path.basename(path).lower()
        if not any(token in basename for token in variant_tokens):
            continue
        if not any(kind in basename for kind in kind_aliases):
            continue
        if not any(basename.startswith(prefix + "_") for prefix in prefixes):
            continue
        if not any(alias in basename for alias in phase_aliases):
            continue

        score = 0
        for idx, prefix in enumerate(prefixes):
            if basename.startswith(prefix + "_"):
                score += 100 - idx
        for idx, alias in enumerate(phase_aliases):
            if alias in basename:
                score += 20 - idx
        for idx, kind in enumerate(kind_aliases):
            if kind in basename:
                score += 10 - idx
        if basename.endswith(f"{variant_key}.csv"):
            score += 20
        candidates.append((score, path))

    if not candidates:
        return None
    candidates.sort(key=lambda x: (-x[0], x[1]))
    return candidates[0][1]


def find_data_file(view_key: str, phase_key: str, variant_key: str) -> Optional[str]:
    return find_table_file(view_key, phase_key, variant_key, table_kind="ev_table")


def find_player_share_file(view_key: str, phase_key: str, variant_key: str) -> Optional[str]:
    return find_table_file(view_key, phase_key, variant_key, table_kind="player_share")


def get_available_player_ids(df: pd.DataFrame, column: str) -> List[int]:
    if column not in df.columns:
        return []
    return sorted(df[column].dropna().astype(int).unique().tolist())


def apply_player_filters(df: pd.DataFrame, player_info_map: Dict[int, dict]) -> Tuple[pd.DataFrame, Dict[str, int]]:
    filtered = df.copy()
    selected_players: Dict[str, int] = {}

    for spec in PLAYER_FILTER_SPECS:
        col = spec["column"]
        if col not in filtered.columns:
            continue

        player_ids = get_available_player_ids(filtered, col)
        if not player_ids:
            continue

        selected = st.sidebar.selectbox(
            spec["label"],
            player_ids,
            key=f"player_filter_{col}",
            format_func=lambda x, _map=player_info_map: player_display(x, _map),
        )
        filtered = filtered[filtered[col].astype(int) == int(selected)].copy()
        selected_players[col] = int(selected)

    return filtered, selected_players


def plot_ev_usage(df: pd.DataFrame):
    df = df.sort_values("EV", ascending=False).reset_index(drop=True)
    x = np.arange(len(df))
    colors = df["Strategy_Confidence"].apply(confidence_color)

    fig, ax1 = plt.subplots(figsize=(14, 6))
    ax1.bar(x, df["EV"], color=colors)
    ax1.set_ylim(0, 1.05)
    ax1.set_ylabel("Expected Value (EV)")
    ax1.legend(handles=legend_elements, title="策略估計可信度 (95% CI)", loc="upper right", frameon=True)

    for i, v in enumerate(df["EV"]):
        ax1.text(i, v + 0.015, f"{v:.3f}", ha="center", fontsize=9)

    if "usage_rate" in df.columns and not df["usage_rate"].isna().all():
        ax2 = ax1.twinx()
        ax2.plot(x, df["usage_rate"], color="black", marker="o")
        ax2.set_ylabel("Usage Rate")
        usage_max = max(df["usage_rate"].max(), 1e-6)
        for i, u in enumerate(df["usage_rate"]):
            if pd.notna(u):
                ax2.text(i, u + usage_max * 0.03, f"{u * 100:.1f}%", ha="center", fontsize=9)

    ax1.set_xticks(x)
    ax1.set_xticklabels(df["C_label"], rotation=45, ha="right")
    plt.tight_layout()
    st.pyplot(fig)


def render_no_data(message: str = "沒資料"):
    st.warning(message)


def render_strategy_section(df_sel: pd.DataFrame, use_spin: bool, section_key: str) -> Optional[pd.Series]:
    if df_sel.empty:
        render_no_data("沒資料")
        return None

    df_sel = prepare_strategy_df(df_sel, use_spin)

    required_cols = ["EV", "count", "C_actionId"]
    missing_cols = [col for col in required_cols if col not in df_sel.columns]
    if missing_cols:
        st.error(f"資料缺少必要欄位：{', '.join(missing_cols)}")
        return None

    plot_ev_usage(df_sel)
    st.caption("EV 為策略層級勝率估計值；長條顏色代表估計信心度（依 Wilson 信賴區間寬度；無樣本以灰色顯示）")
    st.markdown("#### 策略估計可信度")

    summary_cols = ["C_label", "EV", "count"]
    if "ci_low" in df_sel.columns and "ci_high" in df_sel.columns:
        df_sel["95% CI"] = df_sel.apply(
            lambda r: "沒資料" if pd.isna(r.ci_low) or pd.isna(r.ci_high) else f"[{r.ci_low:.2f}, {r.ci_high:.2f}]",
            axis=1,
        )
        summary_cols.append("95% CI")
    if "Strategy_Confidence" in df_sel.columns:
        summary_cols.append("Strategy_Confidence")
    if "train_winrate" in df_sel.columns:
        summary_cols.append("train_winrate")
    if "usage_rate" in df_sel.columns:
        summary_cols.append("usage_rate")

    rename_map = {
        "C_label": "策略",
        "count": "樣本數",
        "Strategy_Confidence": "信心度",
        "train_winrate": "Train Win Rate",
        "usage_rate": "Usage Rate",
    }
    summary_df = df_sel[summary_cols].rename(columns=rename_map).sort_values("EV", ascending=False).reset_index(drop=True)
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


def render_detail_card(c_row: pd.Series, use_spin: bool):
    st.markdown("#### 目前條件下的策略摘要")

    detail_df = c_row.to_frame().T.copy()
    if "A1_actionId" in detail_df.columns:
        detail_df["A_action"] = detail_df["A1_actionId"].apply(safe_action_name)
    if "C_actionId" in detail_df.columns:
        detail_df["C_action"] = detail_df["C_actionId"].apply(safe_action_name)
    if use_spin and "A1_spinId" in detail_df.columns:
        detail_df["A_spin"] = detail_df["A1_spinId"].apply(safe_spin_name)
    if use_spin and "C_spinId" in detail_df.columns:
        detail_df["C_spin"] = detail_df["C_spinId"].apply(safe_spin_name)

    ordered_cols = []
    for col in ["A_action", "A_spin", "C_action", "C_spin", "EV", "count", "train_winrate", "usage_rate"]:
        if col in detail_df.columns:
            ordered_cols.append(col)

    rename_map = {
        "A_action": "先手",
        "A_spin": "先手旋轉",
        "C_action": "後續策略",
        "C_spin": "後續旋轉",
        "count": "樣本數",
        "train_winrate": "Train Win Rate",
        "usage_rate": "Usage Rate",
    }
    if ordered_cols:
        st.dataframe(detail_df[ordered_cols].rename(columns=rename_map), width="stretch")


def render_global_player_share(view_key: str, phase_key: str, variant_key: str, A_action: int, A_spin: Optional[int], c_row: pd.Series, player_info_map: Dict[int, dict]):
    if view_key != "global":
        return

    share_path = find_player_share_file(view_key, phase_key, variant_key)
    if share_path is None or not os.path.exists(share_path):
        return

    pdf = load_csv(share_path).copy()
    if pdf.empty:
        return

    pdf = ensure_numeric_columns(
        pdf,
        ["A1_actionId", "A1_spinId", "C_actionId", "C_spinId", "A1_playerId", "use_count", "usage_share", "win_rate"],
    )

    required_cols = ["A1_playerId", "A1_actionId", "C_actionId"]
    if any(col not in pdf.columns for col in required_cols):
        return

    pdf = pdf[pdf["A1_actionId"] == int(A_action)].copy()
    pdf = pdf[pdf["C_actionId"] == int(c_row["C_actionId"])].copy()

    use_spin = VARIANT_OPTIONS[variant_key]["use_spin"]
    if use_spin:
        if "A1_spinId" not in pdf.columns or "C_spinId" not in pdf.columns:
            return
        if A_spin is None or "C_spinId" not in c_row.index or pd.isna(c_row["C_spinId"]):
            return
        pdf = pdf[pdf["A1_spinId"] == int(A_spin)].copy()
        pdf = pdf[pdf["C_spinId"] == int(c_row["C_spinId"])].copy()

    if pdf.empty:
        st.markdown("#### 前 5 高使用率選手（此策略）")
        render_no_data("沒資料")
        return

    sort_col = "usage_share" if "usage_share" in pdf.columns and not pdf["usage_share"].isna().all() else "use_count"
    if sort_col not in pdf.columns:
        return

    top_players = pdf.sort_values(sort_col, ascending=False).head(5).copy()
    top_players["Player"] = top_players["A1_playerId"].apply(lambda pid: get_player_info(int(pid), player_info_map)["player_name"])

    if "use_count" in top_players.columns:
        top_players["Use Count"] = top_players["use_count"].fillna(0).astype(int)
    if "usage_share" in top_players.columns:
        top_players["Usage Share (%)"] = (top_players["usage_share"] * 100).round(2)
    if "win_rate" in top_players.columns:
        top_players["Win Rate (%)"] = (top_players["win_rate"] * 100).round(1)

    display_cols = ["Player"]
    for col in ["Use Count", "Usage Share (%)", "Win Rate (%)"]:
        if col in top_players.columns:
            display_cols.append(col)

    st.markdown("#### 前 5 高使用率選手（此策略）")
    st.dataframe(top_players[display_cols].reset_index(drop=True), width="stretch")


def build_header_markdown(view_key: str, phase_key: str, variant_key: str, selected_players: Dict[str, int], player_info_map: Dict[int, dict], A_action: int, A_spin: Optional[int]) -> str:
    lines = [
        f"**分頁：** {VIEW_OPTIONS[view_key]}",
        f"**Phase：** {PHASE_OPTIONS[phase_key]}",
        f"**策略組合：** {VARIANT_OPTIONS[variant_key]['label']}",
    ]

    for spec in PLAYER_FILTER_SPECS:
        col = spec["column"]
        if col in selected_players:
            info = get_player_info(selected_players[col], player_info_map)
            lines.append(
                f"**{spec['label']}：** {info['player_name']} (ID: {selected_players[col]})  "
                f"Rally Count: {format_count_value(info['rally_count'])} / Match Count: {format_count_value(info['match_count'])}"
            )

    lines.append(f"**A_action：** {safe_action_name(A_action)}")
    spin_text = safe_spin_name(A_spin) if A_spin is not None else "未區分旋轉"
    lines.append(f"**A_spin：** {spin_text}")
    return "  \n".join(lines)


# =========================================================
# Main
# =========================================================
player_name_map = load_player_mapping()

st.sidebar.header("EV 評估視角")
view_key = st.sidebar.radio(
    "分頁",
    list(VIEW_OPTIONS.keys()),
    format_func=lambda k: VIEW_OPTIONS[k],
)
phase_key = st.sidebar.radio(
    "Phase",
    list(PHASE_OPTIONS.keys()),
    format_func=lambda k: PHASE_OPTIONS[k],
)
variant_key = st.sidebar.radio(
    "策略組合",
    list(VARIANT_OPTIONS.keys()),
    format_func=lambda k: VARIANT_OPTIONS[k]["label"],
)

st.title("桌球策略期望值 (EV) Dashboard")

csv_path = find_data_file(view_key, phase_key, variant_key)
if csv_path is None:
    render_no_data("沒資料")
    st.stop()

if not os.path.exists(csv_path):
    render_no_data("沒資料")
    st.stop()

df = load_csv(csv_path)
df = normalize_ev_columns(df)

if df.empty:
    render_no_data("沒資料")
    st.stop()

use_spin = VARIANT_OPTIONS[variant_key]["use_spin"]

# player filters: 只要資料裡有 player 欄位就自動提供篩選
filtered_df, selected_players = apply_player_filters(df, player_name_map)
if filtered_df.empty:
    render_no_data("沒資料")
    st.stop()

if "A1_actionId" not in filtered_df.columns:
    st.error("資料缺少 A1_actionId 欄位")
    st.stop()

filtered_df = ensure_numeric_columns(filtered_df, ["A1_actionId", "A1_spinId", "C_actionId", "C_spinId"])
SERVE_PHASES = {"front"}
phase_is_serve = phase_key in SERVE_PHASES
action_pool = SERVE_ACTIONS if phase_is_serve else NON_SERVE_ACTIONS
available_actions = sorted(set(filtered_df["A1_actionId"].dropna().astype(int)).intersection(action_pool))
if not available_actions:
    render_no_data("沒資料")
    st.stop()

A_action = st.sidebar.selectbox(
    "A_action (先手動作)",
    available_actions,
    format_func=lambda x: safe_action_name(x),
)

if use_spin:
    if "A1_spinId" not in filtered_df.columns:
        st.error("資料缺少 A1_spinId 欄位")
        st.stop()
    spin_candidates = sorted(
        filtered_df.loc[filtered_df["A1_actionId"] == A_action, "A1_spinId"].dropna().astype(int).unique().tolist()
    )
    if not spin_candidates:
        render_no_data("沒資料")
        st.stop()
    A_spin = st.sidebar.selectbox(
        "A_spin (旋轉)",
        spin_candidates,
        format_func=lambda x: safe_spin_name(x),
    )
    df_sel = filtered_df[(filtered_df["A1_actionId"] == A_action) & (filtered_df["A1_spinId"] == A_spin)].copy()
else:
    A_spin = None
    df_sel = filtered_df[filtered_df["A1_actionId"] == A_action].copy()

st.markdown(build_header_markdown(view_key, phase_key, variant_key, selected_players, player_name_map, A_action, A_spin))

if df_sel.empty:
    render_no_data("沒資料")
    st.stop()

c_row = render_strategy_section(df_sel, use_spin, f"{view_key}_{phase_key}_{variant_key}")
if c_row is not None:
    render_detail_card(c_row, use_spin)
    render_global_player_share(view_key, phase_key, variant_key, A_action, A_spin, c_row, player_name_map)
