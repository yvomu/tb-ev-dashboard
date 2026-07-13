# Table Tennis EWP Dashboard

## 環境安裝

安裝套件：

```bash
pip install -r requirements.txt
```

## 設定 private Hugging Face Dataset

Dashboard 預設從 private Dataset `wobuchida/tb-ewp-dashboard` 讀取 CSV。程式只會下載目前選到的視角、階段與球種組合，下載結果由 `huggingface_hub` 快取。

本機可先登入 Hugging Face：

```bash
hf auth login
```

也可以使用僅有該 Dataset 讀取權限的 token：

```bash
export HF_TOKEN="hf_..."
```

部署到 Streamlit 時，請在部署平台的 Secrets 設定（不要將 token commit 到 Git）：

```toml
HF_TOKEN = "hf_..."
HF_DATASET_REPO = "wobuchida/tb-ewp-dashboard"
```

`HF_DATASET_REPO` 可省略；預設值就是上述 Dataset。若需要固定某個 commit 或 branch，可另設 `HF_DATASET_REVISION`，預設為 `main`。

## 啟動 dashboard

```bash
streamlit run app.py
```

## tb-ev-dashboard

| 檔案/資料夾 | 說明 |
|---|---|
| [tb-ev-dashboard/app.py](/home/ubu/EWR/tb-ev-dashboard/app.py) | Streamlit dashboard |
| [tb-ev-dashboard/requirements.txt](/home/ubu/EWR/tb-ev-dashboard/requirements.txt) | dashboard 套件需求 (也可用EWP的requirement環境)|
| Hugging Face Dataset | dashboard 使用的 conditional CSV |
| `tb-ev-dashboard/fonts/NotoSansCJK-Regular.ttc` | 中文字型 |
| `tb-ev-dashboard/player_id_mapping.csv` | dashboard 用 player mapping 備份 |

## Dashboard 資料命名補充

Hugging Face Dataset repo 根目錄需要放置以下格式的檔案：

```text
{view}_{phase}_conditional_response_table_{feature_type}.csv
```

例子：

```text
global_late_conditional_response_table_action.csv
```

檔名映射：

| dashboard | pipeline |
|---|---|
| `front` | `front3` |
| `last` | `late` |

`view` 為 `global`、`self_player` 或 `opponent`；`feature_type` 為 `action` 或 `action_spin`，因此共有 12 種 conditional CSV。
