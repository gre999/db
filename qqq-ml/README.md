# qqq-ml

用機器學習改善 QQQ 規則策略（ORB / VWAP）的進場時機與部位大小。14 週專題。

## 環境
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest
```

## 結構
- `data/raw/`：原始資料，永不修改（不進 git）
- `data/processed/`：清理後的 parquet（不進 git）
- `src/`：`data_loader`、`features`、`validation`、`strategies`、`backtest`、`models/`
- `notebooks/`、`reports/`、`tests/`
