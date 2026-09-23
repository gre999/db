# qqq-ml

用機器學習改善 QQQ 規則策略（ORB / VWAP）的進場時機與部位大小。14 週專題。

## 環境（Windows cmd）
```cmd
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python -m pytest
```

## 資料管線
```cmd
:: 1. 從 IBKR 下載原始資料到 data\raw\ibkr\（TWS 需開啟 API）
.venv\Scripts\python scripts\download_ibkr.py --port 7496 --start 2015-01-01

:: 2. 清理並產生 data\processed\*.parquet（有快取；--force 強制重建）
.venv\Scripts\python -m src.data_loader

:: 3. 產生 reports\week01_data_quality.md 與圖表
.venv\Scripts\python scripts\make_week01_report.py
```

下游程式讀資料：
```python
from src.data_loader import load_minute_rth, load_5min, load_daily, load_extended
bars = load_5min(exclude_half_days=True)   # ts = bar 開始；bar_end 之後才可使用
```

## 結構
- `data/raw/`：原始資料，永不修改（不進 git）
- `data/processed/`：清理後的 parquet（不進 git）
- `src/`：`data_loader`、`features`、`validation`、`strategies`、`backtest`、`models/`
- `scripts/`：下載、檢查原始資料、產生報告
- `notebooks/`、`reports/`、`tests/`

## 底線：不可有未來資訊洩漏
詳見 `src/data_loader.py` 開頭說明。
