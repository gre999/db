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

:: 4.（第 2 週）VIX_History.csv、VXN_History.csv 放到 data\raw\cboe\ 後建特徵矩陣
.venv\Scripts\python -m src.features
.venv\Scripts\python scripts\make_week02_report.py

:: 5.（第 3 週）HAR 與 random walk，walk-forward 預測存到 data\processed\predictions\
.venv\Scripts\python -m src.models.har
.venv\Scripts\python scripts\make_week03_report.py

:: 6.（第 4 週）評估切片在 config\evaluation.toml；HAR-X、XGBoost、隨機森林
.venv\Scripts\python -m src.features
.venv\Scripts\python scripts\check_week04_setup.py
.venv\Scripts\python -m src.models.week4
.venv\Scripts\python scripts\make_week04_report.py
```

下游程式讀資料：
```python
from src.data_loader import load_minute_rth, load_5min, load_daily, load_extended
bars = load_5min(exclude_half_days=True)   # ts = bar 開始；bar_end 之後才可使用

X = pd.read_parquet("data/processed/features_prev_close.parquet")  # 前一日收盤可知
from src.validation import WalkForwardSplit, make_pipeline, walk_forward_predict
print(WalkForwardSplit().describe(X.dropna().index))
```

## 期末報告(PDF)

```cmd
:: 一次跑完整個管線(週1到週14)並產生 reports\final_report.pdf
.venv\Scripts\python scripts\run_all.py

:: CNN(週11-13)訓練很慢,可跳過重訓、沿用 data\processed\predictions\ 已存的預測檔
.venv\Scripts\python scripts\run_all.py --skip-cnn

:: 只重新產生期末報告(套用 results_summary.json 最新數字)與 PDF,不重跑任何分析
.venv\Scripts\python scripts\run_all.py --pdf-only
```

PDF 轉換需要事先安裝(僅需一次):
```cmd
winget install JohnMacFarlane.Pandoc
winget install MiKTeX.MiKTeX
```
中文字型用系統內建的 Noto Sans TC(Windows 內附,`C:\Windows\Fonts\NotoSansTC-VF.ttf`)——`reports\final_report_template.md` 的 YAML 開頭已指定 `mainfont`/`CJKmainfont: "Noto Sans TC"`,不需另外安裝字型。`reports\final_report.md` 是範本(`final_report_template.md`)套用 `scripts\make_final_report.py` 從 `reports\results_summary.json` 填入數字後產生的,不要直接編輯 `final_report.md` 本身。

## 可重現性

`requirements.txt`/`requirements-dev.txt` 鎖死版本(對照 Python 3.12 的乾淨 venv 實際測過)。`scripts/run_all.py` 從資料清理一路跑到 `reports/final_report.pdf`,第14週在一個全新建立的 venv(不是既有的 `.venv`)、依鎖定版本重新安裝全部套件後,實際跑過一次 `--skip-cnn` 全流程並計時:

| 步驟 | 耗時 |
|---|---|
| 安裝鎖定版本套件(含 CPU 版 torch) | 約 13 分鐘 |
| 完整管線(週1–10 + 週14,跳過CNN重訓) | 約 126 分鐘 |

跑完後核對 `reports/results_summary.json`:除了浮點數在小數點後第13位左右的雜訊(不同執行緒/BLAS加總順序造成,例如 `-0.0025998316877138272` 對 `-0.0025998316877140493`)以外,全部數字一致——四捨五入到報告實際顯示的位數後完全相同,`final_report.md`/`final_report.pdf` 逐字未變。

**已知現象,不是bug**:週1–4的報告腳本(`make_week01_report.py` 等)讀的是**目前**的 `src/features.py`/`src/data_loader.py`,不是「那一週當時」凍結的版本——這幾個模組後來的週次持續加了新特徵欄位、新測試。所以今天重新產生週2報告,會比 git 裡週2當初提交的版本多出後來才加的特徵(`overnight_gap_1d`、`atr_20d` 等)與更多測試筆數,這是正常的、預期中的行為(這些腳本本來就沒有依週版本化凍結),重新產生**不會**覆蓋 git 裡週2-4當初提交的歷史報告版本——本次驗證時特別檢查過這點,把這四份報告的意外變動 revert 回去,只保留 `requirements*.txt` 鎖版本這個真正要改的地方。真正代表「可重現性」驗證對象的是 `results_summary.json` 與 `final_report.pdf`,兩者在乾淨環境下確認過對得上。

## 結構
- `data/raw/`：原始資料，永不修改（不進 git）
- `data/processed/`：清理後的 parquet（不進 git）
- `src/`：`data_loader`、`features`、`validation`、`strategies`、`backtest`、`models/`
- `scripts/`：下載、檢查原始資料、產生報告
- `notebooks/`、`reports/`、`tests/`

## 底線：不可有未來資訊洩漏
詳見 `src/data_loader.py` 開頭說明。
