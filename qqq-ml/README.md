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

## 口試簡報(PPTX)

```cmd
:: 只重新產生 reports\defense_slides.pptx(含它需要的兩張圖),results_summary.json 數字更新時可以重出
.venv\Scripts\python scripts\run_all.py --slides
```

`scripts\make_defense_slides.py`(14頁,對照 `reports\defense_outline.md`)全部數字從 `reports\results_summary.json` 讀取,不手打;`scripts\make_defense_figures.py` 先補產生兩張既有分析沒存過圖的圖(週8門檻曲線疊隨機篩選區間、週12逐折AUC),兩者都不做新分析。字型同樣用 Noto Sans TC。逐頁視覺檢查(文字有沒有溢出、圖有沒有被切掉、中文/符號有沒有方框)用 LibreOffice 把 pptx 轉成 PDF 看過一次:
```cmd
winget install TheDocumentFoundation.LibreOffice
"C:\Program Files\LibreOffice\program\soffice.exe" --headless --convert-to pdf --outdir reports reports\defense_slides.pptx
```
(這個轉出來的 PDF 只是檢查用的暫存檔,不是交付物,檢查完就刪了,不進 git。)第一版檢查時就抓到一張圖(週14「為什麼難」那張)在投影片底部被切掉——只指定寬度、沒限制高度,圖的長寬比在那個版位下會超出投影片邊界,已改成用高度反推寬度並置中。

## 可重現性

`requirements.txt`/`requirements-dev.txt` 鎖死版本(對照 Python 3.12 的乾淨 venv 實際測過)。`scripts/run_all.py` 從資料清理一路跑到 `reports/final_report.pdf`,第14週在一個全新建立的 venv(不是既有的 `.venv`)、依鎖定版本重新安裝全部套件後,實際跑過一次 `--skip-cnn` 全流程並計時:

| 步驟 | 耗時 |
|---|---|
| 安裝鎖定版本套件(含 CPU 版 torch) | 約 13 分鐘 |
| 完整管線(週1–10 + 週14,跳過CNN重訓) | 約 126 分鐘 |

跑完後核對 `reports/results_summary.json`:除了浮點數在小數點後第13位左右的雜訊(不同執行緒/BLAS加總順序造成,例如 `-0.0025998316877138272` 對 `-0.0025998316877140493`)以外,全部數字一致——四捨五入到報告實際顯示的位數後完全相同,`final_report.md`/`final_report.pdf` 逐字未變。

**已知現象,不是bug**:週1–4的報告腳本(`make_week01_report.py` 等)讀的是**目前**的 `src/features.py`/`src/data_loader.py`,不是「那一週當時」凍結的版本——這幾個模組後來的週次持續加了新特徵欄位、新測試。所以今天重新產生週2報告,會比 git 裡週2當初提交的版本多出後來才加的特徵(`overnight_gap_1d`、`atr_20d` 等)與更多測試筆數。第一次做這個驗證時是事後把這四份報告的意外變動手動 revert 回去;現在 `scripts/run_all.py` 預設**只重建週1–4的 `data/processed/`,不覆蓋這四份報告**(週5以後的管線只需要前者),要重新產生報告本身得另外加 `--rebuild-early-reports` 選項才會動到,不會在照 README 正常重跑時意外改到當週報告。真正代表「可重現性」驗證對象的是 `results_summary.json` 與 `final_report.pdf`,兩者在乾淨環境下確認過對得上。

**CNN 訓練確定性抽驗**:上面的 `--skip-cnn` 全流程只驗證了「用已存的CNN預測檔重算後面的分析,結果一致」,沒有驗證「CNN訓練本身跨環境可重現」。另外在一個乾淨 venv(同樣鎖定版本、含CPU版torch)裡,用 `scripts/verify_cnn_determinism.py --fold 1` 重新訓練任務A第1折的CNN(5個固定種子、確定性模式,跟 `run_week12_cnn_walkforward.py` 同一套程式碼路徑),逐日比對跟 `data/processed/predictions/week12_task_a_cnn.parquet` 已存的機率——**250天全部逐位元相同(max_abs_diff=0.0)**,不是量級接近,是完全一致。確認 CNN 訓練在固定種子、`torch.use_deterministic_algorithms(True)` 下,換一台全新安裝的環境重跑也不會漂移。

## 結構
- `data/raw/`：原始資料，永不修改（不進 git）
- `data/processed/`：清理後的 parquet（不進 git）
- `src/`：`data_loader`、`features`、`validation`、`strategies`、`backtest`、`models/`
- `scripts/`：下載、檢查原始資料、產生報告
- `notebooks/`、`reports/`、`tests/`

## 底線：不可有未來資訊洩漏
詳見 `src/data_loader.py` 開頭說明。
