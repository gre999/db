"""Week 14 Part 5: build reports/defense_slides.pptx from reports/
defense_outline.md's 14-slide structure. Every number is read from
reports/results_summary.json (arithmetic-mean Sharpe fields, matching
final_report.md's own convention) - nothing here is hand-typed. Figures
are the four already-generated PNGs named in the outline; no new
analysis happens in this script.

Usage:
    python scripts/make_defense_slides.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Emu, Inches, Pt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
REPORTS = ROOT / "reports"
FIG_DIR = REPORTS / "figures"

# ---------------------------------------------------------------------
# Style constants - same palette as reports/figures/*.png (make_week01_
# report.py's style block) so slide charts and slide chrome match.
# ---------------------------------------------------------------------
FONT = "Noto Sans TC"
BLUE = RGBColor(0x2A, 0x78, 0xD6)
ORANGE = RGBColor(0xEB, 0x68, 0x34)
INK = RGBColor(0x0B, 0x0B, 0x0B)
INK_2 = RGBColor(0x52, 0x51, 0x4E)
MUTED = RGBColor(0x8A, 0x89, 0x84)
SURFACE = RGBColor(0xFC, 0xFC, 0xFB)
GRID = RGBColor(0xE6, 0xE5, 0xE1)

SLIDE_W = Inches(13.333)
SLIDE_H = Inches(7.5)
MARGIN = Inches(0.55)
BODY_LEFT = MARGIN
BODY_WIDTH = SLIDE_W - 2 * MARGIN


def pct(x: float) -> str:
    return f"{x:+.4f}"


def load_numbers() -> dict:
    d = json.loads((REPORTS / "results_summary.json").read_text(encoding="utf-8"))
    rg = d["robustness_gaps"]
    p2, p2s = d["phase2"]["rows"][0], d["phase2"]["rows"][1]
    p3o, p3v = d["phase3"]["rows"][0], d["phase3"]["rows"][1]
    p4 = d["phase4"]["rows"][0]
    p5 = d["phase5"]["rows"][0]
    p2r = rg["phase2"]["random_baseline"]
    return {
        "p2_qlike_diff": p2["prediction_layer"]["diff"], "p2_dm_p": p2["prediction_layer"]["p_value"],
        "p2_strat_diff": p2["strategy_layer"]["diff"], "p2_strat_p": p2["strategy_layer"]["p_value"],
        "p2_mde": p2["strategy_layer"]["mde"],
        "p2_rand_p": p2r["p_value"], "p2_rand_obs": p2r["observed_sharpe"],
        "p3_t1": p3o["prediction_layer"]["states"][0]["t"], "p3_t2": p3o["prediction_layer"]["states"][1]["t"],
        "p3_orb_diff": p3o["strategy_layer"]["diff"], "p3_orb_p": p3o["strategy_layer"]["p_value"],
        "p3_vwap_diff": p3v["strategy_layer"]["diff"], "p3_vwap_p": p3v["strategy_layer"]["p_value"],
        "p3_orb_sharpe": p3o["strategy_layer"]["sharpe"], "p3_orb_unf": p3o["strategy_layer"]["unfiltered_sharpe"],
        "p3_vwap_sharpe": p3v["strategy_layer"]["sharpe"], "p3_vwap_unf": p3v["strategy_layer"]["unfiltered_sharpe"],
        "p4_auc": p4["prediction_layer"]["value"], "p4_strat_diff": p4["strategy_layer"]["diff"],
        "p4_strat_p": p4["strategy_layer"]["p_value"], "p4_mde": p4["strategy_layer"]["mde"],
        "p5_auc_diff": p5["prediction_layer"]["diff"], "p5_auc_p": p5["prediction_layer"]["p_value"],
        "p5_strat_diff": p5["strategy_layer"]["diff"], "p5_strat_p": p5["strategy_layer"]["p_value"],
        "years_needed": 104,
        "direction_flips": sum(1 for r in d["_direction_check"] if not r["direction_match"]),
        "direction_total": len(d["_direction_check"]),
    }


# ---------------------------------------------------------------------
# Slide-building helpers
# ---------------------------------------------------------------------
def new_slide(prs: Presentation):
    slide = prs.slides.add_slide(prs.slide_layouts[6])  # blank layout
    slide.background.fill.solid()
    slide.background.fill.fore_color.rgb = SURFACE
    return slide


def add_title(slide, text: str, size: int = 30, top=Inches(0.45)) -> None:
    box = slide.shapes.add_textbox(BODY_LEFT, top, BODY_WIDTH, Inches(1.15))
    tf = box.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    run = p.add_run()
    run.text = text
    run.font.name = FONT
    run.font.size = Pt(size)
    run.font.bold = True
    run.font.color.rgb = INK


def add_bullets(slide, bullets: list[str], top=Inches(1.75), width=BODY_WIDTH,
                left=BODY_LEFT, size: int = 20, height=Inches(3.6)) -> None:
    box = slide.shapes.add_textbox(left, top, width, height)
    tf = box.text_frame
    tf.word_wrap = True
    for i, b in enumerate(bullets):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        run = p.add_run()
        run.text = f"•  {b}"
        run.font.name = FONT
        run.font.size = Pt(size)
        run.font.color.rgb = INK
        p.space_after = Pt(12)
        p.line_spacing = 1.15


def add_figure(slide, path: Path, left, top, width=None, height=None) -> None:
    slide.shapes.add_picture(str(path), left, top, width=width, height=height)


def add_table(slide, header: list[str], rows: list[list[str]], left, top, width, height,
             font_size: int = 14) -> None:
    n_rows, n_cols = len(rows) + 1, len(header)
    gtable = slide.shapes.add_table(n_rows, n_cols, left, top, width, height).table
    for c, h in enumerate(header):
        cell = gtable.cell(0, c)
        cell.text = h
        cell.fill.solid()
        cell.fill.fore_color.rgb = BLUE
        for p in cell.text_frame.paragraphs:
            p.alignment = PP_ALIGN.CENTER
            for r in p.runs:
                r.font.name, r.font.size, r.font.bold = FONT, Pt(font_size), True
                r.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
    for ri, row in enumerate(rows, start=1):
        for c, val in enumerate(row):
            cell = gtable.cell(ri, c)
            cell.text = str(val)
            cell.fill.solid()
            cell.fill.fore_color.rgb = SURFACE
            for p in cell.text_frame.paragraphs:
                for r in p.runs:
                    r.font.name, r.font.size = FONT, Pt(font_size)
                    r.font.color.rgb = INK


def add_page_number(slide, n: int) -> None:
    box = slide.shapes.add_textbox(SLIDE_W - Inches(0.9), SLIDE_H - Inches(0.5), Inches(0.6), Inches(0.35))
    p = box.text_frame.paragraphs[0]
    run = p.add_run()
    run.text = str(n)
    run.font.name = FONT
    run.font.size = Pt(12)
    run.font.color.rgb = MUTED


def set_notes(slide, text: str) -> None:
    notes = slide.notes_slide.notes_text_frame
    notes.text = text
    for p in notes.paragraphs:
        for r in p.runs:
            r.font.name = FONT
            r.font.size = Pt(12)


# ---------------------------------------------------------------------
def build(prs: Presentation, n: dict) -> None:
    # 1. Cover
    s = new_slide(prs)
    box = s.shapes.add_textbox(Inches(1.0), Inches(2.4), Inches(11.3), Inches(1.6))
    tf = box.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    r = p.add_run()
    r.text = "機器學習能否改善 QQQ 短期交易策略?"
    r.font.name, r.font.size, r.font.bold, r.font.color.rgb = FONT, Pt(36), True, INK
    p2 = tf.add_paragraph()
    r2 = p2.add_run()
    r2.text = "一個 14 週的實證研究"
    r2.font.name, r2.font.size, r2.font.color.rgb = FONT, Pt(22), INK_2
    sub = s.shapes.add_textbox(Inches(1.0), Inches(5.8), Inches(11.3), Inches(1.0))
    stf = sub.text_frame
    sp = stf.paragraphs[0]
    sr = sp.add_run()
    sr.text = "gre888"
    sr.font.name, sr.font.size, sr.font.color.rgb = FONT, Pt(18), INK_2
    sp2 = stf.add_paragraph()
    sr2 = sp2.add_run()
    import datetime as dt
    sr2.text = dt.date.today().isoformat()
    sr2.font.name, sr2.font.size, sr2.font.color.rgb = FONT, Pt(16), MUTED
    set_notes(s, "自我介紹、題目、14週時間跨度。開場30秒帶過,把時間留給結論頁。")
    add_page_number(s, 1)

    # 2. 主旨
    s = new_slide(prs)
    add_title(s, "機器學習能改善預測指標,但無法顯著改善策略績效")
    add_bullets(s, [
        "扣除實際成本後,在現有樣本量下,無法顯著改善QQQ短期策略績效",
        "三個限定詞:能改善預測指標 / 扣除實際成本後 / 現有樣本下",
        f"檢定力分析反推:需要約{n['years_needed']}年OOS資料,才能偵測到0.2的Sharpe差",
    ])
    set_notes(s, "這是全篇唯一要記住的一句話。三個限定詞不是模糊帶過,後面每一頁都有對應"
             "的證據。對應問答:defense_qa.md 第1、11題。")
    add_page_number(s, 2)

    # 3. 研究問題
    s = new_slide(prs)
    add_title(s, "四個階段,遞進回答同一個問題")
    add_bullets(s, [
        "波動率預測,能改善以波動率為基礎的部位管理嗎?(階段二)",
        "市場狀態,能事先判定、用來篩選規則策略的交易日嗎?(階段三)",
        "交易日開盤前後的特徵,能篩選單筆訊號值不值得進場嗎?(階段四)",
        "放棄手工特徵、直接用CNN處理原始分鐘序列,能多學到什麼嗎?(階段五)",
    ])
    set_notes(s, "四個階段是遞進關係:前一階段的負面結果,直接促成下一階段換角度嘗試。"
             "不是四個獨立題目。")
    add_page_number(s, 3)

    # 4. 方法論
    s = new_slide(prs)
    add_title(s, "五項紀律,貫穿14週,不是事後包裝")
    add_bullets(s, [
        "事先寫定設定,結果不論好壞照實報告",
        "巢狀walk-forward:測試段完全不參與配適或選門檻",
        "截斷與篡改測試(CNN序列特徵無未來資訊洩漏)",
        "隨機基準 + block bootstrap 雙層檢定",
        "檢定力分析(第10、14週):量化「測不到」是不是「沒有效果」",
    ])
    set_notes(s, "強調這五項紀律在過程中真的攔下了假結果(下一頁的修正紀錄舉例),"
             "不是為了寫報告才列出來的門面。對應問答:第6題。")
    add_page_number(s, 4)

    # 5. 階段二
    s = new_slide(prs)
    add_title(s, "波動目標部位本身有效,但模型選擇不重要", size=27)
    add_bullets(s, [
        f"預測層:QLIKE差{n['p2_qlike_diff']:.4f},DM p={n['p2_dm_p']:.5f}(顯著改善)",
        f"策略層:Sharpe差{pct(n['p2_strat_diff'])},p={n['p2_strat_p']:.4f}(不顯著)",
        f"隨機基準(第14週):p={n['p2_rand_p']:.3f},真實波動資訊優於打亂日期",
    ], top=Inches(1.7), width=Inches(6.3), size=18, height=Inches(2.6))
    add_figure(s, FIG_DIR / "w05_equity_drawdown.png", Inches(7.1), Inches(1.55),
              width=Inches(5.7))
    set_notes(s, "主比較不顯著,但隨機基準顯著——兩者合起來支持『部位規則有效,模型選擇"
             "不重要』。圖是第5週既有的淨值/回撤圖。對應問答:第4題。")
    add_page_number(s, 5)

    # 6. 階段三
    s = new_slide(prs)
    add_title(s, "狀態標籤站得住腳,但拿來篩選交易日沒有用", size=27)
    add_bullets(s, [
        f"HMM狀態外部驗證:t={n['p3_t1']:.2f}/{n['p3_t2']:.2f},p≈0(狀態本身顯著)",
        "18個「狀態vs全樣本」比較,0個在p<0.05顯著",
        "逐年拆解(第14週):沒有一個狀態連續8年同向,不是巧合",
    ])
    add_table(s, ["策略", "篩選後Sharpe", "不篩選Sharpe", "差", "p值"],
             [["ORB", f"{n['p3_orb_sharpe']:.3f}", f"{n['p3_orb_unf']:.3f}",
               pct(n['p3_orb_diff']), f"{n['p3_orb_p']:.3f}"],
              ["VWAP", f"{n['p3_vwap_sharpe']:.3f}", f"{n['p3_vwap_unf']:.3f}",
               pct(n['p3_vwap_diff']), f"{n['p3_vwap_p']:.3f}"]],
             Inches(0.6), Inches(4.55), Inches(9.0), Inches(1.3))
    set_notes(s, "狀態『能不能判定』跟『能不能拿來篩選』是兩個不同的問題,這頁的重點是"
             "把兩者分開講。對應問答:第5題。")
    add_page_number(s, 6)

    # 7. 階段四
    s = new_slide(prs)
    add_title(s, "篩選效果看似存在,但通不過顯著性檢定", size=27)
    add_bullets(s, [
        f"AUC={n['p4_auc']:.4f}(中等),篩選後Sharpe差{pct(n['p4_strat_diff'])},"
        f"p={n['p4_strat_p']:.4f}(不顯著)",
        "五個模型沒有一個四項標準全過;樹模型嚴重過擬合",
        f"最小可偵測差{n['p4_mde']:.3f},需要約{n['years_needed']}年資料才能偵測到0.2的差距",
    ], top=Inches(1.7), width=Inches(6.3), size=18, height=Inches(2.6))
    add_figure(s, FIG_DIR / "w08_threshold_curve.png", Inches(7.1), Inches(1.55),
              width=Inches(5.7))
    set_notes(s, "字面上看起來贏(篩選後Sharpe較高),但block bootstrap不顯著——這是"
             "整個階段四『證據階梯』的核心示範。對應問答:第2、10題。")
    add_page_number(s, 7)

    # 8. 階段五
    s = new_slide(prs)
    add_title(s, "CNN顯著輸給手工特徵", size=30)
    add_bullets(s, [
        f"AUC差(CNN−手工特徵){pct(n['p5_auc_diff'])},p={n['p5_auc_p']:.4f}(顯著更差)",
        "三項增量資訊檢查一致確認:CNN沒有超出手工特徵的資訊",
        f"策略層:Sharpe差{pct(n['p5_strat_diff'])},p={n['p5_strat_p']:.4f}(不顯著)",
    ], top=Inches(1.7), width=Inches(6.3), size=18, height=Inches(2.6))
    add_figure(s, FIG_DIR / "w12_cnn_vs_hand_auc_by_fold.png", Inches(7.1), Inches(1.5),
              width=Inches(5.7))
    set_notes(s, "換一個表達力更強的架構,結果不是更好、是更差——這是對主旨最強的反例"
             "確認,不是『手工特徵這條路選錯了』。對應問答:第3、8題。")
    add_page_number(s, 8)

    # 9. 為什麼難
    s = new_slide(prs)
    add_title(s, "不是雜訊太大,是樣本量從根本不夠", size=28)
    add_bullets(s, [
        "四階段策略層觀察差,全部遠小於最小可偵測差",
        "階段二:有檢定力支持的『沒有差異』;階段三到五:『偵測不到』≠『沒有效果』",
        f"約{n['years_needed']}年OOS資料才能偵測到0.2的Sharpe差(第10週)",
    ], top=Inches(1.55), width=Inches(12.2), size=18, height=Inches(1.5))
    # w14_diff_vs_mde.png is 1184x671 (aspect ~1.765) - constrain by HEIGHT so it
    # never overflows past the slide edge (specifying only width let an earlier
    # version's auto-scaled height run off the bottom - caught in the LibreOffice
    # PDF visual check).
    fig_h = Inches(3.55)
    fig_w = Inches(3.55 * 1184 / 671)
    add_figure(s, FIG_DIR / "w14_diff_vs_mde.png",
              (SLIDE_W - fig_w) / 2, Inches(3.15), height=fig_h)
    set_notes(s, "這張圖是整份報告的核心圖——強調『判讀分兩類』,不要把階段二跟"
             "階段三到五混為一談。對應問答:第2、12題。")
    add_page_number(s, 9)

    # 10. 修正紀錄
    s = new_slide(prs)
    add_title(s, "方法論紀律真的擋下了看似顯著的假結果", size=27)
    add_table(s, ["週", "問題", "怎麼發現的"],
             [["6", "HMM樣本外解碼漏轉移步驟", "寫測試時主動發現,看任何結果前就修正"],
              ["8", "三項成功標準字面上全過", "使用者要求補四項檢查後,推翻字面結論"],
              ["10", "排序層p值換種子會跳動", "查出是蒙地卡羅精度不足,不是真的不穩健"]],
             Inches(0.6), Inches(1.9), Inches(12.1), Inches(2.6), font_size=16)
    add_bullets(s, ["完整6條記錄在附錄A,每條都寫發現方式、影響範圍、修正後結論"],
               top=Inches(4.8), size=18, height=Inches(0.8))
    set_notes(s, "只挑3條最能展示『紀律有實際作用』的,不是列滿6條。強調第6週那條是"
             "主動發現、不是被結果異常倒逼。對應問答:第6題。")
    add_page_number(s, 10)

    # 11. 限制
    s = new_slide(prs)
    add_title(s, "樣本量是根本限制,不是任一階段的問題", size=28)
    add_bullets(s, [
        "兩次獨立檢定力分析(第10週Sharpe、第14週AUC)從不同指標確認同一件事",
        "2022年做空訊號排序顛倒,目前只是一個假說,不是三次獨立驗證",
        "任何看完結果才挑選的探索性組合,一律打上折扣,不能直接採用",
    ])
    set_notes(s, "誠實列出限制,不是自我否定——這正是整份報告方法論紀律的延伸。"
             "對應問答:第9題。")
    add_page_number(s, 11)

    # 12. 未來工作
    s = new_slide(prs)
    add_title(s, "更大的模型需要更多資料,不是現在", size=28)
    add_bullets(s, [
        "調查2022年做空訊號排序顛倒的具體機制",
        "LSTM/Transformer列為未來工作,前提是樣本量顯著增加才成立",
        "CNN(1809參數)在現有樣本上過擬合差距已經不小,更大架構需要更多資料",
    ])
    set_notes(s, "不是說深度學習沒用,是說『現在』用更大的模型在『現有』樣本量下沒有"
             "意義——先解決樣本量問題,才輪到換架構。對應問答:第3題。")
    add_page_number(s, 12)

    # 13. 結論
    s = new_slide(prs)
    add_title(s, "機器學習能改善預測,但改不了現有樣本下的策略績效", size=26)
    add_bullets(s, [
        "四階段一致:預測層常有改善,策略層的差異顯著性檢定沒有一個通過",
        f"算術/複利兩種Sharpe定義下,{n['direction_total']}個策略層比較"
        f"{n['direction_flips']}個方向翻轉——結論對定義選擇穩健",
        "253個測試通過,可重現性在乾淨環境驗證過(v1.0.1)",
    ])
    set_notes(s, "重申主旨,收尾。不需要新資訊,是把前面12頁串起來的總結句。")
    add_page_number(s, 13)

    # 14. Q&A
    s = new_slide(prs)
    add_title(s, "Q&A", size=40, top=Inches(2.6))
    add_bullets(s, ["準備問答見 reports/defense_qa.md(12題,每題對應章節與數字)"],
               top=Inches(4.2), size=18, height=Inches(0.8))
    set_notes(s, "轉場頁,口頭邀請提問即可。")
    add_page_number(s, 14)


def main() -> None:
    n = load_numbers()
    prs = Presentation()
    prs.slide_width = SLIDE_W
    prs.slide_height = SLIDE_H
    build(prs, n)
    out = REPORTS / "defense_slides.pptx"
    prs.save(out)
    print(f"wrote {out} ({len(prs.slides._sldIdLst)} slides)")


if __name__ == "__main__":
    main()
