# Week 9 (階段四第二週)交易日篩選:隨機森林與 XGBoost

延續第八週的管線(開盤 09:35 特徵、巢狀切分、保留比例網格、不重新配適),換用隨機森林、
XGBoost 分類器,並改用四項更嚴格的成功標準(篩選前後夏普差 block bootstrap p<0.05、隨機
篩選經確 p<0.05、每折每年≥20筆交易、≥6/9折篩選較好且排除貢獻最大一年後仍正)。

設定(`config/week09_filter.toml`)在跑任何模型前寫定,包含兩次跟您討論後的修改:XGBoost
深度從草案的 3 改為 2(附錄留深度 3 對照);early stopping 與第三層巢狀切分整個取消,改用
固定 `n_estimators=200`、學習率 0.05、`min_child_weight=10`,在完整配適段(~2年)訓練——
理由是配適段最後 6 個月只有約 30 筆獲利交易,early stopping 訊號太雜,且會犧牲四分之一
訓練資料。

## 三、模型與過擬合檢查

AUC(全期 OOS 合併):

| model | auc |
|---|---|
| logistic | 0.597 |
| random_forest | 0.607 |
| xgboost | 0.585 |
| xgboost_depth3 | 0.586 |

**過擬合檢查(完成標準要求:樹模型明顯好於 logistic 時先查這個)**——配適段(in-sample)
vs 測試段(OOS)AUC:

| model | mean_fit_auc | mean_test_auc | mean_gap |
|---|---|---|---|
| logistic | 0.667 | 0.612 | 0.056 |
| random_forest | 0.828 | 0.62 | 0.207 |
| xgboost | 0.901 | 0.591 | 0.31 |
| xgboost_depth3 | 0.965 | 0.591 | 0.373 |

logistic 的配適/測試 AUC 差距只有 0.056,樹模型即使深度只有 2、葉節點/子節點權重限制
都刻意設大,配適/測試差距仍高達 0.21(隨機森林)到 0.37(XGBoost 深度3)——**嚴重過擬合**。
這解釋了為什麼隨機森林測試段 AUC(0.607)看起來比 logistic(0.597)略高,但下一節會看到
篩選後夏普反而更差:配適段學到的排序在測試段站不住腳。

## 四、四項成功標準逐一檢查

| model | filtered_sharpe | unfiltered_sharpe | c1_bb_p | c2_rand_p | c4_folds | c4_diff_excl_biggest_year | success |
|---|---|---|---|---|---|---|---|
| logistic | 0.707 | 0.517 | 0.484 | 0.03 | 7/9 | 0.006 |  |
| random_forest | 0.46 | 0.517 | 0.810 (Holm 0.810) | 0.202 | 6/9 | -0.229 |  |
| xgboost | 0.295 | 0.517 | 0.364 (Holm 0.729) | 0.379 | 5/9 | -0.344 |  |
| xgboost_depth3 | 0.538 | 0.517 | 0.953 | 0.035 | 6/9 | -0.077 |  |
| huber | 0.612 | 0.517 | 0.545 | 0.059 | 4/9 | -0.012 |  |

**沒有任何模型四項全過**(0/5)。標準1(block bootstrap)是唯一沒有模型通過的
一項——隨機森林、XGBoost 對不篩選的 block bootstrap p 值經 Holm 校正後分別是
0.810、0.729,
校正前就已經遠高於 0.05,校正只會讓門檻更嚴,結論不變。

**隨機森林、XGBoost 對 logistic 的夏普差(block bootstrap,兩兩比較,不是對不篩選)**:

| model | tree_sharpe | logistic_sharpe | sharpe_diff | block_bootstrap_p |
|---|---|---|---|---|
| random_forest | 0.46 | 0.707 | -0.241 | 0.235 |
| xgboost | 0.295 | 0.707 | -0.398 | 0.078 |
| xgboost_depth3 | 0.538 | 0.707 | -0.159 | 0.494 |

被篩掉的大賺日比例:

| model | n_big_win_days | n_dropped | frac_dropped |
|---|---|---|---|
| logistic | 140 | 61 | 0.436 |
| random_forest | 140 | 82 | 0.586 |
| xgboost | 140 | 68 | 0.486 |
| xgboost_depth3 | 140 | 62 | 0.443 |
| huber | 140 | 62 | 0.443 |

**逐折標準化 logistic 係數**(沿用第八週,列在這裡對照樹模型的 permutation importance):

| feature | mean_abs_coef | sign_consistency |
|---|---|---|
| open5m_body_ratio | 0.427 | 1.0 |
| vxn_1d | 0.358 | 0.667 |
| gap_atr_ratio | 0.245 | 0.778 |
| har_logrv_1d | 0.235 | 1.0 |
| vix_1d | 0.228 | 0.556 |
| open5m_range_rel | 0.206 | 0.556 |
| overnight_gap | 0.172 | 0.556 |
| har_logrv_5d | 0.166 | 0.778 |
| open5m_direction | 0.121 | 0.889 |
| ret_cc_1d | 0.117 | 0.556 |
| har_logrv_22d | 0.108 | 0.556 |
| open5m_rel_volume | 0.101 | 0.556 |
| open5m_gap_agree | 0.09 | 0.778 |
| day_of_week | 0.064 | 0.667 |

**測試段 permutation importance**(scoring=roc_auc,20 次重複,由高到低前 5 名):

- logistic: open5m_body_ratio=0.0914, vxn_1d=0.0225, har_logrv_1d=0.0185, open5m_range_rel=0.0099, vix_1d=0.0098
- random_forest: open5m_body_ratio=0.0800, open5m_range_rel=0.0136, har_logrv_1d=0.0061, open5m_rel_volume=0.0048, ret_cc_1d=0.0029
- xgboost: open5m_body_ratio=0.0723, open5m_range_rel=0.0092, har_logrv_1d=0.0078, open5m_rel_volume=0.0065, gap_atr_ratio=0.0017
- xgboost_depth3: open5m_body_ratio=0.0716, open5m_range_rel=0.0108, har_logrv_1d=0.0101, open5m_rel_volume=0.0066, gap_atr_ratio=0.0046

logistic 穩定倚賴 `open5m_body_ratio`、`har_logrv_1d`(9 折同號)兩個特徵;樹模型的
permutation importance 前幾名不一定是同一組特徵,且本節開頭已經確認樹模型嚴重過擬合,
這些重要性排序的可信度本身要打折扣。

## 五、2022 年診斷(拆多空)

2022 年 ORB 做空訊號(132 筆,全樣本平均淨R為正)拆開來看:

| model | direction | subset | n | win_rate | mean_r_net |
|---|---|---|---|---|---|
| logistic | -1 | kept | 7 | 0.143 | -0.742 |
| logistic | -1 | dropped | 124 | 0.258 | 0.194 |
| logistic | 1 | kept | 43 | 0.279 | 0.015 |
| logistic | 1 | dropped | 76 | 0.211 | 0.185 |
| random_forest | -1 | kept | 25 | 0.12 | -0.706 |
| random_forest | -1 | dropped | 106 | 0.283 | 0.345 |
| random_forest | 1 | kept | 35 | 0.257 | 0.341 |
| random_forest | 1 | dropped | 84 | 0.226 | 0.033 |
| xgboost | -1 | kept | 23 | 0.13 | -0.68 |
| xgboost | -1 | dropped | 108 | 0.278 | 0.32 |
| xgboost | 1 | kept | 28 | 0.25 | 0.143 |
| xgboost | 1 | dropped | 91 | 0.231 | 0.118 |

三個模型(logistic、隨機森林、XGBoost)**一致地**把 2022 年賺錢的空單丟掉、留下賠錢的空單
——留下的空單平均淨R全部是負的,丟掉的空單平均淨R全部是正的,方向完全顛倒。三個獨立配適
的模型在同一年、同一個方向犯一樣的錯,不像是單一折門檻選擇運氣不好,比較像是模型學到的
排序規則在 2022 年那種持續空方動能的環境下,對空方訊號整體是反著的——是機制層面的方向性
弱點,不是門檻選擇的雜訊。做多訊號則沒有這個現象。

## 六、2019 年後子樣本對照(補第八週)

extras 只從 2019 年起有值,子樣本走自己的 WalkForwardSplit 邏輯,只剩 4 折
(測試段 2023-01-03..2026-09-22)。logistic、隨機森林在這個共同子樣本上,加回 HAR+殘差XGB預測與HMM
狀態前後:

| model | features | filtered_sharpe | unfiltered_sharpe |
|---|---|---|---|
| logistic | without_extras | 0.856 | 0.104 |
| logistic | with_extras | 0.642 | 0.104 |
| random_forest | without_extras | 0.778 | 0.104 |
| random_forest | with_extras | 0.766 | 0.104 |

logistic 加回這兩個特徵後夏普從 0.856
**降到** 0.642,
隨機森林大致持平。兩個被第八週排除在主分析外的特徵,沒有被低估的價值。

## 七、探索性:兩特徵 logistic

**這是看完第三、四節樹模型/係數結果後才挑選的組合,不列入任何成功標準判斷**,只做描述性
報告:`open5m_body_ratio` + `har_logrv_1d` 兩特徵 logistic,AUC=0.626(比全特徵
logistic 的 0.597 還高),篩選後夏普
0.761(目前所有模型裡點估計最好),隨機篩選 p=0.025
過關,但 **block bootstrap p=0.381 仍不顯著,只有 5/
9 折篩選後較好**——連點估計最好的模型都撐不過四項標準,這正是把它明確
標成探索性的原因。

## 八、結論與完成標準逐項回答

**哪個模型在四項標準下達成顯著改善?沒有。** logistic、隨機森林、XGBoost(深度2、深度3)、
Huber 迴歸,五個模型沒有一個四項全部通過,標準1(block bootstrap p<0.05,對篩選前後夏普差
最嚴謹的檢定)是所有模型都沒過的一項。

**樹模型是否明顯好於 logistic?沒有——反而更差,且已查過擬合。** 隨機森林、XGBoost(深度2)
的篩選後夏普都低於不篩選(隨機森林 0.460、XGBoost
0.295,對照不篩選 0.517),
配適/測試 AUC 差距(0.21、0.31)確認是過擬合造成,不是訊號更弱這麼簡單——即使深度只有 2、
正則化參數刻意設保守,小樣本上樹模型還是比線性模型更容易背答案。

**本週結論延續第八週、且更確定**:用更嚴格、預先設定、包含差異顯著性檢定的標準,無論是
logistic 還是更複雜的樹模型,目前都沒有找到統計上站得住腳的篩選效益。2022 年的診斷顯示
問題不只是樣本雜訊,至少在做空訊號上有可重現的方向性弱點,值得後續著手處理,而不是繼續
換更複雜的模型。
