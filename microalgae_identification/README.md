# 微藻识别模块

从原始显微图到按流道/腔室的量化结果。换批次只需改 `configs/<批次>.yaml`。

## 目录

```
microalgae_identification/
├── config.py              配置加载：YAML → 环境变量 → 命令行
├── run_pipeline.py        一键执行入口（--from / --to / --only / --dry-run）
├── preprocessing/         ① 原始图分拣与归集
├── detection/             ② 识别（detect_batch.py + chamber_template.png）
└── postprocessing/        ③④⑤ 汇总 → 标准化 → 可视化
```

## 一键运行

```bash
python microalgae_identification/run_pipeline.py --config configs/batch_20261001.yaml
python microalgae_identification/run_pipeline.py --config configs/batch_20261001.yaml --dry-run
python microalgae_identification/run_pipeline.py --config configs/batch_20261001.yaml --from detect
```

阶段名：`sort` `move` `detect` `merge` `standardize` `visualize`。

## 各步职责

| 步骤 | 脚本 | 输入 | 产物 |
|---|---|---|---|
| ①-1 | `preprocessing/01_select_ch.py` | `<photo_root>/2026-Week-N/` | 子目录重排并补 `(001)` 序号后缀 |
| ①-2 | `preprocessing/02_move_photoes.py` | `<photo_root>/HEIDSTAR.COM.x.y/` | `CH{z}/IMG001xNNN/CH{z}_CB{c}_H{h}.tif`（`z=3(x-1)+y`） |
| ② | `detection/detect_batch.py` | `CH1…CH12` | `CH{z}/IMG001xNNN/results/CH{z}_CB{c}.xlsx` + `R_*.png` |
| ③ | `postprocessing/00_merge_excel_summary.py` | 上一步 `results/` | `数据汇总/01_原始数据/CH{n}.xlsx` |
| ④ | `postprocessing/01_standardize_relative_cell_number.py` | ③ 的产物 | `数据汇总/02_标准化数据/CH{n}_标准化.xlsx` |
| ⑤ | `postprocessing/08_growth_curve_batch.py` | ④ 的产物 | `数据汇总/03_可视化结果/<N>小时/CH{n}_可视化结果_<N>.xlsx` |

## 注意事项

- **`chamber_template.png` 必须与 `detect_batch.py` 在同一目录**。腔室掩膜过滤依赖它，
  缺失会导致过滤失效、识别结果偏移。也可在配置里用 `model.chamber_template` 显式指定路径。
- ③ 步会跳过「目标数量最大值 ≤ 3」或「存在 0 值」的腔室，这是原有筛选规则。
- ⑤ 步只处理在 `summary.channel_conditions` 中登记过的流道，未登记的会跳过并告警。
- 各脚本顶部保留了原有的 `MICROALGAE_*` 环境变量开关（线程数、批大小、GC 频率等），
  用于性能调优，与批次路径无关。

完整的目录结构与使用说明见仓库根 [`README.md`](../README.md)。
