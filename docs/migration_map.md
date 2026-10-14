# 重构迁移映射表

2026-10 重构：把 `data_analysis` 并入本仓库，并按「微藻识别 / 数据分析」两个模块重新划分目录。
本表记录每个文件的**旧路径 → 新路径**，便于对照历史提交、脚本与外部调度。

> 除本表列出的改动外，所有脚本的**算法逻辑、参数默认值、产物命名与目录层级均未改变**。

---

## 一、识别主流程（核心，逻辑未变）

| 旧路径 | 新路径 | 改动 |
|---|---|---|
| `Data_Preprocessing/pretreatment/01_selectCH.py` | `microalgae_identification/preprocessing/01_select_ch.py` | 改名；`__main__` 改为读取集中配置 |
| `Data_Preprocessing/pretreatment/02_move_photoes.py` | `microalgae_identification/preprocessing/02_move_photoes.py` | 仅改 `__main__` 的路径来源 |
| `identification_of_microalgae/test_model_17.py` | `microalgae_identification/detection/detect_batch.py` | 改名；仅替换顶部 `MODEL_PATH` / `DATE_STR` / `ROOT_FOLDERS` 的来源，其余 1300+ 行未动 |
| `identification_of_microalgae/chamber_template.png` | `microalgae_identification/detection/chamber_template.png` | **随脚本同步迁移**（脚本按 `__file__` 同目录查找） |
| `Data_Preprocessing/processed/00_merge_excel_summary.py` | `microalgae_identification/postprocessing/00_merge_excel_summary.py` | 仅改 `__main__` 的路径来源 |
| `Data_Preprocessing/processed/01_standardize_relative_cell_number.py` | `microalgae_identification/postprocessing/01_standardize_relative_cell_number.py` | 同上 |
| `Data_Preprocessing/processed/08_processed_batch.py` | `microalgae_identification/postprocessing/08_growth_curve_batch.py` | 改名；`INPUT_FOLDER` / `OUTPUT_FOLDER` / `PARAM_MAPPING` 改为配置注入，原映射块保留为台账 |

新增：

| 新文件 | 用途 |
|---|---|
| `configs/pipeline.example.yaml` | 批次配置模板 |
| `configs/batch_20261001.yaml` | 20261001 批次配置实例 |
| `microalgae_identification/config.py` | 配置加载（YAML → 环境变量 → 命令行） |
| `microalgae_identification/run_pipeline.py` | 一键执行入口 |

---

## 二、训练、标注与论文配图

| 旧路径 | 新路径 |
|---|---|
| `train_model.py` | `training/train_model.py` |
| `dump_conf_curves.py` | `training/dump_conf_curves.py` |
| `Data_Preprocessing/dataset_label/*.py`（5 个） | `training/dataset_label/` |
| `plot_training_curves.py` | `training/figures/plot_training_curves.py` |
| `plot_s3_revised.py` | `training/figures/plot_s3_revised.py` |
| `plot_s3_supplementary.py` | `training/figures/plot_s3_supplementary.py` |
| `plot_simulated_curves.py` | `training/figures/plot_simulated_curves.py` |
| `generate_figures/kla_fit_S1.py` + 其输入/产物 | `training/figures/` |

---

## 三、数据分析类（从识别链剥离）

| 旧路径 | 新路径 |
|---|---|
| `Data_Preprocessing/processed/06_growth_kinetics_pipeline.py` | `data_analysis/kinetics/06_growth_kinetics_pipeline.py` |
| `Data_Preprocessing/processed/97_get_3data.py` | `data_analysis/exploratory/97_get_3data.py` |
| `Data_Preprocessing/processed/98_extract_miumax_data.py` | `data_analysis/exploratory/98_extract_miumax_data.py` |
| `Data_Preprocessing/plot_for_paper/` 有效脚本 + `dataset/` + 产物图 | `data_analysis/exploratory/paper_figures/` |
| `E:/pythonProject/data_analysis/batch_grouping/` | `data_analysis/batch_grouping/` |
| `E:/pythonProject/data_analysis/ccf_chlorella/` | `data_analysis/ccf_chlorella/` |
| `E:/pythonProject/data_analysis/exp_batch_aggregator/` | `data_analysis/exp_batch_aggregator/` |
| `E:/pythonProject/data_analysis/simdata_pipeline/` | `data_analysis/simdata_pipeline/` |
| `E:/pythonProject/data_analysis/_backup/` | `data_analysis/_backup/`（已 gitignore） |
| `E:/pythonProject/data_analysis/模拟数据处理_part1/` | `data_analysis/模拟数据处理_part1/`（已 gitignore） |
| `E:/pythonProject/data_analysis/requirements.txt` | `requirements-data-analysis.txt` |

路径修正：`simdata_pipeline/config.py` 的 `PROJECT_ROOT` 改为由文件位置自动推导；
`ccf_chlorella/optimum_meaning*.py`、`prediction_interval.py`、`simdata_pipeline/_verify_selected.py`
中的绝对路径已指向新仓库位置。

---

## 四、已删除

| 旧路径 | 原因 |
|---|---|
| `identification_of_microalgae/test_model.py` | 版本迭代废弃（要删除项） |
| `identification_of_microalgae/test_model_02.py` ~ `test_model_16.py` | 同上 |
| `identification_of_microalgae/microalgae_dense_hybrid_00.py` | 同上（要删除项） |

---

## 五、已归档到 legacy/（未删除，可追溯）

| 旧路径 | 新路径 | 原因 |
|---|---|---|
| `identification_of_microalgae/microalgae_dense_hybrid_01/02/03.py` | `legacy/detection/` | `detect_batch.py` 的前身版本 |
| `Data_Preprocessing/processed/02_processed_batch.py` | `legacy/processed_versions/` | 算法迭代快照（20260609 批次） |
| `Data_Preprocessing/processed/05_processed_batch.py` | `legacy/processed_versions/` | 同上（20260726 批次） |
| `Data_Preprocessing/processed/07_processed_batch.py` | `legacy/processed_versions/` | 同上（20260712 批次） |
| `run_inference_full.py` | `legacy/research/` | 依赖已删除的 `microalgae_dense_hybrid_00` |
| `run_inference_for_s35.py` | `legacy/research/` | 同上 |
| `sel_conf_and_s35.py` | `legacy/research/` | 同上 |
| `sweep_conf_pipeline.py` | `legacy/research/` | 同上 |
| `main.py` | `legacy/research/` | 正则调试片段，非入口 |
| `Data_Preprocessing/pretreatment/03_delete_photoes.py` | `legacy/research/` | 一次性清理，硬编码 20251104 老批次，且会删除文件 |
| `Data_Preprocessing/pretreatment/100_replace_subfolders_name.py` | `legacy/research/` | 一次性改名 |
| `Data_Preprocessing/rename.py` | `legacy/research/` | 一次性改名 |
| `Data_Preprocessing/data_summary.py` | `legacy/research/` | 旧目录结构，已被 `00_merge_excel_summary.py` 取代 |
| `Data_Preprocessing/processed/99_merge_excel.py` | `legacy/research/` | 旧约定，与当前目录结构脱节 |
| `Data_Preprocessing/processed/100_getfigures.py` | `legacy/research/` | ReRun 存档 |
| `Data_Preprocessing/processed/_growth_pipeline_test/` | `legacy/research/_growth_pipeline_test/` | 测试残留产物 |
| `Data_Preprocessing/test2.py`、`test_code.py` | `legacy/research/` | 数据内联的一次性分析 |
| `Data_Preprocessing/plot_for_paper/test.py`、`3.4_主图代码.py` | `legacy/research/` | 被同名 V2 / 最新版取代 |
| `generate_figures/Variance.py` + 产物 | `legacy/research/` | 与微藻项目无关 |

已删除的空目录：`Data_Preprocessing/`、`identification_of_microalgae/`、`generate_figures/`。

---

## 六、兼容性检查清单

| 检查项 | 状态 |
|---|---|
| `chamber_template.png` 与 `detect_batch.py` 同目录 | ✅ 已确认 |
| 「汇总统计」页列名（`图片名称`/`处理状态`/`目标数量`/`总面积(μm²)`） | ✅ 未变，与 ③ 步要求一致 |
| 识别产物目录 `results/` 与文件名 `CH{z}_CB{c}.xlsx` | ✅ 未变 |
| 数据汇总三级目录 `01_原始数据` / `02_标准化数据` / `03_可视化结果/<N>小时` | ✅ 未变 |
| 算法超参（MERGE_IOU=0.60、TILE 512/192、44.3×42.8 μm 等） | ✅ 未变 |
| 默认权重 `runs/segment/train4/weights/best.pt` | ✅ 未变，已入库 |
| `08` 的工况映射默认值（N287…N900） | ✅ 未变，现已可由配置覆盖 |
