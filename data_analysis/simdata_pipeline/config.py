# -*- coding: utf-8 -*-
"""
可配置路径参数
==============
所有路径集中在本文件，换机器 / 换批次时只需改这里。
"""
from pathlib import Path

# ---------------------------------------------------------------- 根目录
# 项目根（data_analysis/）。由本文件位置自动推导，整仓搬家后无需手改；
# 如需指向别处，直接改写这一行即可。
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# part1 根目录：下面挂着「原始数据」和「数据汇总分析结果」
PART1_ROOT = PROJECT_ROOT / "模拟数据处理_part1"

# ---------------------------------------------------------------- 输入
# 原始数据：μmax_revision_原始数据/<YYYYMMDD>/数据汇总/03_可视化结果/<N>小时/
RAW_DATA_DIR = PART1_ROOT / "μmax_revision_原始数据"

# exp_batch_aggregator + batch_grouping 的产物：数据汇总分析结果/<工况>/18_batch_*/
AGG_RESULT_DIR = PART1_ROOT / "数据汇总分析结果"

# CCF 映射表：ccf_chlorella 选定的 20 组（含「原始数据来源」列）
# 该表随 ccf_chlorella 模块走，故路径指向 ccf_chlorella/data/
CCF_MAPPING_XLSX = PROJECT_ROOT / "ccf_chlorella" / "data" / "μmax拟合-20 Groups.xlsx"
CCF_SHEET_NAME = "CCF"

# ---------------------------------------------------------------- 输出
# 「20组模拟数据」输出目录（建在 part1 下）
OUTPUT_DIR = PART1_ROOT / "20组模拟数据"

# 「未使用数据」输出目录：存放未被 CCF 20 组选中的 selected__groups_*.xlsx
UNUSED_DIR = PART1_ROOT / "未使用数据"

# ---------------------------------------------------------------- 二次分组
# 对「尚未参与任何数据分组的腔室」做二次分组的输出根目录
SECOND_ROUND_DIR = PART1_ROOT / "所有未参与数据分组腔室数据"

# 未参与分组腔室数据的汇总工作簿（超 MAX_SHEETS_PER_FILE 个页签时拆分 _part2/_part3…）
UNUSED_CHAMBER_XLSX = "未参与数据分组腔室数据.xlsx"
# 被生长规律筛查剔除的腔室
EXCLUDED_CHAMBER_XLSX = "剔除腔室数据.xlsx"
# 逐条记录每组成员来源的文档
DOC_SOURCE_XLSX = "文档数据来源.xlsx"

# 单个工作簿最多容纳的腔室页签数（超出则拆分）
MAX_SHEETS_PER_FILE = 50
# True  : 首份沿用原文件名，后续为 _part2 / _part3 …
# False : 全部带后缀，即 _part1 / _part2 …
SPLIT_FIRST_PART_PLAIN = True

# 二次分组产出的 pptx 命名（{folder} = 数据组文件夹名）
SECOND_PPTX_NAME_TEMPLATE = "{folder}.pptx"
# 生成 selected__groups_*.xlsx 用的底稿
REFERENCE_XLSX = AGG_RESULT_DIR / "参考输出.xlsx"

# 本管线生成的数据表格文件名后缀（与「20组模拟数据 / 未使用数据」保持一致）
GENERATED_SUFFIX = "_可视化结果.xlsx"

# ---------------------------------------------------------------- 分组参数
# 与 batch_grouping/run_full_batch.py 的 TARGET_* 保持一致
GROUP_MIN_SIZE = 12          # 最小组规模
GROUP_MAX_SIZE = 18          # 目标完整组规模
GROUP_CV_LIMIT = 0.15        # 组内 μmax 变异系数上限
GROUP_P_LIMIT = 0.05         # Jarque-Bera 正态性检验 p 值下限
# batch_grouping 的严格候选条件（与 excel_io.collect_candidates 一致）
STRICT_MIN_N0 = 0.0
STRICT_MIN_R2 = 0.90

# 未使用数据子文件夹命名：取 summary 页签 mu_max_mean_h^-1 的数值，
# 去掉前导「0.」后取前 N 位（例：0.0657777… → 0657）
UNUSED_FOLDER_DIGITS = 4
UNUSED_SUMMARY_METRIC = "mu_max_mean_h^-1"
UNUSED_SUMMARY_SHEET = "summary"

# ---------------------------------------------------------------- 参数
# 培养时长（小时），对应原始数据中的 <N>小时 目录
CULTURE_HOURS = 96

# batch 子目录前缀（batch_grouping 的输出文件夹名前缀，如 18_batch_20260915）
BATCH_DIR_PREFIX = "18_batch_"

# 期望每组腔室数（用于日志提示，不等于该值只告警不中断）
EXPECTED_CHAMBERS = 18

# ---------------------------------------------------------------- 数据组回链
# link_group_assets.py 用：把 20组模拟数据 的每个子文件夹，回链到 batch_grouping
# 产出的 selected__groups_*.xlsx 与 growth_curves pptx 的对应页。
BATCH_DIR_NAME = "18_batch_20260915"    # batch 子目录名；不存在时取 18_batch_* 里最新的
GROUP_FILE_PATTERN = "selected__groups_*.xlsx"          # 分组文件命名
CURVE_PPTX_PATTERN = "selected_groups_growth_curves_*.pptx"   # 生长曲线汇总 pptx
PPTX_NAME_TEMPLATE = "{condition}.pptx"  # 新建 pptx 的命名（自变量文件夹名）
LINK_ON_EXIST = "skip"                   # 目标已存在时的策略：skip | backup | overwrite
MU_MATCH_TOL = 5e-4                      # 回退匹配（μmax 均值 + 腔室数）的容差

# ---------------------------------------------------------------- 编码 → 实际水平
# CCF 编码值 -1 / 0 / +1 到因子实际水平的映射
LEVEL_LIGHT = {-1: "L30", 0: "L120", 1: "L210"}          # 光照强度 μmol/m²/s
LEVEL_NITROGEN = {-1: "N20", 0: "N160", 1: "N300"}       # 氨氮浓度 mg/L
LEVEL_IC = {-1: "IC0.5%", 0: "IC5.25%", 1: "IC10%"}     # 无机碳（CO2 浓度）

# 工况目录名中的连字符。原始数据目录用的是 Unicode 连字符 U+2011，
# 与 ASCII '-' 不同，拼接目录名时必须一致，否则匹配不上。
COND_SEP = "\u2011"

# 目录名 → 编码值 的反查表
NAME_TO_LEVEL_LIGHT = {v: k for k, v in LEVEL_LIGHT.items()}
NAME_TO_LEVEL_NITROGEN = {v: k for k, v in LEVEL_NITROGEN.items()}
NAME_TO_LEVEL_IC = {v: k for k, v in LEVEL_IC.items()}

# ---------------------------------------------------------------- 汇总字段
# 与 05_processed_batch.py 的 _empty_summary_row 一致（本任务不做腔室筛选，
# 只是把每个腔室页签里的拟合参数汇总成一行）。
# 左：输出表头；右：腔室页签「参数名」列中的对应名称。
SUMMARY_FIELDS = [
    ("腔室名称", None),
    ("最大比生长速率μmax (h^-1)", "最大比生长速率uMAX(h^-1)"),
    ("相对平台倍数 Krel (无量纲)", "相对平台倍数 Krel (无量纲)"),
    ("拟合起始相对数量 Nrel(0) (无量纲)", "拟合起始相对数量 Nrel(0) (无量纲)"),
    ("滞止期参数λ(h)", "滞止期参数λ(h)"),
    ("拟合优度R² (Y=ln(Nrel)尺度)", "拟合优度R² (Y=ln(Nrel)尺度)"),
    ("平均细胞周期T_d(h)", "平均细胞周期T_d(h)"),
    ("增殖倍数 F", "增殖倍数 F"),
    ("生长效率 η (1/F)", "生长效率 η (1/F)"),
    ("滞后期时长(h)", "滞后期时长(h)"),
    ("对数期时长(h)", "对数期时长(h)"),
    ("稳定期时长(h)", "稳定期时长(h)"),
    ("稳定期起始t90(h)", "稳定期起始t90(h)"),
]

# ---------------------------------------------------------------- 合并拟合
# 「汇总」页签的第三张表：全部数据可用的腔室拼接后一次性拟合（不做腔室筛选）。
# 字段顺序与参考实现 08_processed_batch.py 的「合并拟合结果」完全一致。
SUMMARY_MERGED_FIELDS = [
    "参与拟合的页签数量",
    "总数据点数量",
    "最大比生长速率μmax (h^-1)",
    "相对平台倍数 Krel (无量纲)",
    "拟合起始相对数量 Nrel(0) (无量纲)",
    "滞止期参数λ(h)",
    "拟合优度R² (Y=ln(Nrel)尺度)",
    "滞后期时长(h)",
    "对数期时长(h)",
    "稳定期时长(h)",
    "稳定期起始t90(h)",
]

# 腔室页签中「目标数量」所在列（0-based，A 列 = 0）
COUNTS_COLUMN_INDEX = 2

# 腔室页签中「总面积(μm²)」所在列（已按 0 h 归一化，可直接当相对值用）
AREA_COLUMN_INDEX = 3

# ---------------------------------------------------------------- 可视化图
# 参考实现 08_processed_batch.py 会把 matplotlib 图以 PNG 形式嵌入工作簿：
#   腔室页 4 张（H2/H24/R2/R24），汇总页 3 张（与「合并拟合结果」同一行，右侧横向排开）。
# 下列参数控制尺寸、分辨率与显示大小，均取自参考实现的实际输出。
PLOT_ENABLED = True                # False 则只写数据表，不嵌图
PLOT_DPI = 150                     # 出图分辨率（与显示尺寸无关）；150 约为 300 的 1/4 体积
PLOT_AREA_POINTS = 96              # 面积类图取前 96 个点（对齐参考实现的 valid_num=96）
PLOT_CHAMBER_SIZE = (600, 400)     # 腔室页图在 Excel 中的显示尺寸（像素）
PLOT_MERGED_SIZE = (700, 500)      # 汇总页图在 Excel 中的显示尺寸（像素）
PLOT_CHAMBER_ANCHORS = ("H2", "H24", "R2", "R24")   # 腔室页：单拟合 / 合并对比 / 总面积 / 平均面积

# 汇总页 3 张图的排布：
#   行   —— 「合并拟合结果」标题所在行（+ PLOT_MERGED_ROW_OFFSET）
#   列   —— 从 PLOT_MERGED_START_COLUMN 起，按实际列宽依次向右排，
#           每张图之间至少留 PLOT_MERGED_GAP_PX 像素，保证不重叠。
PLOT_MERGED_START_COLUMN = "C"     # 表格占 A/B 两列（字段名 / 数值），图从 C 列起
PLOT_MERGED_GAP_PX = 60            # 相邻两张图之间的最小空白（像素）
PLOT_MERGED_ROW_OFFSET = 0         # 相对「合并拟合结果」标题行的行偏移
PLOT_MERGED_BAND_WIDTH = 18        # 图区域未设宽度的列统一设为该宽度（默认 8.43 太窄会挤在一起）
PLOT_FONT_FAMILY = ["Times New Roman", "serif"]     # 参考实现使用的字体

# 参与合并拟合所需的最少数据点数（单个腔室）；不足则计入「未参与拟合数据」
MERGE_MIN_POINTS = 5

# 稳定期起始 t90 的判定阈值（达到平台倍数的比例）
MERGE_STABLE_FRACTION = 0.90

# ---------------------------------------------------------------- 异常腔室筛查
# 分组前按微藻生长规律剔除明显异常的腔室。判定顺序即下列顺序，
# 命中第一条即停止（原因写进「剔除腔室数据.xlsx」）。
# 阈值按「已被 batch_grouping 接受（标红）的 749 个腔室」的实际分布反推：
#   已接受组 fold 最低 4.05、end_peak 的 1% 分位 0.436、jumps 中位数 2。
# 下列阈值取在已接受组的边界稍外侧，只剔除比「专家从未接受过的」更差的腔室。
SCREEN_MIN_POINTS = 30            # 有效数据点少于该值 → 数据缺失
SCREEN_MAX_MISSING_RATIO = 0.20   # 「目标数量」列缺失/空值占比上限
SCREEN_MAX_FAIL_RATIO = 0.05      # 「处理状态」非「成功」的行占比上限
SCREEN_MIN_GROWTH_FOLD = 4.0      # 峰值/初始 < 该值 → 无生长（已接受组最低 4.05）
SCREEN_MIN_END_PEAK_RATIO = 0.40  # 末期均值/峰值 < 该值 → 长期下降（已接受组 p1 = 0.436）
SCREEN_MAX_DECLINE_RUN_RATIO = 0.35   # 最长连续下降段占比 > 该值 → 长期下降（实测最大 0.155）
SCREEN_JUMP_RATIO = 0.60          # 单点相对变化超过该比例记为一次「突跳」
SCREEN_MAX_JUMP_COUNT = 8         # 突跳次数 > 该值 → 剧烈非典型波动
SCREEN_MAX_NOISE_RATIO = 0.15     # 相对移动中位数(窗口 5)的残差中位数占比上限（实测最大 0.143）
SCREEN_ENABLE = True              # False 则跳过生长规律筛查
# batch_grouping 的严格候选条件（N0 > 0 且 R² > 0.9）不属于「生长异常」：
# 不达标的腔室仍保留在「未参与数据分组腔室数据.xlsx」里，只是不参与分组，
# 数量单独统计到日志。设为 True 可把它们一并写进「剔除腔室数据.xlsx」。
SCREEN_DROP_POOR_FIT = False
