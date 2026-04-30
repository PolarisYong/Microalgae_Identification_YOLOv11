import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from matplotlib.patches import Ellipse, Rectangle
import warnings
warnings.filterwarnings('ignore')

# 设置全局字体为Times New Roman，用于英文期刊
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 10  # 整体字体缩小1号，减少占用空间
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 9  # 图例字体缩小
plt.rcParams['figure.titlesize'] = 16
plt.rcParams['figure.dpi'] = 300

# 设置数学字体
import matplotlib
matplotlib.rcParams['mathtext.default'] = 'regular'

# ============================
# 数据准备：基于您提供的Excel数据
# ============================
data = {
    'Strategy_Type': [],  # 策略类型
    'Light_Intensity': [],  # 平均光强
    'Species': [],  # 藻种名称
    'mu': [],  # 平均比生长速率 μ (h⁻¹)
    'LUE': [],  # 光能利用率 LUE (%)
    'Regulation_Strategy': [],  # 光强调控策略
    'Reference': []  # 文献来源
}

# 静态低光 (I<150) - 11个点（作为低性能基准，不参与相关性分析）
low_light_data = [
    (18, "Microcystis spp.", 0.0026, 3.83, "Static constant", "New ref. 2"),
    (35, "Microcystis aeruginosa FACHB-1203", 0.002, 1.7, "Static constant", "New ref. 5"),
    (40, "Chlorella emersonii", 0.00125, 0.3, "Static constant", "New ref. 7"),
    (54, "Microcystis spp.", 0.0034, 1.77, "Static constant", "New ref. 2"),
    (68, "Chlorella emersonii", 0.01167, 2.1, "Static constant", "New ref. 7"),
    (70, "Microcystis aeruginosa FACHB-1203", 0.00433, 2.09, "Static constant", "New ref. 5"),
    (82, "Chlorella sp. (ATCC 14854)", 0.0111, 2.59, "Static constant", "New ref. 2"),
    (100, "Scenedesmus quadricauda", 0.0191, 4.61, "Static constant", "New ref. 1"),
    (105, "Microcystis aeruginosa FACHB-1203", 0.00628, 2.35, "Static constant", "New ref. 5"),
    (108, "Microcystis spp.", 0.0038, 1.03, "Static constant", "New ref. 2"),
    (135, "Chlorella emersonii", 0.02667, 3.8, "Static constant", "New ref. 7")
]

# 静态中光 (150<I<400) - 10个点（参与相关性分析）
medium_light_data = [
    (160, "Chlorella vulgaris FACHB-31", 0.08973, 6.6, "Static constant", "New ref. 8"),
    (200, "Scenedesmus quadricauda", 0.0212, 2.91, "Static constant", "New ref. 1"),
    (216, "Microcystis spp.", 0.0039, 0.54, "Static constant", "New ref. 2"),
    (260, "Chlorella sp. (ATCC 14854)", 0.0185, 2.04, "Static constant", "New ref. 2"),
    (260, "Chlorella vulgaris FACHB-31", 0.10059, 4.62, "Static constant", "New ref. 8"),
    (300, "Scenedesmus quadricauda", 0.0185, 1.45, "Static constant", "New ref. 1"),
    (300, "Chlorella sp. (ATCC 14854)", 0.0202, 1.35, "Static constant", "New ref. 2"),
    (360, "Chlorella vulgaris FACHB-31", 0.11395, 4.59, "Static constant", "New ref. 8"),
    (360, "Chlorella vulgaris FACHB-31", 0.146, 3.3, "Static constant", "Literature"),
    (368, "Chlorella sp. (ATCC 14854)", 0.0194, 1.51, "Static constant", "New ref. 2")
]

# 静态高光 (I>400) - 8个点（参与相关性分析）
high_light_data = [
    (460, "Chlorella vulgaris FACHB-31", 0.1236, 3.75, "Static constant", "New ref. 8"),
    (560, "Chlorella vulgaris FACHB-31", 0.13437, 3.4, "Static constant", "New ref. 8"),
    (590, "Chlorella sp. (ATCC 14854)", 0.0217, 1.3, "Static constant", "New ref. 2"),
    (635, "Arthrospira platensis BP", 0.0146, 6.56, "Static constant", "New ref. 4"),
    (660, "Chlorella vulgaris FACHB-31", 0.11102, 2.05, "Static constant", "New ref. 8"),
    (980, "Arthrospira platensis BP", 0.0208, 4.4, "Static constant", "New ref. 4"),
    (1300, "Arthrospira platensis BP", 0.0254, 4.12, "Static constant", "New ref. 4"),
    (2300, "Arthrospira platensis BP", 0.0104, 1.57, "Static constant", "New ref. 4")
]

# 传统动态光 - 18个点（参与相关性分析）
dynamic_data = [
    (100, "Porphyridium purpureum", 0.01, 5.39, "Dynamic stepwise", "New ref. 1"),
    (116.7, "Chlorella sp. (ATCC 14854)", 0.0149, 2.9, "Dynamic stepwise", "New ref. 2"),
    (135, "Porphyridium purpureum", 0.01, 9.92, "Dynamic stepwise", "New ref. 1"),
    (138, "Porphyridium purpureum", 0.009, 2.34, "Dynamic stepwise", "New ref. 1"),
    (141.7, "Chlorella sp. (ATCC 14854)", 0.0197, 3.18, "Dynamic stepwise", "New ref. 2"),
    (141.7, "Chlorella sp. (ATCC 14854)", 0.0238, 3.84, "Dynamic stepwise", "New ref. 2"),
    (158.3, "Chlorella sp. (ATCC 14854)", 0.0168, 2.46, "Dynamic stepwise", "New ref. 2"),
    (158.3, "Chlorella sp. (ATCC 14854)", 0.0221, 3.24, "Dynamic stepwise", "New ref. 2"),
    (158.3, "Chlorella sp. (ATCC 14854)", 0.0261, 3.82, "Dynamic stepwise", "New ref. 2"),
    (173.3, "Chlorella sp. (ATCC 14854)", 0.021, 2.78, "Dynamic stepwise", "New ref. 2"),
    (209.5, "Chlorella sp. (ATCC 14854)", 0.0279, 3.14, "Dynamic stepwise", "New ref. 2"),
    (250, "Chlorella sp. (ATCC 14854)", 0.0259, 2.38, "Dynamic stepwise", "New ref. 2"),
    (336.7, "Chlorella sp. (ATCC 14854)", 0.0195, 1.36, "Dynamic stepwise", "New ref. 2"),
    (540, "Porphyridium purpureum", 0.00842, 1.5, "Dynamic stepwise", "New ref. 1"),
    (550, "Chlorella sp. (ATCC 14854)", 0.0225, 0.96, "Dynamic stepwise", "New ref. 2"),
    (1160, "Porphyridium purpureum", 0.00842, 1.5, "Dynamic stepwise", "New ref. 1"),
    ("Dynamic", "Mixed species", 0.028, 7.88, "SWI-80-240", "Literature"),
    ("Dynamic", "Mixed species", 0.09, 6.37, "SWI-90-225-360", "Literature"),
    ("Dynamic", "Mixed species", 0.117, 8.54, "ILR-Iave689-τc20-ε0.8", "Literature")
]

# PHM12S (创新降光策略) - 1个点
phm12s_data = [("Dynamic downgrading", "Chlorella vulgaris", 0.129, 12.97, "PHM12S", "This study")]

# 将所有数据合并到一个数据框中
all_data = []
for item in low_light_data:
    all_data.append(['Static low-light (I<150)'] + list(item))
for item in medium_light_data:
    all_data.append(['Static medium-light (150<I<400)'] + list(item))
for item in high_light_data:
    all_data.append(['Static high-light (I>400)'] + list(item))
for item in dynamic_data:
    all_data.append(['Conventional dynamic'] + list(item))
for item in phm12s_data:
    all_data.append(['PHM12S'] + list(item))

# 创建DataFrame并处理数据类型
df = pd.DataFrame(all_data, columns=[
    'Strategy_Type', 'Light_Intensity', 'Species', 'mu', 'LUE',
    'Regulation_Strategy', 'Reference'
])
df['mu'] = pd.to_numeric(df['mu'], errors='coerce')
df['LUE'] = pd.to_numeric(df['LUE'], errors='coerce')
df['Combined_Efficiency'] = df['mu'] * df['LUE']

# ============================
# 关键计算：传统中/高光及动态策略的相关系数
# ============================
traditional_strategies_for_corr = [
    'Static medium-light (150<I<400)',
    'Static high-light (I>400)',
    'Conventional dynamic'
]
traditional_for_corr = df[df['Strategy_Type'].isin(traditional_strategies_for_corr)]
corr, p_value = stats.pearsonr(traditional_for_corr['mu'], traditional_for_corr['LUE'])
slope, intercept, r_value, p_reg, std_err = stats.linregress(
    traditional_for_corr['mu'], traditional_for_corr['LUE']
)

# PHM12S数据提取
phm12s_point = df[df['Strategy_Type'] == 'PHM12S'].iloc[0]
phm12s_mu = phm12s_point['mu']
phm12s_lue = phm12s_point['LUE']
phm12s_ratio = phm12s_lue / phm12s_mu
A, B, C = slope, -1, intercept
phm12s_distance = np.abs(A * phm12s_mu + B * phm12s_lue + C) / np.sqrt(A ** 2 + B ** 2)

# ============================
# 核心修改：优化颜色、标记和布局配置
# ============================
# 简化策略名称（减少图例文本长度）
strategy_labels = {
    'PHM12S': 'PHM12S (This study)',
    'Conventional dynamic': 'Conventional dynamic',
    'Static high-light (I>400)': 'Static high-light',
    'Static medium-light (150<I<400)': 'Static medium-light',
    'Static low-light (I<150)': 'Static low-light (baseline)'
}

# 颜色和标记保持不变，确保辨识度
strategy_colors = {
    'PHM12S': '#e41a1c',
    'Conventional dynamic': '#377eb8',
    'Static high-light (I>400)': '#ff7f00',
    'Static medium-light (150<I<400)': '#984ea3',
    'Static low-light (I<150)': '#4daf4a'
}
strategy_markers = {
    'PHM12S': '*',
    'Conventional dynamic': '^',
    'Static high-light (I>400)': 'D',
    'Static medium-light (150<I<400)': 's',
    'Static low-light (I<150)': 'o'
}

# ============================
# 优化后的双面板图表
# ============================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# ============================
# 左图: 三线分析 - 解决标签重叠的核心修改
# ============================
# 1. 绘制散点（调整大小，避免点重叠）
for strategy_type in df['Strategy_Type'].unique():
    strategy_data = df[df['Strategy_Type'] == strategy_type]
    alpha = 0.4 if strategy_type == 'Static low-light (I<150)' else 0.8
    size = 200 if strategy_type == 'PHM12S' else 60  # 减小散点大小，避免遮挡
    ax1.scatter(
        strategy_data['mu'],
        strategy_data['LUE'],
        s=size,
        c=strategy_colors[strategy_type],
        marker=strategy_markers[strategy_type],
        label=strategy_labels[strategy_type],  # 使用简化标签
        alpha=alpha,
        edgecolors='black' if strategy_type == 'PHM12S' else 'none',
        linewidth=1.2 if strategy_type == 'PHM12S' else 0.8,
        zorder=10 if strategy_type == 'PHM12S' else 5
    )

# 2. 绘制参考线（保持不变）
μ_range = np.linspace(0, 0.15, 100)
ax1.plot(μ_range, 100 * μ_range, '--', color='#4daf4a', linewidth=2.5, alpha=0.7,
         label='Ideal balance: LUE=100×μ')  # 简化线标签
ax1.plot(μ_range, 50 * μ_range, ':', color='#ff7f00', linewidth=2, alpha=0.5,
         label='Low efficiency: LUE=50×μ')
ax1.plot(μ_range, 200 * μ_range, '-.', color='#984ea3', linewidth=2, alpha=0.7,
         label='High efficiency: LUE=200×μ')

# 3. 优化PHM12S标注位置（向右上方移动，避免遮挡）
ax1.annotate(f'Efficiency ratio = {phm12s_ratio:.1f}',
             xy=(phm12s_mu, phm12s_lue),
             xytext=(phm12s_mu + 0.005, phm12s_lue + 0.8),  # 调整文本位置
             arrowprops=dict(arrowstyle='->', color='red', lw=1.2, alpha=0.8),
             fontsize=9, fontweight='bold',
             bbox=dict(boxstyle="round,pad=0.2", facecolor='white', edgecolor='red', alpha=0.9))

# 4. 调整"高效前沿区"标注位置（向左移动，避免与图例重叠）
ax1.text(0.085, 13.8, 'High-μ–High-LUE\nFrontier', fontsize=9, fontweight='bold',  # 简化文本
         bbox=dict(boxstyle="round,pad=0.2", facecolor='white', edgecolor='#e41a1c', alpha=0.8))

# 5. 调整矩形标注（缩小范围，避免遮挡数据）
ax1.add_patch(Rectangle((0.08, 0.5), 0.065, 9,  # 调整矩形位置和大小
                        alpha=0.05, color='gray',
                        label='Conventional performance'))

# 6. 图例位置调整到左上角（原右上角重叠，左上角空间充足）
ax1.legend(loc='upper left', fontsize=8.5, ncol=1, framealpha=0.9)  # 单列布局，减少宽度

# 7. 轴标签和标题（保持不变）
ax1.set_xlabel('Average specific growth rate μ (h⁻¹)', fontsize=14, fontweight='bold')
ax1.set_ylabel('Light use efficiency LUE (%)', fontsize=14, fontweight='bold')
ax1.set_title('Efficiency Balance Analysis of All Strategies', fontsize=16, fontweight='bold', pad=20)
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.set_xlim(0, 0.15)
ax1.set_ylim(0, 15)

# ============================
# 右图: 相关性分析 - 核心修改
# ============================
# 1. 绘制静态低光基准（保持浅色背景）
low_light_data = df[df['Strategy_Type'] == 'Static low-light (I<150)']
if not low_light_data.empty:
    ax2.scatter(low_light_data['mu'], low_light_data['LUE'],
                c='lightgray', marker='o', s=40, alpha=0.3,
                label=strategy_labels['Static low-light (I<150)'])

# 2. 绘制传统策略数据点（减小大小）
for strategy_type in traditional_strategies_for_corr:
    strategy_data = df[df['Strategy_Type'] == strategy_type]
    ax2.scatter(strategy_data['mu'], strategy_data['LUE'],
                c=strategy_colors[strategy_type],
                marker=strategy_markers[strategy_type],
                s=60,  # 减小散点大小
                label=strategy_labels[strategy_type],
                alpha=0.8,
                edgecolors='none')

# 3. 绘制回归线和置信区间（保持不变）
μ_fit = np.linspace(traditional_for_corr['mu'].min(), traditional_for_corr['mu'].max(), 100)
lue_fit = slope * μ_fit + intercept
ax2.plot(μ_fit, lue_fit, '--', color='black', linewidth=2.5, alpha=0.7,
         label=f'Conventional trade-off (r={corr:.2f}, p={p_value:.3f})')  # 简化标签
y_err = std_err * np.sqrt(1 / len(traditional_for_corr['mu']) +
                          (μ_fit - np.mean(traditional_for_corr['mu'])) ** 2 /
                          np.sum((traditional_for_corr['mu'] - np.mean(traditional_for_corr['mu'])) ** 2))
ax2.fill_between(μ_fit, lue_fit - 1.96 * y_err, lue_fit + 1.96 * y_err,
                 alpha=0.2, color='gray', label='95% CI')  # 简化置信区间标签

# 4. 绘制PHM12S点（保持突出）
ax2.scatter(phm12s_mu, phm12s_lue,
            c='#e41a1c', marker='*', s=250,
            label=strategy_labels['PHM12S'],
            edgecolors='black', linewidth=1.5, zorder=10)

# 5. 优化PHM12S偏离距离标注（向右上方移动）
ax2.annotate(f'Deviation = {phm12s_distance:.2f}',  # 简化文本
             xy=(phm12s_mu, phm12s_lue),
             xytext=(phm12s_mu + 0.005, phm12s_lue + 0.8),  # 调整位置
             arrowprops=dict(arrowstyle='->', color='red', lw=1.2, alpha=0.8),
             fontsize=9, fontweight='bold',
             bbox=dict(boxstyle="round,pad=0.2", facecolor='white', edgecolor='red', alpha=0.9))

# 6. 调整文本标注位置（分散布局）
ax2.text(0.015, 3.0, 'Conventional\nPerformance Boundary',  # 向左移动
         fontsize=9, fontweight='bold', ha='left',
         bbox=dict(boxstyle="round,pad=0.2", facecolor='white', edgecolor='black', alpha=0.8))
ax2.text(0.085, 13.8, 'Breakthrough\nRegion',  # 简化文本，调整位置
         fontsize=9, fontweight='bold', ha='center',
         bbox=dict(boxstyle="round,pad=0.2", facecolor='white', edgecolor='#e41a1c', alpha=0.8))

# 7. 图例位置调整到左下角（避免与数据点重叠）
ax2.legend(loc='lower left', fontsize=8.5, ncol=1, framealpha=0.9)

# 8. 轴标签和标题（保持不变）
ax2.set_xlabel('Average specific growth rate μ (h⁻¹)', fontsize=14, fontweight='bold')
ax2.set_ylabel('Light use efficiency LUE (%)', fontsize=14, fontweight='bold')
ax2.set_title('Breaking the Conventional Trade-off: PHM12S vs. Traditional Strategies',
              fontsize=16, fontweight='bold', pad=20)
ax2.grid(True, alpha=0.3, linestyle='--')
ax2.set_xlim(0, 0.15)
ax2.set_ylim(0, 15)

# 调整子图间距，避免整体重叠
plt.tight_layout(pad=3.0)
plt.savefig('Figure5_Revised_Trade-off_Analysis_Fixed.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================
# 输出关键统计信息（保持不变）
# ============================
print("=== 关键统计信息汇总 ===")
print(f"传统策略数据点数 (用于相关性分析): {len(traditional_for_corr)}")
print(f"Pearson相关系数 r: {corr:.3f}")
print(f"p-value: {p_value:.4f}")
print(f"回归线方程: LUE = {slope:.2f} × μ + {intercept:.2f}")
print(f"PHM12S效率比 (LUE/μ): {phm12s_ratio:.1f}")
print(f"PHM12S偏离传统趋势距离: {phm12s_distance:.3f}")
print(f"PHM12S综合效率 (μ×LUE): {phm12s_mu * phm12s_lue:.3f}")

traditional_max_efficiency = traditional_for_corr['Combined_Efficiency'].max()
phm12s_efficiency = phm12s_mu * phm12s_lue
improvement = ((phm12s_efficiency - traditional_max_efficiency) / traditional_max_efficiency) * 100
print(f"传统策略最高综合效率: {traditional_max_efficiency:.3f}")
print(f"PHM12S综合效率改进: {improvement:.1f}%")

print("\n=== 分策略类型统计 ===")
for strategy_type in df['Strategy_Type'].unique():
    strategy_data = df[df['Strategy_Type'] == strategy_type]
    print(f"\n{strategy_type}:")
    print(f"  数据点数: {len(strategy_data)}")
    print(f"  μ范围: [{strategy_data['mu'].min():.4f}, {strategy_data['mu'].max():.4f}]")
    print(f"  LUE范围: [{strategy_data['LUE'].min():.2f}, {strategy_data['LUE'].max():.2f}]")
    print(f"  平均μ: {strategy_data['mu'].mean():.4f}")
    print(f"  平均LUE: {strategy_data['LUE'].mean():.2f}%")