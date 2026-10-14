import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from matplotlib.patches import Ellipse, Rectangle
import warnings

warnings.filterwarnings('ignore')
# 设置全局字体为Times New Roman，用于英文期刊
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 16
plt.rcParams['figure.dpi'] = 300
# 设置数学字体
import matplotlib

matplotlib.rcParams['mathtext.default'] = 'regular'
# ============================
# 数据准备：基于您提供的Excel数据
# ============================
# 创建数据框（基于您提供的Excel表格数据）
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
    # (光强, 藻种, μ, LUE, 策略, 文献)
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
    (360, "Chlorella vulgaris FACHB-31", 0.146, 3.3, "Static constant", "Literature"),  # Const-360
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
    # 额外补充的传统动态策略数据
    ("Dynamic", "Mixed species", 0.028, 7.88, "SWI-80-240", "Literature"),
    ("Dynamic", "Mixed species", 0.09, 6.37, "SWI-90-225-360", "Literature"),
    ("Dynamic", "Mixed species", 0.117, 8.54, "ILR-Iave689-τc20-ε0.8", "Literature")
]
# PHM12S (创新降光策略) - 1个点
phm12s_data = [("Dynamic downgrading", "Chlorella vulgaris", 0.129, 12.97, "PHM12S", "This study")]
# 将所有数据合并到一个数据框中
all_data = []
# 添加静态低光数据（标记为基准组）
for item in low_light_data:
    all_data.append(['Static low-light (I<150)'] + list(item))
# 添加静态中光数据（标记为传统策略）
for item in medium_light_data:
    all_data.append(['Static medium-light (150<I<400)'] + list(item))
# 添加静态高光数据（标记为传统策略）
for item in high_light_data:
    all_data.append(['Static high-light (I>400)'] + list(item))
# 添加传统动态光数据（标记为传统策略）
for item in dynamic_data:
    all_data.append(['Conventional dynamic'] + list(item))
# 添加PHM12S数据
for item in phm12s_data:
    all_data.append(['PHM12S'] + list(item))
# 创建DataFrame
df = pd.DataFrame(all_data, columns=[
    'Strategy_Type', 'Light_Intensity', 'Species', 'mu', 'LUE',
    'Regulation_Strategy', 'Reference'
])
# 确保数值列类型正确
df['mu'] = pd.to_numeric(df['mu'], errors='coerce')
df['LUE'] = pd.to_numeric(df['LUE'], errors='coerce')
# 计算综合效率指标
df['Combined_Efficiency'] = df['mu'] * df['LUE']
print("数据概览:")
print(f"总数据点数量: {len(df)}")
print(f"各策略类型数据点数量:")
print(df['Strategy_Type'].value_counts())
print()
# ============================
# 关键计算：传统中/高光及动态策略的相关系数
# ============================
# 定义参与相关性分析的策略类型
traditional_strategies_for_corr = [
    'Static medium-light (150<I<400)',
    'Static high-light (I>400)',
    'Conventional dynamic'
]
# 筛选传统策略数据（用于相关性分析）
traditional_for_corr = df[df['Strategy_Type'].isin(traditional_strategies_for_corr)]
# 计算Pearson相关系数
corr, p_value = stats.pearsonr(traditional_for_corr['mu'], traditional_for_corr['LUE'])
# 计算线性回归
slope, intercept, r_value, p_reg, std_err = stats.linregress(
    traditional_for_corr['mu'], traditional_for_corr['LUE']
)
print("=== 传统策略相关性分析结果 ===")
print(f"参与分析的数据点数: {len(traditional_for_corr)}")
print(f"Pearson相关系数 r: {corr:.4f}")
print(f"p-value: {p_value:.6f}")
print(f"线性回归斜率: {slope:.4f}")
print(f"线性回归截距: {intercept:.4f}")
print(f"回归线方程: LUE = {slope:.2f} × μ + {intercept:.2f}")
print()
# PHM12S数据
phm12s_point = df[df['Strategy_Type'] == 'PHM12S'].iloc[0]
phm12s_mu = phm12s_point['mu']
phm12s_lue = phm12s_point['LUE']
phm12s_ratio = phm12s_lue / phm12s_mu
# 计算PHM12S点到传统策略回归线的垂直距离
A, B, C = slope, -1, intercept
phm12s_distance = np.abs(A * phm12s_mu + B * phm12s_lue + C) / np.sqrt(A ** 2 + B ** 2)
print(f"PHM12S数据点:")
print(f"  μ = {phm12s_mu:.3f} h⁻¹")
print(f"  LUE = {phm12s_lue:.2f}%")
print(f"  效率比 (LUE/μ) = {phm12s_ratio:.1f}")
print(f"  偏离传统趋势距离 = {phm12s_distance:.3f}")
print()
# ============================
# Figure 5: Main dual-panel figure
# ============================
# 设置颜色和标记
strategy_colors = {
    'PHM12S': '#e41a1c',  # 红色 - 突出显示
    'Conventional dynamic': '#377eb8',  # 蓝色
    'Static high-light (I>400)': '#ff7f00',  # 橙色
    'Static medium-light (150<I<400)': '#984ea3',  # 紫色
    'Static low-light (I<150)': '#4daf4a'  # 绿色 - 作为基准，颜色较浅
}
strategy_markers = {
    'PHM12S': '*',  # 五角星
    'Conventional dynamic': '^',  # 三角形
    'Static high-light (I>400)': 'D',  # 菱形
    'Static medium-light (150<I<400)': 's',  # 正方形
    'Static low-light (I<150)': 'o'  # 圆形
}
# 创建图形
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
# ============================
# 左图: 三线分析 - 显示所有策略
# ============================
# 绘制散点（所有策略）
for strategy_type in df['Strategy_Type'].unique():
    strategy_data = df[df['Strategy_Type'] == strategy_type]

    # 对于静态低光策略，使用较低的alpha值以表示其为基准
    alpha = 0.4 if strategy_type == 'Static low-light (I<150)' else 0.8

    # 对于PHM12S，使用更大的标记
    size = 250 if strategy_type == 'PHM12S' else 70

    ax1.scatter(
        strategy_data['mu'],
        strategy_data['LUE'],
        s=size,
        c=strategy_colors[strategy_type],
        marker=strategy_markers[strategy_type],
        label=strategy_type,
        alpha=alpha,
        edgecolors='black' if strategy_type == 'PHM12S' else 'none',
        linewidth=1.5 if strategy_type == 'PHM12S' else 1.0,
        zorder=10 if strategy_type == 'PHM12S' else 5
    )
# 绘制三条关键参考线
μ_range = np.linspace(0, 0.15, 100)
# 线1：平衡参考线 LUE = 100*μ (理想平衡)
ax1.plot(μ_range, 100 * μ_range, '--', color='#4daf4a', linewidth=2.5, alpha=0.7,
         label='Ideal balance: LUE = 100 × μ')
# 线2：低效率线 LUE = 50*μ
ax1.plot(μ_range, 50 * μ_range, ':', color='#ff7f00', linewidth=2, alpha=0.5,
         label='Low efficiency: LUE = 50 × μ')
# 线3：高效率线 LUE = 200*μ
ax1.plot(μ_range, 200 * μ_range, '-.', color='#984ea3', linewidth=2, alpha=0.7,
         label='High efficiency: LUE = 200 × μ')
# 标记PHM12S的效率比
ax1.annotate(f'Efficiency ratio = {phm12s_ratio:.1f}',
             xy=(phm12s_mu, phm12s_lue),
             xytext=(phm12s_mu - 0.03, phm12s_lue + 2),
             arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
             fontsize=11, fontweight='bold',
             bbox=dict(boxstyle="round,pad=0.3", facecolor='white', edgecolor='red'))
# 添加"高效前沿区"标注
ax1.text(0.10, 13.5, 'High-μ–High-LUE\nFrontier Region', fontsize=11, fontweight='bold',
         bbox=dict(boxstyle="round,pad=0.3", facecolor='white', edgecolor='#e41a1c', alpha=0.8))
# 用矩形标注传统策略的性能范围
ax1.add_patch(Rectangle((0.08, 0), 0.07, 10,
                        alpha=0.05, color='gray',
                        label='Conventional performance envelope'))
ax1.set_xlabel('Average specific growth rate μ (h⁻¹)', fontsize=14, fontweight='bold')
ax1.set_ylabel('Light use efficiency LUE (%)', fontsize=14, fontweight='bold')
ax1.set_title('Efficiency Balance Analysis of All Strategies', fontsize=16, fontweight='bold', pad=20)
ax1.legend(loc='upper right', fontsize=9, ncol=2)
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.set_xlim(0, 0.15)
ax1.set_ylim(0, 15)
# ============================
# 右图: 相关性分析 - 聚焦传统高性能策略
# ============================
# 首先，用浅色显示静态低光策略（作为背景基准）
low_light_data = df[df['Strategy_Type'] == 'Static low-light (I<150)']
if not low_light_data.empty:
    ax2.scatter(low_light_data['mu'], low_light_data['LUE'],
                c='lightgray', marker='o', s=50,
                alpha=0.3, label='Static low-light (baseline)')
# 绘制传统中/高光及动态策略数据点（参与相关性分析）
for strategy_type in traditional_strategies_for_corr:
    strategy_data = df[df['Strategy_Type'] == strategy_type]
    ax2.scatter(strategy_data['mu'], strategy_data['LUE'],
                c=strategy_colors[strategy_type],
                marker=strategy_markers[strategy_type],
                s=80,
                label=strategy_type,
                alpha=0.8,
                edgecolors='none')
# 绘制传统策略的回归线
μ_fit = np.linspace(traditional_for_corr['mu'].min(),
                    traditional_for_corr['mu'].max(), 100)
lue_fit = slope * μ_fit + intercept
ax2.plot(μ_fit, lue_fit, '--', color='black', linewidth=2.5, alpha=0.7,
         label=f'Conventional trade-off line\n(r = {corr:.2f}, p = {p_value:.3f})')
# 添加回归线周围的置信区间阴影
# 计算预测值的标准误差
y_err = std_err * np.sqrt(1 / len(traditional_for_corr['mu']) +
                          (μ_fit - np.mean(traditional_for_corr['mu'])) ** 2 /
                          np.sum((traditional_for_corr['mu'] - np.mean(traditional_for_corr['mu'])) ** 2))
ax2.fill_between(μ_fit, lue_fit - 1.96 * y_err, lue_fit + 1.96 * y_err,
                 alpha=0.2, color='gray', label='95% confidence interval')
# 绘制PHM12S点
ax2.scatter(phm12s_mu, phm12s_lue,
            c='#e41a1c', marker='*', s=300,
            label='PHM12S (This study)',
            edgecolors='black', linewidth=2, zorder=10)
# 标记PHM12S的偏离距离
ax2.annotate(f'Deviation from conventional trend\n= {phm12s_distance:.2f}',
             xy=(phm12s_mu, phm12s_lue),
             xytext=(phm12s_mu - 0.04, phm12s_lue + 2),
             arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
             fontsize=11, fontweight='bold',
             bbox=dict(boxstyle="round,pad=0.3", facecolor='white', edgecolor='red'))
# 添加"传统性能边界"标注
ax2.text(0.08, 3.0, 'Conventional\nPerformance Boundary',
         fontsize=10, fontweight='bold', ha='center',
         bbox=dict(boxstyle="round,pad=0.3", facecolor='white', edgecolor='black', alpha=0.8))
# 添加"突破区域"标注
ax2.text(0.10, 11.5, 'Breakthrough Region\n(PHM12S)',
         fontsize=11, fontweight='bold', ha='center',
         bbox=dict(boxstyle="round,pad=0.3", facecolor='white', edgecolor='#e41a1c', alpha=0.8))
ax2.set_xlabel('Average specific growth rate μ (h⁻¹)', fontsize=14, fontweight='bold')
ax2.set_ylabel('Light use efficiency LUE (%)', fontsize=14, fontweight='bold')
ax2.set_title('Breaking the Conventional Trade-off: PHM12S vs. Traditional Strategies',
              fontsize=16, fontweight='bold', pad=20)
ax2.legend(loc='upper right', fontsize=9)
ax2.grid(True, alpha=0.3, linestyle='--')
ax2.set_xlim(0, 0.15)
ax2.set_ylim(0, 15)
plt.tight_layout()
plt.savefig('Figure5_Revised_Trade-off_Analysis.png', dpi=300, bbox_inches='tight')
plt.show()
# ============================
# 输出关键统计信息
# ============================
print("\n=== 关键统计信息汇总 ===")
print(f"传统策略数据点数 (用于相关性分析): {len(traditional_for_corr)}")
print(f"Pearson相关系数 r: {corr:.3f}")
print(f"p-value: {p_value:.4f}")
print(f"回归线方程: LUE = {slope:.2f} × μ + {intercept:.2f}")
print(f"PHM12S效率比 (LUE/μ): {phm12s_ratio:.1f}")
print(f"PHM12S偏离传统趋势距离: {phm12s_distance:.3f}")
print(f"PHM12S综合效率 (μ×LUE): {phm12s_mu * phm12s_lue:.3f}")
# 计算PHM12S相对于传统策略的改进
# 找到传统策略中最高的综合效率
traditional_max_efficiency = traditional_for_corr['Combined_Efficiency'].max()
phm12s_efficiency = phm12s_mu * phm12s_lue
improvement = ((phm12s_efficiency - traditional_max_efficiency) / traditional_max_efficiency) * 100
print(f"传统策略最高综合效率: {traditional_max_efficiency:.3f}")
print(f"PHM12S综合效率改进: {improvement:.1f}%")
# 分策略类型统计
print("\n=== 分策略类型统计 ===")
for strategy_type in df['Strategy_Type'].unique():
    strategy_data = df[df['Strategy_Type'] == strategy_type]
    print(f"\n{strategy_type}:")
    print(f"  数据点数: {len(strategy_data)}")
    print(f"  μ范围: [{strategy_data['mu'].min():.4f}, {strategy_data['mu'].max():.4f}]")
    print(f"  LUE范围: [{strategy_data['LUE'].min():.2f}, {strategy_data['LUE'].max():.2f}]")
    print(f"  平均μ: {strategy_data['mu'].mean():.4f}")
    print(f"  平均LUE: {strategy_data['LUE'].mean():.2f}%")
