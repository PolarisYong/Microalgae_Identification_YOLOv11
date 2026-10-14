# 导入所需库
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Circle
import matplotlib.transforms as transforms
from matplotlib.font_manager import FontProperties
# 1. 基础文本字体（文本模式）
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
plt.rcParams['axes.unicode_minus'] = False
# 2. 关键：设置数学模式字体为Times New Roman
plt.rcParams['mathtext.fontset'] = 'custom'  # 自定义数学字体
plt.rcParams['mathtext.rm'] = 'Times New Roman'  # 数学模式罗马体=Times New Roman
plt.rcParams['mathtext.it'] = 'Times New Roman:italic'  # 斜体（可选，保持一致）
plt.rcParams['mathtext.bf'] = 'Times New Roman:bold'  # 粗体（可选）

# 定义Times New Roman字体属性
tnr_font = FontProperties(
    family='Times New Roman',  # 衬线字体类别
)


# 定义95%置信椭圆函数（仅用于传统主动调光组）
def confidence_ellipse(x, y, ax, n_std=1.96, facecolor='none', **kwargs):
    cov = np.cov(x, y)
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, **kwargs)
    mean_x = np.mean(x)
    mean_y = np.mean(y)
    scale_x = np.sqrt(cov[0, 0]) * n_std
    scale_y = np.sqrt(cov[1, 1]) * n_std
    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)
    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)

# -------------------------- 2. 读取所有数据（已修正路径格式）--------------------------
# 传统策略组（静态+动态）
df_traditional = pd.read_excel(r'dataset\传统光策略 完整数据集 final.xlsx')
# 新策略PHM12S
df_new = pd.read_excel(r'dataset\新策略 PHM12S 的单点数据.xlsx')
# 恒定低光基准组（替换为你的实际路径）
df_low_light = pd.read_excel(r'dataset\恒定低光组 完整数据集 final.xlsx')

# 数据筛选
static_high = df_traditional[df_traditional['策略类型'] == "传统静态提高光强策略"]  # 传统静态调光组(n=9)
traditional_dynamic = df_traditional[df_traditional['策略类型'] == "传统动态光策略"]  # 传统动态调光组(n=20)
phm12s = df_new.iloc[0]  # PHM12S(n=1)
low_light = df_low_light  # 恒定低光组(n=11)

# 传统主动调光组合并数据（仅静态+动态，用于绘制置信椭圆，共29样本）
df_traditional_active = pd.concat([static_high, traditional_dynamic], ignore_index=True)
x_traditional_active = df_traditional_active['平均u (h-1)']
y_traditional_active = df_traditional_active['LUE (%)']

# -------------------------- 1. 提取图表/数据核心参数 --------------------------
# 步骤1：获取传统组的统计量（从原始数据/图中估算）
mean_x = 0.08  # 灰色椭圆中心的μ值（横轴≈0.08）
mean_y = 4.0   # 灰色椭圆中心的LUE值（纵轴≈4.0）
# 协方差矩阵（从原始数据计算，或根据椭圆扁率/方向估算）
cov = np.array([[0.0016, -0.012],  # cov(μ,μ)、cov(μ,LUE)（负相关匹配图中椭圆方向）
                [-0.012,  4.0]])   # cov(LUE,μ)、cov(LUE,LUE)
cov_inv = np.linalg.inv(cov)       # 协方差矩阵的逆（马氏距离核心）

# 步骤2：红星（PHM12S）的坐标（从图中读取）
red_star_x = 0.18  # 红星横轴≈0.18 h⁻¹
red_star_y = 9.0   # 红星纵轴≈9.0 %

# -------------------------- 2. 计算统计偏离度 --------------------------
def calc_mahalanobis_deviation(x_p, y_p, mean_x, mean_y, cov_inv):
    """计算点相对于95%置信椭圆的统计偏离度"""
    # 构造坐标向量
    p = np.array([[x_p], [y_p]])    # 红星坐标
    mu = np.array([[mean_x], [mean_y]])  # 椭圆中心（传统组均值）
    # 核心：计算马氏距离（MD）
    diff = p - mu
    md_squared = diff.T @ cov_inv @ diff  # 马氏距离的平方
    md = np.sqrt(md_squared[0, 0])
    # 转换为百分比偏离度（1.96是95%置信对应的马氏距离阈值）
    # >0 = 椭圆外，=0 = 椭圆上，<0 = 椭圆内
    deviation = (md - 1.96) / 1.96 * 100
    return md, deviation

# 计算红星的偏离度
md_star, dev_star = calc_mahalanobis_deviation(red_star_x, red_star_y, mean_x, mean_y, cov_inv)
print(f"红星的马氏距离：{md_star:.4f}")  # 量化偏离分布中心的程度
print(f"红星相对于95%置信椭圆的偏离度：{dev_star:.4f}%")  # 直观的百分比偏离

# 动态补光+PHM12S合并数据（用于相关性标注）
dynamic_plus_phm = pd.concat([traditional_dynamic, df_new], ignore_index=True)

# -------------------------- 3. 创建2张子图（横向排列）--------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), dpi=100)
fig.suptitle('μ-LUE Distribution, Correlation and Efficiency Trade-off of Different Light Strategies',
             fontsize=16, fontweight='bold', y=1.02)

# -------------------------- 子图a（左图）：μ-LUE分布及相关性 --------------------------
# 3.1 绘制各策略散点
# 恒定低光基准组
ax1.scatter(
    low_light['平均u (h-1)'], low_light['LUE (%)'],
    c='forestgreen', marker='s', s=60, alpha=0.8,
    label='Constant low-$I_0$ group (n=11, CL01$\sim$11)',
    edgecolors='darkgreen', linewidth=0.5
)
# 传统静态调光组
ax1.scatter(
    static_high['平均u (h-1)'], static_high['LUE (%)'],
    c='steelblue', marker='o', s=60, alpha=0.7,
    label='Traditional static $I_0$ regulation group (n=9, ST01$\sim$09)',
    edgecolors='darkblue', linewidth=0.5
)
# 传统动态调光组
ax1.scatter(
    traditional_dynamic['平均u (h-1)'], traditional_dynamic['LUE (%)'],
    c='darkorange', marker='^', s=60, alpha=0.7,
    label='Traditional dynamic $I_0$ regulation group (n=17, DY01$\sim$17)',
    edgecolors='darkred', linewidth=0.5
)
# 本研究创新型PHM12S策略
ax1.scatter(
    phm12s['平均u (h-1)'], phm12s['LUE (%)'],
    c='crimson', marker='*', s=300,
    label='Innovative PHM12S strategy from this study (n=1)',
    edgecolors='black', linewidth=1
)

# 3.2 绘制传统主动调光组95%置信椭圆
confidence_ellipse(
    x_traditional_active, y_traditional_active, ax1,
    n_std=1.96, facecolor='#ECEFF1', alpha=0.35,
    edgecolor='#666666', linestyle=':', linewidth=1.6,
    label='95% confidence ellipse for the traditional $I_0$ regulation strategies (n=26)'
)

# 3.3 添加各组回归线
# 恒定低光组
low_light_z = np.polyfit(low_light['平均u (h-1)'], low_light['LUE (%)'], 1)
low_light_p = np.poly1d(low_light_z)
ax1.plot(low_light['平均u (h-1)'], low_light_p(low_light['平均u (h-1)']),
         color='#2E7D32', linestyle='-', linewidth=1.2)
# 传统静态调光组
static_z = np.polyfit(static_high['平均u (h-1)'], static_high['LUE (%)'], 1)
static_p = np.poly1d(static_z)
ax1.plot(static_high['平均u (h-1)'], static_p(static_high['平均u (h-1)']),
         color='#1976D2', linestyle='-', linewidth=1.2)
# 传统动态调光组
dynamic_z = np.polyfit(traditional_dynamic['平均u (h-1)'], traditional_dynamic['LUE (%)'], 1)
dynamic_p = np.poly1d(dynamic_z)
ax1.plot(traditional_dynamic['平均u (h-1)'], dynamic_p(traditional_dynamic['平均u (h-1)']),
         color='#F57C00', linestyle='-', linewidth=1.2)

# 3.6 子图a细节优化
ax1.set_xlabel(r'Average specific growth rate of biomass ($\bar{\mu}$, h$^{-1}$)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Light use efficiency (LUE, %)', fontsize=12, fontweight='bold')
ax1.set_title('(a) μ-LUE Distribution and Correlation', fontsize=13, fontweight='bold')
ax1.set_xlim(-0.01, 0.20)
ax1.set_ylim(-1, 20)
ax1.set_xticks(np.arange(0, 0.20, 0.04))
ax1.set_yticks(np.arange(0, 20, 4))
ax1.grid(False)
ax1.legend(loc='upper left', fontsize=9, framealpha=0.9, bbox_to_anchor=(0.01, 0.99), prop=tnr_font)
# 图注说明
ax1.text(0.01, -1.2, 'Dashed lines represent linear regression lines of each group', fontsize=9, color='black',
         bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8),
         transform=ax1.transAxes)

# -------------------------- 子图b（右图）：μ-LUE效率权衡及高效区 --------------------------
# 4.1 绘制高μ-高LUE高效区
x_high = np.linspace(0.08, 0.22, 100)
high_patch = ax2.fill_between(x_high, 8, 20, color='#FFB6C1', alpha=0.2, label=r'High $\bar{\mu}$-high LUE high-efficiency region')

# 4.2 绘制所有原始数据点
# 恒定低光组
scatter_low = ax2.scatter(
    low_light['平均u (h-1)'], low_light['LUE (%)'],
    c='forestgreen', marker='s', s=60, alpha=0.5,
    label='Constant low-$I_0$ group (n=11, CL01$\sim$11)', edgecolors='darkgreen', linewidth=0.5
)
# 传统静态调光组
scatter_static = ax2.scatter(
    static_high['平均u (h-1)'], static_high['LUE (%)'],
    c='steelblue', marker='o', s=50, alpha=0.5,
    label='Traditional static $I_0$ regulation group (n=9, ST01$\sim$09)', edgecolors='darkblue', linewidth=0.5
)
# 传统动态调光组
scatter_dynamic = ax2.scatter(
    traditional_dynamic['平均u (h-1)'], traditional_dynamic['LUE (%)'],
    c='darkorange', marker='^', s=50, alpha=0.5,
    label='Traditional dynamic $I_0$ regulation group (n=17, DY01$\sim$17)', edgecolors='darkred', linewidth=0.5
)
# PHM12S策略
scatter_phm = ax2.scatter(
    phm12s['平均u (h-1)'], phm12s['LUE (%)'],
    c='crimson', marker='*', s=300,
    label='Innovative PHM12S strategy from this study (n=1)', edgecolors='black', linewidth=1
)

# 4.3 绘制效率参考线
x_ref = np.linspace(0, 0.22, 100)
line_low = ax2.plot(x_ref, 50 * x_ref, color='#b0b0b0', linestyle='--', linewidth=1.5, label=r'$\eta_r$ = 50')
line_balance = ax2.plot(x_ref, 100 * x_ref, color='#ff7f0e', linestyle='-.', linewidth=1.5, label=r'$\eta_r$ = 100')
line_high = ax2.plot(x_ref, 200 * x_ref, color='#2ca02c', linestyle='--', linewidth=1.5, label=r'$\eta_r$ = 200')

# 4.4 双图例设置
# 左上角散点图例
scatter_handles = [scatter_low, scatter_static, scatter_dynamic, scatter_phm]
scatter_labels = [h.get_label() for h in scatter_handles]
scatter_legend = ax2.legend(
    handles=scatter_handles,
    labels=scatter_labels,
    loc='upper left',
    fontsize=9,
    framealpha=0.9,
    bbox_to_anchor=(0.01, 0.99)
)
ax2.add_artist(scatter_legend)
# 右下角效率线图例
line_handles = line_low + line_balance + line_high
line_labels = [h.get_label() for h in line_handles]
ax2.legend(handles=line_handles, labels=line_labels,
           loc='lower right', fontsize=9, framealpha=0.9, bbox_to_anchor=(1.005, -0.005),
           title=r'Isopleths of efficiency ratio ($\eta_r$ = LUE / $\bar{\mu}$)', title_fontsize=9)

# 4.5 子图b细节优化
ax2.set_xlabel(r'Average specific growth rate of biomass ($\bar{\mu}$, h$^{-1}$)', fontsize=12, fontweight='bold')
ax2.set_ylabel('Light use efficiency (LUE, %)', fontsize=12, fontweight='bold')
ax2.set_title('(b) μ-LUE Efficiency Trade-off & High-Efficiency Region', fontsize=13, fontweight='bold')
ax2.set_xlim(-0.01, 0.20)
ax2.set_ylim(-1, 20)
ax2.set_xticks(np.arange(0, 0.20, 0.04))
ax2.set_yticks(np.arange(0, 20, 4))
ax2.grid(False)

# -------------------------- 4. 保存高清图表 --------------------------
plt.tight_layout()
plt.savefig(r'不同光策略 μ-LUE 分布相关性与效率权衡_双图绘制_双图.png',
            dpi=600, bbox_inches='tight', facecolor='white')
plt.close()

# 运行完成提示
print("\nFinal version of the fully optimized dual plot has been generated!")
print(f"Data verification: Constant Low Light (n={len(low_light)}), Static High Light (n={len(static_high)}), Dynamic Light (n={len(traditional_dynamic)}), Dynamic + PHM12S (n={len(dynamic_plus_phm)})")