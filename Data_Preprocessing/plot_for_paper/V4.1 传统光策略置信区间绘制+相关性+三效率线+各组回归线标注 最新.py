# 导入所需库
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Circle
import matplotlib.transforms as transforms

# -------------------------- 1. 基础设置（解决中文/负号显示）--------------------------
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.family'] = 'sans-serif'

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
df_traditional = pd.read_excel(r'E:\pythonProject\Microalgae_Identification_YOLOv11\Data_Preprocessing\plot_for_paper\dataset\传统光策略 完整数据集 final.xlsx')
# 新策略PHM12S
df_new = pd.read_excel(r'E:\pythonProject\Microalgae_Identification_YOLOv11\Data_Preprocessing\plot_for_paper\dataset\新策略 PHM12S 的单点数据.xlsx')
# 恒定低光基准组（替换为你的实际路径）
df_low_light = pd.read_excel(r'E:\pythonProject\Microalgae_Identification_YOLOv11\Data_Preprocessing\plot_for_paper\dataset\恒定低光组 完整数据集 final.xlsx')

# 数据筛选
static_high = df_traditional[df_traditional['策略类型'] == "传统静态提高光强策略"]  # 传统静态调光组(n=9)
traditional_dynamic = df_traditional[df_traditional['策略类型'] == "传统动态光策略"]  # 传统动态调光组(n=20)
phm12s = df_new.iloc[0]  # PHM12S(n=1)
low_light = df_low_light  # 恒定低光组(n=11)

# 传统主动调光组合并数据（仅静态+动态，用于绘制置信椭圆，共29样本）
df_traditional_active = pd.concat([static_high, traditional_dynamic], ignore_index=True)
x_traditional_active = df_traditional_active['平均u (h-1)']
y_traditional_active = df_traditional_active['LUE (%)']

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
    c='forestgreen', marker='s', s=80, alpha=0.8,
    label='恒定低光组（n=11）',
    edgecolors='darkgreen', linewidth=0.8
)
# 传统动态调光组
ax1.scatter(
    traditional_dynamic['平均u (h-1)'], traditional_dynamic['LUE (%)'],
    c='darkorange', marker='^', s=60, alpha=0.7,
    label='传统动态调光组（n=20）',
    edgecolors='darkred', linewidth=0.5
)
# 传统静态调光组
ax1.scatter(
    static_high['平均u (h-1)'], static_high['LUE (%)'],
    c='steelblue', marker='o', s=60, alpha=0.7,
    label='传统静态调光组（n=9）',
    edgecolors='darkblue', linewidth=0.5
)
# 本研究创新型PHM12S策略
ax1.scatter(
    phm12s['平均u (h-1)'], phm12s['LUE (%)'],
    c='crimson', marker='*', s=300,
    label='本研究创新型PHM12S策略（n=1）',
    edgecolors='black', linewidth=2
)

# 3.2 绘制传统主动调光组95%置信椭圆（深灰色边界+低透明度填充）
confidence_ellipse(
    x_traditional_active, y_traditional_active, ax1,
    n_std=1.96, facecolor='#ECEFF1', alpha=0.2,
    edgecolor='#666666', linestyle='--', linewidth=1.5,  # 深灰色边界
    label='传统主动调光策略95%置信椭圆（n=29）'
)

# 3.3 添加各组回归线（统一为虚线）
# 恒定低光组（改为虚线）
low_light_z = np.polyfit(low_light['平均u (h-1)'], low_light['LUE (%)'], 1)
low_light_p = np.poly1d(low_light_z)
ax1.plot(low_light['平均u (h-1)'], low_light_p(low_light['平均u (h-1)']),
         color='#2E7D32', linestyle='--', linewidth=1.2)  # ✅ 改成虚线

# 传统静态调光组（保持虚线）
static_z = np.polyfit(static_high['平均u (h-1)'], static_high['LUE (%)'], 1)
static_p = np.poly1d(static_z)
ax1.plot(static_high['平均u (h-1)'], static_p(static_high['平均u (h-1)']),
         color='#1976D2', linestyle='--', linewidth=1.2)

# 传统动态调光组（保持虚线）
dynamic_z = np.polyfit(traditional_dynamic['平均u (h-1)'], traditional_dynamic['LUE (%)'], 1)
dynamic_p = np.poly1d(dynamic_z)
ax1.plot(traditional_dynamic['平均u (h-1)'], dynamic_p(traditional_dynamic['平均u (h-1)']),
         color='#F57C00', linestyle='--', linewidth=1.2)

# 3.4 添加各组相关性标注（调整位置避免遮挡）
ax1.text(0.14, 6, '静态调光组：r=-0.681, p<0.05', fontsize=9, fontweight='bold', color='steelblue',
         bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
ax1.text(0.02, 4, '动态调光组：r=-0.0887, p=0.71', fontsize=9, fontweight='bold', color='darkorange',
         bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
ax1.text(0.02, 1.5, '恒定低光组：r=0.683, p<0.05', fontsize=9, fontweight='bold', color='forestgreen',
         bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
ax1.text(0.1, 10, '传统动态+PHM12S：r=0.320, p=0.156', fontsize=9, fontweight='bold', color='crimson',
         bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

# 3.5 PHM12S偏离度标注
deviation = np.round(np.linalg.norm([phm12s["平均u (h-1)"]-np.mean(x_traditional_active),
                                    phm12s["LUE (%)"]-np.mean(y_traditional_active)]), 2)
ax1.annotate(f'显著偏离传统策略分布\n（偏离度：{deviation}', xy=(phm12s['平均u (h-1)'], phm12s['LUE (%)']),
             xytext=(0.12, 12), arrowprops=dict(arrowstyle='->', color='crimson', linewidth=1.5),
             fontsize=10, fontweight='bold', color='crimson',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

# 3.6 子图a细节优化（去掉网格线）
ax1.set_xlabel('Average Specific Growth Rate μ (h-1)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Light Use Efficiency LUE (%)', fontsize=12, fontweight='bold')
ax1.set_title('(a) μ-LUE Distribution and Correlation', fontsize=13, fontweight='bold')
ax1.set_xlim(-0.01, 0.20)
ax1.set_ylim(-1, 20)
ax1.set_xticks(np.arange(0, 0.20, 0.04))
ax1.set_yticks(np.arange(0, 20, 4))
ax1.grid(False)  # 去掉网格线
ax1.legend(loc='upper left', fontsize=9, framealpha=0.9, bbox_to_anchor=(0.02, 0.98))

# 3.7 添加图注说明（虚线表示回归线）
ax1.text(0.01, -1.2, '虚线表示各组的线性回归线', fontsize=9, color='black',
         bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8),
         transform=ax1.transAxes)  # 用相对坐标定位在底部

# -------------------------- 子图b（右图）：μ-LUE效率权衡及高效区 --------------------------
# 4.1 绘制高μ-高LUE高效区（精准对齐μ=0.08和LUE=8）
x_high = np.linspace(0.08, 0.22, 100)
high_patch = ax2.fill_between(x_high, 8, 20, color='#FFB6C1', alpha=0.2, label='高μ-高LUE高效区')

# 4.2 绘制所有原始数据点（降低透明度避免遮挡）
# 恒定低光组
scatter_low = ax2.scatter(
    low_light['平均u (h-1)'], low_light['LUE (%)'],
    c='forestgreen', marker='s', s=60, alpha=0.5,
    label='恒定低光组（n=11）', edgecolors='darkgreen', linewidth=0.5
)
# 传统动态调光组
scatter_dynamic = ax2.scatter(
    traditional_dynamic['平均u (h-1)'], traditional_dynamic['LUE (%)'],
    c='darkorange', marker='^', s=50, alpha=0.5,
    label='传统动态调光组（n=20）', edgecolors='darkred', linewidth=0.5
)
# 传统静态调光组
scatter_static = ax2.scatter(
    static_high['平均u (h-1)'], static_high['LUE (%)'],
    c='steelblue', marker='o', s=50, alpha=0.5,
    label='传统静态调光组（n=9）', edgecolors='darkblue', linewidth=0.5
)
# 本研究创新型PHM12S策略
scatter_phm = ax2.scatter(
    phm12s['平均u (h-1)'], phm12s['LUE (%)'],
    c='crimson', marker='*', s=300,
    label='本研究创新型PHM12S策略（n=1）', edgecolors='black', linewidth=2
)

# 4.3 绘制3条效率参考线（按要求修改样式）
x_ref = np.linspace(0, 0.22, 100)
line_low = ax2.plot(x_ref, 50 * x_ref, color='#b0b0b0', linestyle='--', linewidth=1.5, label='低效线 (LUE=50μ)')
line_balance = ax2.plot(x_ref, 100 * x_ref, color='#ff7f0e', linestyle='-.', linewidth=1.5, label='平衡线 (LUE=100μ)')
line_high = ax2.plot(x_ref, 200 * x_ref, color='#2ca02c', linestyle='--', linewidth=1.5, label='高效线 (LUE=200μ)')

# 4.4 高效区文本标注（位置微调，确保在高效区内）
ax2.text(0.12, 12, '高μ-高LUE高效区', fontsize=10, color='#D81F26', fontweight='bold',
         bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

# 4.5 子图b细节优化（图例移到左上角，不遮挡内容）
ax2.set_xlabel('Average Specific Growth Rate μ (h-1)', fontsize=12, fontweight='bold')
ax2.set_ylabel('Light Use Efficiency LUE (%)', fontsize=12, fontweight='bold')
ax2.set_title('(b) μ-LUE Efficiency Trade-off & High-Efficiency Region', fontsize=13, fontweight='bold')
# 统一坐标轴范围（与子图a完全一致）
ax2.set_xlim(-0.01, 0.20)
ax2.set_ylim(-1, 20)
# 统一刻度间隔（与子图a完全一致）
ax2.set_xticks(np.arange(0, 0.20, 0.04))
ax2.set_yticks(np.arange(0, 20, 4))
ax2.grid(False)  # 去掉网格线

# 拆分图例：左上角放散点，右下角放高效区和线
# 左上角图例（散点）
scatter_handles = [scatter_low, scatter_dynamic, scatter_static, scatter_phm]
scatter_labels = [h.get_label() for h in scatter_handles]
ax2.legend(handles=scatter_handles, labels=scatter_labels,
           loc='upper left', fontsize=9, framealpha=0.9, bbox_to_anchor=(0.02, 0.98))

# 右下角图例（高效区+线）
line_handles = [high_patch] + line_low + line_balance + line_high
line_labels = [h.get_label() for h in line_handles]
ax2.legend(handles=line_handles, labels=line_labels,
           loc='lower right', fontsize=9, framealpha=0.9, bbox_to_anchor=(0.98, 0.02))

# -------------------------- 4. 保存高清图表 --------------------------
plt.tight_layout()
plt.savefig(r'最终版_双图_完整优化.png',
            dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

# 运行完成提示
print("最终版完整优化双图已生成！")
print(f"数据核对：恒定低光(n={len(low_light)})、静态调光(n={len(static_high)})、动态调光(n={len(traditional_dynamic)})、动态+PHM12S(n={len(dynamic_plus_phm)})")