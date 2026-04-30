import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pandas as pd
from sklearn.metrics import r2_score

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# 定义Haldane模型函数
def haldane_model(S, umax, Ks, Ki):
    """
    Haldane底物抑制模型
    S: 底物浓度 (氨氮浓度)
    umax: 最大比生长速率
    Ks: 半饱和常数
    Ki: 抑制常数
    """
    return (umax * S) / (Ks + S + (S ** 2 / Ki))


# 准备数据 - 根据您提供的表格
# 浓度 (mg/L)
concentrations = [20, 80, 150, 300, 400, 500]

# 各浓度下的μmax数据 (18个重复)
umax_data = [
    # 20 mg/L
    [0.1107, 0.0931, 0.0948, 0.0995, 0.0918, 0.0773, 0.0816, 0.0941, 0.0905,
     0.0952, 0.0982, 0.0982, 0.105, 0.1042, 0.0867, 0.0833, 0.1029, 0.0874],
    # 80 mg/L
    [0.1279, 0.1036, 0.0989, 0.1259, 0.1267, 0.1202, 0.1303, 0.1237, 0.1244,
     0.1267, 0.1201, 0.1285, 0.1179, 0.1305, 0.1282, 0.1065, 0.1158, 0.1064],
    # 150 mg/L
    [0.1233, 0.1296, 0.1223, 0.1346, 0.1219, 0.1286, 0.1244, 0.1286, 0.1145,
     0.1398, 0.1264, 0.1308, 0.1222, 0.1278, 0.1265, 0.1281, 0.1202, 0.1375],
    # 300 mg/L
    [0.133, 0.1413, 0.1623, 0.1138, 0.1128, 0.1478, 0.1332, 0.1285, 0.144,
     0.1596, 0.1437, 0.1133, 0.1415, 0.1481, 0.1349, 0.1133, 0.1384, 0.1364],
    # 400 mg/L
    [0.0709, 0.0798, 0.0758, 0.0793, 0.077, 0.0687, 0.0794, 0.0747, 0.0786,
     0.0807, 0.0741, 0.0787, 0.0783, 0.0756, 0.072, 0.0793, 0.0765, 0.0792],
    # 500 mg/L
    [0.048, 0.0601, 0.0646, 0.052, 0.083, 0.0779, 0.0646, 0.0722, 0.0624,
     0.0528, 0.079, 0.0632, 0.0825, 0.0746, 0.0604, 0.0617, 0.0792, 0.0728]
]

# 将数据整理为适合拟合的格式
S_data = []  # 浓度数据
umax_flat = []  # 对应的生长速率数据

for i, conc in enumerate(concentrations):
    for value in umax_data[i]:
        S_data.append(conc)
        umax_flat.append(value)

S_data = np.array(S_data)
umax_flat = np.array(umax_flat)

print(f"总数据点数: {len(S_data)}")

# 设置参数初始值和界限
initial_guess = [0.14, 80, 350]  # [umax, Ks, Ki]
bounds = ([0.08, 10, 200], [0.18, 200, 800])  # 下限和上限

# 执行拟合
try:
    popt, pcov = curve_fit(haldane_model, S_data, umax_flat,
                           p0=initial_guess, bounds=bounds, maxfev=5000)

    # 提取拟合参数和标准误差
    umax_fit, Ks_fit, Ki_fit = popt
    perr = np.sqrt(np.diag(pcov))  # 参数的标准误差

    # 计算预测值
    S_range = np.linspace(10, 550, 100)  # 用于绘制平滑曲线
    umax_pred = haldane_model(S_range, *popt)

    # 计算R²
    umax_pred_data = haldane_model(S_data, *popt)
    r_squared = r2_score(umax_flat, umax_pred_data)

    # 输出拟合结果
    print("\n=== Haldane模型拟合结果 ===")
    print(f"最大比生长速率 (μmax): {umax_fit:.4f} ± {perr[0]:.4f} h⁻¹")
    print(f"半饱和常数 (Ks): {Ks_fit:.2f} ± {perr[1]:.2f} mg/L")
    print(f"抑制常数 (Ki): {Ki_fit:.2f} ± {perr[2]:.2f} mg/L")
    print(f"R²: {r_squared:.4f}")
    print(f"调整R²: {1 - (1 - r_squared) * (len(S_data) - 1) / (len(S_data) - 3 - 1):.4f}")

    # 计算每个浓度的均值和标准差用于箱线图
    concentrations_array = np.array(concentrations)
    means = [np.mean(umax_data[i]) for i in range(len(concentrations))]
    stds = [np.std(umax_data[i]) for i in range(len(concentrations))]

    # 绘制结果
    plt.figure(figsize=(12, 8))

    # 主图：拟合曲线和所有数据点
    plt.subplot(2, 1, 1)

    # 绘制所有数据点（轻微抖动避免重叠）
    jitter = np.random.normal(0, 2, len(S_data))  # 添加微小抖动
    plt.scatter(S_data + jitter, umax_flat, alpha=0.6, color='blue', label='实验数据点', s=30)

    # 绘制拟合曲线
    plt.plot(S_range, umax_pred, 'r-', linewidth=2, label='Haldane模型拟合')

    # 绘制每个浓度的均值点
    plt.scatter(concentrations, means, color='red', s=80, marker='D',
                label='浓度均值', zorder=5)

    plt.xlabel('氨氮浓度 (mg/L)')
    plt.ylabel('最大比生长速率 (h⁻¹)')
    plt.title('氨氮浓度对微藻最大比生长速率的影响 - Haldane模型拟合')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 添加拟合方程和参数
    equation_text = f'拟合方程: $\\mu = \\frac{{{umax_fit:.3f} \\cdot S}}{{{Ks_fit:.1f} + S + S^2/{Ki_fit:.1f}}}$\n'
    equation_text += f'$R^2 = {r_squared:.4f}$'
    plt.text(0.02, 0.98, equation_text, transform=plt.gca().transAxes,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # 子图：箱线图显示数据分布
    plt.subplot(2, 1, 2)

    # 创建箱线图
    box_plot = plt.boxplot(umax_data, positions=concentrations, widths=30,
                           patch_artist=True)

    # 设置箱线图颜色
    colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightsalmon', 'plum', 'wheat']
    for patch, color in zip(box_plot['boxes'], colors):
        patch.set_facecolor(color)

    plt.xlabel('氨氮浓度 (mg/L)')
    plt.ylabel('最大比生长速率 (h⁻¹)')
    plt.title('各浓度下最大比生长速率的分布')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # 输出各浓度的统计信息
    print("\n=== 各浓度统计信息 ===")
    for i, conc in enumerate(concentrations):
        data = umax_data[i]
        print(f"{conc} mg/L: 均值={np.mean(data):.4f}, 标准差={np.std(data):.4f}, "
              f"变异系数={(np.std(data) / np.mean(data) * 100):.2f}%")

except Exception as e:
    print(f"拟合过程中出现错误: {e}")
    print("尝试调整初始参数或界限...")

    # 备用方案：使用更宽松的界限
    try:
        bounds_loose = ([0.05, 1, 100], [0.25, 500, 1500])
        popt, pcov = curve_fit(haldane_model, S_data, umax_flat,
                               p0=initial_guess, bounds=bounds_loose, maxfev=5000)

        umax_fit, Ks_fit, Ki_fit = popt
        perr = np.sqrt(np.diag(pcov))

        umax_pred_data = haldane_model(S_data, *popt)
        r_squared = r2_score(umax_flat, umax_pred_data)

        print("\n=== 使用宽松界限的拟合结果 ===")
        print(f"μmax: {umax_fit:.4f} ± {perr[0]:.4f} h⁻¹")
        print(f"Ks: {Ks_fit:.2f} ± {perr[1]:.2f} mg/L")
        print(f"Ki: {Ki_fit:.2f} ± {perr[2]:.2f} mg/L")
        print(f"R²: {r_squared:.4f}")

    except Exception as e2:
        print(f"宽松界限拟合也失败: {e2}")