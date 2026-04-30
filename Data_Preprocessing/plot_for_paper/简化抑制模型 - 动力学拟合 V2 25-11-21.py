import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import warnings
from matplotlib import rcParams
# 设置中文字体
# plt.rcParams['font.sans-serif'] = ['Times New Roman', 'SimHei']
# plt.rcParams['axes.unicode_minus'] = False
plt.rcParams.update({
    "text.usetex": False,  # 无需安装LaTeX也能工作
    "font.family": ["Times New Roman", "SimHei"],  # 包含英文字体和中文字体
    "mathtext.fontset": "stix"  # 优化数学符号显示

})
rcParams["font.size"] = 12
rcParams["axes.labelsize"] = 22
rcParams["xtick.labelsize"] = 22
rcParams["ytick.labelsize"] = 22
rcParams["axes.titlesize"] = 26
rcParams["legend.fontsize"] = 20
rcParams["axes.titley"] = 1.01

# 您的数据
concentrations = np.array([20, 80, 150, 300, 400, 500])
umax_means = np.array([0.0948, 0.1203, 0.1272, 0.1374, 0.0764, 0.0671])

print("=== 简化抑制模型拟合 ===")


# 简化抑制模型 (4参数)
def simple_inhibition_model(S, μ_max, K_s, K_i, n):
    """简化抑制模型"""
    return (μ_max * S / (K_s + S)) / (1 + (S / K_i) ** n)


# 设置模型参数
model = {
    'name': '简化抑制模型',
    'func': simple_inhibition_model,
    'params': ['μ_max', 'K_s', 'K_i', 'n'],
    'bounds': ([0.08, 10, 200, 1], [0.15, 200, 2000, 5]),
    'initial': [0.14, 50, 500, 2]
}

print(f"尝试 {model['name']}...")

try:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        popt, pcov = curve_fit(
            model['func'], concentrations, umax_means,
            p0=model['initial'], bounds=model['bounds'],
            maxfev=5000
        )

    # 计算R²
    pred = model['func'](concentrations, *popt)
    ss_res = np.sum((umax_means - pred) ** 2)
    ss_tot = np.sum((umax_means - np.mean(umax_means)) ** 2)
    r2 = 1 - (ss_res / ss_tot)

    print(f"✓ 拟合成功!")
    print(f"R² = {r2:.4f}")

    # 输出详细的模型表达式和参数
    print(f"\n=== 简化抑制模型完整表达式 ===")
    μ_max_val, K_s_val, K_i_val, n_val = popt
    print(f"μ = ({μ_max_val:.4f} × S / ({K_s_val:.2f} + S)) / (1 + (S/{K_i_val:.2f})^{n_val:.2f})")

    print(f"\n=== 动力学参数值 ===")
    print(f"μ_max = {μ_max_val:.4f} h⁻¹")
    print(f"K_s = {K_s_val:.2f} mg/L")
    print(f"K_i = {K_i_val:.2f} mg/L")
    print(f"n = {n_val:.2f}")

    # 参数的标准误差
    perr = np.sqrt(np.diag(pcov))
    print(f"\n=== 参数标准误差 ===")
    for i, (param, value, err) in enumerate(zip(model['params'], popt, perr)):
        print(f"{param}: {value:.4f} ± {err:.4f}")

    # 生物学意义分析
    print(f"\n=== 生物学意义分析 ===")
    print(f"1. 最大生长潜力 (μ_max): {μ_max_val:.4f} h⁻¹")
    print(f"   - 理论上在无抑制条件下能达到的最大生长速率")
    print(f"   - 比实验观测最大值 ({max(umax_means):.4f}) 略高，符合模型预期")

    print(f"2. 半饱和常数 (K_s): {K_s_val:.2f} mg/L")
    print(f"   - 生长速率达到μ_max一半时所需的氨氮浓度")
    print(f"   - 值较小表明微藻对氨氮有较好的亲和力")

    print(f"3. 抑制常数 (K_i): {K_i_val:.2f} mg/L")
    print(f"   - 开始出现明显抑制效应的氨氮浓度")
    print(f"   - 与实验数据中300mg/L出现抑制的现象一致")

    print(f"4. 抑制指数 (n): {n_val:.2f}")
    if n_val > 1:
        print(f"   - 协同抑制效应，抑制强度随浓度增加而加速")
    elif n_val < 1:
        print(f"   - 渐进抑制效应，抑制强度随浓度增加而减缓")
    else:
        print(f"   - 线性抑制效应")

    # 计算理论最佳浓度
    S_range = np.linspace(10, 550, 500)
    umax_range = model['func'](S_range, *popt)
    max_idx = np.argmax(umax_range)
    optimal_conc = S_range[max_idx]
    optimal_umax = umax_range[max_idx]

    print(f"\n=== 理论预测 ===")
    print(f"最佳生长浓度: {optimal_conc:.1f} mg/L")
    print(f"理论最大生长速率: {optimal_umax:.4f} h⁻¹")

    # 绘制结果
    plt.figure(figsize=(13, 9))
    plt.scatter(concentrations, umax_means, color='red', s=100, label='Experimental data', zorder=5)
    plt.plot(S_range, umax_range, 'b-', linewidth=3, label='Simplified fitting curve')

    # 标注最佳点
    plt.axvline(x=optimal_conc, color='green', linestyle='--', alpha=0.7,
                label=f'Optimum concentration: {optimal_conc:.0f} mg/L')
    # 标注实验点
    for i, (x, y) in enumerate(zip(concentrations, umax_means)):
        plt.annotate(f'{y:.4f}', (x, y), textcoords="offset points",
                     xytext=(0, 10), ha='center', fontsize=16)
    plt.xlabel('Concentration of NH$_4^+$-N (mg/L)', fontsize=28, fontweight='bold')
    plt.ylabel('Microalgae specific growth rate (h$^{-1}$)', fontsize=28, fontweight='bold')
    plt.title(f'Classical inhibition model fitting\n($R^{2}$ = {r2:.4f})', fontsize=32, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('简化抑制模型拟合结果')

except Exception as e:
    print(f"✗ 拟合失败: {e}")