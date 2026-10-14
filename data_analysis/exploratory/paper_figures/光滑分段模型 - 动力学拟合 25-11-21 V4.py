import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import warnings
import matplotlib
from matplotlib import rcParams
# 在导入 pyplot 之前设置后端

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

# 实验数据
concentrations = np.array([20, 80, 150, 300, 400, 500])
umax_means = np.array([0.0948, 0.1203, 0.1272, 0.1374, 0.0764, 0.0671])

print("=== 光滑分段模型拟合 ===")


def smooth_piecewise_model(S, μ_max, K_s, K_i, S_transition, smoothness):
    """
    光滑分段模型
    参数:
    - S: 氨氮浓度
    - μ_max: 最大生长速率
    - K_s: 半饱和常数
    - K_i: 抑制常数
    - S_transition: 转折点浓度
    - smoothness: 光滑度参数，控制过渡的平滑程度
    """
    S = np.array(S)

    # 使用sigmoid函数实现平滑过渡
    # smoothness越小，过渡越陡峭；smoothness越大，过渡越平缓
    transition_weight = 1 / (1 + np.exp(-(S - S_transition) / smoothness))

    # 促进阶段（纯Monod增长）
    promotion_phase = μ_max * S / (K_s + S)

    # 抑制阶段（基于转折点处的生长速率进行抑制）
    base_growth = μ_max * S_transition / (K_s + S_transition)
    inhibition_phase = base_growth / (1 + (S - S_transition) / K_i)

    # 平滑混合两个阶段
    return (1 - transition_weight) * promotion_phase + transition_weight * inhibition_phase


# 尝试光滑分段模型拟合
print("尝试光滑分段模型...")
try:
    # 参数: μ_max, K_s, K_i, S_transition, smoothness
    # 初始猜测值
    initial_guess = [0.14, 50, 100, 300, 10]

    # 参数边界
    bounds = (
        [0.08, 10, 50, 250, 1],  # 下限
        [0.15, 200, 500, 350, 50]  # 上限
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        popt, pcov = curve_fit(smooth_piecewise_model, concentrations, umax_means,
                               p0=initial_guess, bounds=bounds, maxfev=10000)

    # 计算R²
    pred = smooth_piecewise_model(concentrations, *popt)
    ss_res = np.sum((umax_means - pred) ** 2)
    ss_tot = np.sum((umax_means - np.mean(umax_means)) ** 2)
    r2 = 1 - (ss_res / ss_tot)

    μ_max, K_s, K_i, S_transition, smoothness = popt

    print(f"✓ 光滑分段模型拟合成功!")
    print(f"R² = {r2:.4f}")

    print(f"\n=== 模型参数 ===")
    print(f"μ_max = {μ_max:.4f} h⁻¹")
    print(f"K_s = {K_s:.2f} mg/L")
    print(f"K_i = {K_i:.2f} mg/L")
    print(f"S_transition = {S_transition:.1f} mg/L")
    print(f"smoothness = {smoothness:.2f}")

    # 参数标准误差
    perr = np.sqrt(np.diag(pcov))
    print(f"\n=== 参数标准误差 ===")
    param_names = ['μ_max', 'K_s', 'K_i', 'S_transition', 'smoothness']
    for name, value, err in zip(param_names, popt, perr):
        print(f"{name}: {value:.4f} ± {err:.4f}")

    # 生物学解释
    print(f"\n=== 生物学意义 ===")
    print(f"1. 最大生长潜力: {μ_max:.4f} h⁻¹")
    print(f"   - 理论上在无抑制条件下能达到的最大生长速率")

    print(f"2. 半饱和常数: {K_s:.2f} mg/L")
    print(f"   - 生长速率达到μ_max一半时所需的氨氮浓度")
    print(f"   - 值较小表明微藻对氨氮有较好的亲和力")

    print(f"3. 抑制常数: {K_i:.2f} mg/L")
    print(f"   - 控制抑制效应的强度，值越小抑制效应越强")

    print(f"4. 转折点浓度: {S_transition:.1f} mg/L")
    print(f"   - 从促进为主转变为抑制为主的过渡浓度")
    print(f"   - 与实验观察的最佳浓度点(~300mg/L)一致")

    print(f"5. 光滑度: {smoothness:.2f}")
    print(f"   - 控制从促进到抑制的过渡平滑程度")
    print(f"   - 值越小过渡越陡峭，值越大过渡越平缓")

    # 生成预测曲线
    S_range = np.linspace(10, 550, 500)
    μ_range = smooth_piecewise_model(S_range, *popt)

    # 计算理论最佳点
    max_idx = np.argmax(μ_range)
    optimal_conc = S_range[max_idx]
    optimal_umax = μ_range[max_idx]

    print(f"\n=== 理论预测 ===")
    print(f"最佳生长浓度: {optimal_conc:.1f} mg/L")
    print(f"理论最大生长速率: {optimal_umax:.4f} h⁻¹")

    # 可视化过渡函数
    transition_weight_range = 1 / (1 + np.exp(-(S_range - S_transition) / smoothness))

    # 绘制综合结果
    plt.figure(figsize=(15, 10))

    # 1. 主拟合图
    """figure 1"""

    plt.scatter(concentrations, umax_means, color='red', s=100, label='Experimental data', zorder=5)
    plt.plot(S_range, μ_range, 'b-', linewidth=3, label='Fitting curve')
    plt.axvline(x=S_transition, color='green', linestyle='--', alpha=0.7,
                label=f'$S_{{N,transition}}$: {S_transition:.0f} mg/L')
    # --- 关键设置：将图例放在右上角 ---
    plt.legend(loc='upper right', fontsize=12)

    # 标注实验点
    for i, (x, y) in enumerate(zip(concentrations, umax_means)):
        plt.annotate(f'{y:.4f}', (x, y), textcoords="offset points",
                     xytext=(0, 10), ha='center', fontsize=16)

    plt.xlabel('Concentration of NH$_4^+$-N (mg/L)', fontsize=28, fontweight='bold')
    plt.ylabel('Microalgae specific growth rate (h$^{-1}$)', fontsize=28, fontweight='bold')
    plt.title(f'Smooth piecewise model fitting\n($R^{2}$ = {r2:.4f})', fontsize=32, fontweight='bold')
    plt.grid(True, alpha=0.3)

    # # 2. 残差图
    # plt.subplot(2, 3, 2)
    # residuals = umax_means - pred
    # plt.scatter(concentrations, residuals, color='blue', s=80)
    # plt.axhline(y=0, color='red', linestyle='--', alpha=0.7)
    # plt.xlabel('氨氮浓度 (mg/L)')
    # plt.ylabel('残差')
    # plt.title('残差分析')
    # plt.grid(True, alpha=0.3)
    #
    # # 3. 过渡权重函数
    # plt.subplot(2, 3, 3)
    # plt.plot(S_range, transition_weight_range, 'purple', linewidth=2)
    # plt.axvline(x=S_transition, color='green', linestyle='--', alpha=0.7)
    # plt.xlabel('氨氮浓度 (mg/L)')
    # plt.ylabel('过渡权重')
    # plt.title('过渡函数 (Sigmoid)')
    # plt.grid(True, alpha=0.3)
    #
    # # 4. 模型分解：促进阶段和抑制阶段
    # plt.subplot(2, 3, 4)
    # promotion_phase_range = μ_max * S_range / (K_s + S_range)
    # base_growth = μ_max * S_transition / (K_s + S_transition)
    # inhibition_phase_range = base_growth / (1 + (S_range - S_transition) / K_i)
    #
    # plt.plot(S_range, promotion_phase_range, 'g--', alpha=0.7, label='促进阶段')
    # plt.plot(S_range, inhibition_phase_range, 'r--', alpha=0.7, label='抑制阶段')
    # plt.plot(S_range, μ_range, 'b-', linewidth=2, label='组合模型')
    # plt.axvline(x=S_transition, color='green', linestyle='--', alpha=0.7)
    # plt.xlabel('氨氮浓度 (mg/L)')
    # plt.ylabel('最大比生长速率 (h⁻¹)')
    # plt.title('模型分解')
    # plt.legend()
    # plt.grid(True, alpha=0.3)
    #
    # # 5. 灵敏度分析 - 光滑度参数的影响
    # plt.subplot(2, 3, 5)
    # smoothness_values = [1, 5, smoothness, 20, 50]
    # colors = ['red', 'orange', 'blue', 'green', 'purple']

    # for smooth_val, color in zip(smoothness_values, colors):
    #     if abs(smooth_val - smoothness) < 0.1:
    #         label = f'smoothness = {smooth_val:.1f} (最优)'
    #         linewidth = 3
    #     else:
    #         label = f'smoothness = {smooth_val:.1f}'
    #         linewidth = 1.5
    #
    #     μ_test = smooth_piecewise_model(S_range, μ_max, K_s, K_i, S_transition, smooth_val)
    #     plt.plot(S_range, μ_test, color=color, linewidth=linewidth, label=label, alpha=0.8)
    #
    # plt.xlabel('氨氮浓度 (mg/L)')
    # plt.ylabel('Microalgae specific growth rate ($h^{-1}$)')
    # plt.title('光滑度参数灵敏度分析')
    # plt.legend(fontsize=8)
    # plt.grid(True, alpha=0.3)

    # # 6. 预测表
    # plt.subplot(2, 3, 6)
    # test_concentrations = [20, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500]
    # predictions = [smooth_piecewise_model(x, *popt) for x in test_concentrations]
    #
    # # 创建简单的表格显示
    # table_data = []
    # for conc, pred_val in zip(test_concentrations, predictions):
    #     if conc <= S_transition - 50:
    #         status = "促进阶段"
    #     elif conc <= S_transition + 50:
    #         status = "过渡阶段"
    #     else:
    #         status = "抑制阶段"
    #     table_data.append([f"{conc} mg/L", f"{pred_val:.4f}", status])
    #
    # # 简单文本显示
    # plt.text(0.05, 0.95, "浓度预测表:", fontsize=12, weight='bold', transform=plt.gca().transAxes)
    # for i, row in enumerate(table_data):
    #     plt.text(0.05, 0.85 - i * 0.07, f"{row[0]}: {row[1]} ({row[2]})",
    #              fontsize=9, transform=plt.gca().transAxes)
    #
    # plt.axis('off')
    # plt.title('浓度-生长速率预测')
    #
    # plt.tight_layout()
    plt.savefig('光滑分段模型')

    # 模型方程输出
    print(f"\n=== 模型方程 ===")
    print(f"μ(S) = [1 - w(S)] × [μ_max × S / (K_s + S)] + w(S) × [Base × (1 / (1 + (S - S_transition)/K_i))]")
    print(f"其中:")
    print(f"  w(S) = 1 / (1 + exp(-(S - {S_transition:.1f}) / {smoothness:.2f}))")
    print(f"  Base = μ_max × S_transition / (K_s + S_transition) = {base_growth:.4f}")
    print(f"  μ_max = {μ_max:.4f}, K_s = {K_s:.2f}, K_i = {K_i:.2f}")

except Exception as e:
    print(f"光滑分段模型拟合失败: {e}")
    print(f"错误详情: {e}")

    # 简化版本作为备选
    print("\n尝试简化版本...")
    try:
        # 固定smoothness，只拟合其他参数
        def simplified_smooth_model(S, μ_max, K_s, K_i, S_transition):
            return smooth_piecewise_model(S, μ_max, K_s, K_i, S_transition, smoothness=10)


        initial_guess_simple = [0.14, 50, 100, 300]
        bounds_simple = ([0.08, 10, 50, 250], [0.15, 200, 500, 350])

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            popt_simple, pcov_simple = curve_fit(simplified_smooth_model, concentrations, umax_means,
                                                 p0=initial_guess_simple, bounds=bounds_simple, maxfev=5000)

        pred_simple = simplified_smooth_model(concentrations, *popt_simple)
        r2_simple = 1 - np.sum((umax_means - pred_simple) ** 2) / np.sum((umax_means - np.mean(umax_means)) ** 2)

        μ_max_s, K_s_s, K_i_s, S_transition_s = popt_simple
        print(f"✓ 简化版本拟合成功! R² = {r2_simple:.4f}")
        print(f"参数: μ_max = {μ_max_s:.4f}, K_s = {K_s_s:.2f}, K_i = {K_i_s:.2f}, S_transition = {S_transition_s:.1f}")

    except Exception as e2:
        print(f"简化版本也失败: {e2}")

# 数据质量评估
print(f"\n=== 数据质量评估 ===")
print(f"数据点数量: {len(concentrations)}")
print(f"浓度覆盖范围: {min(concentrations)}-{max(concentrations)} mg/L")
print(f"生长速率范围: {min(umax_means):.4f}-{max(umax_means):.4f} h⁻¹")
print(f"数据变异系数: {np.std(umax_means) / np.mean(umax_means) * 100:.1f}%")