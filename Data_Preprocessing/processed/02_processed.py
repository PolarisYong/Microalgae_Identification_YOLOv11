import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import openpyxl
from openpyxl.drawing.image import Image
import io
from matplotlib import rcParams

# ---------------------- 全局字体配置 ----------------------
rcParams["font.family"] = ["Times New Roman", "serif"]
rcParams["axes.unicode_minus"] = False
rcParams["font.size"] = 10
rcParams["axes.labelsize"] = 20
rcParams["xtick.labelsize"] = 20
rcParams["ytick.labelsize"] = 20
rcParams["axes.titlesize"] = 24
rcParams["legend.fontsize"] = 18
rcParams["axes.titley"] = 1.01

# ---------------------- 1. 定义修正Logistic生长模型 ----------------------
def modified_logistic_model(t, mu_max, K, N0):
    if K == 0 or N0 == 0:
        return np.zeros_like(t)
    term = (K / N0 - 1) * np.exp(-mu_max * t)
    N_t = K / (1 + term)
    return N_t


def calculate_growth_phases(t, mu_max, K, N0):
    term = (K / N0 - 1) * np.exp(-mu_max * t)
    N_t = K / (1 + term)

    dt = np.diff(t).mean() if len(t) > 1 else 1
    slope = np.diff(N_t) / dt
    slope = np.insert(slope, 0, 0)

    max_slope = np.max(slope) if np.max(slope) > 0 else 1e-6
    threshold = 0.05 * max_slope

    phase_flags = []
    for i in range(len(t)):
        current_slope = slope[i]
        current_N = N_t[i]
        if current_slope < threshold and abs(current_N - K) / K < 0.05:
            phase_flags.append("稳定期")
        elif current_slope >= threshold:
            phase_flags.append("对数期")
        else:
            phase_flags.append("滞后期")

    phases = []
    current_phase = phase_flags[0]
    start_idx = 0
    for i in range(1, len(phase_flags)):
        if phase_flags[i] != current_phase:
            start_time = t[start_idx]
            end_time = t[i - 1]
            duration = end_time - start_time
            phases.append({
                "阶段": current_phase,
                "开始时间(h)": start_time,
                "结束时间(h)": end_time,
                "时长(h)": duration
            })
            current_phase = phase_flags[i]
            start_idx = i
    start_time = t[start_idx]
    end_time = t[-1]
    duration = end_time - start_time
    phases.append({
        "阶段": current_phase,
        "开始时间(h)": start_time,
        "结束时间(h)": end_time,
        "时长(h)": duration
    })

    phase_duration = {
        "滞后期时长(h)": 0,
        "对数期时长(h)": 0,
        "稳定期时长(h)": 0
    }
    for phase in phases:
        key = f"{phase['阶段']}时长(h)"
        if key in phase_duration:
            phase_duration[key] = round(phase["时长(h)"], 2) + 1
    return phase_duration, phases


# ---------------------- 2. 生成趋势线函数 ----------------------
def generate_trendline(x, y, degree=2):
    z = np.polyfit(x, y, degree)
    p = np.poly1d(z)
    return p(x)


# ---------------------- 3. 数据读取与处理主函数 ----------------------
def process_cell_growth(excel_path, result_excel_path="enhanced_cell_growth_results.xlsx",
                        min_data_points=5, nitrogen_concentration="test"):
    skip_sheet = "数据汇总"
    wb_original = openpyxl.load_workbook(excel_path)
    sheet_names = wb_original.sheetnames
    individual_summary = []
    merged_t = []
    merged_cell_counts = []
    valid_sheets = []
    individual_data = []

    # 第一步：处理各页签并进行单独拟合
    for sheet in sheet_names:
        if sheet == skip_sheet:
            print(f"跳过页签: {sheet}")
            continue
        print(f"正在处理页签: {sheet}")
        plt_name = nitrogen_concentration + "_" + sheet[4:]
        ws = wb_original[sheet]
        try:
            df = pd.read_excel(excel_path, sheet_name=sheet)
            required_columns = ["目标数量", "总面积(μm²)", "相对平均细胞面积"]
            missing_cols = [col for col in required_columns if col not in df.columns]
            if missing_cols:
                print(f"页签 {sheet} 缺少必要列: {missing_cols}，跳过\n")
                individual_summary.append({
                    "腔室名称": sheet,
                    "最大比生长速率μmax (h^-1)": np.nan,
                    "环境容纳量K (个/腔室)": np.nan,
                    "初始细胞数量N0 (个/腔室)": np.nan,
                    "拟合优度R²": np.nan,
                    "平均细胞周期T_d(h)": np.nan,
                    "增殖倍数 F": np.nan,
                    "生长效率 η (1/增殖倍数)": np.nan,
                    "滞后期时长(h)": np.nan,
                    "对数期时长(h)": np.nan,
                    "稳定期时长(h)": np.nan
                })
                continue
            cell_counts = df["目标数量"].values
            if np.all(cell_counts == 0):
                print(f"页签 {sheet} 所有细胞数量为0，跳过\n")
                individual_summary.append({
                    "腔室名称": sheet,
                    "最大比生长速率μmax (h^-1)": np.nan,
                    "环境容纳量K (个/腔室)": np.nan,
                    "初始细胞数量N0 (个/腔室)": np.nan,
                    "拟合优度R²": np.nan,
                    "平均细胞周期T_d(h)": np.nan,
                    "增殖倍数 F": np.nan,
                    "生长效率 η (1/增殖倍数)": np.nan,
                    "滞后期时长(h)": np.nan,
                    "对数期时长(h)": np.nan,
                    "稳定期时长(h)": np.nan
                })
                continue
            t = np.arange(1, len(cell_counts) + 1)
            if len(cell_counts) >= min_data_points and not np.all(cell_counts == 0):
                merged_t.extend(t)
                merged_cell_counts.extend(cell_counts)
                valid_sheets.append(sheet)
                individual_data.append({
                    "sheet": sheet,
                    "t": t,
                    "cell_counts": cell_counts,
                    "area": df["总面积(μm²)"].values,
                    "avg_cell_area": df["相对平均细胞面积"].values
                })
            N0_guess = cell_counts[0] if cell_counts[0] != 0 else 1e-6
            K_guess = np.max(cell_counts) if np.max(cell_counts) != 0 else 1.0
            mu_max_guess = 0.1
            initial_guess = [mu_max_guess, K_guess, N0_guess]
            bounds = (
                [1e-6, 0.1, 1e-6],
                [1.0, np.max(cell_counts) * 1.5 if np.max(cell_counts) != 0 else 100,
                 np.max(cell_counts) * 0.5 if np.max(cell_counts) != 0 else 10]
            )
            try:
                popt, pcov = curve_fit(
                    f=modified_logistic_model,
                    xdata=t,
                    ydata=cell_counts,
                    p0=initial_guess,
                    bounds=bounds,
                    maxfev=10000
                )
                mu_max_fit, K_fit, N0_fit = popt
                y_fit = modified_logistic_model(t, mu_max_fit, K_fit, N0_fit)
                ss_res = np.sum((cell_counts - y_fit) ** 2)
                ss_tot = np.sum((cell_counts - np.mean(cell_counts)) ** 2)
                r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else np.nan
                t_d = np.log(2) / mu_max_fit if mu_max_fit != 0 else np.nan
                F_last_cell_number = df['目标数量'].iloc[-1]
                total_count = len(df['目标数量'])
                growth_rate = F_last_cell_number / total_count if (F_last_cell_number is not None and not np.isnan(F_last_cell_number)) else np.nan
                phase_duration, _ = calculate_growth_phases(t, mu_max_fit, K_fit, N0_fit)
                lag_duration = phase_duration["滞后期时长(h)"]
                log_duration = phase_duration["对数期时长(h)"]
                stable_duration = phase_duration["稳定期时长(h)"]

                individual_summary.append({
                    "腔室名称": sheet,
                    "最大比生长速率μmax (h^-1)": round(mu_max_fit, 4),
                    "环境容纳量K (个/腔室)": round(K_fit, 4),
                    "初始细胞数量N0 (个/腔室)": round(N0_fit, 4),
                    "拟合优度R²": round(r2, 2),
                    "平均细胞周期T_d(h)": round(t_d, 4) if not np.isnan(t_d) else np.nan,
                    "增殖倍数 F": round(F_last_cell_number, 4) if not np.isnan(F_last_cell_number) else np.nan,
                    "生长效率 η (1/增殖倍数)": round(growth_rate, 4) if not np.isnan(growth_rate) else np.nan,
                    "滞后期时长(h)": lag_duration,
                    "对数期时长(h)": log_duration,
                    "稳定期时长(h)": stable_duration
                })
                print(f"页签 {sheet} 单独拟合完成\n")

                plt.figure(figsize=(10, 6))
                plt.scatter(t, cell_counts, label="Actual data", color="blue", alpha=0.6)
                plt.plot(t, y_fit, label="Fitted curve", color="#F2BA02", linewidth=2)
                ax = plt.gca()
                ax.set_xlabel("Cultivation time (h)")
                ax.set_ylabel("Relative cell number (normalized to 0 h)")
                ax.set_title(f"{plt_name} Cell growth curve fitting")
                ax.xaxis.set_major_locator(MultipleLocator(12))
                ax.xaxis.set_minor_locator(MultipleLocator(3))
                y_tick_interval = (ax.get_yticks()[1] - ax.get_yticks()[0]) if len(ax.get_yticks()) > 1 else 1
                ax.yaxis.set_minor_locator(MultipleLocator(y_tick_interval / 5))
                param_text = (f"$\it{{μ}}_{{\mathrm{{max}}}}$= {round(mu_max_fit, 2)} h⁻¹\n"
                              f"$\it{{K}}$ = {round(K_fit, 2)}\n"
                              f"$\it{{N}}_{{0}}$ = {round(N0_fit, 2)}\n"
                              f"R² = {round(r2, 2)}")
                plt.text(0.05, 0.95, param_text, transform=ax.transAxes,
                         verticalalignment='top', fontsize=20,
                         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                plt.legend()
                plt.grid(alpha=0.3)
                img_buffer = io.BytesIO()
                plt.savefig(img_buffer, format='png', dpi=300, bbox_inches="tight")
                img_buffer.seek(0)
                plt.close()

                param_col_start = 5
                ws.cell(row=1, column=param_col_start).value = "参数名"
                ws.cell(row=1, column=param_col_start + 1).value = "参数值"
                ws.cell(row=2, column=param_col_start).value = "最大比生长速率uMAX(h^-1)"
                ws.cell(row=2, column=param_col_start + 1).value = round(mu_max_fit, 4)
                ws.cell(row=3, column=param_col_start).value = "环境容纳量K(个)"
                ws.cell(row=3, column=param_col_start + 1).value = round(K_fit, 4)
                ws.cell(row=4, column=param_col_start).value = "初始细胞数量N0(个)"
                ws.cell(row=4, column=param_col_start + 1).value = round(N0_fit, 4)
                ws.cell(row=5, column=param_col_start).value = "拟合优度R²"
                ws.cell(row=5, column=param_col_start + 1).value = round(r2, 4)
                ws.cell(row=6, column=param_col_start).value = "平均细胞周期T_d(h)"
                ws.cell(row=6, column=param_col_start + 1).value = round(t_d, 4) if not np.isnan(t_d) else np.nan
                ws.cell(row=7, column=param_col_start).value = "增殖倍数 F"
                ws.cell(row=7, column=param_col_start + 1).value = round(F_last_cell_number, 4) if not np.isnan(F_last_cell_number) else np.nan
                ws.cell(row=8, column=param_col_start).value = "生长效率 η (1/增殖倍数)"
                ws.cell(row=8, column=param_col_start + 1).value = round(growth_rate, 4) if not np.isnan(
                    growth_rate) else np.nan
                ws.cell(row=9, column=param_col_start).value = "滞后期时长(h)"
                ws.cell(row=9, column=param_col_start + 1).value = lag_duration
                ws.cell(row=10, column=param_col_start).value = "对数期时长(h)"
                ws.cell(row=10, column=param_col_start + 1).value = log_duration
                ws.cell(row=11, column=param_col_start).value = "稳定期时长(h)"
                ws.cell(row=11, column=param_col_start + 1).value = stable_duration

                img = Image(img_buffer)
                img.width = 600
                img.height = 400
                ws.add_image(img, "H02")

            except Exception as e:
                print(f"页签 {sheet} 单独拟合失败：{str(e)}\n")
                individual_summary.append({
                    "腔室名称": sheet,
                    "最大比生长速率μmax (h^-1)": np.nan,
                    "环境容纳量K (个/腔室)": np.nan,
                    "初始细胞数量N0 (个/腔室)": np.nan,
                    "拟合优度R²": np.nan,
                    "平均细胞周期T_d(h)": np.nan,
                    "增殖倍数 F": np.nan,
                    "生长效率 η (1/增殖倍数)": np.nan,
                    "滞后期时长(h)": np.nan,
                    "对数期时长(h)": np.nan,
                    "稳定期时长(h)": np.nan
                })
        except Exception as e:
            print(f"处理页签 {sheet} 时出错：{str(e)}\n")
            individual_summary.append({
                "腔室名称": sheet,
                "最大比生长速率μmax (h^-1)": np.nan,
                "环境容纳量K (个/腔室)": np.nan,
                "初始细胞数量N0 (个/腔室)": np.nan,
                "拟合优度R²": np.nan,
                "平均细胞周期T_d(h)": np.nan,
                "增殖倍数 F": np.nan,
                "生长效率 η (1/增殖倍数)": np.nan,
                "滞后期时长(h)": np.nan,
                "对数期时长(h)": np.nan,
                "稳定期时长(h)": np.nan
            })

    # 第二步：合并拟合
    merged_params = None
    merged_img_buffer = None
    if len(merged_t) >= min_data_points * 2 and len(valid_sheets) >= 2:
        original_merged_data = [
            {"sheet": d["sheet"], "t": d["t"], "cell_counts": d["cell_counts"]}
            for d in individual_data
        ]
        current_valid_sheets = [d["sheet"] for d in original_merged_data]
        best_r2 = -np.inf
        best_merged_params = None
        best_merged_t = None
        best_merged_counts = None
        best_valid_sheets = []
        max_iterations = len(current_valid_sheets) - 1
        iteration = 0
        while iteration < max_iterations and len(current_valid_sheets) >= 2:
            current_merged_t = []
            current_merged_counts = []
            for d in original_merged_data:
                if d["sheet"] in current_valid_sheets:
                    current_merged_t.extend(d["t"])
                    current_merged_counts.extend(d["cell_counts"])
            current_merged_t = np.array(current_merged_t)
            current_merged_counts = np.array(current_merged_counts)
            try:
                N0_guess = np.mean([
                    d["cell_counts"][0] for d in original_merged_data
                    if d["sheet"] in current_valid_sheets and d["cell_counts"][0] > 0
                ])
                K_guess = np.max(current_merged_counts)
                mu_max_guess = 0.1
                initial_guess = [mu_max_guess, K_guess, N0_guess]
                bounds = (
                    [1e-6, 0.1, 1e-6],
                    [1.0, np.max(current_merged_counts) * 1.5, np.max(current_merged_counts) * 0.5]
                )
                popt, pcov = curve_fit(
                    f=modified_logistic_model,
                    xdata=current_merged_t,
                    ydata=current_merged_counts,
                    p0=initial_guess,
                    bounds=bounds,
                    maxfev=10000
                )
                mu_max, K, N0 = popt
                y_fit = modified_logistic_model(current_merged_t, mu_max, K, N0)
                ss_res = np.sum((current_merged_counts - y_fit) ** 2)
                ss_tot = np.sum((current_merged_counts - np.mean(current_merged_counts)) ** 2)
                r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else -np.inf
                if r2 > best_r2:
                    best_r2 = r2
                    best_merged_params = {
                        "mu_max": mu_max,
                        "K": K,
                        "N0": N0,
                        "r2": r2
                    }
                    best_merged_t = current_merged_t
                    best_merged_counts = current_merged_counts
                    best_valid_sheets = current_valid_sheets.copy()
                if r2 >= 0.8:
                    break
                chamber_residuals = {}
                for d in original_merged_data:
                    if d["sheet"] not in current_valid_sheets:
                        continue
                    y_fit_chamber = modified_logistic_model(d["t"], mu_max, K, N0)
                    res = np.sum((d["cell_counts"] - y_fit_chamber) ** 2)
                    chamber_residuals[d["sheet"]] = res
                if chamber_residuals:
                    worst_chamber = max(chamber_residuals, key=chamber_residuals.get)
                    current_valid_sheets.remove(worst_chamber)
                    print(
                        f"迭代{iteration + 1}：剔除腔室 {worst_chamber}（残差过大），剩余{len(current_valid_sheets)}个腔室")
            except Exception as e:
                print(f"拟合失败：{str(e)}，尝试剔除下一个腔室")
                if current_valid_sheets:
                    worst_chamber = current_valid_sheets[0]
                    current_valid_sheets.remove(worst_chamber)
                    print(f"剔除腔室 {worst_chamber} 后重试")
            iteration += 1
        if best_merged_params and best_merged_params["r2"] >= 0.8:
            merged_params = best_merged_params
            merged_params["valid_sheets"] = best_valid_sheets
            merged_t = best_merged_t
            merged_cell_counts = best_merged_counts
            print(
                f"合并拟合完成（已剔除异常腔室）：μmax={merged_params['mu_max']:.6f}, K={merged_params['K']:.6f}, N0={merged_params['N0']:.6f}, R²={merged_params['r2']:.6f}")
            plt.figure(figsize=(12, 7))
            plt.scatter(merged_t, merged_cell_counts, label="Merged actual data", color="blue", alpha=0.5, s=30)
            sorted_t = np.sort(merged_t)
            plt.plot(sorted_t, modified_logistic_model(sorted_t, merged_params["mu_max"], merged_params["K"],
                                                       merged_params["N0"]),
                     label="Merged fitted curve", color="red", linewidth=2)
            ax = plt.gca()
            ax.set_xlabel("Cultivation time (h)")
            ax.set_ylabel("Relative cell number (normalized to 0 h)")
            ax.set_title(
                f"Growth curve (merged data, {len(best_valid_sheets)} chambers, NH$_4^+$-N={nitrogen_concentration})")
            ax.xaxis.set_major_locator(MultipleLocator(12))
            ax.xaxis.set_minor_locator(MultipleLocator(3))
            y_tick_interval = (ax.get_yticks()[1] - ax.get_yticks()[0]) if len(ax.get_yticks()) > 1 else 1
            ax.yaxis.set_minor_locator(MultipleLocator(y_tick_interval / 5))
            merged_phase_duration, _ = calculate_growth_phases(sorted_t, merged_params["mu_max"], merged_params["K"],
                                                               merged_params["N0"])
            param_text = (f"$\it{{μ}}_{{\mathrm{{max}}}}$ = {merged_params['mu_max']:.2f} h⁻¹\n"
                          f"$\it{{K}}$ = {merged_params['K']:.2f}\n"
                          f"$\it{{N}}_{{0}}$ = {merged_params['N0']:.2f}\n"
                          f"R² = {merged_params['r2']:.2f}")
            plt.text(0.05, 0.95, param_text, transform=ax.transAxes,
                     verticalalignment='top', fontsize=20, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            plt.legend()
            plt.grid(alpha=0.3)
            merged_img_buffer = io.BytesIO()
            plt.savefig(merged_img_buffer, format='png', dpi=300, bbox_inches="tight")
            merged_img_buffer.seek(0)
            plt.close()
        else:
            print(f"无法通过剔除腔室使R²≥0.8，最佳R²为{best_r2:.2f}，保留所有有效腔室")

    # 第三步：创建汇总结果页签
    if "汇总结果" in wb_original.sheetnames:
        del wb_original["汇总结果"]
    summary_ws = wb_original.create_sheet(title="汇总结果", index=0)
    summary_ws["A1"] = "腔室名称"
    summary_ws["B1"] = "最大比生长速率μmax (h^-1)"
    summary_ws["C1"] = "环境容纳量K (个/腔室)"
    summary_ws["D1"] = "初始细胞数量N0 (个/腔室)"
    summary_ws["E1"] = "拟合优度R²"
    summary_ws["F1"] = "平均细胞周期T_d(h)"
    summary_ws["G1"] = "增殖倍数 F"
    summary_ws["H1"] = "生长效率 η (1/增殖倍数)"
    summary_ws["I1"] = "滞后期时长(h)"
    summary_ws["J1"] = "对数期时长(h)"
    summary_ws["K1"] = "稳定期时长(h)"

    valid_sheets = merged_params.get("valid_sheets", []) if merged_params else []
    valid_data = [item for item in individual_summary if item["腔室名称"] in valid_sheets]
    non_valid_data = [item for item in individual_summary if item["腔室名称"] not in valid_sheets]

    current_row = 2
    for data in valid_data:
        summary_ws[f"A{current_row}"] = data["腔室名称"]
        summary_ws[f"B{current_row}"] = data["最大比生长速率μmax (h^-1)"]
        summary_ws[f"C{current_row}"] = data["环境容纳量K (个/腔室)"]
        summary_ws[f"D{current_row}"] = data["初始细胞数量N0 (个/腔室)"]
        summary_ws[f"E{current_row}"] = data["拟合优度R²"]
        summary_ws[f"F{current_row}"] = data["平均细胞周期T_d(h)"]
        summary_ws[f"G{current_row}"] = data["增殖倍数 F"]
        summary_ws[f"H{current_row}"] = data["生长效率 η (1/增殖倍数)"]
        summary_ws[f"I{current_row}"] = data["滞后期时长(h)"]
        summary_ws[f"J{current_row}"] = data["对数期时长(h)"]
        summary_ws[f"K{current_row}"] = data["稳定期时长(h)"]
        current_row += 1

    if non_valid_data:
        summary_ws[f"A{current_row}"] = "未参与拟合数据"
        summary_ws.merge_cells(f"A{current_row}:K{current_row}")
        summary_ws[f"A{current_row}"].font = openpyxl.styles.Font(bold=True, color="FF0000")
        current_row += 1

    for data in non_valid_data:
        summary_ws[f"A{current_row}"] = data["腔室名称"]
        summary_ws[f"B{current_row}"] = data["最大比生长速率μmax (h^-1)"]
        summary_ws[f"C{current_row}"] = data["环境容纳量K (个/腔室)"]
        summary_ws[f"D{current_row}"] = data["初始细胞数量N0 (个/腔室)"]
        summary_ws[f"E{current_row}"] = data["拟合优度R²"]
        summary_ws[f"F{current_row}"] = data["平均细胞周期T_d(h)"]
        summary_ws[f"G{current_row}"] = data["增殖倍数 F"]
        summary_ws[f"H{current_row}"] = data["生长效率 η (1/增殖倍数)"]
        summary_ws[f"I{current_row}"] = data["滞后期时长(h)"]
        summary_ws[f"J{current_row}"] = data["对数期时长(h)"]
        summary_ws[f"K{current_row}"] = data["稳定期时长(h)"]
        current_row += 1

    for col in ["A", "B", "C", "D", "E", "F", "G", "H"]:
        summary_ws.column_dimensions[col].width = 25
    for col in ["I", "J", "K"]:
        summary_ws.column_dimensions[col].width = 20

    # 保留数据使用到的行
        reserve_row = current_row + 1

    # 第四步：在汇总结果页签添加合并拟合数据和曲线
    if merged_params:
        last_row = current_row + 1
        summary_ws[f"A{last_row}"] = "合并拟合结果"
        summary_ws[f"A{last_row}"].font = openpyxl.styles.Font(bold=True)
        merged_phase_duration, _ = calculate_growth_phases(merged_t, merged_params["mu_max"], merged_params["K"],
                                                           merged_params["N0"])
        param_rows = {
            "参与拟合的页签数量": len(merged_params["valid_sheets"]),
            "总数据点数量": len(merged_t),
            "最大比生长速率μmax (h^-1)": round(merged_params["mu_max"], 6),
            "环境容纳量K (个/腔室)": round(merged_params["K"], 6),
            "初始细胞数量N0 (个/腔室)": round(merged_params["N0"], 6),
            "拟合优度R²": round(merged_params["r2"], 6),
            "滞后期时长(h)": merged_phase_duration["滞后期时长(h)"],
            "对数期时长(h)": merged_phase_duration["对数期时长(h)"],
            "稳定期时长(h)": merged_phase_duration["稳定期时长(h)"]
        }
        current_row = last_row + 1
        for param, value in param_rows.items():
            summary_ws[f"A{current_row}"] = param
            summary_ws[f"B{current_row}"] = value
            current_row += 1
        if merged_img_buffer:
            summary_ws[f"C{last_row}"] = "参与拟合的腔室"
            summary_ws[f"D{last_row}"] = ", ".join(merged_params["valid_sheets"])
            merged_img = Image(merged_img_buffer)
            merged_img.width = 700
            merged_img.height = 500
            summary_ws.add_image(merged_img, f"C{last_row + 1}")

    # 第五步：为所有有效页签添加与合并拟合曲线的对比图
    if merged_params and len(valid_sheets) > 0:
        for sheet_data in individual_data:
            sheet_name = sheet_data["sheet"]
            plt_name = nitrogen_concentration + "_" + sheet_name[4:]
            t_data = sheet_data["t"]
            counts_data = sheet_data["cell_counts"]
            plt.figure(figsize=(10, 6))
            plt.scatter(t_data, counts_data, label=f"{plt_name} actual data", color="blue", alpha=0.6)
            merged_curve = modified_logistic_model(t_data, merged_params["mu_max"], merged_params["K"],
                                                   merged_params["N0"])
            plt.plot(t_data, merged_curve, label="Merged fitted curve", color="red", linewidth=2)
            ax = plt.gca()
            ax.set_xlabel("Cultivation time (h)")
            ax.set_ylabel("Relative cell number (normalized to 0 h)")
            ax.set_title(f"{plt_name} data vs merged fitting curve")
            ax.xaxis.set_major_locator(MultipleLocator(12))
            ax.xaxis.set_minor_locator(MultipleLocator(3))
            y_tick_interval = (ax.get_yticks()[1] - ax.get_yticks()[0]) if len(ax.get_yticks()) > 1 else 1
            ax.yaxis.set_minor_locator(MultipleLocator(y_tick_interval / 5))
            merged_phase_duration, _ = calculate_growth_phases(t_data, merged_params["mu_max"], merged_params["K"],
                                                               merged_params["N0"])
            param_text = (f"$\it{{μ}}_{{\mathrm{{max}}}}$ = {merged_params['mu_max']:.2f} h⁻¹\n"
                          f"$\it{{K}}$ = {merged_params['K']:.2f}\n"
                          f"$\it{{N}}_{{0}}$ = {merged_params['N0']:.2f}\n"
                          f"R² = {merged_params['r2']:.2f}")
            plt.text(0.05, 0.95, param_text, transform=ax.transAxes,
                     verticalalignment='top', fontsize=20, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            plt.legend()
            plt.grid(alpha=0.3)
            img_buffer = io.BytesIO()
            plt.savefig(img_buffer, format='png', dpi=300, bbox_inches="tight")
            img_buffer.seek(0)
            plt.close()
            try:
                ws = wb_original[sheet_name]
                img = Image(img_buffer)
                img.width = 600
                img.height = 400
                ws.add_image(img, "H24")
                print(f"已为 {sheet_name} 添加合并拟合对比图")
            except Exception as e:
                print(f"为 {sheet_name} 添加对比图失败：{str(e)}")

    # 第六步：为每个有效页签添加额外的2个图
    for sheet_data in individual_data:
        sheet_name = sheet_data["sheet"]
        t_data = sheet_data["t"]
        cell_counts = sheet_data["cell_counts"]
        area = sheet_data["area"]
        avg_cell_area = sheet_data["avg_cell_area"]
        try:
            ws = wb_original[sheet_name]
            plt_title = nitrogen_concentration + "_" + sheet_name[4:]
            # 1. 总面积散点图
            plt.figure(figsize=(10, 6))
            plt.scatter(t_data, area, label="Relative total cell area", color="green", alpha=0.6)
            ax = plt.gca()
            ax.set_xlabel("Cultivation time (h)")
            ax.set_ylabel("Relative total cell area (normalized to 0 h)")
            ax.set_title(f"{plt_title} - Relative total cell area variation")
            ax.xaxis.set_major_locator(MultipleLocator(12))
            ax.xaxis.set_minor_locator(MultipleLocator(3))
            y_tick_interval = (ax.get_yticks()[1] - ax.get_yticks()[0]) if len(ax.get_yticks()) > 1 else 1
            ax.yaxis.set_minor_locator(MultipleLocator(y_tick_interval / 5))
            plt.legend()
            plt.grid(alpha=0.3)
            area_buffer = io.BytesIO()
            plt.savefig(area_buffer, format='png', dpi=300, bbox_inches="tight")
            area_buffer.seek(0)
            plt.close()
            area_img = Image(area_buffer)
            area_img.width = 600
            area_img.height = 400
            ws.add_image(area_img, "R2")

            # 2. 平均细胞面积点线图
            plt.figure(figsize=(10, 6))
            plt.scatter(t_data, avg_cell_area, label="Relative average cell area", color="purple", alpha=0.6, s=30)
            plt.plot(t_data, avg_cell_area, color="purple", alpha=0.8, linestyle="-", linewidth=1.5, marker="",
                     markersize=5)
            ax = plt.gca()
            ax.set_xlabel("Cultivation time (h)")
            ax.set_ylabel("Relative average cell area (normalized to 0 h)")
            ax.set_title(f"{plt_title} - Relative average cell area variation")
            ax.xaxis.set_major_locator(MultipleLocator(12))
            ax.xaxis.set_minor_locator(MultipleLocator(3))
            y_tick_interval = (ax.get_yticks()[1] - ax.get_yticks()[0]) if len(ax.get_yticks()) > 1 else 1
            ax.yaxis.set_minor_locator(MultipleLocator(y_tick_interval / 5))
            plt.legend()
            plt.grid(alpha=0.3)
            avg_buffer = io.BytesIO()
            plt.savefig(avg_buffer, format='png', dpi=300, bbox_inches="tight")
            avg_buffer.seek(0)
            plt.close()
            avg_img = Image(avg_buffer)
            avg_img.width = 600
            avg_img.height = 400
            ws.add_image(avg_img, "R24")
            print(f"已为 {sheet_name} 添加2个图表")
        except Exception as e:
            print(f"为 {sheet_name} 添加趋势图表失败：{str(e)}")

    # 第七步 新增功能：汇总所有参与拟合腔室的总面积和相对平均细胞面积散点图并添加到汇总结果页签
    if merged_params and len(valid_sheets) > 0:
        # 收集所有参与拟合腔室的数据
        all_t = []
        all_area = []
        all_avg_area = []
        for sheet_data in individual_data:
            if sheet_data["sheet"] in merged_params["valid_sheets"]:
                all_t.extend(sheet_data["t"])
                all_area.extend(sheet_data["area"])
                all_avg_area.extend(sheet_data["avg_cell_area"])

        # 1. 创建总面积汇总散点图
        plt.figure(figsize=(12, 7))
        plt.scatter(all_t, all_area, label="Relative total area data", color="green", alpha=0.6, s=30)
        ax = plt.gca()
        ax.set_xlabel("Cultivation time (h)")
        ax.set_ylabel("Relative total cell area (normalized to 0 h)")
        ax.set_title(f"Relative total area distribution (merged data, {len(merged_params['valid_sheets'])} chambers, NH$_4^+$-N={nitrogen_concentration})")
        ax.xaxis.set_major_locator(MultipleLocator(12))
        ax.xaxis.set_minor_locator(MultipleLocator(3))
        y_tick_interval = (ax.get_yticks()[1] - ax.get_yticks()[0]) if len(ax.get_yticks()) > 1 else 1
        ax.yaxis.set_minor_locator(MultipleLocator(y_tick_interval / 5))
        plt.legend()
        plt.grid(alpha=0.3)
        total_area_buffer = io.BytesIO()
        plt.savefig(total_area_buffer, format='png', dpi=300, bbox_inches="tight")
        total_area_buffer.seek(0)
        plt.close()

        # 2. 创建相对平均细胞面积汇总散点图
        plt.figure(figsize=(12, 7))
        plt.scatter(all_t, all_avg_area, label="Relative average cell area data", color="purple", alpha=0.6, s=30)
        ax = plt.gca()
        ax.set_xlabel("Cultivation time (h)")
        ax.set_ylabel("Relative average cell area (normalized to 0 h)")
        ax.set_title(f"Relative average cell area distribution (merged data, {len(merged_params['valid_sheets'])} chambers, NH$_4^+$-N={nitrogen_concentration})")
        ax.xaxis.set_major_locator(MultipleLocator(12))
        ax.xaxis.set_minor_locator(MultipleLocator(3))
        y_tick_interval = (ax.get_yticks()[1] - ax.get_yticks()[0]) if len(ax.get_yticks()) > 1 else 1
        ax.yaxis.set_minor_locator(MultipleLocator(y_tick_interval / 5))
        plt.legend()
        plt.grid(alpha=0.3)
        avg_area_buffer = io.BytesIO()
        plt.savefig(avg_area_buffer, format='png', dpi=300, bbox_inches="tight")
        avg_area_buffer.seek(0)
        plt.close()

        # 确定汇总结果页签中放置图表的位置
        # 找到当前汇总结果页签的最后一行
        current_row = reserve_row + 1  # 留出一些空白

        # 添加总面积汇总图
        total_area_img = Image(total_area_buffer)
        total_area_img.width = 700
        total_area_img.height = 500
        summary_ws.add_image(total_area_img, f"G{current_row}")

        # 添加相对平均细胞面积汇总图
        avg_area_img = Image(avg_area_buffer)
        avg_area_img.width = 700
        avg_area_img.height = 500
        summary_ws.add_image(avg_area_img, f"L{current_row}")

        print(f"已在汇总结果页签添加2个汇总散点图")

    # 保存最终结果
    wb_original.save(result_excel_path)
    wb_original.close()
    print(f"所有处理完成，结果已保存到：{result_excel_path}")
    return pd.DataFrame(individual_summary)


# ---------------------- 脚本执行入口 ----------------------
if __name__ == "__main__":
    EXCEL_FILE_PATH = r"F:\Microalgae_Photoes\20251104\数据处理结果\标准化数据\CH6_标准化.xlsx"
    # RESULT_EXCEL_PATH = EXCEL_FILE_PATH[:-5] + "_数据处理.xlsx"
    RESULT_EXCEL_PATH = r"F:\Microalgae_Photoes\20251104\数据处理结果\最终结果\L100_500.xlsx"
    Nitrogen = "500 mg/L"
    result = process_cell_growth(
        excel_path=EXCEL_FILE_PATH,
        result_excel_path=RESULT_EXCEL_PATH,
        min_data_points=5,
        nitrogen_concentration=Nitrogen
    )
    if result is not None:
        print("\n单独拟合结果预览（含生长阶段时长）：")
        print(result[["腔室名称", "滞后期时长(h)", "对数期时长(h)", "稳定期时长(h)"]].head())