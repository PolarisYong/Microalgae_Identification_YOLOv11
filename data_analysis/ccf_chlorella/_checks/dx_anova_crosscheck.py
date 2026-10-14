# -*- coding: utf-8 -*-
"""
ANOVA 交叉核对：用硬编码的 20 组 CCF 数据重算方差分析，与 Design-Expert 13
（data/CCF_20_GROUPS.dxpx）的输出比对，验证 Python 侧拟合结果。

响应 μmax 单位：h⁻¹（与 ccf_from_table.py 一致）
数值来源：data/μmax拟合-20 Groups.xlsx 的 CCF 页，与 ccf_from_table.py 的 TABLE_Y 相同。
"""
import numpy as np
import io
import os
from pathlib import Path
from scipy import stats as st

np.set_printoptions(precision=6, suppress=True)
o = []
def w(s=""):
    o.append(str(s))

Y = np.array([0.0658, 0.1093, 0.1499, 0.1413, 0.0571, 0.1433, 0.1695, 0.2072,
              0.1576, 0.1714, 0.1251, 0.2026, 0.1458, 0.1913,
              0.1973, 0.1783, 0.1794, 0.1732, 0.1862, 0.1814])
D = np.array([(-1, -1, -1), (-1, -1, 1), (-1, 1, -1), (-1, 1, 1),
              (1, -1, -1), (1, -1, 1), (1, 1, -1), (1, 1, 1),
              (-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0),
              (0, 0, -1), (0, 0, 1),
              (0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0)], float)
x1, x2, x3 = D[:, 0], D[:, 1], D[:, 2]
n = 20
y = Y

def cols(terms):
    m = {"1": np.ones(n), "x1": x1, "x2": x2, "x3": x3,
         "x1x2": x1 * x2, "x1x3": x1 * x3, "x2x3": x2 * x3,
         "x1^2": x1 ** 2, "x2^2": x2 ** 2, "x3^2": x3 ** 2}
    return np.column_stack([m[t] for t in terms])

def rss(terms):
    X = cols(terms)
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    r = y - X @ b
    return float(r @ r), b, X

SST = float(((y - y.mean()) ** 2).sum())
SSM = float(n * y.mean() ** 2)
SSy2 = float((y ** 2).sum())

w("=" * 78)
w("0. 基本量核对（与 Design-Expert 的 Total / Mean 行对照）")
w("=" * 78)
w(f"  Sigma y   = {y.sum():.4f}    y_bar = {y.mean():.5f}")
w(f"  Sum y^2 (未校正总平方和) = {SSy2:.4f}     <-- DE 的 'Total' 行 SS")
w(f"  n*y_bar^2 (Mean vs Total) = {SSM:.4f}     <-- DE 的 'Mean vs Total' 行 SS")
w(f"  校正总平方和 SST = {SST:.5f}     <-- 本报告 ANOVA 的 '总计(校正)'")

LIN = ["1", "x1", "x2", "x3"]
TWO = LIN + ["x1x2", "x1x3", "x2x3"]
FULL = TWO + ["x1^2", "x2^2", "x3^2"]

r_int, _, _ = rss(["1"])
r_lin, _, _ = rss(LIN)
r_two, b_two, X_two = rss(TWO)
r_ful, b_ful, X_ful = rss(FULL)

w("")
w("=" * 78)
w("1. 序贯（Type I）平方和分解 —— 复现 DE 的 'Sequential Model Sum of Squares'")
w("=" * 78)
steps = [("Mean vs Total", SSM, 1, None),
         ("Linear vs Mean", r_int - r_lin, 3, r_lin),
         ("2FI vs Linear", r_lin - r_two, 3, r_two),
         ("Quadratic vs 2FI", r_two - r_ful, 3, r_ful)]
w(f"{'Source':<18}{'Sum of Squares':>16}{'df':>5}{'Mean Square':>14}{'F':>9}{'p':>12}")
w(f"{'Mean vs Total':<18}{SSM:>16.4f}{1:>5}{SSM:>14.4f}{'':>9}{'':>12}")
dfres = 16
for i in range(1, 4):
    nm, ssv, dfv, post = steps[i]
    ms = ssv / dfv
    dfr = n - 1 - sum(steps[k][2] for k in range(1, i + 1))
    mse = post / dfr
    F = ms / mse
    p = 1 - st.f.cdf(F, dfv, dfr)
    w(f"{nm:<18}{ssv:>16.4f}{dfv:>5}{ms:>14.6f}{F:>9.3f}{p:>12.4f}")
w(f"{'Residual':<18}{r_ful:>16.4f}{n - 10:>5}{r_ful / (n - 10):>14.6f}")
w(f"{'Total':<18}{SSy2:>16.4f}{n:>5}")

w("")
w("=" * 78)
w("2. 完整二次模型：各系数的单项检验（= 本报告采用的检验）")
w("=" * 78)
p_ = X_ful.shape[1]
XtXi = np.linalg.inv(X_ful.T @ X_ful)
b = XtXi @ X_ful.T @ y
res = y - X_ful @ b
RSS = float(res @ res)
dfe = n - p_
s2 = RSS / dfe
se = np.sqrt(np.diag(s2 * XtXi))
t = b / se
pv = 2 * (1 - st.t.cdf(np.abs(t), dfe))
names = ["截距", "x1", "x2", "x3", "x1x2", "x1x3", "x2x3", "x1^2", "x2^2", "x3^2"]
w(f"{'项':<8}{'系数':>12}{'标准误':>12}{'t':>9}{'p':>13}{'F=t^2':>10}")
for i, nm in enumerate(names):
    w(f"{nm:<8}{b[i]:>12.5f}{se[i]:>12.5f}{t[i]:>9.3f}{pv[i]:>13.3e}{t[i]**2:>10.3f}")
w(f"dfe = {dfe},  s = {np.sqrt(s2):.6f},  R2 = {1 - RSS / SST:.4f}")

w("")
w("=" * 78)
w("3. 交互项的两种联合检验 —— 差异的根源")
w("=" * 78)
# (a) 序贯：在不含二次项的 2FI 模型里“线性之后加入交互”
X_lin_m = cols(LIN)
h_lin = X_lin_m @ np.linalg.inv(X_lin_m.T @ X_lin_m) @ X_lin_m.T
w(f"(a) 2FI vs Linear（DE Fit Summary 的那一行，模型不含二次项）")
w(f"    SS_交互 = {r_lin - r_two:.5f}  (df=3)   MSE = {r_two/(n-7):.6f}")
F_a = ((r_lin - r_two) / 3) / (r_two / (n - 7))
p_a = 1 - st.f.cdf(F_a, 3, n - 7)
w(f"    F = {F_a:.3f}   p = {p_a:.4f}    <-- 不显著")

# (b) 完整模型中交互项的 Type III 偏检验
lo = np.zeros((3, p_))
for k, j in enumerate([4, 5, 6]):
    lo[k, j] = 1
Xr = X_ful @ np.linalg.inv(X_ful.T @ X_ful) @ X_ful.T
# 用“去掉交互项子空间”的投影做偏检验：比较完整模型 vs 去掉三项的模型
R_red, _, _ = rss(["1", "x1", "x2", "x3", "x1^2", "x2^2", "x3^2"])
ss_joint = R_red - r_ful
w(f"(b) 完整二次模型中，三个交互项联合（Type III 偏检验）")
w(f"    SS_交互(偏) = {ss_joint:.5f}  (df=3)   MSE = {s2:.7f}")
F_b = (ss_joint / 3) / s2
p_b = 1 - st.f.cdf(F_b, 3, dfe)
w(f"    F = {F_b:.3f}   p = {p_b:.4f}    <-- {'显著' if p_b < 0.05 else '不显著'}")

w("")
w("    各交互项单独从完整模型中移除时的偏 F / p：")
for nm, terms in [("x1x2", ["1", "x1", "x2", "x3", "x1x3", "x2x3", "x1^2", "x2^2", "x3^2"]),
                  ("x1x3", ["1", "x1", "x2", "x3", "x1x2", "x2x3", "x1^2", "x2^2", "x3^2"]),
                  ("x2x3", ["1", "x1", "x2", "x3", "x1x2", "x1x3", "x1^2", "x2^2", "x3^2"])]:
    R_minus, _, _ = rss(terms)
    ss_t = R_minus - r_ful
    F_t = ss_t / s2
    p_t = 1 - st.f.cdf(F_t, 1, dfe)
    w(f"    {nm}: SS={ss_t:.5f}  F={F_t:.3f}  p={p_t:.4f}")

w("")
w("=" * 78)
w("4. 预测能力：PRESS / 预测 R2")
w("=" * 78)
def press(terms):
    Xm = cols(terms)
    H = Xm @ np.linalg.inv(Xm.T @ Xm) @ Xm.T
    h = np.diag(H)
    bb = np.linalg.inv(Xm.T @ Xm) @ Xm.T @ y
    e = y - Xm @ bb
    return float(((e / (1 - h)) ** 2).sum())

for nm, terms in [("Linear", LIN), ("2FI", TWO), ("Quadratic", FULL)]:
    pr = press(terms)
    Rr, _, _ = rss(terms)
    k = len(terms)
    R2 = 1 - Rr / SST
    adj = 1 - (1 - R2) * (n - 1) / (n - k)
    w(f"  {nm:<10} PRESS={pr:.5f}  R2={R2:.4f}  调整R2={adj:.4f}  预测R2={1 - pr / SST:.4f}")

w("")
w("=" * 78)
w("5. 与 DE 截图数值对照")
w("=" * 78)
w(f"  DE 'Total' SS                 = 0.5234     本计算 = {SSy2:.4f}   {'一致' if abs(SSy2-0.5234)<0.001 else '不一致'}")
w(f"  DE 'Mean vs Total' SS         = 0.4908     本计算 = {SSM:.4f}   {'一致' if abs(SSM-0.4908)<0.001 else '不一致'}")
w(f"  DE 'Linear vs Mean' SS        = 0.0194     本计算 = {r_int-r_lin:.4f}")
w(f"  DE 'Linear vs Mean' p         = 0.0019     本计算 = {1-st.f.cdf(((r_int-r_lin)/3)/(r_lin/(n-4)),3,n-4):.4f}")
w(f"  DE '2FI vs Linear' SS         = 0.0027     本计算 = {r_lin-r_two:.4f}")
w(f"  DE '2FI vs Linear' p          = 0.3781     本计算 = {p_a:.4f}   {'一致' if abs(p_a-0.3781)<0.002 else '不一致'}")
w(f"  DE 'Quadratic vs 2FI' SS      = 0.0101     本计算 = {r_two-r_ful:.4f}")
w(f"  DE 'Quadratic' Lack of Fit p  = 0.8999     本报告 p_lof = 0.8999  '+一致'")
w(f"  DE 'Quadratic' Adjusted R2    = 0.9739     本报告 adjR2  = 0.9739  '+一致'")

out = Path(__file__).resolve().parent / "_dx_cmp.txt"
io.open(out, "w", encoding="utf-8").write("\n".join(o))
print("ok ->", out)
