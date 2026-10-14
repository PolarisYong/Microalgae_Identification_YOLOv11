# -*- coding: utf-8 -*-
"""在驻点 (182, 287, 461) 处做确认实验，μmax 的合理范围。"""
import json, os
import numpy as np
from scipy import stats

BASE = r"E:\pythonProject\Microalgae_Identification_YOLOv11\data_analysis\ccf_chlorella"
OUT = os.path.join(BASE, "_pred_pi.txt")
d = json.load(open(os.path.join(BASE, "summary.json"), encoding="utf-8"))
rows = d["design"]; beta = np.array(d["beta"])
s = d["s"]; xs = np.array(d["xs"]); yopt = d["yopt"]

def rw(x1, x2, x3):
    return [1.0, x1, x2, x3, x1*x2, x1*x3, x2*x3, x1*x1, x2*x2, x3*x3]
X = np.array([rw(r["x1"], r["x2"], r["x3"]) for r in rows])
y = np.array([r["y"] for r in rows])
n, p = X.shape
dfres = n - p
XtXi = np.linalg.inv(X.T @ X)
tcrit = stats.t.ppf(0.975, dfres)

b0 = beta[0]; b = beta[1:4]; bi = beta[4:7]; bq = beta[7:10]
Bm = np.array([[bq[0], bi[0]/2, bi[1]/2],
               [bi[0]/2, bq[1], bi[2]/2],
               [bi[1]/2, bi[2]/2, bq[2]]])
yhat = b0 + b @ xs + xs @ Bm @ xs
x0 = np.array(rw(*xs))
v_lev = float(x0 @ XtXi @ x0)
se_mean = s * np.sqrt(v_lev)
se_pred = np.sqrt(s**2 + se_mean**2)

L = []
def w(t=""): L.append(t)

w("=" * 78)
w("确认实验点: 光照 182 umol/m2/s, 氨氮 287 mg/L, 无机碳 461 mg/L")
w("=" * 78)
w(f"编码坐标 xs = ({xs[0]:.4f}, {xs[1]:.4f}, {xs[2]:.4f})")
w(f"模型预测 yhat = {yhat:.5f} h^-1")
w(f"残差标准差 s = {s:.5f}   残差自由度 df = {dfres}   t(0.975,{dfres}) = {tcrit:.4f}")
w(f"杠杆值 h = x0'(X'X)^-1 x0 = {v_lev:.5f}   (20 点平均杠杆 = {p/n:.3f})")
w(f"均值预测标准误 se_mean = s*sqrt(h) = {se_mean:.5f}")
w(f"单次观测预测标准误 se_pred = sqrt(s^2 + se_mean^2) = {se_pred:.5f}")
w(f"  其中 model 部分占 {se_mean/se_pred:.1%}, 随机误差部分占 {s/se_pred:.1%}")

w("")
w("-" * 78)
w("1) 单次实验（1 个反应器）的合理范围 = 95% 预测区间 PI")
w("-" * 78)
lo, hi = yhat - tcrit*se_pred, yhat + tcrit*se_pred
w(f"  PI = {yhat:.5f} +/- {tcrit:.4f} x {se_pred:.5f} = [{lo:.5f}, {hi:.5f}]")
w(f"  半宽 {tcrit*se_pred:.5f} h^-1 (相对预测值 {tcrit*se_pred/yhat:.1%})")
w(f"  换算倍增时间 td = ln2/mu : {np.log(2)/hi:.2f} ~ {np.log(2)/lo:.2f} h")
w(f"  68% 区间 (±1 se_pred): [{yhat-se_pred:.5f}, {yhat+se_pred:.5f}]")
w(f"  99% 区间: [{yhat-stats.t.ppf(0.995,dfres)*se_pred:.5f}, "
  f"{yhat+stats.t.ppf(0.995,dfres)*se_pred:.5f}]")

w("")
w("-" * 78)
w("2) 重复 n 次取均值的合理范围")
w("-" * 78)
w("  n    均值标准误      95% 区间              区间宽")
for nn in (1, 2, 3, 4, 6, 8):
    se_n = np.sqrt(s**2/nn + se_mean**2)
    a, bb = yhat - tcrit*se_n, yhat + tcrit*se_n
    w(f"  {nn:2d}    {se_n:.5f}      [{a:.5f}, {bb:.5f}]     {bb-a:.5f}")
w("")
w("  注: 重复只压缩随机误差项 s/sqrt(n)，模型自身的不确定 se_mean 不会因重复而减小。")

w("")
w("-" * 78)
w("3) 参照系: 已有实测值能提供什么")
w("-" * 78)
w(f"  20 组观测最大值 = {y.max():.4f} (第 8 组, +1,+1,+1 = 210/300/565)")
w(f"  20 组观测最小值 = {y.min():.4f} (第 5 组, +1,-1,-1 = 210/20/75)")
cen = y[[i for i, r in enumerate(rows) if r["type"] == "中心点"]]
w(f"  6 个中心点重复: 均值 {cen.mean():.5f}, SD {cen.std(ddof=1):.5f}, "
  f"CV {cen.std(ddof=1)/cen.mean():.1%}")
w(f"  -> 本套数据体现的重复精度 SD = {cen.std(ddof=1):.5f} h^-1")
w("")
w("  距确认点最近的 3 个已做实验点:")
dist = []
for r, yy in zip(rows, y):
    dd = np.linalg.norm(np.array([r["x1"], r["x2"], r["x3"]]) - xs)
    dist.append((dd, yy, r))
for dd, yy, r in sorted(dist, key=lambda z: z[0])[:3]:
    w(f"    ({r['x1']:+.0f},{r['x2']:+.0f},{r['x3']:+.0f}) "
      f"{r['light']:.0f}/{r['nh4']:.0f}/{r['ic']:.0f}  实测 {yy:.4f}  编码距离 {dd:.3f}")
w(f"  确认点本身没有实测值 —— 是模型在立方体内部的插值。")

w("")
w("-" * 78)
w("4) 若以原实验汇总表的真实重复变异为准（关键限定）")
w("-" * 78)
w("  原表同一条件的重复间 CV 可达 17.7%（本次构造用的 6 个中心点 CV 仅 4.5%）")
for cv in (0.045, 0.10, 0.177):
    sd = yhat * cv
    w(f"    CV={cv:.1%} -> SD={sd:.4f} -> 单次观测 95% 范围 "
      f"[{yhat-1.96*sd:.4f}, {yhat+1.96*sd:.4f}]  (宽 {(yhat+1.96*sd)-(yhat-1.96*sd):.4f})")
w("  即: 实验操作稳定性才是决定'合理范围'宽窄的主因，模型不确定度只占小部分。")

w("")
w("-" * 78)
w("5) 判读规则")
w("-" * 78)
w(f"  实测落在 [{lo:.4f}, {hi:.4f}] 内 -> 与模型一致，正常")
w(f"  实测略超界(如 {hi:.4f}~{hi+0.01:.4f}) -> 约 5% 概率纯属随机，可复测确认")
w(f"  3 次以上重复均值仍系统性偏离 > {tcrit*np.sqrt(s**2/3+se_mean**2):.4f} -> 模型在该点失拟")
w(f"  实测 < {y.max():.4f}(观测最大值) 不一定说明模型错了 —— 预测峰值本就允许高于任一单次观测")

open(OUT, "w", encoding="utf-8").write("\n".join(L))
print("ok")
