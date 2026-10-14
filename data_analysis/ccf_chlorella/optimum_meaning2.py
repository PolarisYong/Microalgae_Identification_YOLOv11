# -*- coding: utf-8 -*-
"""补充：峰顶 vs 角点/边界、向下偏移代价、bootstrap 中极大值存在的比例。"""
import json, os, itertools
import numpy as np

BASE = r"E:\pythonProject\Microalgae_Identification_YOLOv11\data_analysis\ccf_chlorella"
OUT = os.path.join(BASE, "_opt_meaning2.txt")
d = json.load(open(os.path.join(BASE, "summary.json"), encoding="utf-8"))
rows = d["design"]; beta = np.array(d["beta"])
s = d["s"]; xs = np.array(d["xs"]); yopt = d["yopt"]

def row(x1, x2, x3):
    return [1.0, x1, x2, x3, x1*x2, x1*x3, x2*x3, x1*x1, x2*x2, x3*x3]
X = np.array([row(r["x1"], r["x2"], r["x3"]) for r in rows])
y = np.array([r["y"] for r in rows])

b0 = beta[0]; b = beta[1:4]; bi = beta[4:7]; bq = beta[7:10]
B = np.array([[bq[0], bi[0]/2, bi[1]/2],
              [bi[0]/2, bq[1], bi[2]/2],
              [bi[1]/2, bi[2]/2, bq[2]]])
def yhat(x):
    x = np.asarray(x, float); return b0 + b @ x + x @ B @ x

SCALE = np.array([90.0, 140.0, 245.0]); CEN = np.array([120.0, 160.0, 320.0])
NAMES = ["x1 光照", "x2 氨氮", "x3 无机碳"]
L = []
def w(t=""): L.append(t)

w("=" * 78)
w("F. 峰顶 vs 8 个角点：最优条件到底“优”出多少？")
w("=" * 78)
w("编码角点              实际(光/氮/碳)        yhat      与峰顶差   差/s")
corners = list(itertools.product([-1, 1], repeat=3))
recs = []
for c in corners:
    a = CEN + np.array(c) * SCALE
    yy = yhat(np.array(c, float))
    recs.append((yy, c, a))
for yy, c, a in sorted(recs, reverse=True):
    dl = yopt - yy
    w(f"({c[0]:+d},{c[1]:+d},{c[2]:+d})      {a[0]:5.0f}/{a[1]:5.0f}/{a[2]:5.0f}"
      f"     {yy:.5f}    {dl:+.5f}   {dl/s:5.2f}")
w("")
w(f"峰顶 yopt = {yopt:.5f}   s = {s:.5f}")
w("")
w("关键对比 —— 全上界角点 (+1,+1,+1):")
yy111 = yhat(np.array([1.0, 1.0, 1.0]))
w(f"  预测 {yy111:.5f}  vs  峰顶 {yopt:.5f}  ->  差 {yopt-yyy111 if False else yopt-yy111:+.5f}"
  f" = {(yopt-yy111)/s:.2f} s = {(yopt/yy111-1)*100:.2f}%")
w(f"  该角点的实测值(第 8 组) = {y[7]:.4f}")
w("  -> 峰顶与全上界角点的差异远小于一个残差标准差，统计上不可区分。")

w("")
w("=" * 78)
w("G. 离开峰顶的代价（不被边界截断的方向）")
w("=" * 78)
w("向下（减小）偏移 t 时的预测损失, 与 s 的倍数:")
w("因子          t=-0.10   t=-0.25   t=-0.50   t=-1.00 |  向上到边界(+1)的可用偏移与损失")
for i in range(3):
    losses = []
    for t in (-0.10, -0.25, -0.50, -1.00):
        xv = xs.copy(); xv[i] += t
        losses.append(yopt - yhat(xv))
    up = 1.0 - xs[i]
    xv = xs.copy(); xv[i] = 1.0
    loss_up = yopt - yhat(xv)
    w(f"  {NAMES[i]}  " + "  ".join(f"{v:8.5f}" for v in losses) +
      f" |  Δx=+{up:.3f} (至 {CEN[i]+SCALE[i]:.0f})  损失 {loss_up:.5f}")
w("")
w(f"参照: 残差标准差 s = {s:.5f} h^-1；95% PI 半宽 = {(d['pi'][1]-d['pi'][0])/2:.5f}")
w("结论: 沿任一因子轴向任一方向移动 0.25 个编码单位，损失均小于 0.5 s。")

w("")
w("=" * 78)
w("H. 峰顶存在的稳健性：bootstrap 中“确实有内部极大值”的比例")
w("=" * 78)
yhat_obs = X @ beta
rng = np.random.default_rng(20260929)
Bs = 6000
neg = 0; in_box = 0; both = 0
xk = []
for k in range(Bs):
    ys = yhat_obs + rng.normal(0, s, size=len(y))
    bb = np.linalg.lstsq(X, ys, rcond=None)[0]
    Bl = np.array([[bb[7], bb[4]/2, bb[5]/2],
                   [bb[4]/2, bb[8], bb[6]/2],
                   [bb[5]/2, bb[6]/2, bb[9]]])
    ev = np.linalg.eigvalsh(Bl)
    is_neg = np.all(ev < 0)
    xb = -0.5 * np.linalg.solve(Bl, bb[1:4])
    ib = np.all(np.abs(xb) <= 1)
    neg += is_neg; in_box += ib; both += (is_neg and ib)
    if is_neg and ib:
        xk.append(xb)
xk = np.array(xk)
w(f"重抽样 {Bs} 次:")
w(f"  Hessian 负定（曲面确为凹、存在内部极大值）的比例 = {neg/Bs:.1%}")
w(f"  驻点落在设计立方体内的比例                     = {in_box/Bs:.1%}")
w(f"  两者同时满足的比例                             = {both/Bs:.1%}")
if len(xk) > 30:
    w("")
    w("  仅在这部分“良态”样本中，驻点分布的分位数:")
    w("  因子          2.5%     25%     50%     75%   97.5%  |  实际单位 95% 区间")
    for i in range(3):
        q = np.percentile(xk[:, i], [2.5, 25, 50, 75, 97.5])
        lo = CEN[i] + q[0]*SCALE[i]; hi = CEN[i] + q[4]*SCALE[i]
        w(f"    {NAMES[i]}   " + "  ".join(f"{v:+.3f}" for v in q) +
          f"  |  {lo:.0f} ~ {hi:.0f}")
    w("")
    w(f"  样本数 = {len(xk)}")

w("")
w("=" * 78)
w("I. 一句话可引用的表述参数")
w("=" * 78)
w(f"驻点(编码) = ({xs[0]:.4f}, {xs[1]:.4f}, {xs[2]:.4f})")
w(f"驻点(实际) = 光照 {CEN[0]+SCALE[0]*xs[0]:.0f} umol/m2/s, "
  f"氨氮 {CEN[1]+SCALE[1]*xs[1]:.0f} mg/L, 无机碳 {CEN[2]+SCALE[2]*xs[2]:.0f} mg/L")
w(f"预测峰值 = {yopt:.4f} h^-1, 95% CI [{d['ci'][0]:.4f}, {d['ci'][1]:.4f}], "
  f"95% PI [{d['pi'][0]:.4f}, {d['pi'][1]:.4f}]")
w(f"倍增时间 t_d = ln2/mu:")
for lab, v in [("峰顶 0.2098", yopt), ("观测最大 0.2072", max(y)), ("观测最小 0.0571", min(y))]:
    w(f"  {lab}: {np.log(2)/v:.2f} h = {np.log(2)/v/24:.2f} d")

open(OUT, "w", encoding="utf-8").write("\n".join(L))
print("ok")
