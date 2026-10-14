# -*- coding: utf-8 -*-
"""解析响应曲面"最优点"的多重含义：梯度、条件依赖、平台区、统计不确定性。"""
import json, os
import numpy as np

BASE = r"E:\pythonProject\Microalgae_Identification_YOLOv11\data_analysis\ccf_chlorella"
OUT = os.path.join(BASE, "_opt_meaning.txt")

d = json.load(open(os.path.join(BASE, "summary.json"), encoding="utf-8"))
rows = d["design"]
beta = np.array(d["beta"])
s = d["s"]
xs = np.array(d["xs"])
yopt = d["yopt"]
ci, pi = d["ci"], d["pi"]

def design_row(x1, x2, x3):
    return [1.0, x1, x2, x3, x1*x2, x1*x3, x2*x3, x1*x1, x2*x2, x3*x3]

X = np.array([design_row(r["x1"], r["x2"], r["x3"]) for r in rows])
y = np.array([r["y"] for r in rows])

b0 = beta[0]; b = beta[1:4]; bi = beta[4:7]; bq = beta[7:10]
# 二次型矩阵: yhat = b0 + b.x + x' B x
B = np.array([[bq[0], bi[0]/2, bi[1]/2],
              [bi[0]/2, bq[1], bi[2]/2],
              [bi[1]/2, bi[2]/2, bq[2]]])
H = 2*B

def yhat(x):
    x = np.asarray(x, float)
    return b0 + b @ x + x @ B @ x

def grad(x):
    x = np.asarray(x, float)
    return b + 2*B @ x

L = []
def w(s=""):
    L.append(s)

w("=" * 78)
w("A. 驻点身份核验")
w("=" * 78)
w("模型（编码单位）:")
w(f"  b0={b0:.6f}  b=[{b[0]:.6f},{b[1]:.6f},{b[2]:.6f}]")
w(f"  b_int=[{bi[0]:.6f},{bi[1]:.6f},{bi[2]:.6f}]  b_quad=[{bq[0]:.6f},{bq[1]:.6f},{bq[2]:.6f}]")
w("")
w(f"驻点 xs = ({xs[0]:.6f}, {xs[1]:.6f}, {xs[2]:.6f})")
w(f"驻点处梯度 = ({grad(xs)[0]:.2e}, {grad(xs)[1]:.2e}, {grad(xs)[2]:.2e})   <- 应为 0")
w(f"驻点处预测 yhat = {yhat(xs):.6f}   (summary yopt={yopt:.6f})")
ev, evec = np.linalg.eigh(H)
w(f"Hessian 特征值 = [{ev[0]:.6f}, {ev[1]:.6f}, {ev[2]:.6f}]   全部为负 -> 极大值")
w("Hessian 特征向量（列 = 方向，按特征值从小到大）:")
for j in range(3):
    v = evec[:, j]
    w(f"  lam={ev[j]:+.5f}  v=({v[0]:+.4f}, {v[1]:+.4f}, {v[2]:+.4f})"
      f"  -> 半轴 sqrt(2d/|lam|)")
w("")
w("实际单位换算: light=120+90*x1   nh4=160+140*x2   ic=320+245*x3")
act = [120+90*xs[0], 160+140*xs[1], 320+245*xs[2]]
w(f"  驻点实际值: 光照 {act[0]:.1f} umol/m2/s, 氨氮 {act[1]:.1f} mg/L, 无机碳 {act[2]:.1f} mg/L")
w(f"  在设计区间内的相对位置: 光照 {(act[0]-30)/180:.1%}, 氨氮 {(act[1]-20)/280:.1%}, 无机碳 {(act[2]-75)/490:.1%}")
w("")
w(f"预测峰值 95% CI（均值）= [{ci[0]:.6f}, {ci[1]:.6f}]   半宽 {(ci[1]-ci[0])/2:.6f}")
w(f"预测峰值 95% PI（单次观测）= [{pi[0]:.6f}, {pi[1]:.6f}]   半宽 {(pi[1]-pi[0])/2:.6f}")
w(f"残差标准差 s = {s:.6f}    观测最大值 = {max(y):.4f} (第 8 组, +1,+1,+1)")
w(f"预测峰值比观测最大值高 {yopt-max(y):+.6f} ({(yopt/max(y)-1)*100:+.2f}%)")

w("")
w("=" * 78)
w("B. 最优点不是“单因子最优的叠加”：条件最优点随其他因子移动")
w("=" * 78)
w("条件最优: d(yhat)/dxi = 0  ->  xi* = -(bi + bij*xj + bik*xk) / (2*bii)")
names = ["x1 光照", "x2 氨氮", "x3 无机碳"]
lin = b; quad = bq
INT = {(0,1): bi[0], (0,2): bi[1], (1,2): bi[2]}
def cond_opt(i, xj, xk):
    j, k = [t for t in range(3) if t != i]
    bij = INT[tuple(sorted((i,j)))]; bik = INT[tuple(sorted((i,k)))]
    return -(lin[i] + bij*xj + bik*xk) / (2*quad[i])
w("")
w("背景 (其余两因子水平)      x1* 光照   x2* 氨氮   x3* 无机碳")
for lab, v in [("(-1,-1) 都低", -1), ("(0,0) 都居中", 0), ("(+1,+1) 都高", 1)]:
    vals = [cond_opt(i, v, v) for i in range(3)]
    w(f"{lab:22s}   {vals[0]:+8.3f}   {vals[1]:+8.3f}   {vals[2]:+8.3f}")
w("")
w("逐个因子的条件最优（其余两因子固定为 -1 / 0 / +1）:")
for i in range(3):
    j, k = [t for t in range(3) if t != i]
    r = []
    for v in (-1, 0, 1):
        r.append(cond_opt(i, v, v))
    span = r[2] - r[0]
    w(f"  {names[i]}: 背景-1 -> {r[0]:+.3f} | 背景0 -> {r[1]:+.3f} | 背景+1 -> {r[2]:+.3f}"
      f"   移动幅度 {span:+.3f} 编码单位")
w("")
w("对比：忽略交互项时的“独立最优” xi** = -bi/(2*bii):")
ind = np.array([-lin[i]/(2*quad[i]) for i in range(3)])
w(f"  ({ind[0]:+.3f}, {ind[1]:+.3f}, {ind[2]:+.3f})   该处 yhat = {yhat(ind):.6f}")
w(f"  真实驻点 ({xs[0]:+.3f}, {xs[1]:+.3f}, {xs[2]:+.3f})  yhat = {yhat(xs):.6f}")
w(f"  差异:   Δ=({xs[0]-ind[0]:+.3f}, {xs[1]-ind[1]:+.3f}, {xs[2]-ind[2]:+.3f})")
for i in range(3):
    w(f"    {names[i]}: 交互把它从 {ind[i]:+.3f} 推到 {xs[i]:+.3f}"
      f"  (实际单位 {(120,160,320)[i] + (90,140,245)[i]*ind[i]:.0f} -> {(120,160,320)[i] + (90,140,245)[i]*xs[i]:.0f})")

w("")
w("=" * 78)
w("C. 最优点是一片“平台”，不是一个点")
w("=" * 78)
w("在驻点处 yhat(xs) - yhat(xs+t*d) = sum_i (|lam_i|/2) * u_i^2  (u = 沿特征向量的坐标)")
for delta in (0.002, 0.005, s):
    half = np.sqrt(2*delta/np.abs(ev))
    w(f"  容差 delta={delta:.5f} h^-1 : 等值椭球半轴 = ({half[0]:.3f}, {half[1]:.3f}, {half[2]:.3f})")
w("")
w("单因子容差（其余两因子固定在驻点，沿编码轴移动多远仍使 yhat 下降 <= delta）:")
for delta in (0.002, 0.005):
    w(f"  delta={delta:.4f}:")
    for i in range(3):
        t = np.sqrt(delta/abs(bq[i]))
        scale = (90, 140, 245)[i]
        lo = (120,160,320)[i] + (xs[i]-t)*scale
        hi = (120,160,320)[i] + (xs[i]+t)*scale
        w(f"    {names[i]}: |Δx|<={t:.3f}  -> 实际 {lo:.0f} ~ {hi:.0f}"
          f"  (区间宽 {hi-lo:.0f}, 占设计范围 {(hi-lo)/(2*scale):.0%})")
w("")
rng = np.random.default_rng(20260929)
N = 400000
P = rng.uniform(-1, 1, size=(N, 3))
yp = b0 + P @ b + np.einsum('ij,jk,ik->i', P, B, P)
for delta in (0.002, 0.005, s):
    frac = np.mean(yp >= yopt - delta)
    w(f"  蒙特卡洛: 设计立方体内 yhat >= yopt-{delta:.5f} 的体积占比 = {frac:.1%}")
w(f"  (立方体内 yhat 均值 {yp.mean():.5f}, 最大 {yp.max():.5f}, 最小 {yp.min():.5f})")

w("")
w("=" * 78)
w("D. 驻点本身是估计值：参数 bootstrap 不确定性 (B=4000)")
w("=" * 78)
yhat_obs = X @ beta
resid = y - yhat_obs
Bs = 4000
boot = np.zeros((Bs, 3))
inside = 0
rng2 = np.random.default_rng(7)
for k in range(Bs):
    ystar = yhat_obs + rng2.normal(0, s, size=len(y))
    bb = np.linalg.lstsq(X, ystar, rcond=None)[0]
    Bl = np.array([[bb[7], bb[4]/2, bb[5]/2],
                   [bb[4]/2, bb[8], bb[6]/2],
                   [bb[5]/2, bb[6]/2, bb[9]]])
    bl = bb[1:4]
    try:
        xb = -0.5 * np.linalg.solve(Bl, bl)
    except np.linalg.LinAlgError:
        xb = np.array([np.nan]*3)
    boot[k] = xb
    if np.all(np.abs(xb) <= 1):
        inside += 1
ok = ~np.isnan(boot[:, 0])
boot = boot[ok]
w(f"有效重抽样 {len(boot)} / {Bs}")
w(f"bootstrap 驻点落在设计立方体内的比例 = {inside/len(boot):.1%}")
qs = [2.5, 25, 50, 75, 97.5]
w("")
w("因子        2.5%      25%      50%      75%    97.5%   |  实际单位 95% 区间")
for i in range(3):
    q = np.percentile(boot[:, i], qs)
    scale = (90, 140, 245)[i]; cen = (120, 160, 320)[i]
    lo = cen + q[0]*scale; hi = cen + q[4]*scale
    w(f"  {names[i]}  " + "  ".join(f"{v:+.3f}" for v in q) +
      f"   |  {lo:.0f} ~ {hi:.0f}")
# 预测峰值的不确定性
yb = np.array([b0 + boot[k] @ b + boot[k] @ B @ boot[k] for k in range(len(boot))])
w("")
w(f"bootstrap 峰顶高度: 中位数 {np.median(yb):.5f}, 95% 区间 "
  f"[{np.percentile(yb,2.5):.5f}, {np.percentile(yb,97.5):.5f}]")
w(f"  与解析解 yopt={yopt:.5f}、CI=[{ci[0]:.5f},{ci[1]:.5f}] 对照")

w("")
w("=" * 78)
w("E. 从峰顶出发的单因子敏感性（离开最优点的代价）")
w("=" * 78)
w("编码偏移 t=0.25 / 0.50 / 1.00 时的预测损失 (h^-1):")
w("因子          t=0.25    t=0.50    t=1.00   |  t=1.00 时到达的实际条件")
for i in range(3):
    losses = []
    for t in (0.25, 0.5, 1.0):
        xv = xs.copy(); xv[i] += t
        xv[i] = min(xv[i], 1.0)
        losses.append(yopt - yhat(xv))
    xv = xs.copy(); xv[i] = 1.0
    acti = (120,160,320)[i] + (90,140,245)[i]*1.0
    w(f"  {names[i]}   " + "  ".join(f"{v:8.5f}" for v in losses) + f"   |  {acti:.0f}")
w("")
w("注: 向上偏移受设计边界 (+1) 截断，故 t=1.00 实际为到达边界的偏移。")

open(OUT, "w", encoding="utf-8").write("\n".join(L))
print("written", OUT)
