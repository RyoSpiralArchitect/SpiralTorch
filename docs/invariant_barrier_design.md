# Invariant barrier gating and contraction notes

> Japanese notes from the SpiralTorch dynamics journal, organised so they can be slotted into the runtime design docs.

## A. バリア設計：\( \mu_{\mathrm{eff}} \le 0 \) を不変集合化

### A-0. 前提

A 系（収束型）の骨格：
\[
\begin{aligned}
\dot z &= (\mu_0+\gamma\,\hat c(u,s))\,z - \nu \lvert z \rvert^2 z + i\,\omega z, \\
\dot u &= \kappa(\alpha\,\Re z - \beta\,\Im z - \theta) - \tau u, \\
\dot s &= -\lambda s + \rho\, \Im z.
\end{aligned}
\]
外部強制はロジスティック飽和：
\[
\hat c(u,s)=\frac{c_{\max}}{1+\exp\!\bigl(-[u-\sigma_s s]\bigr)}.
\]
有効成長率 \( \mu_{\mathrm{eff}} = \mu_0 + \gamma\,\hat c \)。

### A-1. ハード・バリア（十分条件）

設計：
\[
\boxed{\ \mu_0+\gamma c_{\max} \le 0\ }
\]
を満たすように \( c_{\max} \)（または \( \gamma \) の符号）を設定。このとき \( \hat c \le c_{\max} \) より常に \( \mu_{\mathrm{eff}}\le 0 \)。
不変集合は状態空間全体になり、Lyapunov \( V=\tfrac12 \lvert z \rvert^2 \) に対して
\[
\dot V=(\mu_0+\gamma\hat c)\lvert z \rvert^2-\nu\lvert z \rvert^4 \le -\nu\lvert z \rvert^4\le0,
\]
LaSalle より \( z \to 0 \)（大域安定）。最も堅牢な方法。

### A-2. ソフト・バリア（容器ゲーティング付の上限制御）

上限を状態依存に落とす：
\[
\hat c(u,s)=\frac{c_{\text{base}}}{(1+\kappa_b s)\,(1+e^{-[u-\sigma_s s]})}.
\]
このとき
\[
\sup_{u}\hat c(u,s)=\frac{c_{\text{base}}}{1+\kappa_b s} \le \frac{c_{\text{base}}}{1+\kappa_b \cdot 0}=c_{\text{base}}.
\]
設計：
\[
\boxed{\ c_{\text{base}} \le c_\star := -\frac{\mu_0+m}{\gamma} \quad (m>0)\ }
\]
を選べば \( \forall s \ge 0 \) で \( \hat c \le c_\star \) → \( \mu_{\mathrm{eff}} \le -m < 0 \)。
「容器」 \( s \) が増えるほど上限がさらに下がり、安全側に引き込む。

### A-3. バリア微分不等式（CBF 的な検証）

安全関数 \( h(u,s):=-(\mu_0+\gamma\hat c(u,s)) \)。安全集合 \( \mathcal{S} = \{h\ge0\} = \{\mu_{\mathrm{eff}} \le 0\} \)。
ロジスティックの導関数：
\[
\partial_u \hat c = \hat c\Bigl(1-\frac{\hat c}{c_{\max}}\Bigr), \qquad
\partial_s \hat c = -\sigma_s\,\partial_u \hat c.
\]
境界 \( h=0 \) で
\[
\dot h = -\gamma\,\partial_u\hat c\, (\dot u - \sigma_s \dot s).
\]
十分条件として
\[
\boxed{\ \dot u - \sigma_s\dot s \le 0 \quad (\text{on } h=0)\ }
\]
が成り立てば \( \dot h \ge 0 \)（前方不変）。代入すると
\[
\kappa(\alpha\,\Re z-\beta\,\Im z-\theta) - \tau u + \sigma_s\lambda s - \sigma_s\rho\,\Im z \le 0.
\]
設計指針：\( \tau \) を十分大、\( \lambda \le \tau \)、\( \sigma_s\rho \) 小、\( \kappa(|\alpha|,|\beta|) \) 小、等で境界上の最悪値を負に押さえる。
実務上は A-1 / A-2 を採り、A-3 は検証用として併用するのが現実的。

## B. 安定振幅 \( r^{\circ} \) の閉形式（呼吸型 B）

B 系の半径力学（\( r = \lvert z \rvert \)）：
\[
\dot r = \Bigl[\underbrace{\mu_0-\eta}_{A} + \gamma\,\frac{c_1}{1+q r^2} - \nu r^2 - \gamma\sigma_s s\Bigr] r.
\]
非自明平衡 \( r^{\circ}>0 \) は角括弧の中が 0 で与えられる。\( y := r^2 \ge 0 \) と置くと
\[
g(y)=A+\frac{\gamma c_1}{1+q y}-\nu y-\gamma\sigma_s s=0.
\]
両辺に \( 1+qy \) を掛けて二次式に落とす：
\[
\gamma c_1=(1+qy)(\nu y-A+\gamma\sigma_s s)
= q\nu y^2+\bigl(\nu-qA+q\gamma\sigma_s s\bigr)y+\bigl(-A+\gamma\sigma_s s\bigr).
\]
よって
\[
\boxed{\ a_2 y^2+a_1 y+a_0=0,\ }
\]
\[
a_2 = q\nu>0, \quad
a_1 = \nu-qA+q\gamma\sigma_s s, \quad
\!a_0 = (-A+\gamma\sigma_s s) - \gamma c_1.
\]
根（物理的には \( y^{\circ}=r^{\circ 2}\ge0 \) の方）：
\[
\boxed{\ r^{\circ 2}=y^{\circ}=\frac{-a_1+\sqrt{a_1^2-4a_2 a_0}}{2a_2} \ (\ge0).\ }
\]

### 一意性と安定性

\( g'(y) = -\frac{\gamma c_1 q}{(1+qy)^2}-\nu<0 \) なので \( g \) は単調減少。\( \lim_{y\to\infty}g(y) = -\infty \)。さらに
\[
g(0) = A+\gamma c_1-\gamma\sigma_s s.
\]
条件
\[
\boxed{\ g(0)>0 \iff \mu_0-\eta+\gamma c_1>\gamma\sigma_s s\ }
\]
なら唯一の正根 \( y^{\circ} \) が存在（従って \( r^{\circ}>0 \) 一意）。単調減少関数の零点なので、この平衡は安定（\( g'(y^{\circ})<0 \)）。

### 比較静学（容器 \( s \) が揺らぎを抑える）

暗黙関数定理で
\[
\frac{\mathrm{d} y^{\circ}}{\mathrm{d}s}=-\frac{\partial g/\partial s}{\partial g/\partial y}
=\frac{\gamma\sigma_s}{-\frac{\gamma c_1 q}{(1+qy^{\circ})^2}-\nu}<0.
\]
したがって \( s \) が増えると \( r^{\circ 2} \) は厳密に減少（容器は振幅を絞る）。

## C. 収縮率 \( \varepsilon \) の下界（Gershgorin / ノルム境界）

C 系（収縮型）の線形部：
\[
J=\begin{bmatrix}
-a & -\omega & \ \gamma & 0 \\
\ \omega & -a & 0 & \sigma_s \\
-\kappa\alpha & \kappa\beta & -\tau & -\sigma_s \\
0 & -\rho & 0 & -\lambda
\end{bmatrix},
\]
非線形減衰（\( (\psi,\phi) \) ブロックに \(-\nu(\psi^2+\phi^2)\) 由来の三次減衰、等）を付加。
2-ノルムの行列測度は \( \mu_2(A)=\lambda_{\max}((A+A^\top)/2) \)。
よって収縮条件は
\[
\lambda_{\max}\!\Bigl(\frac{\partial f}{\partial x}+\frac{\partial f}{\partial x}^\top\Bigr)/2 \le -\varepsilon < 0.
\]

### 対称部 \( S=(J+J^\top)/2 \)

\[
S=\begin{bmatrix}
-a & 0 & \tfrac{\gamma-\kappa\alpha}{2} & 0 \\
0 & -a & \tfrac{\kappa\beta}{2} & \tfrac{\sigma_s-\rho}{2} \\
\tfrac{\gamma-\kappa\alpha}{2} & \tfrac{\kappa\beta}{2} & -\tau & -\tfrac{\sigma_s}{2} \\
0 & \tfrac{\sigma_s-\rho}{2} & -\tfrac{\sigma_s}{2} & -\lambda
\end{bmatrix}.
\]

### Gershgorin による上界と \( \varepsilon \) の下界

各行の「中心＋半径」の最大値が \( \lambda_{\max}(S) \) の上界。行 \( i \) の中心 \( c_i \)、半径 \( R_i \)：
\[
\begin{aligned}
&c_1=-a, && R_1=\tfrac12\lvert\gamma-\kappa\alpha\rvert,\\
&c_2=-a, && R_2=\tfrac12\bigl(\lvert\kappa\beta\rvert+\lvert\sigma_s-\rho\rvert\bigr),\\
&c_3=-\tau, && R_3=\tfrac12\bigl(\lvert\gamma-\kappa\alpha\rvert+\lvert\kappa\beta\rvert+\lvert\sigma_s\rvert\bigr),\\
&c_4=-\lambda, && R_4=\tfrac12\bigl(\lvert\sigma_s-\rho\rvert+\lvert\sigma_s\rvert\bigr).
\end{aligned}
\]
したがって
\[
\lambda_{\max}(S) \le \max\{c_1+R_1, c_2+R_2, c_3+R_3, c_4+R_4\}.
\]
この最大値を \( U \) と書くと、収縮率の下界は
\[
\boxed{\ \varepsilon \ge -U = \min\!\left\{\begin{aligned}
&a-\tfrac12\lvert\gamma-\kappa\alpha\rvert,\\
&a-\tfrac12\bigl(\lvert\kappa\beta\rvert+\lvert\sigma_s-\rho\rvert\bigr),\\
&\tau-\tfrac12\bigl(\lvert\gamma-\kappa\alpha\rvert+\lvert\kappa\beta\rvert+\lvert\sigma_s\rvert\bigr),\\
&\lambda-\tfrac12\bigl(\lvert\sigma_s-\rho\rvert+\lvert\sigma_s\rvert\bigr)
\end{aligned}\right\}.\ }
\]
右辺が正なら \( S \preceq -\varepsilon I \)（線形部だけで収縮）。

### 非線形減衰が与える追加の負性

\( (\psi,\phi) \) ブロックの三次減衰 \(-\nu[\psi(\psi^2+\phi^2),\ \phi(\psi^2+\phi^2)]\) のヤコビアンは
\[
-\nu
\begin{bmatrix}
3\psi^2+\phi^2 & 2\psi\phi \\
2\psi\phi & \psi^2+3\phi^2
\end{bmatrix}
\]
で、固有値は \(-\nu\{3r^2, r^2\}\)（\( r^2 = \psi^2+\phi^2 \)）。
よって対称部の最大固有値はさらに \( \le -\nu r^2 \) だけ下がる（Weyl の不等式）。
つまり実際の収縮率は
\[
\varepsilon_{\text{actual}} \ge \varepsilon_{\text{lin}} + \underline{\nu r^2\ \text{（状態依存の利得）}},
\]
で、特に小振幅域でも \( \varepsilon_{\text{lin}} \) が正なら大域収縮。

---

## まとめ（設計チェックリスト）

- **A（バリア）**
  - まずはハード：\( \mu_0+\gamma c_{\max}\le 0 \)。
  - 余裕を持たせるなら \( \mu_0+\gamma c_{\text{base}}\le -m \) かつ \( \kappa_b>0 \)（容器でさらに安全側）。
  - CBF 条件 \( \dot u-\sigma_s\dot s\le0 \) を境界で点検（\( \tau \) 大、\( \lambda\le\tau \)、\( \kappa, \sigma_s\rho \) 小）。
- **B（安定振幅）**
  - \( r^{\circ 2}=\dfrac{-a_1+\sqrt{a_1^2-4a_2 a_0}}{2a_2} \)（係数は上記）。
  - 存在条件：\( \mu_0-\eta+\gamma c_1>\gamma\sigma_s s \)。
  - 単調性：\( \dfrac{\mathrm{d} r^{\circ 2}}{\mathrm{d}s}<0 \)（容器で振幅が縮む）。
- **C（収縮率）**
  - \( \varepsilon \ge \min\{ a-\tfrac12\lvert\gamma-\kappa\alpha\rvert, a-\tfrac12(\lvert\kappa\beta\rvert+\lvert\sigma_s-\rho\rvert), \tau-\tfrac12(\lvert\gamma-\kappa\alpha\rvert+\lvert\kappa\beta\rvert+\lvert\sigma_s\rvert), \lambda-\tfrac12(\lvert\sigma_s-\rho\rvert+\lvert\sigma_s\rvert) \} \).
  - これを正に保つようパラメータを選ぶ。非線形減衰はさらに有利。

