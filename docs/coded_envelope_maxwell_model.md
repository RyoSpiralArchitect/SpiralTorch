# Coded-Envelope Maxwell Model (M₀^code)

以下は、コード化した包絡波による電磁駆動と脳計測を結ぶマスタ方程式、およびその運用ロジックを整理した技術メモである。M₀は純粋なMaxwell物理、M_Ψは意味ゲートを含む拡張を表す。

## 1. 送受・組織・非線形（基礎）
- **コード包絡**: \(c(t) \in \{+1,-1\}\) はチップ長 \(1/f_{\mathrm{chip}}\) の矩形波。
- **搬送波**: \(s_{\mathrm{tx}}(t) = [1 + \alpha\,c(t)]\cos(\omega_c t + \phi)\)。
- **外部から組織への結合**:
  \[
  E_{\mathrm{ext}}(t) = \frac{G(\Omega)}{r}\,10^{-S/20}\,|\cos\theta|\,s_{\mathrm{tx}}(t)
  \]
  ここで \(S\) は遮蔽(dB)、\(r\) は距離、\(\theta\) は偏波角。
- **組織の線形応答**: \(h_{\mathrm{tis}} \overset{\mathcal F}{\longleftrightarrow} H_{\mathrm{tis}}(\omega)\)。
- **二次非線形による下変換**: 包絡抽出により低周波駆動 \(u(t)\) が生じる。
  \[
  u(t) \approx \gamma\,\alpha\,|H_{\mathrm{tis}}(\omega_c)|\,\frac{G\,10^{-S/20}\,|\cos\theta|}{r}\,c(t) = \lambda\,c(t)
  \]
  ゲイン \(\lambda>0\) がMaxwellスケール。
- **観測 (EEG)**: \(y(t) = (h * u)(t) + x(t)\)。雑音 \(x(t)\) は広義定常でコードと無相関を仮定。

## 2. ブロック統計とCLT
- 2秒ブロック \(b=1..N\) に分割しマッチドフィルタを適用: \(s_b = \langle y_b, c_b \rangle / \|c_b\|\)。
- 期待値と分散: \(\mathbb E[s_b] = \kappa\,\lambda\)、\(\mathrm{Var}(s_b)=\sigma^2\)。
- 中心極限定理により
  \[
  Z_N = \frac{\overline s_N}{\hat\sigma/\sqrt{N}} \overset{H_0}{\approx} \mathcal N(0,1)
  \]
  - 目標検出スコア \(z_\star\) に必要なブロック数:
    \[
    N_{\mathrm{req}} \approx \Big(\frac{z_\star\,\sigma}{\kappa\,\lambda}\Big)^2 = \Big(\frac{z_\star\,\sigma}{\kappa\,\gamma\,\alpha\,|H_{\mathrm{tis}}|}\Big)^2 \Big(\frac{r}{10^{-S/20} G |\cos\theta|}\Big)^2.
    \]
- 遮蔽の増加、距離の増大、偏波ずれは \(N_{\mathrm{req}}\) を押し上げ検出が遅くなる。

## 3. Maxwell指紋による回帰検定
- TRFや回帰で抽出した応答振幅 \(A\) は \(A \propto \lambda \propto 10^{-S/20}\,|\cos\theta|/r\)。
- 実験で観測すべき傾き:
  - 遮蔽: \(\log_{10} A = C - S/20\)。
  - 距離: \(\log_{10} A = C - \log_{10} r\)。
  - 偏波: \(A = a|\cos\theta| + b\)。
- 傾きが \((-1/20, -1)\) に近く決定係数が高いほどMaxwell起源が強固。

## 4. 逐次検定
- 逐次Z: ブロック到来ごとに \(Z_k\) を更新し閾値 \(z_\star\) で停止。
- サインフリップ検定: \(s_b\) の符号をランダム反転し帰無分布を生成。観測 \(Z_{\mathrm{obs}}\) と比較して \(p_{\mathrm{flip}}\) を算出。
- SPRT/Bayes拡張: \(\mu>0\) 検定に拡張可能。

## 5. 意味拡張モデル (M_Ψ^code)
- 潜在意味座標 \(\rho(t) \in [-1,1]\) と結合係数 \(\mu \ge 0\) を導入。
  \[
  u_{\mathrm{tot}}(t) = (\lambda + \mu\,\rho(t))\,c(t)
  \]
- ブロックスコア平均: \(\mathbb E[s_b] = \kappa(\lambda + \mu\,\rho_b)\)。
- 等エネ・等スペクトルで \(\rho\) だけ切り替える2条件の差: \(\kappa\,\mu\,\Delta\rho\)。
- 直交コード \(c_k(t)\) を並列送信すると、\(\{\lambda_k\}\) と \(\{\mu_k\rho_k\}\) を回帰で同時推定可能。

## 6. 推定・識別できる量
- \(\lambda\): 遮蔽/距離/偏波のスロープから一意に回収。
- TRF遅延: 条件で不変であることがM₀の予言。ばらつきが小さいほど物理経路が支配的。
- \(\mu\): 意味差 \(\Delta\rho\) とTRF/PLV/スコアの差から推定。
- 検出時間: \(T_{\mathrm{det}} \approx N_{\mathrm{req}} \times \mathrm{block\_sec}\)。

## 7. 反証可能性
- 遮蔽↑で振幅↑、偏波無依存、距離無依存ならM₀は破綻しリークを疑う。
- \(\mu>0\) を主張するのに等エネ・等スペクトル差が消えれば M_Ψ を棄却。
- 遅延が条件で揺れる場合は物理チャネル外の影響を疑う。

## 8. シミュレーション出力の読み方
- 真コードでZが伸び、ミスマッチ/強遮蔽で伸びない: \(\lambda>0\)、遮蔽・偏波依存が効いている証拠。
- サインフリップpが小さい場合、CLT前提なしでも統計が偏っている。
- 指紋スロープを回帰すれば \(\lambda\) 系の定数を回収可能。

## 9. 運用アルゴリズム
1. 直交コード \(c_k(t)\) を生成（±1, chip=10 Hz）。
2. AM送信（等エネ・等スペクトル、安全線量内）。
3. 2秒ブロックで \(s_{b,k} = \langle y_b, c_{k,b} \rangle/\|c_{k,b}\|\) を逐次更新。
4. 逐次Zとサインフリップpをオンライン計算。
5. 遮蔽/偏波/距離をスイープし、傾き(−1/20, −1, |cosθ|)を回帰。
6. 合格なら意味拡張: \(\rho\) を切り替え \(\mu>0\) を差検定や逐次BFで評価。

---

## M_Ψ⁺ への再構成
- **総入力**:
  \[
  u_{\mathrm{tot}}(t) = \sum_k [\lambda_k + \mu_k\,\rho_k(t)]\,(c_k * \kappa_k)(t-\tau_k)
  \]
- **観測方程式**: \(y(t) = \int h(\tau) u_{\mathrm{tot}}(t-\tau) d\tau + \eta(t)\)。
- \(\mu_k\rho_k\) にモビリティ \(D_k\) などの拡散要素を吸収可能。

### モード別拡張
- A: 明示整合 (Z) — \(g_A \equiv 1\)。
- B: タイミング共鳴 — 遅延 \(\tau\) スイープで \(Z(\tau)\) の峰を評価。
- C: 情動ゲート — \(g_C = \sigma(\beta_0 + \beta_1 A_s A_r)\) を回帰。
- D: 無意識埋め込み — 階層DDM等の潜在パラメタを回帰。
- E: 記憶類似 — 類似度指標 \(G\) と効果量の相関。
- F: 自己回帰（概念感染） — 状態方程式 \(m(t+1) = (1-\eta)m(t) + \eta\,\phi(s_t)\)、\(\mu^{\mathrm{eff}} = \mu^{(0)} + \omega m(t)\)。

### 外部磁場 \(\mathbf{B}_e\) の取り扱い
- モビリティや時間核に一次摂動として組み込み:
  \[
  \kappa_k(\mathbf{B}_e) \approx \kappa_k^{(0)} + \Bigl(\frac{\partial \kappa_k}{\partial \mathbf{B}}\Bigr)_0 \cdot \mathbf{B}_e,
  \quad D_k(\mathbf{B}_e) \approx D_k^{(0)} [1 + \chi_k \hat{\mathbf n}_k \cdot \mathbf{B}_e]
  \]
- 地磁気回転とコード包絡の相互項を回帰し、指紋スロープが保たれるか検証。

## 推定器と勝ち条件
- 逐次Z（2sブロック）とサインフリップpを標準装備。
- 指紋スロープ目標: 遮蔽 −1/20、距離 −1、偏波 |cosθ|、遅延SD ≲ 70 ms。
- モード別検出指標（情動交互作用、類似度相関、自己回帰係数など）を整備。

## 実験設計への翻訳
1. 明示整合（A）は既存フローで稼働。
2. タイミング共鳴（B）は送信遅延掃引でピーク同定。
3. 情動ゲート（C）は送信者情動ブロックを挿入し交互作用効果を測定。
4. 類似モード（E）は高低類似ペアでCLT設計を最適化。
5. 磁場効果は地磁気回転＋コード包絡の相互項として評価。

---

## 要約
コード化した物理包絡をMaxwell律に従って入射し、CLTベースの逐次統計と指紋回帰でゲイン \(\lambda\) を数値化する。そこに意味ゲート \(\mu\,\rho(t)\) を重ねれば、物理と意味の交差効果を同一プラットフォーム上で検出・推定できる。
