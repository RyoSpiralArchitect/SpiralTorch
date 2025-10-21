# SpinoTensorVector Formalism in Z-Space

## 1. Dirac/STV in General Symmetric Spatial Tensors

Consider the block matrices
\[
S=\begin{pmatrix}s_0 & 0\\ 0 & A\end{pmatrix},\quad W=\begin{pmatrix}0 & \mathbf{E}^\top\\ \mathbf{E} & \Omega(\boldsymbol{\omega})\end{pmatrix},\quad T=S+W,
\]
where $A=A^\top\in\mathbb{R}^{3\times3}$ and $\Omega(\boldsymbol{\omega})x=\boldsymbol{\omega}\times x$.

### 1.1 Determinant Identities

The determinant decomposes as
\[
\det(T)=s_0\det(D)-\mathbf{E}^\top \operatorname{adj}(D)\mathbf{E},\quad D=A+\Omega(\boldsymbol{\omega}).
\]
For any symmetric $A$ we have
\[
\det(A+\Omega)=\det A+\boldsymbol{\omega}^\top A\boldsymbol{\omega},\quad
\mathbf{E}^\top\operatorname{adj}(A+\Omega)\mathbf{E}=\mathbf{E}^\top\operatorname{adj}(A)\mathbf{E}+(\mathbf{E}\cdot\boldsymbol{\omega})^2,
\]
leading to
\[
\det(T)=s_0\Big(\det A+\boldsymbol{\omega}^\top A\boldsymbol{\omega}\Big)-\Big(\mathbf{E}^\top\operatorname{adj}(A)\mathbf{E}+(\mathbf{E}\cdot\boldsymbol{\omega})^2\Big).
\]

### 1.2 Kernel Conditions

The kernel exists iff either $\det(D)=0$ and $\mathbf{E}^\top\operatorname{adj}(D)\mathbf{E}=0$, or
\[
 s_0=\alpha:=\mathbf{E}^\top D^{-1}\mathbf{E}=\frac{\mathbf{E}^\top\operatorname{adj}(D)\mathbf{E}}{\det D}
 =\frac{\mathbf{E}^\top\operatorname{adj}(A)\mathbf{E}+(\mathbf{E}\cdot\boldsymbol{\omega})^2}{\det A+\boldsymbol{\omega}^\top A\boldsymbol{\omega}}
\]
when $\det(D)\neq 0$.

For $\det(D)\neq 0$, a kernel vector can be written as
\[
 j=\begin{pmatrix}1\\ -D^{-1}\mathbf{E}\end{pmatrix} j^0.
\]
Its Minkowski norm with $g=\operatorname{diag}(+,-,-,-)$ is proportional to
\[
\beta:=\mathbf{E}^\top(DD^\top)^{-1}\mathbf{E},\quad j^\top g j\propto 1-\beta.
\]
Accordingly the kernel is timelike, lightlike, or spacelike when $\beta<1$, $\beta=1$, or $\beta>1$.

## 2. Typology Map: Kernel Hyperplane and Lightlike Ellipsoid

Define
\[
\mathcal{Q}_\alpha:\ \mathbf{E}^\top\operatorname{adj}(D)\mathbf{E}=s_0\det D,
\qquad
\mathcal{E}_\beta:\ \mathbf{E}^\top(DD^\top)^{-1}\mathbf{E}=1.
\]
The causal class of the kernel is determined by the position of $\mathbf{E}$ relative to $\mathcal{E}_\beta$ while constrained to $\mathcal{Q}_\alpha$.

## 3. Intersection Parameterisation

To describe $\mathcal{Q}_\alpha\cap\mathcal{E}_\beta$ parametrically, whiten the ellipsoid using the eigendecomposition $(DD^\top)^{-1}=R^\top\Gamma R$. Setting $\mathbf{y}=\Gamma^{1/2}R\mathbf{E}$ gives $\|\mathbf{y}\|=1$. Introducing the symmetric matrix $B=\tfrac{1}{2}(\operatorname{adj}D+\operatorname{adj}D^\top)$ and $C=\Gamma^{-1/2}RBR^\top\Gamma^{-1/2}$, diagonalise $C=Q^\top\operatorname{diag}(c_1,c_2,c_3)Q$. In the new coordinates $\mathbf{z}=Q\mathbf{y}$ the constraints are
\[
\|\mathbf{z}\|^2=1,\qquad c_1 z_1^2+c_2 z_2^2+c_3 z_3^2=\kappa,
\]
with $\kappa=s_0\det D$. Eliminating $z_3$ yields a conic section whose non-degenerate parametrisation is
\[
 z_1(t)=a_1\cos t,\quad z_2(t)=a_2\sin t,\quad z_3^{(\pm)}(t)=\pm\sqrt{1-a_1^2\cos^2 t-a_2^2\sin^2 t}
\]
for $t\in[0,2\pi)$. The original coordinates recover via
\[
 \mathbf{E}(t,\pm)=R^\top\Gamma^{-1/2}Q^\top\begin{pmatrix}a_1\cos t\\ a_2\sin t\\ z_3^{(\pm)}(t)\end{pmatrix}.
\]

## 4. Concrete Diagonal Example

For $A=\operatorname{diag}(a,b,c)$ and $\boldsymbol{\omega}=(0,0,\omega)$ we have
\[
 D=\begin{pmatrix}a & -\omega & 0\\ \omega & b & 0\\ 0 & 0 & c\end{pmatrix},\quad
 DD^\top=\begin{pmatrix}a^2+\omega^2 & \omega(a+b) & 0\\ \omega(a+b) & b^2+\omega^2 & 0\\ 0 & 0 & c^2\end{pmatrix}.
\]
The $2\times2$ block in the $xy$-plane is diagonalised by a rotation $R_{xy}(\varphi)$ with
\[
 \tan 2\varphi=\frac{2\omega(a+b)}{b^2-a^2}.
\]
The matrix $B$ is diagonal,
\[
 B=\operatorname{diag}(bc,ac,ab+\omega^2),
\]
so the general construction above yields explicit parameters for the intersection curve by combining $R_{xy}(\varphi)$ and the eigenstructure of $C$.

## 5. Minimal-Norm Electric Field or Vorticity

With $K(\boldsymbol{\omega})=\operatorname{adj}(A)+\boldsymbol{\omega}\boldsymbol{\omega}^\top$ the kernel constraint implies
\[
 \mathbf{E}^\top K(\boldsymbol{\omega})\mathbf{E}=s_0\Big(\det A+\boldsymbol{\omega}^\top A\boldsymbol{\omega}\Big).
\]
The minimal electric field magnitude is
\[
 \min \|\mathbf{E}\|^2=\frac{s_0\left(\det A+\boldsymbol{\omega}^\top A\boldsymbol{\omega}\right)}{\lambda_{\max}\big(K(\boldsymbol{\omega})\big)},
\]
attained along the dominant eigenvector of $K(\boldsymbol{\omega})$.

Conversely, fixing $\mathbf{E}$ and defining $K_\omega=s_0 A-\mathbf{E}\mathbf{E}^\top$ and $\mu=\mathbf{E}^\top\operatorname{adj}(A)\mathbf{E}-s_0\det A$, the minimal vorticity satisfies
\[
 \min \|\boldsymbol{\omega}\|^2=
 \begin{cases}
  \mu/\lambda_{\max}(K_\omega), & \mu>0,\\
  \mu/\lambda_{\min}(K_\omega), & \mu<0,\\
  0, & \mu=0.
 \end{cases}
\]

### 5.1 Numerical Illustration

Take $A=\operatorname{diag}(2,1,3)$, $s_0=1.5$, and $\boldsymbol{\omega}=(0,0,1)$. Then $K=\operatorname{diag}(3,6,3)$ and $\det A+\boldsymbol{\omega}^\top A\boldsymbol{\omega}=9$, so the minimal electric field obeying the kernel constraint has magnitude squared $\|\mathbf{E}\|^2=2.25$ along the $y$-axis.

Fixing $\mathbf{E}=(0,\sqrt{2},0)$ with the same $A$ and $s_0$, the vorticity matrix $K_\omega=\operatorname{diag}(3,-0.5,4.5)$ and $\mu=3$, yielding $\|\boldsymbol{\omega}\|^2=2/3$ realised on the $z$-axis.

## 6. Normalising Kernel Vectors

For $\beta<1$ the timelike kernel $j=(1,-D^{-1}\mathbf{E})$ normalises to
\[
 \hat{j}=\frac{(1,-D^{-1}\mathbf{E})}{\sqrt{1-\beta}},
\]
while lightlike kernels ($\beta=1$) require an auxiliary null vector $\ell$ to form a projector $P=\frac{j\,j^\top g}{j^\top g\,\ell}$.

## 7. Reconstruction and Variational Stability

A three-dimensional Bloch vector $j$ lifts to a Pauli spinor via
\[
 \psi(\theta,\phi)=\begin{pmatrix}\cos(\theta/2)\\ e^{i\phi}\sin(\theta/2)\end{pmatrix},
\]
while four-dimensional reconstruction leverages Fierz identities linking bilinears $(S,P,A^\mu,T^{\mu\nu})$. Variational stability follows from positive definiteness of the Hessians $V''(T_0)$ and suitable couplings.

## 8. Definition of STV

SpinoTensorVector aggregates a spinor, a tensor, and their induced vector:
\[
\operatorname{STV}:\quad \Phi=(\psi,T,v),\qquad v=T\big(j(\psi)\big).
\]

This packaging preserves equivariance, phase invariance, and degree constraints without collapsing the spinor degrees of freedom.
