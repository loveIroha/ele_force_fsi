$\boldsymbol{\sigma}^e=J^{-1} \mathbf{P}^e \mathbf{F}^T$

$\boldsymbol{\sigma}^e=\boldsymbol{\sigma}^p+\boldsymbol{\sigma}^a$ ,其中$\boldsymbol{\sigma}^p$为被动弹性响应应力, $\boldsymbol{\sigma}^a$为主动收缩应力。

$\mathbf{P}^p=\frac{\partial W}{\partial \mathbf{F}}-a e^{b\left(I_1-3\right)} \mathbf{F}^{-T}$

$\boldsymbol{\sigma}^p=J^{-1} \frac{\partial W}{\partial \mathbf{F}} \mathbf{F}^T-J^{-1} a e^{b\left(I_1-3\right)} \mathbf{I},$  相应的被动弹性响应应力

$\boldsymbol{\sigma}^a=T \mathbf{f} \otimes \mathbf{f},$ 相应的主动收缩应力

$\mathbf{P}^a=J T \mathbf{F f}_0 \otimes \mathbf{f}_0$, 相应的主动应力张量，$T(\mathbf{X}, t)$为主动收缩力，与心肌纤维拉伸$\lambda_f$, 变化率$\frac{\partial \lambda_f}{\partial t}$以及细胞内钙离子浓度$\left[\mathrm{Ca}^{2+}\right]_i$ 有关。



在肌节中，收缩力是通过细丝肌动蛋白丝和粗丝肌动蛋白丝之间的交叉桥循环生成的。当细胞处于静息状态时，沿着肌动蛋白丝的肌动蛋白结合位点是不可及的，因为它们被肌动蛋白凹槽中的一种高度延展的蛋白质——肌动凝蛋白所覆盖。这些肌动蛋白结合位点通过与肌动蛋白和肌动凝蛋白相关的蛋白质——肌钙蛋白的作用而显露出来。当结合钙离子时，肌钙蛋白发生构象变化，作用是移动肌动凝蛋白，从而允许肌动蛋白-肌动蛋白交叉桥循环和主动张力生成。高水平的细胞内钙离子由细胞的电激活引发，其中内向跨膜钙离子电流诱导肌质网钙库的释放（钙诱导的钙释放）。有关主动收缩和兴奋-收缩耦合生理学的详细信息，请参阅Bers（2001）的专著。

在我们的模型中，细胞内钙离子浓度 [Ca]i 被假定为空间均匀，并满足动力学方程
$$
\frac{\mathrm{d}[\mathrm{Ca}]_{\mathrm{i}}}{\mathrm{d} t}=\frac{[\mathrm{Ca}]_{\mathrm{i}, \max }-[\mathrm{Ca}]_{\mathrm{i}, 0}}{\tau_{\mathrm{Ca}}} \exp ^{\left(1-t / \tau_{\mathrm{Ca}}\right)}\left(1-\frac{t}{\tau_{\mathrm{Ca}}}\right),
$$
其中

| 参数                                |      | 单位                                                         |
| ----------------------------------- | ---- | ------------------------------------------------------------ |
| $[\mathrm{Ca}]_{\mathrm{i}, \max }$ |      | 1.0                                                                                           $\mu \mathrm{M}$ |
| $[\mathrm{Ca}]_{\mathrm{i}, 0}$     |      | 0.01                                                                                           $\mu \mathrm{M}$ |
| $\tau_{\mathrm{Ca}}$                |      | 0.06                                                                                             $s$ |

在我们的模拟中，肌节中活跃张力的发展由 Niederer 等人（2006）的模型控制，该模型将主动收缩张力的动力学描述为 $[Ca]_{i}$、纤维伸展 $λ_f$ 以及纤维拉伸的时间变化率的函数。在该模型中，结合肌钙蛋白的钙离子浓度 $[Ca]_{trpn}$ 的动力学由下述公式描述：
$$
\begin{aligned}
\frac{\mathrm{d}[\mathrm{Ca}]_{\text {trpn }}}{\mathrm{d} t} & =k_{\text {on }}[\mathrm{Ca}]_{\mathrm{i}}\left([\mathrm{Ca}]_{\text {trpn,max }}-[\mathrm{Ca}]_{\text {trpn }}\right)-k_{\text {off }}[\mathrm{Ca}]_{\text {trpn }} \\
k_{\text {off }} & =k_{\text {off,ref }}\left(1-\frac{T}{\gamma T_{\text {ref }}}\right)
\end{aligned}
$$
其中，$k_{on}$是结合率，$k_{off}$ 是非结合率，$k_{off,ref}$是在没有主动张力的情况下解离速率，$T$ 是主动收缩张力，$T_{ref}$是在剩余肌节长度处的最大主动收缩力。而原肌球蛋白钙离子浓度 $z$ 满足如下方程
$$
\frac{\mathrm{d} z}{\mathrm{~d} t}=\alpha_0\left(\frac{[\mathrm{Ca}]_{\operatorname{trpn}}}{[\mathrm{Ca}]_{\operatorname{trpn}, 50}}\right)^n(1-z)-\alpha_{\mathrm{r} 1} z-\alpha_{\mathrm{r} 2} \frac{z^{n_{\mathrm{r}}}}{z^{n_{\mathrm{r}}}+K_z^{n_{\mathrm{r}}}}
$$
控制。其中，$[Ca]_{trpn,50}$是半激活状态下肌钙蛋白结合的钙离子浓度,
$$
[\mathrm{Ca}]_{\text {trpn }, 50}=[\mathrm{Ca}]_{\text {trpn,max }}[\mathrm{Ca}]_{50} /\left([\mathrm{Ca}]_{50}+\frac{k_{\text {off,ref }}}{k_{\text {on }}}\left(1-\frac{1+\beta_0\left(\lambda_{\mathrm{f}}-1\right)}{2 \gamma}\right)\right),
$$
其中$[\mathrm{Ca}]_{50}=[\mathrm{Ca}]_{50, \text { ref }}\left(1+\beta_1\left(\lambda_{\mathrm{f}}-1\right)\right)$. 等长张力 $T_0=T_{\text {ref }}\left(1+\beta_0\left(\lambda_{\mathrm{f}}-1\right)\right) \frac{z}{z_{\max }}$, 其中，$λ_{f}$ 是纤维方向的拉伸，而 $z_{max}$ 由下式确定
$$
z_{\max }=\left(\frac{\alpha_0}{\left([\mathrm{Ca}]_{\operatorname{trpn}, 50} /[\mathrm{Ca}]_{\operatorname{trpn}, \max }\right)^n}-K_2\right) /\left(\alpha_{\mathrm{r} 1}+K_1+\frac{\alpha_0}{\left([\mathrm{Ca}]_{\mathrm{trpn}, 50} /[\mathrm{Ca}]_{\mathrm{trpn}, \max }\right)^n}\right)
$$
其中
$$
\begin{aligned}
& K_1=\frac{\alpha_{\mathrm{r} 2} z_{\mathrm{p}}^{n_{\mathrm{r}}-1} n_{\mathrm{r}} K_z^{n_{\mathrm{r}}}}{\left(z_{\mathrm{p}}^{n_{\mathrm{r}}}+K_z^{n_{\mathrm{r}}}\right)^2}, \\
& K_2=\alpha_{\mathrm{r} 2} \frac{z_{\mathrm{p}}^{n_{\mathrm{r}}}}{z_{\mathrm{p}}^{n_{\mathrm{r}}}+K_z^{n_{\mathrm{r}}}}\left(1-\frac{n_{\mathrm{r}} K_z^{n_{\mathrm{r}}}}{z_{\mathrm{p}}^{n_{\mathrm{r}}}+K_z^{n_{\mathrm{r}}}}\right) .
\end{aligned}
$$
主动收缩张力$T$ 是通过一个衰减记忆模型计算得出
$$
T=T_0 \times \begin{cases}\frac{1+a \sum_{i=1}^3 Q_i}{1-\sum_{i=1}^3 Q_i} & \text { if } \sum_{i=1}^3 Q_i<0, \\ \frac{1+(2+a) \sum_{i=1}^3 Q_i}{1+\sum_{i=1}^3 Q_i} & \text { otherwise, }\end{cases}
$$
其中$Q_{i}$ 由下式决定
$$
\frac{\mathrm{d} Q_i}{\mathrm{~d} t}=A_i \frac{\mathrm{d} \lambda_{\mathrm{f}}}{\mathrm{d} t}-\alpha_i Q_i
$$
根据Niederer *et al.* (2006), 相关的参数如下：

|            参数            |                           | 单位               |
| :------------------------: | :-----------------------: | ------------------ |
|            $a$             |           0.35            |                    |
|          $A_{1}$           |            -29            |                    |
|          $A_{2}$           |            138            |                    |
|          $A_{3}$           |            129            |                    |
|        $\alpha_{0}$        |             8             | $s^{-1}$           |
|        $\alpha_{1}$        |            130            | $s^{-1}$           |
|        $\alpha_{2}$        |            625            | $s^{-1}$           |
|       $\alpha_{r1}$        |            2.0            | $s^{-1}$           |
|       $\alpha_{r2}$        |           1.75            | $s^{-1}$           |
|        $\beta_{0}$         |            4.9            |                    |
|        $\beta_{1}$         |           -4.0            |                    |
|  $[\mathrm{Ca}]_{50,ref}$  |           1.05            | $\mu M$            |
| $[\mathrm{Ca}]_{trpn,max}$ |            70             | $\mu M$            |
|          $\gamma$          |            2.0            |                    |
|          $k_{on}$          |            100            | $\mu M^{-1}s^{-1}$ |
|       $k_{off, ref}$       |            200            | $s^{-1}$           |
|          $K_{z}$           |           0.15            |                    |
|            $n$             |             3             |                    |
|          $n_{r}$           |             3             |                    |
|          $z_{p}$           |           0.85            |                    |
|         $T_{ref}$          | $T_{scale}\times 56.2kPa$ |                    |
|        $T_{scale}$         |             1             |                    |
|                            |                           |                    |

其中，无量纲比例因子$T_{scale}$的值是通过将模型左心室的收缩末期体积与通过磁共振成像非侵入性确定的体内左心室体积相匹配而确定的。当 $T_{scale} = 1$ 时，$T_{ref}$ 的值与 Niederer 等人（2006）使用的值相对应。









第一步：计算钙离子浓度$ [Ca]_{i} $:
$$
\frac{\mathrm{d}[\mathrm{Ca}]_{\mathrm{i}}}{\mathrm{d} t}=\frac{[\mathrm{Ca}]_{\mathrm{i}, \max }-[\mathrm{Ca}]_{\mathrm{i}, 0}}{\tau_{\mathrm{Ca}}} \exp ^{\left(1-t / \tau_{\mathrm{Ca}}\right)}\left(1-\frac{t}{\tau_{\mathrm{Ca}}}\right),
$$
参数：

| 参数                                |      | 单位                                                         |
| ----------------------------------- | ---- | ------------------------------------------------------------ |
| $[\mathrm{Ca}]_{\mathrm{i}, \max }$ |      | 1.0                                                                                           $\mu \mathrm{M}$ |
| $[\mathrm{Ca}]_{\mathrm{i}, 0}$     |      | 0.01                                                                                           $\mu \mathrm{M}$ |
| $\tau_{\mathrm{Ca}}$                |      | 0.06                                                                                             $s$ |

第二步：计算肌钙蛋白的钙离子浓度 $[Ca]_{trpn}$
$$
\begin{aligned}
\frac{\mathrm{d}[\mathrm{Ca}]_{\text {trpn }}}{\mathrm{d} t} & =k_{\text {on }}[\mathrm{Ca}]_{\mathrm{i}}\left([\mathrm{Ca}]_{\text {trpn,max }}-[\mathrm{Ca}]_{\text {trpn }}\right)-k_{\text {off }}[\mathrm{Ca}]_{\text {trpn }} \\
k_{\text {off }} & =k_{\text {off,ref }}\left(1-\frac{T}{\gamma T_{\text {ref }}}\right)
\end{aligned}
$$
第三步：计算半激活状态下肌钙蛋白结合的钙离子浓度$[Ca]_{trpn,50}$
$$
[\mathrm{Ca}]_{\text {trpn }, 50}=[\mathrm{Ca}]_{\text {trpn,max }}[\mathrm{Ca}]_{50} /\left([\mathrm{Ca}]_{50}+\frac{k_{\text {off,ref }}}{k_{\text {on }}}\left(1-\frac{1+\beta_0\left(\lambda_{\mathrm{f}}-1\right)}{2 \gamma}\right)\right),
$$
其中$[\mathrm{Ca}]_{50}=[\mathrm{Ca}]_{50, \text { ref }}\left(1+\beta_1\left(\lambda_{\mathrm{f}}-1\right)\right)$.  $[\mathrm{Ca}]_{50,ref}=1.05\mu M$,  $[\mathrm{Ca}]_{trpn,max}=70\mu M$, $k_{on}=100\mu M^{-1}s^{-1}$,  $\beta_{0}=4.9$, $k_{off, ref}=200s^{-1}$,$\gamma=2.0$

第四步：原肌球蛋白钙离子浓度 $z$ 
$$
\frac{\mathrm{d} z}{\mathrm{~d} t}=\alpha_0\left(\frac{[\mathrm{Ca}]_{\operatorname{trpn}}}{[\mathrm{Ca}]_{\operatorname{trpn}, 50}}\right)^n(1-z)-\alpha_{\mathrm{r} 1} z-\alpha_{\mathrm{r} 2} \frac{z^{n_{\mathrm{r}}}}{z^{n_{\mathrm{r}}}+K_z^{n_{\mathrm{r}}}}
$$


第五步：等容积收缩张力
$$
T_0=T_{\text {ref }}\left(1+\beta_0\left(\lambda_{\mathrm{f}}-1\right)\right) \frac{z}{z_{\max }}
$$
其中
$$
z_{\max }=\left(\frac{\alpha_0}{\left([\mathrm{Ca}]_{\operatorname{trpn}, 50} /[\mathrm{Ca}]_{\operatorname{trpn}, \max }\right)^n}-K_2\right) /\left(\alpha_{\mathrm{r} 1}+K_1+\frac{\alpha_0}{\left([\mathrm{Ca}]_{\mathrm{trpn}, 50} /[\mathrm{Ca}]_{\mathrm{trpn}, \max }\right)^n}\right)
$$

$$
其中\begin{aligned}
& K_1=\frac{\alpha_{\mathrm{r} 2} z_{\mathrm{p}}^{n_{\mathrm{r}}-1} n_{\mathrm{r}} K_z^{n_{\mathrm{r}}}}{\left(z_{\mathrm{p}}^{n_{\mathrm{r}}}+K_z^{n_{\mathrm{r}}}\right)^2}, \\
& K_2=\alpha_{\mathrm{r} 2} \frac{z_{\mathrm{p}}^{n_{\mathrm{r}}}}{z_{\mathrm{p}}^{n_{\mathrm{r}}}+K_z^{n_{\mathrm{r}}}}\left(1-\frac{n_{\mathrm{r}} K_z^{n_{\mathrm{r}}}}{z_{\mathrm{p}}^{n_{\mathrm{r}}}+K_z^{n_{\mathrm{r}}}}\right) .
\end{aligned}
$$

第六步： 主动收缩张力$T$ 是通过一个衰减记忆模型计算得出
$$
T=T_0 \times \begin{cases}\frac{1+a \sum_{i=1}^3 Q_i}{1-\sum_{i=1}^3 Q_i} & \text { if } \sum_{i=1}^3 Q_i<0, \\ \frac{1+(2+a) \sum_{i=1}^3 Q_i}{1+\sum_{i=1}^3 Q_i} & \text { otherwise, }\end{cases}
$$
其中$Q_{i}$ 由下式决定
$$
\frac{\mathrm{d} Q_i}{\mathrm{~d} t}=A_i \frac{\mathrm{d} \lambda_{\mathrm{f}}}{\mathrm{d} t}-\alpha_i Q_i
$$














