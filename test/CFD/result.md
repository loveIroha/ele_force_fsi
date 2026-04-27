
### TEST 1
在验证投影方法之前，先验证一下方程$(1)$ 和方程 $(2)$ 能否正确求解。
$$
\boldsymbol{u}_t-\mu\Delta\boldsymbol{u}=\boldsymbol{f}\tag{1}
$$
$$
\Delta p = g
$$

- 测试算例为`features/finite_difference/3D/generate_demo.py`生成的`features/finite_difference/3D/ns_demo.json`。
- 需要注意计算 $\mathbf{f}$ 时没有 $\nabla p$ ， $p$ 不能为二次及以下的多项式。
- 边界条件为非齐次Dirichlet边界
- 测试结果显示出时间精度为一阶，空间精度为二阶。

**GIT提交**:

```
Commit: 6706238c9978268829aab618337366ac464ee77d
Parents: 927903bf6f3f69488080a74c98cd0367caf2eb89
Author: Ma Pengfei <code@pengfeima.cn>
Committer: Ma Pengfei <code@pengfeima.cn>
Date: Sat Oct 21 2023 13:07:11 GMT+0800 (中国标准时间)
 

[result] 3D projection method for N-S equaitons (1)
```
**运行命令**:

```
make read_ns_demo -j8
./test/read_ns_demo
```
**运行结果**:

```
Result for ns = 4:
           nt = 4:
sum_eh_squared = 1.65353282832054748752e-02
sum_eh_squared = 7.85289470419641907895e-03
sum_eh_squared = 7.85289470419641734422e-03
sum_eh_squared = 7.85289470419641214005e-03
           nt = 8:
sum_eh_squared = 1.65353282832054748752e-02
sum_eh_squared = 8.11539210476943696781e-03
sum_eh_squared = 8.11539210476943523309e-03
sum_eh_squared = 8.11539210476943349837e-03
           nt = 16:
sum_eh_squared = 1.65353282832054748752e-02
sum_eh_squared = 8.23979663309891245671e-03
sum_eh_squared = 8.23979663309891072198e-03
sum_eh_squared = 8.23979663309890898726e-03
           nt = 32:
sum_eh_squared = 1.65353282832054748752e-02
sum_eh_squared = 8.30045543933358165312e-03
sum_eh_squared = 8.30045543933358338784e-03
sum_eh_squared = 8.30045543933358165312e-03
           nt = 64:
sum_eh_squared = 1.65353282832054748752e-02
sum_eh_squared = 8.33040691382306783264e-03
sum_eh_squared = 8.33040691382306956736e-03
sum_eh_squared = 8.33040691382306609791e-03
           nt = 128:
sum_eh_squared = 1.65353282832054748752e-02
sum_eh_squared = 8.34528905782876487263e-03
sum_eh_squared = 8.34528905782876487263e-03
sum_eh_squared = 8.34528905782876140318e-03
Result for ns = 8:
           nt = 4:
sum_eh_squared = 3.93612632528241718211e-03
sum_eh_squared = 2.08301648845785666170e-03
sum_eh_squared = 2.08301648845785319225e-03
sum_eh_squared = 2.08301648845785232489e-03
           nt = 8:
sum_eh_squared = 3.93612632528241718211e-03
sum_eh_squared = 2.25719741272618771091e-03
sum_eh_squared = 2.25719741272618857827e-03
sum_eh_squared = 2.25719741272618597619e-03
           nt = 16:
sum_eh_squared = 3.93612632528241718211e-03
sum_eh_squared = 2.34975535400619960080e-03
sum_eh_squared = 2.34975535400620133553e-03
sum_eh_squared = 2.34975535400620090185e-03
           nt = 32:
sum_eh_squared = 3.93612632528241718211e-03
sum_eh_squared = 2.39673046842543051652e-03
sum_eh_squared = 2.39673046842543095020e-03
sum_eh_squared = 2.39673046842543138388e-03
           nt = 64:
sum_eh_squared = 3.93612632528241718211e-03
sum_eh_squared = 2.42032272182302228558e-03
sum_eh_squared = 2.42032272182302488767e-03
sum_eh_squared = 2.42032272182302532135e-03
           nt = 128:
sum_eh_squared = 3.93612632528241718211e-03
sum_eh_squared = 2.43213939700050044493e-03
sum_eh_squared = 2.43213939700050087861e-03
sum_eh_squared = 2.43213939700049914389e-03
Result for ns = 16:
           nt = 4:
sum_eh_squared = 9.72203542866096844834e-04
sum_eh_squared = 5.55665031516984985437e-04
sum_eh_squared = 5.55665031516986937001e-04
sum_eh_squared = 5.55665031516987262261e-04
           nt = 8:
sum_eh_squared = 9.72203542866096844834e-04
sum_eh_squared = 5.20433294371604182604e-04
sum_eh_squared = 5.20433294371603640503e-04
sum_eh_squared = 5.20433294371604182604e-04
           nt = 16:
sum_eh_squared = 9.72203542866096844834e-04
sum_eh_squared = 5.68404866288850008950e-04
sum_eh_squared = 5.68404866288850334211e-04
sum_eh_squared = 5.68404866288849358429e-04
           nt = 32:
sum_eh_squared = 9.72203542866096844834e-04
sum_eh_squared = 6.03624037464871245236e-04
sum_eh_squared = 6.03624037464868859991e-04
sum_eh_squared = 6.03624037464871245236e-04
           nt = 64:
sum_eh_squared = 9.72203542866096844834e-04
sum_eh_squared = 6.23292417763032726048e-04
sum_eh_squared = 6.23292417763033051309e-04
sum_eh_squared = 6.23292417763031641846e-04
           nt = 128:
sum_eh_squared = 9.72203542866096844834e-04
sum_eh_squared = 6.33557955199488169809e-04
sum_eh_squared = 6.33557955199488711910e-04
sum_eh_squared = 6.33557955199488169809e-04
Result for ns = 32:
           nt = 4:
sum_eh_squared = 2.42309602733810982049e-04
sum_eh_squared = 5.28220407004484945705e-04
sum_eh_squared = 5.28220407004484620445e-04
sum_eh_squared = 5.28220407004483102562e-04
           nt = 8:
sum_eh_squared = 2.42309602733810982049e-04
sum_eh_squared = 2.34653110321486109034e-04
sum_eh_squared = 2.34653110321484564046e-04
sum_eh_squared = 2.34653110321486976396e-04
           nt = 16:
sum_eh_squared = 2.42309602733810982049e-04
sum_eh_squared = 1.39664719227558650413e-04
sum_eh_squared = 1.39664719227559571985e-04
sum_eh_squared = 1.39664719227557701737e-04
           nt = 32:
sum_eh_squared = 2.42309602733810982049e-04
sum_eh_squared = 1.34750814971549198523e-04
sum_eh_squared = 1.34750814971547545115e-04
sum_eh_squared = 1.34750814971547761955e-04
           nt = 64:
sum_eh_squared = 2.42309602733810982049e-04
sum_eh_squared = 1.45899109524727158124e-04
sum_eh_squared = 1.45899109524728133906e-04
sum_eh_squared = 1.45899109524727564700e-04
           nt = 128:
sum_eh_squared = 2.42309602733810982049e-04
sum_eh_squared = 1.54160668433651800715e-04
sum_eh_squared = 1.54160668433650689408e-04
sum_eh_squared = 1.54160668433652153081e-04
```



### TEST 2

继续 **TEST 1** 的测试，测试Neumann边界条件。

- 边界条件为非齐次Neumann边界。

-  $\nabla p=g$ 没有唯一解，$\boldsymbol{u}_t-\mu\Delta\boldsymbol{u}=\boldsymbol{f}\tag{1}$ 有唯一解。
- 测试结果显示，相比Dirichlet边界条件，更小的时间步长才能观察到时间精度为一阶，空间精度为二阶。
-  `test/CFD/read_ns_demo.cpp` 中的变量 `all_boundary_type` 保存着六面边界的类型，`NEUMANN`和`DIRICHLET`类型的边界数值上分别为 `1` 和 `2`。


**GIT提交**:

```
Commit: 89df403967f8cc8228231eedacb53e1cf8e4b502
Parents: 6706238c9978268829aab618337366ac464ee77d
Author: Ma Pengfei <code@pengfeima.cn>
Committer: Ma Pengfei <code@pengfeima.cn>
Date: Sat Oct 21 2023 17:24:22 GMT+0800 (中国标准时间)
 

[result] 3D projection method for N-S equaitons (2)
```
**运行命令**:

```
make read_ns_demo -j8
./test/read_ns_demo
```
**运行结果**:


```
Result for ns = 4:
           nt = 4:
sum_eh_squared = 7.15787588871419529823e+02
sum_eh_squared = 6.41241444262253268427e-02
sum_eh_squared = 6.41241444262253129649e-02
sum_eh_squared = 6.41241444262252713315e-02
           nt = 8:
sum_eh_squared = 7.15787588871419529823e+02
sum_eh_squared = 4.95445702103279872475e-02
sum_eh_squared = 4.95445702103280635753e-02
sum_eh_squared = 4.95445702103281190865e-02
           nt = 16:
sum_eh_squared = 7.15787588871419529823e+02
sum_eh_squared = 4.20257680261840946279e-02
sum_eh_squared = 4.20257680261841154445e-02
sum_eh_squared = 4.20257680261841154445e-02
           nt = 32:
sum_eh_squared = 7.15787588871419529823e+02
sum_eh_squared = 3.82112448200671930842e-02
sum_eh_squared = 3.82112448200671722676e-02
sum_eh_squared = 3.82112448200670681842e-02
           nt = 64:
sum_eh_squared = 7.15787588871419529823e+02
sum_eh_squared = 3.62907433698037296121e-02
sum_eh_squared = 3.62907433698037157344e-02
sum_eh_squared = 3.62907433698037087955e-02
           nt = 128:
sum_eh_squared = 7.15787588871419529823e+02
sum_eh_squared = 3.53272623859018711223e-02
sum_eh_squared = 3.53272623859018988779e-02
sum_eh_squared = 3.53272623859018780612e-02
           nt = 256:
sum_eh_squared = 7.15787588871419529823e+02
sum_eh_squared = 3.48446892927353757408e-02
sum_eh_squared = 3.48446892927354312519e-02
sum_eh_squared = 3.48446892927353479852e-02
           nt = 512:
sum_eh_squared = 7.15787588871419529823e+02
sum_eh_squared = 3.46033959672740754066e-02
sum_eh_squared = 3.46033959672741517344e-02
sum_eh_squared = 3.46033959672740129565e-02
Result for ns = 8:
           nt = 4:
sum_eh_squared = 4.77308119971261817227e+01
sum_eh_squared = 4.56606416837147047194e-02
sum_eh_squared = 4.56606416837146977805e-02
sum_eh_squared = 4.56606416837146283916e-02
           nt = 8:
sum_eh_squared = 4.77308119971261817227e+01
sum_eh_squared = 2.78424998605874343682e-02
sum_eh_squared = 2.78424998605873684487e-02
sum_eh_squared = 2.78424998605874135515e-02
           nt = 16:
sum_eh_squared = 4.77308119971261817227e+01
sum_eh_squared = 1.86467057391121251997e-02
sum_eh_squared = 1.86467057391121043830e-02
sum_eh_squared = 1.86467057391121251997e-02
           nt = 32:
sum_eh_squared = 4.77308119971261817227e+01
sum_eh_squared = 1.39815240308582330181e-02
sum_eh_squared = 1.39815240308582069972e-02
sum_eh_squared = 1.39815240308582087320e-02
           nt = 64:
sum_eh_squared = 4.77308119971261817227e+01
sum_eh_squared = 1.16344275888595211771e-02
sum_eh_squared = 1.16344275888595697493e-02
sum_eh_squared = 1.16344275888595263813e-02
           nt = 128:
sum_eh_squared = 4.77308119971261817227e+01
sum_eh_squared = 1.04580599362391422208e-02
sum_eh_squared = 1.04580599362391595680e-02
sum_eh_squared = 1.04580599362391907931e-02
           nt = 256:
sum_eh_squared = 4.77308119971261817227e+01
sum_eh_squared = 9.86936661227970878085e-03
sum_eh_squared = 9.86936661227970531141e-03
sum_eh_squared = 9.86936661227971571975e-03
           nt = 512:
sum_eh_squared = 4.77308119971261817227e+01
sum_eh_squared = 9.57493545318797122101e-03
sum_eh_squared = 9.57493545318811173361e-03
sum_eh_squared = 9.57493545318801111965e-03
Result for ns = 16:
           nt = 4:
sum_eh_squared = 2.76531566656638938539e+00
sum_eh_squared = 4.11789150427593064752e-02
sum_eh_squared = 4.11789150427593064752e-02
sum_eh_squared = 4.11789150427593689252e-02
           nt = 8:
sum_eh_squared = 2.76531566656638938539e+00
sum_eh_squared = 2.21996280137581082637e-02
sum_eh_squared = 2.21996280137581117331e-02
sum_eh_squared = 2.21996280137581152025e-02
           nt = 16:
sum_eh_squared = 2.76531566656638938539e+00
sum_eh_squared = 1.23977698682739643360e-02
sum_eh_squared = 1.23977698682739279068e-02
sum_eh_squared = 1.23977698682739556624e-02
           nt = 32:
sum_eh_squared = 2.76531566656638938539e+00
sum_eh_squared = 7.42011569708743608242e-03
sum_eh_squared = 7.42011569708746470536e-03
sum_eh_squared = 7.42011569708745776647e-03
           nt = 64:
sum_eh_squared = 2.76531566656638938539e+00
sum_eh_squared = 4.91366024960312314240e-03
sum_eh_squared = 4.91366024960312747921e-03
sum_eh_squared = 4.91366024960303900831e-03
           nt = 128:
sum_eh_squared = 2.76531566656638938539e+00
sum_eh_squared = 3.65724000448534045452e-03
sum_eh_squared = 3.65724000448540680769e-03
sum_eh_squared = 3.65724000448542372124e-03
           nt = 256:
sum_eh_squared = 2.76531566656638938539e+00
sum_eh_squared = 3.02889243779858258546e-03
sum_eh_squared = 3.02889243779856523822e-03
sum_eh_squared = 3.02889243779857954969e-03
           nt = 512:
sum_eh_squared = 2.76531566656638938539e+00
sum_eh_squared = 2.71493414941986104319e-03
sum_eh_squared = 2.71493414941988749772e-03
sum_eh_squared = 2.71493414941985627270e-03
Result for ns = 32:
           nt = 4:
sum_eh_squared = 1.95079209332415681732e-01
sum_eh_squared = 4.03160519093490044007e-02
sum_eh_squared = 4.03160519093489627673e-02
sum_eh_squared = 4.03160519093490182785e-02
           nt = 8:
sum_eh_squared = 1.95079209332415681732e-01
sum_eh_squared = 2.08882863289922178496e-02
sum_eh_squared = 2.08882863289922456052e-02
sum_eh_squared = 2.08882863289922143801e-02
           nt = 16:
sum_eh_squared = 1.95079209332415681732e-01
sum_eh_squared = 1.08536586296325652384e-02
sum_eh_squared = 1.08536586296325478912e-02
sum_eh_squared = 1.08536586296325461565e-02
           nt = 32:
sum_eh_squared = 1.95079209332415681732e-01
sum_eh_squared = 5.75621871679038443759e-03
sum_eh_squared = 5.75621871679040178482e-03
sum_eh_squared = 5.75621871679043040776e-03
           nt = 64:
sum_eh_squared = 1.95079209332415681732e-01
sum_eh_squared = 3.18771549955767499393e-03
sum_eh_squared = 3.18771549955768453491e-03
sum_eh_squared = 3.18771549955767412657e-03
           nt = 128:
sum_eh_squared = 1.95079209332415681732e-01
sum_eh_squared = 1.89888891891754128631e-03
sum_eh_squared = 1.89888891891751331389e-03
sum_eh_squared = 1.89888891891748252255e-03
           nt = 256:
sum_eh_squared = 1.95079209332415681732e-01
sum_eh_squared = 1.25375170239665168978e-03
sum_eh_squared = 1.25375170239666556757e-03
sum_eh_squared = 1.25375170239663499307e-03
           nt = 512:
sum_eh_squared = 1.95079209332415681732e-01
sum_eh_squared = 9.31348141267139941271e-04
sum_eh_squared = 9.31348141267096790025e-04
sum_eh_squared = 9.31348141267153168538e-04
```

### TEST 3
继续 TEST 2 的测试，接下来，我们需要将使用投影法求解 N-S 方程。

- 速度为DIRICHLET边界条件，压强为齐次NEUMANN边界条件。
- 源文件为`test/CFD/3D/analytical_solution_projection.cpp`。
- 能观察到时间精度为一阶，空间精度为二阶。

**GIT提交**:
```
Commit: 11c79528099365a105ba4a0b69805617c7487a84
Parents: 89df403967f8cc8228231eedacb53e1cf8e4b502
Author: Ma Pengfei <code@pengfeima.cn>
Committer: Ma Pengfei <code@pengfeima.cn>
Date: Sat Oct 21 2023 22:31:34 GMT+0800 (中国标准时间)
 

[result] 3D projection method for N-S equaitons (3)
```

**计算结果**:
```
2023-10-21 18:05:56.255 (   0.000s) [         789C000]         StokesFlow3D.h:164   INFO| Lx : 1.0000, Ly : 1.0000, Lz : 1.0000, T  : 1.0000, rho : 1.0000, mu : 1.0000.
2023-10-21 18:05:56.255 (   0.000s) [         789C000]         StokesFlow3D.h:165   INFO| _Nx : 0004, _Ny : 0004, _Nz : 0004, Nt : 00000004, max_iters : 100000.
2023-10-21 18:05:56.255 (   0.000s) [         789C000]         StokesFlow3D.h:166   INFO| dx : 0.250000, dy : 0.250000, dz : 0.250000, dt : 0.25000000.
2023-10-21 18:05:56.259 (   0.004s) [         789C000]         StokesFlow3D.h:200   INFO| p sum_eh_squared = 2.10966817476105489293e-03
2023-10-21 18:05:56.259 (   0.004s) [         789C000]         StokesFlow3D.h:209   INFO| u sum_eh_squared = 5.07573024918226978208e-02
2023-10-21 18:05:56.259 (   0.004s) [         789C000]         StokesFlow3D.h:218   INFO| v sum_eh_squared = 5.07573024918226978208e-02
2023-10-21 18:05:56.259 (   0.004s) [         789C000]         StokesFlow3D.h:227   INFO| w sum_eh_squared = 5.07573024918226978208e-02
2023-10-21 18:05:56.259 (   0.004s) [         789C000]         StokesFlow3D.h:164   INFO| Lx : 1.0000, Ly : 1.0000, Lz : 1.0000, T  : 1.0000, rho : 1.0000, mu : 1.0000.
2023-10-21 18:05:56.259 (   0.004s) [         789C000]         StokesFlow3D.h:165   INFO| _Nx : 0004, _Ny : 0004, _Nz : 0004, Nt : 00000008, max_iters : 100000.
2023-10-21 18:05:56.259 (   0.004s) [         789C000]         StokesFlow3D.h:166   INFO| dx : 0.250000, dy : 0.250000, dz : 0.250000, dt : 0.12500000.
2023-10-21 18:05:56.266 (   0.011s) [         789C000]         StokesFlow3D.h:200   INFO| p sum_eh_squared = 4.34598931758105495871e-03
2023-10-21 18:05:56.266 (   0.011s) [         789C000]         StokesFlow3D.h:209   INFO| u sum_eh_squared = 2.60916629095841907937e-02
2023-10-21 18:05:56.266 (   0.011s) [         789C000]         StokesFlow3D.h:218   INFO| v sum_eh_squared = 2.60916629095841942632e-02
2023-10-21 18:05:56.266 (   0.011s) [         789C000]         StokesFlow3D.h:227   INFO| w sum_eh_squared = 2.60916629095842012021e-02
2023-10-21 18:05:56.266 (   0.011s) [         789C000]         StokesFlow3D.h:164   INFO| Lx : 1.0000, Ly : 1.0000, Lz : 1.0000, T  : 1.0000, rho : 1.0000, mu : 1.0000.
2023-10-21 18:05:56.266 (   0.011s) [         789C000]         StokesFlow3D.h:165   INFO| _Nx : 0004, _Ny : 0004, _Nz : 0004, Nt : 00000016, max_iters : 100000.
2023-10-21 18:05:56.266 (   0.011s) [         789C000]         StokesFlow3D.h:166   INFO| dx : 0.250000, dy : 0.250000, dz : 0.250000, dt : 0.06250000.
2023-10-21 18:05:56.277 (   0.022s) [         789C000]         StokesFlow3D.h:200   INFO| p sum_eh_squared = 8.56749428286291379864e-03
2023-10-21 18:05:56.277 (   0.022s) [         789C000]         StokesFlow3D.h:209   INFO| u sum_eh_squared = 1.43777457612173070051e-02
2023-10-21 18:05:56.277 (   0.022s) [         789C000]         StokesFlow3D.h:218   INFO| v sum_eh_squared = 1.43777457612173087398e-02
2023-10-21 18:05:56.277 (   0.022s) [         789C000]         StokesFlow3D.h:227   INFO| w sum_eh_squared = 1.43777457612173070051e-02
2023-10-21 18:05:56.277 (   0.022s) [         789C000]         StokesFlow3D.h:164   INFO| Lx : 1.0000, Ly : 1.0000, Lz : 1.0000, T  : 1.0000, rho : 1.0000, mu : 1.0000.
2023-10-21 18:05:56.277 (   0.022s) [         789C000]         StokesFlow3D.h:165   INFO| _Nx : 0004, _Ny : 0004, _Nz : 0004, Nt : 00000032, max_iters : 100000.
2023-10-21 18:05:56.277 (   0.022s) [         789C000]         StokesFlow3D.h:166   INFO| dx : 0.250000, dy : 0.250000, dz : 0.250000, dt : 0.03125000.
2023-10-21 18:05:56.296 (   0.041s) [         789C000]         StokesFlow3D.h:200   INFO| p sum_eh_squared = 1.61725640936352318966e-02
2023-10-21 18:05:56.296 (   0.041s) [         789C000]         StokesFlow3D.h:209   INFO| u sum_eh_squared = 9.26875551386044035929e-03
2023-10-21 18:05:56.296 (   0.041s) [         789C000]         StokesFlow3D.h:218   INFO| v sum_eh_squared = 9.26875551386043862456e-03
2023-10-21 18:05:56.296 (   0.041s) [         789C000]         StokesFlow3D.h:227   INFO| w sum_eh_squared = 9.26875551386044035929e-03
2023-10-21 18:05:56.296 (   0.041s) [         789C000]         StokesFlow3D.h:164   INFO| Lx : 1.0000, Ly : 1.0000, Lz : 1.0000, T  : 1.0000, rho : 1.0000, mu : 1.0000.
2023-10-21 18:05:56.296 (   0.041s) [         789C000]         StokesFlow3D.h:165   INFO| _Nx : 0004, _Ny : 0004, _Nz : 0004, Nt : 00000064, max_iters : 100000.
2023-10-21 18:05:56.296 (   0.041s) [         789C000]         StokesFlow3D.h:166   INFO| dx : 0.250000, dy : 0.250000, dz : 0.250000, dt : 0.01562500.
2023-10-21 18:05:56.329 (   0.074s) [         789C000]         StokesFlow3D.h:200   INFO| p sum_eh_squared = 2.88743096716802122303e-02
2023-10-21 18:05:56.329 (   0.074s) [         789C000]         StokesFlow3D.h:209   INFO| u sum_eh_squared = 7.19951992896659172944e-03
2023-10-21 18:05:56.329 (   0.074s) [         789C000]         StokesFlow3D.h:218   INFO| v sum_eh_squared = 7.19951992896659606624e-03
2023-10-21 18:05:56.329 (   0.074s) [         789C000]         StokesFlow3D.h:227   INFO| w sum_eh_squared = 7.19951992896658652527e-03
2023-10-21 18:05:56.329 (   0.074s) [         789C000]         StokesFlow3D.h:164   INFO| Lx : 1.0000, Ly : 1.0000, Lz : 1.0000, T  : 1.0000, rho : 1.0000, mu : 1.0000.
2023-10-21 18:05:56.329 (   0.074s) [         789C000]         StokesFlow3D.h:165   INFO| _Nx : 0004, _Ny : 0004, _Nz : 0004, Nt : 00000128, max_iters : 100000.
2023-10-21 18:05:56.329 (   0.074s) [         789C000]         StokesFlow3D.h:166   INFO| dx : 0.250000, dy : 0.250000, dz : 0.250000, dt : 0.00781250.
2023-10-21 18:05:56.387 (   0.132s) [         789C000]         StokesFlow3D.h:200   INFO| p sum_eh_squared = 4.78307837532068250153e-02
2023-10-21 18:05:56.387 (   0.132s) [         789C000]         StokesFlow3D.h:209   INFO| u sum_eh_squared = 6.20445141104424609940e-03
2023-10-21 18:05:56.387 (   0.132s) [         789C000]         StokesFlow3D.h:218   INFO| v sum_eh_squared = 6.20445141104424696676e-03
2023-10-21 18:05:56.387 (   0.132s) [         789C000]         StokesFlow3D.h:227   INFO| w sum_eh_squared = 6.20445141104424870149e-03
2023-10-21 18:05:56.387 (   0.132s) [         789C000]         StokesFlow3D.h:164   INFO| Lx : 1.0000, Ly : 1.0000, Lz : 1.0000, T  : 1.0000, rho : 1.0000, mu : 1.0000.
2023-10-21 18:05:56.387 (   0.132s) [         789C000]         StokesFlow3D.h:165   INFO| _Nx : 0004, _Ny : 0004, _Nz : 0004, Nt : 00000256, max_iters : 100000.
2023-10-21 18:05:56.387 (   0.132s) [         789C000]         StokesFlow3D.h:166   INFO| dx : 0.250000, dy : 0.250000, dz : 0.250000, dt : 0.00390625.
2023-10-21 18:05:56.493 (   0.238s) [         789C000]         StokesFlow3D.h:200   INFO| p sum_eh_squared = 7.20808164738246798242e-02
2023-10-21 18:05:56.493 (   0.238s) [         789C000]         StokesFlow3D.h:209   INFO| u sum_eh_squared = 5.54836183702060985884e-03
2023-10-21 18:05:56.493 (   0.238s) [         789C000]         StokesFlow3D.h:218   INFO| v sum_eh_squared = 5.54836183702060552203e-03
2023-10-21 18:05:56.493 (   0.238s) [         789C000]         StokesFlow3D.h:227   INFO| w sum_eh_squared = 5.54836183702060552203e-03
2023-10-21 18:05:56.494 (   0.239s) [         789C000]         StokesFlow3D.h:164   INFO| Lx : 1.0000, Ly : 1.0000, Lz : 1.0000, T  : 1.0000, rho : 1.0000, mu : 1.0000.
2023-10-21 18:05:56.494 (   0.239s) [         789C000]         StokesFlow3D.h:165   INFO| _Nx : 0004, _Ny : 0004, _Nz : 0004, Nt : 00000512, max_iters : 100000.
2023-10-21 18:05:56.494 (   0.239s) [         789C000]         StokesFlow3D.h:166   INFO| dx : 0.250000, dy : 0.250000, dz : 0.250000, dt : 0.00195312.
2023-10-21 18:05:56.691 (   0.436s) [         789C000]         StokesFlow3D.h:200   INFO| p sum_eh_squared = 9.76967321960485468590e-02
2023-10-21 18:05:56.691 (   0.436s) [         789C000]         StokesFlow3D.h:209   INFO| u sum_eh_squared = 5.12162715710181778994e-03
2023-10-21 18:05:56.691 (   0.436s) [         789C000]         StokesFlow3D.h:218   INFO| v sum_eh_squared = 5.12162715710181258577e-03
2023-10-21 18:05:56.691 (   0.436s) [         789C000]         StokesFlow3D.h:227   INFO| w sum_eh_squared = 5.12162715710181692258e-03
2023-10-21 18:05:56.691 (   0.436s) [         789C000]         StokesFlow3D.h:164   INFO| Lx : 1.0000, Ly : 1.0000, Lz : 1.0000, T  : 1.0000, rho : 1.0000, mu : 1.0000.
2023-10-21 18:05:56.691 (   0.436s) [         789C000]         StokesFlow3D.h:165   INFO| _Nx : 0004, _Ny : 0004, _Nz : 0004, Nt : 00001024, max_iters : 100000.
2023-10-21 18:05:56.691 (   0.436s) [         789C000]         StokesFlow3D.h:166   INFO| dx : 0.250000, dy : 0.250000, dz : 0.250000, dt : 0.00097656.
2023-10-21 18:05:57.051 (   0.796s) [         789C000]         StokesFlow3D.h:200   INFO| p sum_eh_squared = 1.19690998140538726324e-01
2023-10-21 18:05:57.051 (   0.796s) [         789C000]         StokesFlow3D.h:209   INFO| u sum_eh_squared = 4.92418810764946347780e-03
2023-10-21 18:05:57.051 (   0.796s) [         789C000]         StokesFlow3D.h:218   INFO| v sum_eh_squared = 4.92418810764946000835e-03
2023-10-21 18:05:57.051 (   0.796s) [         789C000]         StokesFlow3D.h:227   INFO| w sum_eh_squared = 4.92418810764946000835e-03
2023-10-21 18:05:57.051 (   0.796s) [         789C000]         StokesFlow3D.h:164   INFO| Lx : 1.0000, Ly : 1.0000, Lz : 1.0000, T  : 1.0000, rho : 1.0000, mu : 1.0000.
2023-10-21 18:05:57.051 (   0.796s) [         789C000]         StokesFlow3D.h:165   INFO| _Nx : 0008, _Ny : 0008, _Nz : 0008, Nt : 00000004, max_iters : 100000.
2023-10-21 18:05:57.051 (   0.796s) [         789C000]         StokesFlow3D.h:166   INFO| dx : 0.125000, dy : 0.125000, dz : 0.125000, dt : 0.25000000.
2023-10-21 18:05:57.080 (   0.825s) [         789C000]         StokesFlow3D.h:200   INFO| p sum_eh_squared = 2.97187101315638847483e-04
2023-10-21 18:05:57.080 (   0.825s) [         789C000]         StokesFlow3D.h:209   INFO| u sum_eh_squared = 5.68975666068891605676e-02
2023-10-21 18:05:57.080 (   0.825s) [         789C000]         StokesFlow3D.h:218   INFO| v sum_eh_squared = 5.68975666068891605676e-02
2023-10-21 18:05:57.080 (   0.825s) [         789C000]         StokesFlow3D.h:227   INFO| w sum_eh_squared = 5.68975666068891605676e-02
2023-10-21 18:05:57.080 (   0.825s) [         789C000]         StokesFlow3D.h:164   INFO| Lx : 1.0000, Ly : 1.0000, Lz : 1.0000, T  : 1.0000, rho : 1.0000, mu : 1.0000.
2023-10-21 18:05:57.080 (   0.825s) [         789C000]         StokesFlow3D.h:165   INFO| _Nx : 0008, _Ny : 0008, _Nz : 0008, Nt : 00000008, max_iters : 100000.
2023-10-21 18:05:57.080 (   0.825s) [         789C000]         StokesFlow3D.h:166   INFO| dx : 0.125000, dy : 0.125000, dz : 0.125000, dt : 0.12500000.
2023-10-21 18:05:57.131 (   0.877s) [         789C000]         StokesFlow3D.h:200   INFO| p sum_eh_squared = 6.93081983153310621573e-04
2023-10-21 18:05:57.132 (   0.877s) [         789C000]         StokesFlow3D.h:209   INFO| u sum_eh_squared = 2.84961934836517466474e-02
2023-10-21 18:05:57.132 (   0.877s) [         789C000]         StokesFlow3D.h:218   INFO| v sum_eh_squared = 2.84961934836517501168e-02
2023-10-21 18:05:57.132 (   0.877s) [         789C000]         StokesFlow3D.h:227   INFO| w sum_eh_squared = 2.84961934836517466474e-02
2023-10-21 18:05:57.132 (   0.877s) [         789C000]         StokesFlow3D.h:164   INFO| Lx : 1.0000, Ly : 1.0000, Lz : 1.0000, T  : 1.0000, rho : 1.0000, mu : 1.0000.
2023-10-21 18:05:57.132 (   0.877s) [         789C000]         StokesFlow3D.h:165   INFO| _Nx : 0008, _Ny : 0008, _Nz : 0008, Nt : 00000016, max_iters : 100000.
2023-10-21 18:05:57.132 (   0.877s) [         789C000]         StokesFlow3D.h:166   INFO| dx : 0.125000, dy : 0.125000, dz : 0.125000, dt : 0.06250000.
2023-10-21 18:05:57.218 (   0.963s) [         789C000]         StokesFlow3D.h:200   INFO| p sum_eh_squared = 1.46021551590248344424e-03
2023-10-21 18:05:57.218 (   0.963s) [         789C000]         StokesFlow3D.h:209   INFO| u sum_eh_squared = 1.43696881316688618302e-02
2023-10-21 18:05:57.218 (   0.963s) [         789C000]         StokesFlow3D.h:218   INFO| v sum_eh_squared = 1.43696881316688583607e-02
2023-10-21 18:05:57.218 (   0.963s) [         789C000]         StokesFlow3D.h:227   INFO| w sum_eh_squared = 1.43696881316688566260e-02
2023-10-21 18:05:57.219 (   0.964s) [         789C000]         StokesFlow3D.h:164   INFO| Lx : 1.0000, Ly : 1.0000, Lz : 1.0000, T  : 1.0000, rho : 1.0000, mu : 1.0000.
2023-10-21 18:05:57.219 (   0.964s) [         789C000]         StokesFlow3D.h:165   INFO| _Nx : 0008, _Ny : 0008, _Nz : 0008, Nt : 00000032, max_iters : 100000.
2023-10-21 18:05:57.219 (   0.964s) [         789C000]         StokesFlow3D.h:166   INFO| dx : 0.125000, dy : 0.125000, dz : 0.125000, dt : 0.03125000.
2023-10-21 18:05:57.361 (   1.106s) [         789C000]         StokesFlow3D.h:200   INFO| p sum_eh_squared = 2.85191977071846654648e-03
2023-10-21 18:05:57.361 (   1.107s) [         789C000]         StokesFlow3D.h:209   INFO| u sum_eh_squared = 7.42605658684001353892e-03
2023-10-21 18:05:57.362 (   1.107s) [         789C000]         StokesFlow3D.h:218   INFO| v sum_eh_squared = 7.42605658684001180420e-03
2023-10-21 18:05:57.362 (   1.107s) [         789C000]         StokesFlow3D.h:227   INFO| w sum_eh_squared = 7.42605658684001006947e-03
2023-10-21 18:05:57.362 (   1.107s) [         789C000]         StokesFlow3D.h:164   INFO| Lx : 1.0000, Ly : 1.0000, Lz : 1.0000, T  : 1.0000, rho : 1.0000, mu : 1.0000.
2023-10-21 18:05:57.362 (   1.107s) [         789C000]         StokesFlow3D.h:165   INFO| _Nx : 0008, _Ny : 0008, _Nz : 0008, Nt : 00000064, max_iters : 100000.
2023-10-21 18:05:57.362 (   1.107s) [         789C000]         StokesFlow3D.h:166   INFO| dx : 0.125000, dy : 0.125000, dz : 0.125000, dt : 0.01562500.
2023-10-21 18:05:57.589 (   1.334s) [         789C000]         StokesFlow3D.h:200   INFO| p sum_eh_squared = 5.23392419357972520783e-03
2023-10-21 18:05:57.589 (   1.334s) [         789C000]         StokesFlow3D.h:209   INFO| u sum_eh_squared = 4.14185414227926859687e-03
2023-10-21 18:05:57.589 (   1.334s) [         789C000]         StokesFlow3D.h:218   INFO| v sum_eh_squared = 4.14185414227926599479e-03
2023-10-21 18:05:57.589 (   1.334s) [         789C000]         StokesFlow3D.h:227   INFO| w sum_eh_squared = 4.14185414227926599479e-03
2023-10-21 18:05:57.590 (   1.335s) [         789C000]         StokesFlow3D.h:164   INFO| Lx : 1.0000, Ly : 1.0000, Lz : 1.0000, T  : 1.0000, rho : 1.0000, mu : 1.0000.
2023-10-21 18:05:57.590 (   1.335s) [         789C000]         StokesFlow3D.h:165   INFO| _Nx : 0008, _Ny : 0008, _Nz : 0008, Nt : 00000128, max_iters : 100000.
2023-10-21 18:05:57.590 (   1.335s) [         789C000]         StokesFlow3D.h:166   INFO| dx : 0.125000, dy : 0.125000, dz : 0.125000, dt : 0.00781250.
2023-10-21 18:05:57.955 (   1.700s) [         789C000]         StokesFlow3D.h:200   INFO| p sum_eh_squared = 9.06479403307398727552e-03
2023-10-21 18:05:57.955 (   1.700s) [         789C000]         StokesFlow3D.h:209   INFO| u sum_eh_squared = 2.71634247595917968296e-03
2023-10-21 18:05:57.955 (   1.700s) [         789C000]         StokesFlow3D.h:218   INFO| v sum_eh_squared = 2.71634247595918098400e-03
2023-10-21 18:05:57.955 (   1.700s) [         789C000]         StokesFlow3D.h:227   INFO| w sum_eh_squared = 2.71634247595918055032e-03
2023-10-21 18:05:57.956 (   1.701s) [         789C000]         StokesFlow3D.h:164   INFO| Lx : 1.0000, Ly : 1.0000, Lz : 1.0000, T  : 1.0000, rho : 1.0000, mu : 1.0000.
2023-10-21 18:05:57.956 (   1.701s) [         789C000]         StokesFlow3D.h:165   INFO| _Nx : 0008, _Ny : 0008, _Nz : 0008, Nt : 00000256, max_iters : 100000.
2023-10-21 18:05:57.956 (   1.701s) [         789C000]         StokesFlow3D.h:166   INFO| dx : 0.125000, dy : 0.125000, dz : 0.125000, dt : 0.00390625.
2023-10-21 18:05:58.568 (   2.314s) [         789C000]         StokesFlow3D.h:200   INFO| p sum_eh_squared = 1.48197563871197614954e-02
2023-10-21 18:05:58.569 (   2.314s) [         789C000]         StokesFlow3D.h:209   INFO| u sum_eh_squared = 2.14065740367769177105e-03
2023-10-21 18:05:58.569 (   2.314s) [         789C000]         StokesFlow3D.h:218   INFO| v sum_eh_squared = 2.14065740367769220473e-03
2023-10-21 18:05:58.569 (   2.314s) [         789C000]         StokesFlow3D.h:227   INFO| w sum_eh_squared = 2.14065740367769263841e-03
2023-10-21 18:05:58.569 (   2.314s) [         789C000]         StokesFlow3D.h:164   INFO| Lx : 1.0000, Ly : 1.0000, Lz : 1.0000, T  : 1.0000, rho : 1.0000, mu : 1.0000.
2023-10-21 18:05:58.569 (   2.314s) [         789C000]         StokesFlow3D.h:165   INFO| _Nx : 0008, _Ny : 0008, _Nz : 0008, Nt : 00000512, max_iters : 100000.
2023-10-21 18:05:58.569 (   2.314s) [         789C000]         StokesFlow3D.h:166   INFO| dx : 0.125000, dy : 0.125000, dz : 0.125000, dt : 0.00195312.
2023-10-21 18:05:59.634 (   3.379s) [         789C000]         StokesFlow3D.h:200   INFO| p sum_eh_squared = 2.27022172459299571845e-02
2023-10-21 18:05:59.634 (   3.379s) [         789C000]         StokesFlow3D.h:209   INFO| u sum_eh_squared = 1.88258071771218657431e-03
2023-10-21 18:05:59.635 (   3.380s) [         789C000]         StokesFlow3D.h:218   INFO| v sum_eh_squared = 1.88258071771218830903e-03
2023-10-21 18:05:59.635 (   3.380s) [         789C000]         StokesFlow3D.h:227   INFO| w sum_eh_squared = 1.88258071771219177848e-03
2023-10-21 18:05:59.635 (   3.380s) [         789C000]         StokesFlow3D.h:164   INFO| Lx : 1.0000, Ly : 1.0000, Lz : 1.0000, T  : 1.0000, rho : 1.0000, mu : 1.0000.
2023-10-21 18:05:59.635 (   3.380s) [         789C000]         StokesFlow3D.h:165   INFO| _Nx : 0008, _Ny : 0008, _Nz : 0008, Nt : 00001024, max_iters : 100000.
2023-10-21 18:05:59.635 (   3.380s) [         789C000]         StokesFlow3D.h:166   INFO| dx : 0.125000, dy : 0.125000, dz : 0.125000, dt : 0.00097656.
2023-10-21 18:06:01.531 (   5.276s) [         789C000]         StokesFlow3D.h:200   INFO| p sum_eh_squared = 3.21343983223467619115e-02
2023-10-21 18:06:01.531 (   5.276s) [         789C000]         StokesFlow3D.h:209   INFO| u sum_eh_squared = 1.74961621632814664715e-03
2023-10-21 18:06:01.531 (   5.276s) [         789C000]         StokesFlow3D.h:218   INFO| v sum_eh_squared = 1.74961621632814816503e-03
2023-10-21 18:06:01.531 (   5.276s) [         789C000]         StokesFlow3D.h:227   INFO| w sum_eh_squared = 1.74961621632814664715e-03
2023-10-21 18:06:01.532 (   5.277s) [         789C000]         StokesFlow3D.h:164   INFO| Lx : 1.0000, Ly : 1.0000, Lz : 1.0000, T  : 1.0000, rho : 1.0000, mu : 1.0000.
2023-10-21 18:06:01.532 (   5.277s) [         789C000]         StokesFlow3D.h:165   INFO| _Nx : 0016, _Ny : 0016, _Nz : 0016, Nt : 00000004, max_iters : 100000.
2023-10-21 18:06:01.532 (   5.277s) [         789C000]         StokesFlow3D.h:166   INFO| dx : 0.062500, dy : 0.062500, dz : 0.062500, dt : 0.25000000.
2023-10-21 18:06:02.150 (   5.895s) [         789C000]         StokesFlow3D.h:200   INFO| p sum_eh_squared = 1.12794142337579729159e-04
2023-10-21 18:06:02.150 (   5.895s) [         789C000]         StokesFlow3D.h:209   INFO| u sum_eh_squared = 5.94952030417834898390e-02
2023-10-21 18:06:02.151 (   5.896s) [         789C000]         StokesFlow3D.h:218   INFO| v sum_eh_squared = 5.94952030417834829001e-02
2023-10-21 18:06:02.151 (   5.896s) [         789C000]         StokesFlow3D.h:227   INFO| w sum_eh_squared = 5.94952030417834829001e-02
2023-10-21 18:06:02.152 (   5.897s) [         789C000]         StokesFlow3D.h:164   INFO| Lx : 1.0000, Ly : 1.0000, Lz : 1.0000, T  : 1.0000, rho : 1.0000, mu : 1.0000.
2023-10-21 18:06:02.152 (   5.897s) [         789C000]         StokesFlow3D.h:165   INFO| _Nx : 0016, _Ny : 0016, _Nz : 0016, Nt : 00000008, max_iters : 100000.
2023-10-21 18:06:02.152 (   5.897s) [         789C000]         StokesFlow3D.h:166   INFO| dx : 0.062500, dy : 0.062500, dz : 0.062500, dt : 0.12500000.
2023-10-21 18:06:03.223 (   6.968s) [         789C000]         StokesFlow3D.h:200   INFO| p sum_eh_squared = 1.05900783183685600871e-04
2023-10-21 18:06:03.224 (   6.969s) [         789C000]         StokesFlow3D.h:209   INFO| u sum_eh_squared = 2.97388765278064869102e-02
2023-10-21 18:06:03.224 (   6.969s) [         789C000]         StokesFlow3D.h:218   INFO| v sum_eh_squared = 2.97388765278064869102e-02
2023-10-21 18:06:03.224 (   6.969s) [         789C000]         StokesFlow3D.h:227   INFO| w sum_eh_squared = 2.97388765278064903796e-02
2023-10-21 18:06:03.225 (   6.970s) [         789C000]         StokesFlow3D.h:164   INFO| Lx : 1.0000, Ly : 1.0000, Lz : 1.0000, T  : 1.0000, rho : 1.0000, mu : 1.0000.
2023-10-21 18:06:03.225 (   6.970s) [         789C000]         StokesFlow3D.h:165   INFO| _Nx : 0016, _Ny : 0016, _Nz : 0016, Nt : 00000016, max_iters : 100000.
2023-10-21 18:06:03.225 (   6.970s) [         789C000]         StokesFlow3D.h:166   INFO| dx : 0.062500, dy : 0.062500, dz : 0.062500, dt : 0.06250000.
2023-10-21 18:06:04.935 (   8.680s) [         789C000]         StokesFlow3D.h:200   INFO| p sum_eh_squared = 2.38559671050893757358e-04
2023-10-21 18:06:04.935 (   8.680s) [         789C000]         StokesFlow3D.h:209   INFO| u sum_eh_squared = 1.48741158191381672715e-02
2023-10-21 18:06:04.936 (   8.681s) [         789C000]         StokesFlow3D.h:218   INFO| v sum_eh_squared = 1.48741158191381707410e-02
2023-10-21 18:06:04.936 (   8.681s) [         789C000]         StokesFlow3D.h:227   INFO| w sum_eh_squared = 1.48741158191381707410e-02
2023-10-21 18:06:04.937 (   8.682s) [         789C000]         StokesFlow3D.h:164   INFO| Lx : 1.0000, Ly : 1.0000, Lz : 1.0000, T  : 1.0000, rho : 1.0000, mu : 1.0000.
2023-10-21 18:06:04.937 (   8.682s) [         789C000]         StokesFlow3D.h:165   INFO| _Nx : 0016, _Ny : 0016, _Nz : 0016, Nt : 00000032, max_iters : 100000.
2023-10-21 18:06:04.937 (   8.682s) [         789C000]         StokesFlow3D.h:166   INFO| dx : 0.062500, dy : 0.062500, dz : 0.062500, dt : 0.03125000.
2023-10-21 18:06:07.655 (  11.400s) [         789C000]         StokesFlow3D.h:200   INFO| p sum_eh_squared = 5.28085788434643280120e-04
2023-10-21 18:06:07.655 (  11.400s) [         789C000]         StokesFlow3D.h:209   INFO| u sum_eh_squared = 7.45278619990127899619e-03
2023-10-21 18:06:07.656 (  11.401s) [         789C000]         StokesFlow3D.h:218   INFO| v sum_eh_squared = 7.45278619990128073092e-03
2023-10-21 18:06:07.656 (  11.401s) [         789C000]         StokesFlow3D.h:227   INFO| w sum_eh_squared = 7.45278619990127986356e-03
2023-10-21 18:06:07.657 (  11.402s) [         789C000]         StokesFlow3D.h:164   INFO| Lx : 1.0000, Ly : 1.0000, Lz : 1.0000, T  : 1.0000, rho : 1.0000, mu : 1.0000.
2023-10-21 18:06:07.657 (  11.402s) [         789C000]         StokesFlow3D.h:165   INFO| _Nx : 0016, _Ny : 0016, _Nz : 0016, Nt : 00000064, max_iters : 100000.
2023-10-21 18:06:07.657 (  11.402s) [         789C000]         StokesFlow3D.h:166   INFO| dx : 0.062500, dy : 0.062500, dz : 0.062500, dt : 0.01562500.
2023-10-21 18:06:11.668 (  15.413s) [         789C000]         StokesFlow3D.h:200   INFO| p sum_eh_squared = 1.01341164514978025572e-03
2023-10-21 18:06:11.669 (  15.414s) [         789C000]         StokesFlow3D.h:209   INFO| u sum_eh_squared = 3.75939677292771090231e-03
2023-10-21 18:06:11.669 (  15.414s) [         789C000]         StokesFlow3D.h:218   INFO| v sum_eh_squared = 3.75939677292770873390e-03
2023-10-21 18:06:11.669 (  15.415s) [         789C000]         StokesFlow3D.h:227   INFO| w sum_eh_squared = 3.75939677292770960126e-03
2023-10-21 18:06:11.670 (  15.415s) [         789C000]         StokesFlow3D.h:164   INFO| Lx : 1.0000, Ly : 1.0000, Lz : 1.0000, T  : 1.0000, rho : 1.0000, mu : 1.0000.
2023-10-21 18:06:11.670 (  15.415s) [         789C000]         StokesFlow3D.h:165   INFO| _Nx : 0016, _Ny : 0016, _Nz : 0016, Nt : 00000128, max_iters : 100000.
2023-10-21 18:06:11.670 (  15.415s) [         789C000]         StokesFlow3D.h:166   INFO| dx : 0.062500, dy : 0.062500, dz : 0.062500, dt : 0.00781250.
2023-10-21 18:06:17.574 (  21.319s) [         789C000]         StokesFlow3D.h:200   INFO| p sum_eh_squared = 1.75479860255731319463e-03
2023-10-21 18:06:17.574 (  21.319s) [         789C000]         StokesFlow3D.h:209   INFO| u sum_eh_squared = 1.94242938176381572946e-03
2023-10-21 18:06:17.575 (  21.320s) [         789C000]         StokesFlow3D.h:218   INFO| v sum_eh_squared = 1.94242938176381486209e-03
2023-10-21 18:06:17.575 (  21.320s) [         789C000]         StokesFlow3D.h:227   INFO| w sum_eh_squared = 1.94242938176381464525e-03
2023-10-21 18:06:17.576 (  21.321s) [         789C000]         StokesFlow3D.h:164   INFO| Lx : 1.0000, Ly : 1.0000, Lz : 1.0000, T  : 1.0000, rho : 1.0000, mu : 1.0000.
2023-10-21 18:06:17.576 (  21.321s) [         789C000]         StokesFlow3D.h:165   INFO| _Nx : 0016, _Ny : 0016, _Nz : 0016, Nt : 00000256, max_iters : 100000.
2023-10-21 18:06:17.576 (  21.321s) [         789C000]         StokesFlow3D.h:166   INFO| dx : 0.062500, dy : 0.062500, dz : 0.062500, dt : 0.00390625.
2023-10-21 18:06:26.523 (  30.268s) [         789C000]         StokesFlow3D.h:200   INFO| p sum_eh_squared = 2.82805438553090059128e-03
2023-10-21 18:06:26.524 (  30.269s) [         789C000]         StokesFlow3D.h:209   INFO| u sum_eh_squared = 1.08022296143477384775e-03
2023-10-21 18:06:26.524 (  30.269s) [         789C000]         StokesFlow3D.h:218   INFO| v sum_eh_squared = 1.08022296143477319723e-03
2023-10-21 18:06:26.525 (  30.270s) [         789C000]         StokesFlow3D.h:227   INFO| w sum_eh_squared = 1.08022296143477601615e-03
2023-10-21 18:06:26.525 (  30.270s) [         789C000]         StokesFlow3D.h:164   INFO| Lx : 1.0000, Ly : 1.0000, Lz : 1.0000, T  : 1.0000, rho : 1.0000, mu : 1.0000.
2023-10-21 18:06:26.525 (  30.270s) [         789C000]         StokesFlow3D.h:165   INFO| _Nx : 0016, _Ny : 0016, _Nz : 0016, Nt : 00000512, max_iters : 100000.
2023-10-21 18:06:26.525 (  30.270s) [         789C000]         StokesFlow3D.h:166   INFO| dx : 0.062500, dy : 0.062500, dz : 0.062500, dt : 0.00195312.
2023-10-21 18:06:40.779 (  44.524s) [         789C000]         StokesFlow3D.h:200   INFO| p sum_eh_squared = 4.33926423030472820791e-03
2023-10-21 18:06:40.780 (  44.525s) [         789C000]         StokesFlow3D.h:209   INFO| u sum_eh_squared = 7.03620915748175601752e-04
2023-10-21 18:06:40.780 (  44.525s) [         789C000]         StokesFlow3D.h:218   INFO| v sum_eh_squared = 7.03620915748175927013e-04
2023-10-21 18:06:40.780 (  44.525s) [         789C000]         StokesFlow3D.h:227   INFO| w sum_eh_squared = 7.03620915748175493332e-04
2023-10-21 18:06:40.781 (  44.526s) [         789C000]         StokesFlow3D.h:164   INFO| Lx : 1.0000, Ly : 1.0000, Lz : 1.0000, T  : 1.0000, rho : 1.0000, mu : 1.0000.
2023-10-21 18:06:40.781 (  44.526s) [         789C000]         StokesFlow3D.h:165   INFO| _Nx : 0016, _Ny : 0016, _Nz : 0016, Nt : 00001024, max_iters : 100000.
2023-10-21 18:06:40.781 (  44.526s) [         789C000]         StokesFlow3D.h:166   INFO| dx : 0.062500, dy : 0.062500, dz : 0.062500, dt : 0.00097656.
2023-10-21 18:07:04.802 (  68.547s) [         789C000]         StokesFlow3D.h:200   INFO| p sum_eh_squared = 6.40100726686586470049e-03
2023-10-21 18:07:04.802 (  68.547s) [         789C000]         StokesFlow3D.h:209   INFO| u sum_eh_squared = 5.54378776744805746787e-04
2023-10-21 18:07:04.802 (  68.548s) [         789C000]         StokesFlow3D.h:218   INFO| v sum_eh_squared = 5.54378776744804120484e-04
2023-10-21 18:07:04.803 (  68.548s) [         789C000]         StokesFlow3D.h:227   INFO| w sum_eh_squared = 5.54378776744804987846e-04
2023-10-21 18:07:04.807 (  68.552s) [         789C000]         StokesFlow3D.h:164   INFO| Lx : 1.0000, Ly : 1.0000, Lz : 1.0000, T  : 1.0000, rho : 1.0000, mu : 1.0000.
2023-10-21 18:07:04.807 (  68.552s) [         789C000]         StokesFlow3D.h:165   INFO| _Nx : 0032, _Ny : 0032, _Nz : 0032, Nt : 00000004, max_iters : 100000.
2023-10-21 18:07:04.807 (  68.552s) [         789C000]         StokesFlow3D.h:166   INFO| dx : 0.031250, dy : 0.031250, dz : 0.031250, dt : 0.25000000.
2023-10-21 18:07:18.288 (  82.033s) [         789C000]         StokesFlow3D.h:200   INFO| p sum_eh_squared = 1.49700937110300335339e-04
2023-10-21 18:07:18.291 (  82.036s) [         789C000]         StokesFlow3D.h:209   INFO| u sum_eh_squared = 6.06357494537010169311e-02
2023-10-21 18:07:18.293 (  82.038s) [         789C000]         StokesFlow3D.h:218   INFO| v sum_eh_squared = 6.06357494537010238700e-02
2023-10-21 18:07:18.295 (  82.040s) [         789C000]         StokesFlow3D.h:227   INFO| w sum_eh_squared = 6.06357494537010238700e-02
2023-10-21 18:07:18.297 (  82.042s) [         789C000]         StokesFlow3D.h:164   INFO| Lx : 1.0000, Ly : 1.0000, Lz : 1.0000, T  : 1.0000, rho : 1.0000, mu : 1.0000.
2023-10-21 18:07:18.297 (  82.042s) [         789C000]         StokesFlow3D.h:165   INFO| _Nx : 0032, _Ny : 0032, _Nz : 0032, Nt : 00000008, max_iters : 100000.
2023-10-21 18:07:18.297 (  82.042s) [         789C000]         StokesFlow3D.h:166   INFO| dx : 0.031250, dy : 0.031250, dz : 0.031250, dt : 0.12500000.
2023-10-21 18:07:41.608 ( 105.353s) [         789C000]         StokesFlow3D.h:200   INFO| p sum_eh_squared = 1.21610978606568917817e-04
2023-10-21 18:07:41.610 ( 105.355s) [         789C000]         StokesFlow3D.h:209   INFO| u sum_eh_squared = 3.03063308559087633831e-02
2023-10-21 18:07:41.612 ( 105.357s) [         789C000]         StokesFlow3D.h:218   INFO| v sum_eh_squared = 3.03063308559087633831e-02
2023-10-21 18:07:41.614 ( 105.359s) [         789C000]         StokesFlow3D.h:227   INFO| w sum_eh_squared = 3.03063308559087633831e-02
2023-10-21 18:07:41.616 ( 105.361s) [         789C000]         StokesFlow3D.h:164   INFO| Lx : 1.0000, Ly : 1.0000, Lz : 1.0000, T  : 1.0000, rho : 1.0000, mu : 1.0000.
2023-10-21 18:07:41.616 ( 105.361s) [         789C000]         StokesFlow3D.h:165   INFO| _Nx : 0032, _Ny : 0032, _Nz : 0032, Nt : 00000016, max_iters : 100000.
2023-10-21 18:07:41.616 ( 105.361s) [         789C000]         StokesFlow3D.h:166   INFO| dx : 0.031250, dy : 0.031250, dz : 0.031250, dt : 0.06250000.
2023-10-21 18:08:19.163 ( 142.908s) [         789C000]         StokesFlow3D.h:200   INFO| p sum_eh_squared = 8.91246715651485254530e-05
2023-10-21 18:08:19.165 ( 142.910s) [         789C000]         StokesFlow3D.h:209   INFO| u sum_eh_squared = 1.51503433786400604211e-02
2023-10-21 18:08:19.167 ( 142.912s) [         789C000]         StokesFlow3D.h:218   INFO| v sum_eh_squared = 1.51503433786400586863e-02
2023-10-21 18:08:19.169 ( 142.914s) [         789C000]         StokesFlow3D.h:227   INFO| w sum_eh_squared = 1.51503433786400604211e-02
2023-10-21 18:08:19.172 ( 142.917s) [         789C000]         StokesFlow3D.h:164   INFO| Lx : 1.0000, Ly : 1.0000, Lz : 1.0000, T  : 1.0000, rho : 1.0000, mu : 1.0000.
2023-10-21 18:08:19.172 ( 142.917s) [         789C000]         StokesFlow3D.h:165   INFO| _Nx : 0032, _Ny : 0032, _Nz : 0032, Nt : 00000032, max_iters : 100000.
2023-10-21 18:08:19.172 ( 142.917s) [         789C000]         StokesFlow3D.h:166   INFO| dx : 0.031250, dy : 0.031250, dz : 0.031250, dt : 0.03125000.
2023-10-21 18:09:14.849 ( 198.594s) [         789C000]         StokesFlow3D.h:200   INFO| p sum_eh_squared = 8.88809597949205998802e-05
2023-10-21 18:09:14.851 ( 198.597s) [         789C000]         StokesFlow3D.h:209   INFO| u sum_eh_squared = 7.57495709792095541640e-03
2023-10-21 18:09:14.854 ( 198.599s) [         789C000]         StokesFlow3D.h:218   INFO| v sum_eh_squared = 7.57495709792095628377e-03
2023-10-21 18:09:14.856 ( 198.601s) [         789C000]         StokesFlow3D.h:227   INFO| w sum_eh_squared = 7.57495709792095628377e-03
2023-10-21 18:09:14.858 ( 198.603s) [         789C000]         StokesFlow3D.h:164   INFO| Lx : 1.0000, Ly : 1.0000, Lz : 1.0000, T  : 1.0000, rho : 1.0000, mu : 1.0000.
2023-10-21 18:09:14.858 ( 198.603s) [         789C000]         StokesFlow3D.h:165   INFO| _Nx : 0032, _Ny : 0032, _Nz : 0032, Nt : 00000064, max_iters : 100000.
2023-10-21 18:09:14.858 ( 198.603s) [         789C000]         StokesFlow3D.h:166   INFO| dx : 0.031250, dy : 0.031250, dz : 0.031250, dt : 0.01562500.
2023-10-21 18:10:32.824 ( 276.569s) [         789C000]         StokesFlow3D.h:200   INFO| p sum_eh_squared = 1.86316656088678919223e-04
2023-10-21 18:10:32.826 ( 276.571s) [         789C000]         StokesFlow3D.h:209   INFO| u sum_eh_squared = 3.78894296807187180159e-03
2023-10-21 18:10:32.828 ( 276.573s) [         789C000]         StokesFlow3D.h:218   INFO| v sum_eh_squared = 3.78894296807187006687e-03
2023-10-21 18:10:32.830 ( 276.575s) [         789C000]         StokesFlow3D.h:227   INFO| w sum_eh_squared = 3.78894296807187006687e-03
2023-10-21 18:10:32.832 ( 276.577s) [         789C000]         StokesFlow3D.h:164   INFO| Lx : 1.0000, Ly : 1.0000, Lz : 1.0000, T  : 1.0000, rho : 1.0000, mu : 1.0000.
2023-10-21 18:10:32.832 ( 276.578s) [         789C000]         StokesFlow3D.h:165   INFO| _Nx : 0032, _Ny : 0032, _Nz : 0032, Nt : 00000128, max_iters : 100000.
2023-10-21 18:10:32.832 ( 276.578s) [         789C000]         StokesFlow3D.h:166   INFO| dx : 0.031250, dy : 0.031250, dz : 0.031250, dt : 0.00781250.
2023-10-21 18:12:19.802 ( 383.547s) [         789C000]         StokesFlow3D.h:200   INFO| p sum_eh_squared = 3.59962546459289569039e-04
2023-10-21 18:12:19.804 ( 383.549s) [         789C000]         StokesFlow3D.h:209   INFO| u sum_eh_squared = 1.89824660820684157879e-03
2023-10-21 18:12:19.806 ( 383.551s) [         789C000]         StokesFlow3D.h:218   INFO| v sum_eh_squared = 1.89824660820684266299e-03
2023-10-21 18:12:19.809 ( 383.554s) [         789C000]         StokesFlow3D.h:227   INFO| w sum_eh_squared = 1.89824660820684244615e-03
2023-10-21 18:12:19.811 ( 383.556s) [         789C000]         StokesFlow3D.h:164   INFO| Lx : 1.0000, Ly : 1.0000, Lz : 1.0000, T  : 1.0000, rho : 1.0000, mu : 1.0000.
2023-10-21 18:12:19.811 ( 383.556s) [         789C000]         StokesFlow3D.h:165   INFO| _Nx : 0032, _Ny : 0032, _Nz : 0032, Nt : 00000256, max_iters : 100000.
2023-10-21 18:12:19.811 ( 383.556s) [         789C000]         StokesFlow3D.h:166   INFO| dx : 0.031250, dy : 0.031250, dz : 0.031250, dt : 0.00390625.
2023-10-21 18:14:48.322 ( 532.067s) [         789C000]         StokesFlow3D.h:200   INFO| p sum_eh_squared = 5.99118907977689175298e-04
2023-10-21 18:14:48.324 ( 532.069s) [         789C000]         StokesFlow3D.h:209   INFO| u sum_eh_squared = 9.56836714174245537776e-04
2023-10-21 18:14:48.327 ( 532.072s) [         789C000]         StokesFlow3D.h:218   INFO| v sum_eh_squared = 9.56836714174245537776e-04
2023-10-21 18:14:48.329 ( 532.074s) [         789C000]         StokesFlow3D.h:227   INFO| w sum_eh_squared = 9.56836714174245754616e-04
2023-10-21 18:14:48.331 ( 532.076s) [         789C000]         StokesFlow3D.h:164   INFO| Lx : 1.0000, Ly : 1.0000, Lz : 1.0000, T  : 1.0000, rho : 1.0000, mu : 1.0000.
2023-10-21 18:14:48.331 ( 532.076s) [         789C000]         StokesFlow3D.h:165   INFO| _Nx : 0032, _Ny : 0032, _Nz : 0032, Nt : 00000512, max_iters : 100000.
2023-10-21 18:14:48.331 ( 532.076s) [         789C000]         StokesFlow3D.h:166   INFO| dx : 0.031250, dy : 0.031250, dz : 0.031250, dt : 0.00195312.
2023-10-21 18:18:23.390 ( 747.135s) [         789C000]         StokesFlow3D.h:200   INFO| p sum_eh_squared = 9.08659354925325053395e-04
2023-10-21 18:18:23.392 ( 747.137s) [         789C000]         StokesFlow3D.h:209   INFO| u sum_eh_squared = 4.92900404612650242485e-04
2023-10-21 18:18:23.394 ( 747.139s) [         789C000]         StokesFlow3D.h:218   INFO| v sum_eh_squared = 4.92900404612652085629e-04
2023-10-21 18:18:23.396 ( 747.141s) [         789C000]         StokesFlow3D.h:227   INFO| w sum_eh_squared = 4.92900404612651218267e-04
2023-10-21 18:18:23.398 ( 747.143s) [         789C000]         StokesFlow3D.h:164   INFO| Lx : 1.0000, Ly : 1.0000, Lz : 1.0000, T  : 1.0000, rho : 1.0000, mu : 1.0000.
2023-10-21 18:18:23.398 ( 747.143s) [         789C000]         StokesFlow3D.h:165   INFO| _Nx : 0032, _Ny : 0032, _Nz : 0032, Nt : 00001024, max_iters : 100000.
2023-10-21 18:18:23.398 ( 747.143s) [         789C000]         StokesFlow3D.h:166   INFO| dx : 0.031250, dy : 0.031250, dz : 0.031250, dt : 0.00097656.
2023-10-21 18:23:55.386 (1079.131s) [         789C000]         StokesFlow3D.h:200   INFO| p sum_eh_squared = 1.30401175153903748499e-03
2023-10-21 18:23:55.388 (1079.133s) [         789C000]         StokesFlow3D.h:209   INFO| u sum_eh_squared = 2.71700930707453425055e-04
2023-10-21 18:23:55.390 (1079.135s) [         789C000]         StokesFlow3D.h:218   INFO| v sum_eh_squared = 2.71700930707454726097e-04
2023-10-21 18:23:55.393 (1079.138s) [         789C000]         StokesFlow3D.h:227   INFO| w sum_eh_squared = 2.71700930707452991374e-04
2023-10-21 18:23:55.394 (1079.139s) [         789C000]        ScopeProfiler.h:106   INFO|    avg   |   min   |   max   |  total  | cnt | tag
2023-10-21 18:23:55.394 (1079.139s) [         789C000]        ScopeProfiler.h:114   INFO|    127670|      182|  3650704| 1043836K| 8176|solve_one_step
2023-10-21 18:23:55.394 (1079.139s) [         789C000]        ScopeProfiler.h:114   INFO|     62324|       88|   525272|  509561K| 8176|solve_p
2023-10-21 18:23:55.394 (1079.139s) [         789C000]        ScopeProfiler.h:114   INFO|     23543|       37|  1134256|  192492K| 8176|solve_w
2023-10-21 18:23:55.394 (1079.139s) [         789C000]        ScopeProfiler.h:114   INFO|     21571|       27|  1029049|  176366K| 8176|solve_v
2023-10-21 18:23:55.394 (1079.139s) [         789C000]        ScopeProfiler.h:114   INFO|     20119|       25|   961620|  164500K| 8176|solve_u
2023-10-21 18:23:55.394 (1079.139s) [         789C000]        ScopeProfiler.h:114   INFO|        24|        0|      157|   202741| 8176|correct_velocity
2023-10-21 18:23:55.394 (1079.139s) [         789C000]        ScopeProfiler.h:114   INFO|       692|      301|     3504|    24929|   36|StokesFlow
```


### TEST 4

LidDriven 为 StokeFlow 的派生类，用于实现三维顶盖驱动流算例。这里需要注意的是的压强边界条件为
$$
\frac{\partial p}{\partial n}=0,
$$
而不是
$$
p|_{\partial\Omega}=0.
$$
这两者的差别可通过流线图Fig. 1观察到。其中，左侧的是正确的结果。

![image-20231022200821490](https://githubimages.pengfeima.cn/images/202310222011374.png)

![image-20231022200956826](https://githubimages.pengfeima.cn/images/202310222011566.png)
**GIT提交**:
```
[result] 3D projection method for N-S equaitons (4)
```

**命令**
```
make 3D_projection_lid_driven -j64 && ./test/3D_projection_lid_driven
```

**计算结果**:

```
2023-10-22 20:16:51.264 (   0.004s) [        62821000]         StokesFlow3D.h:167   INFO| Lx : 1.0000, Ly : 1.0000, Lz : 1.0000, T  : 0.0100, rho : 1.0000, mu : 1.0000.
2023-10-22 20:16:51.264 (   0.004s) [        62821000]         StokesFlow3D.h:168   INFO| _Nx : 0032, _Ny : 0032, _Nz : 0032, Nt : 00000016, max_iters : 100000.
2023-10-22 20:16:51.264 (   0.004s) [        62821000]         StokesFlow3D.h:169   INFO| dx : 0.031250, dy : 0.031250, dz : 0.031250, dt : 0.00062500.
Saving result: 1
Saving result: 1
Saving result: 1
Error at the last step: 1.9415945647534576e-03
2023-10-22 20:17:42.220 (  50.960s) [        62821000]        ScopeProfiler.h:106   INFO|    avg   |   min   |   max   |  total  | cnt | tag
2023-10-22 20:17:42.220 (  50.961s) [        62821000]        ScopeProfiler.h:114   INFO|   3177686|  2842734|  4019871| 50842991|   16|solve_one_step
2023-10-22 20:17:42.220 (  50.961s) [        62821000]        ScopeProfiler.h:114   INFO|   3105654|  2770837|  3978865| 49690473|   16|solve_p
2023-10-22 20:17:42.220 (  50.961s) [        62821000]        ScopeProfiler.h:114   INFO|     25541|    23447|    37862|   408657|   16|solve_u
2023-10-22 20:17:42.220 (  50.961s) [        62821000]        ScopeProfiler.h:114   INFO|     24604|     1142|    27658|   393670|   16|solve_v
2023-10-22 20:17:42.220 (  50.961s) [        62821000]        ScopeProfiler.h:114   INFO|     21379|     1322|    23918|   342070|   16|solve_w
2023-10-22 20:17:42.220 (  50.961s) [        62821000]        ScopeProfiler.h:114   INFO|      4925|     4925|     4925|     4925|    1|StokesFlow
2023-10-22 20:17:42.220 (  50.961s) [        62821000]        ScopeProfiler.h:114   INFO|       129|      112|      159|     2075|   16|correct_velocity
```