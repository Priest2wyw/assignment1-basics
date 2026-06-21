# AdamW resource Account

你这个题本质是 **资源会计题**，不是 optimizer 实现题。核心是把所有东西拆成：
$$
\text{memory} = \text{parameters} + \text{gradients} + \text{optimizer state} + \text{activations}
$$
所有 tensor 都是 float32，所以每个元素 **4 bytes**。

------

## 0. 符号说明

```shell
B=batch_size
T=context_length
L=num_layers
D=d_model
H=num_heads
d_h=D/H
V=vocab_size
```
题目里写的：
$$
d_{ff} = \frac{8}{3} d_{\text{model}}
$$
也就是：
$$
F = d_{ff} = \frac{8}{3}D
$$
你后面所有表达式都用这些符号。

------

## (a) Memory accounting

### 1. Parameters

参数内存就是：
$$
4 \times \#\text{parameters}
$$
你要数的是模型参数数量。可以按模块拆：

**Embedding:**

- token embedding: $V \times D$

**每层 Transformer block:**

- attention 里的 Q/K/V projection
- attention output projection
- SwiGLU 里的 $W_1, W_2, W_3$
- RMSNorm 参数

注意，Q/K/V 如果实现成三个 Linear，每个是：
$$
D \times D
$$
output projection 也是：
$$
D \times D
$$
SwiGLU 的三个矩阵形状要根据作业定义确认。一般是：
$$
W_1: D \to F
$$
所以数量级是：
$$
D F + D F + F D
$$
也就是三份 $DF$。

**最后 final RMSNorm:**

- 一份 $D$

**Output embedding / unembedding:**

要确认作业实现里 output embedding 是否和 input embedding tied。如果没有 tied，一般还有：
$$
D \times V
$$
#### 汇总

$$
param = L(4D^2+3DF+2D)+D+2VD
$$

------

### 2. Gradients

训练时，每个可学习参数通常都有同 shape 的 gradient。

所以梯度内存是：
$$
4 \times \#\text{parameters}
$$
也就是和 parameter memory 一样大。

------

### 3. Optimizer state

AdamW 对每个参数维护两个状态：
$$
m
$$
分别是一阶矩和二阶矩，shape 都和参数一样。

所以 optimizer state 是：
$$
2 \times 4 \times \#\text{parameters}
$$
也就是：
$$
8 \times \#\text{parameters bytes}
$$
如果只算 tensor，AdamW 的状态就是参数量的两倍。`t` 这种 step counter 很小，通常忽略。

------

### 4. Activations

activations 是这个题最容易乱的部分。题目已经限制了只考虑若干组件，所以你不要把所有中间变量都无限展开，只按它列的组件来。

每个 activation 的内存也是：
$$
4 \times \#\text{elements}
$$
你可以按每层 block 建表：

| 组件                        | shape 量级                     |
| --------------------------- | ------------------------------ |
| RMSNorm 输出                | $B \times T \times D$          |
| Q projection                | $B \times T \times D$          |
| K projection                | $B \times T \times D$          |
| V projection                | $B \times T \times D$          |
| $QK^\top$ attention scores  | $B \times H \times T \times T$ |
| softmax attention probs     | $B \times H \times T \times T$ |
| weighted sum of values      | $B \times T \times D$          |
| attention output projection | $B \times T \times D$          |
| SwiGLU $W_1$ branch         | $B \times T \times F$          |
| SwiGLU $W_3$ branch         | $B \times T \times F$          |
| SiLU gate output            | $B \times T \times F$          |
| elementwise product         | $B \times T \times F$          |
| $W_2$ output                | $B \times T \times D$          |

然后乘以层数 $L$。

以上结果合计为：
$$
L(8BTD+2BHT^2+4BTF)
$$


最后还要加：

| 组件                             | shape                                                        |
| -------------------------------- | ------------------------------------------------------------ |
| final RMSNorm                    | $B \times T \times D$                                        |
| output logits / output embedding | $B \times T \times V$                                        |
| cross entropy on logits          | 通常至少和 $B \times T$ 或 $B \times T \times V$ 相关，按题目要求口径确认 |

**$CE$按照$BTV$的参数量算**，那么最后的激活部分的参数量为：
$$ {CE activations}=L(8BTD+2BHT^2+4BTF)+BTD+2BTV$$


这里有两个大头：$BHT^2$来自 attention matrix, 以及：$BTV$来自 logits。

所以一个 sanity check 是：如果 $V$ 很大，logits activation 会非常显眼；如果 $T$ 很大，attention matrix 会显眼。

### 5. 内存总占用

$$\begin{aligned} \text{all-params} &= \text{parameters} + \text{gradients} + \text{optimizer state} + \text{activations}\\&= \text{parameters} +  \text{parameters} + 2 \text{parameters} + \text{activations} \\& =4\text{parameters}+\text{activations} \end{aligned}$$

上面算下来：$$param = L(4D^2+3DF+2D)+D+2VD$$

所以 

$$\begin{aligned} \text{all-params} &=4(L(4D^2+3DF+2D)+D+2VD)+ L(8BTD+2BHT^2+4BTF)+BTD+2BTV \end{aligned}$$

以上参数，在计算过程之中，如果全部以`float32`存储，则总共需要的内存为：

$$M_{all}= 4\text{all-parames} = 4(4P+A) $$

------

## (b) GPT-2 XL 的最大batch_size

根据(a)中得出的结果，我们可以拆分整体的参数量为：

$$M_{all} = 16(L(4D^2+3DF+2D)+D+2VD)+ 4B(L(8TD+2HT^2+4TF)+TD+2TV) $$

将
```shell
V=50257
T=1024
L=48
D=1600
H=25
F=4288 
```

计算出来结果为：


$$
a \cdot B + b
$$
```

```

然后最大 batch size 的逻辑是：
$$
aB + b \leq 80 \text{ GB}
$$
所以：
$$
B_{\max} = \left\lfloor \frac{80\text{GB} - b}{a} \right\rfloor
$$
```python
a = 4*(L*(8*T*D+2*H*T*T+4*T*F)+T*D+2*T*V) # 16373391360, 
b= 16*(L*(4*D*D+3*D*F+2*D)+D+2*V*D)       # 26247244800

max_resource = 80*2**(30)
max_batch_size = (max_resource-b)//a# 向下取整
print(f"a:{a:,}\nb:{b:,}")
print(max_batch_size)
#a:16,373,391,360
#b:26,247,244,800
#3 
```

------

## (c) AdamW 的计算量FLOPs

AdamW 的计算过程，主要跟参数量有关，我们关注的将会是以下格式：
$$
C \times \#\text{parameters}
$$
我们假设：加减乘除、开方都占用一次flops，下面开始计算涉及到参数运算的计算量：

1. weight decay update:2
2. first moment update:3
3. second moment update:4
4. denominator $\sqrt{v}+\epsilon$:2
5. moment-adjusted parameter update:3

按照以上每步的计算量来看，一次更新需要14次参数量的计算，`c=14`

所以总 FLOPs 是：
$$
14 \times \#\text{parameters}
$$
*`sqrt`和除法算不算一次flop?*

------

## (d) 训练时间估计



$$
\text{training step FLOPs}
=
\text{forward FLOPs}
+
\text{backward FLOPs}
+
\text{AdamW step FLOPs}
$$
大约是：
$$
3 \times \text{forward FLOPs}
+
\text{AdamW FLOPs}
$$
如果 AdamW optimizer FLOPs 相比 Transformer forward/backward 很小，有些分析会忽略；但题目既然前面问了 AdamW step FLOPs，最好把它写进公式里，并说明它通常不是主导项。

$$FLOPsforward≈L(8BTD2+4BT2D+6BTDF)+2BTDV+smaller elementwise terms$$

```python
V=50257
T=1024
L=48
D=1600
H=25
F=4288 
B = 1024 #batch_size
S = 400000 # train_step

parameter_count = L*(4*D*D + 3*D*F + 2*D) + D + 2*V*D
adamw_flops_per_step = 14 * parameter_count    # adamw flops

forward_flops_per_step = L*(8*B*T*D*D+4*B*T*T*D+6*B*T*D*F)+2*B*T*D*V # forward flops
all_flops_per_step = 3*forward_flops_per_step +adamw_flops_per_step

mfu = 0.5
head_process_per_s = 495*10**12 

seconds = all_flops_per_step*S/(head_process_per_s*mfu)

print(f"parameter_count is {parameter_count:,}, adamw_flops_per_step is {adamw_flops_per_step:,}, forward_flops_per_step is {forward_flops_per_step:,}")
print(f"need hours: {seconds/3600:,.3},days: {seconds/3600/24:,.3},")
# parameter_count is 1,640,452,800, adamw_flops_per_step is 22,966,339,200, forward_flops_per_step is 3,601,172,371,865,600
# need hours: 4.85e+03,days: 2.02e+02,
```

单卡大概需要202天。