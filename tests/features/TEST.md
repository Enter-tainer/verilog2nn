# verilog2nn 测试用例

## T1: NOT 门

- 输入：单个 NOT 门 `assign y = ~a;`
- 验证：穷举 {0,1}，NN 输出与真值表一致

## T2: 双输入门（AND / OR / XOR）

- 输入：分别测试 `assign y = a & b;`、`assign y = a | b;`、`assign y = a ^ b;`
- 验证：穷举 2-bit 输入（4 种），NN 输出与真值表一致

## T3: 组合表达式

- 输入：`assign y = (a & b) | (~c);`
- 验证：穷举 3-bit 输入（8 种），NN 输出一致

## T4: 多 bit 信号 — 4-bit 加法器

- 输入：4-bit ripple carry adder（`a[3:0] + b[3:0] = sum[4:0]`）
- 验证：穷举 256 种输入（4+4 bit），逐 bit 对比 sum 和 carry

## T5: 8-bit 比较器

- 输入：`assign gt = (a > b); assign eq = (a == b); assign lt = (a < b);`
- 验证：随机采样 1000 组 8-bit 输入，NN 输出与 iverilog 仿真一致

## T6: 多路选择器（MUX）

- 输入：4-to-1 MUX，2-bit sel + 4 个数据输入
- 验证：穷举所有输入组合

## T7: 模块实例化

- 输入：顶层模块实例化多个子模块（如两个全加器组成 2-bit 加法器）
- 验证：穷举验证，确认 Yosys 展平 + NN 输出正确

## T8: Yosys JSON 解析

- 验证：解析 Yosys 输出的 JSON 网表，正确识别所有门、连接和端口

## T9: 拓扑排序与分层

- 验证：DAG 拓扑排序正确，无环检测，同层门确实无依赖关系

## T10: safetensors 输出

- 验证：生成的 safetensors 文件可正常加载，权重形状与网络结构匹配

## T11: 端到端 — 复杂电路（MD5 round）

- 输入：MD5 单轮函数的 Verilog 实现
- 验证：随机采样 100 组 128-bit 输入，NN 输出与 iverilog 仿真 100% 一致
- 目的：验证可扩展性和性能
