# FieldScope implementation contract

日期：2026-07-30

## 时间与速度

项目公开接口只使用 clean-time：

```text
t=0  noise
t=1  encoded image
z_t=(1-t) epsilon + t z0
u_t=z0-epsilon
```

AuraFlow 原生接口使用 `1=noise, 0=image`，其 transformer 输出沿原生时间方向。
`AuraFlowBackend` 因此执行：

```text
native_time = 1 - clean_time
public_velocity = -native_transformer_output
```

该转换必须在真实模型 smoke 中用 scheduler 一致性检查再次验证。

## 张量契约

```text
image        [B,3,H,W] in [0,1]
latent       [B,C,h,w]
probe        [B,R,C,h,w]
response     [B,R,C,h,w]
state nodes  [B,P,3C+2]
response     [B,P,S*R*C*noise_views]
affinity     [B,P,P]
dense token  [B,P,D]
global token [B,D]
```

基础状态为 `[velocity; mismatch; endpoint_residual; ||velocity||; ||mismatch||]`。
响应签名按时间和 antithetic noise view 拼接；关系图先在每个时间/视图计算，再平均。

## 冻结边界

VAE、文本编码器、Flow Transformer 和原生输出层都设置为 eval 且不参与梯度。
缓存生成使用 `torch.no_grad()`；离线训练只更新 state/response 投影、Graph Tokenizer 与任务头。

## Baseline 防泄漏

`state` 及 `z0/zt/velocity/mismatch/endpoint` baseline 使用固定局部网格邻接，不读取响应图。
`response` 与 `full` 才使用 field-response adjacency。所有模式共享相同 tokenizer 深度与任务头。

