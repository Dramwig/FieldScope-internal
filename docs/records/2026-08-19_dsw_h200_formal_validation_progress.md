# DSW H200 正式验证进度审计（2026-08-19）

状态：`active`。本记录只证明固定 revision 正式执行链的阶段进度，不构成方法有效性
结论，也不改变任何既有科学裁决。

## 观测边界

- 目标机：`dsw-h200`；最新里程碑主机时间：`2026-08-20T03:22:22Z`
  （北京时间 `2026-08-20T11:22:22+08:00`）。
- 项目根：`/mnt/omni_ssd/user_workspace/wangzixi/FieldScope`。
- 正式 worktree：`recovery/worktrees/formal-020c1de`；detached 且 clean。
- 分析 worktree：`recovery/worktrees/analysis-020c1de`；detached 且 clean。
- 固定 revision：`020c1de567edd88e0eda245fd085335ffe678f47`。
- 正式 worker PID：`107681`；`CUDA_VISIBLE_DEVICES=6`；显式 `PYTHONPATH` 指向固定
  worktree 的 `src`，并附加已审计的 RNG 兼容 shim。
- watchdog PID：`106745`；recovery PID：`92034`；analyzer PID：`105935`；三者均存活。
- analyzer 已直接核验 `CUDA_VISIBLE_DEVICES=6`，未来回放只允许使用物理 GPU 6。
- watchdog `status=active`，`changes_scientific_verdict=false`。

## ImageNet-100 主矩阵进度

正式 cache 的 train/val/test manifest 均完整，样本数分别为
`116455 / 12940 / 5000`。20 表示 × 3 seed 主矩阵的第一个表示
`random_feature_local` 已完成三个 seed：

| seed | 状态 | epoch | optimizer steps | sample exposures | test top-1 |
|---:|---|---:|---:|---:|---:|
| 4121 | `passed` | 90/90 | 81,900 | 10,480,950 | 0.01 |
| 7319 | `passed` | 90/90 | 81,900 | 10,480,950 | 0.01 |
| 104729 | `passed` | 90/90 | 81,900 | 10,480,950 | 0.01 |

三份训练 history 均从 epoch 1 连续到 epoch 90，revision 与报告一致，
`code_dirty=false`。三个 seed 的 `passed` 只表示训练/测试执行及报告合约通过；三者
test top-1 均为 0.01、top-5 均为 0.05，是必须保留的随机特征负对照，不得解释为
FieldScope 方法有效。第二个表示 `z0` 的三个 seed 也已全部完成 90/90、完整 5,000
样本测试和矩阵注册；seed 4121 / 7319 / 104729 的 test top-1 分别为 `0.1162`、
`0.0522`、`0.0750`，top-5 分别为 `0.3362`、`0.2022`、`0.2598`。这些数字只是
单个表示的三-seed 任务结果，不能单独构成方法有效性结论。第三个表示 `zt` 的 seed
4121 已完成 90/90、完整 5,000 样本测试和矩阵注册；test top-1 为 `0.01`、top-5
为 `0.05`，同样是必须保留的完整负结果，不能因其为负而省略。矩阵随后自动切换到
`zt` seed 7319。该 seed 随后完成 90/90、完整 5,000 样本测试和矩阵注册，test top-1
为 `0.0872`、top-5 为 `0.2728`；这只是单个表示/seed 的完整结果，不单独构成方法
有效性结论。`zt` seed 104729 随后也完成 90/90、完整 5,000 样本测试和矩阵注册；
test top-1/top-5 为 `0.01/0.05`，必须作为完整负结果保留。`trajectory` seed 4121 随后
完成 90/90、5,000 样本测试和矩阵注册；test top-1/top-5 为 `0.0558/0.2036`。随后
`trajectory` seed 7319 也完成 90/90、5,000 样本测试和矩阵注册；test top-1/top-5 为
`0.0904/0.295`。`trajectory` seed 104729 随后完成 90/90、完整 5,000 样本测试和矩阵
注册；test top-1/top-5 为 `0.0502/0.2112`。recovery 无重启地自动切换到 `velocity`
seed 4121；最新强审计覆盖到 epoch 40。

观测时 `matrix_report.json` 已注册十二个完成 run，状态为 `running`。直接读取当前活动
报告得到的总进度为：

- optimizer steps：`1019200 / 51013200`；
- training sample exposures：`130429600 / 725922600`；
- ImageNet-100 已完成 run：`12 / 60`，另有 `velocity` seed 4121 正在执行；
- 全四任务所需 run：`240`。

上述总进度来自 `2026-08-20T03:21:42Z` 对活动 cell epoch 40 的强一致内容审计；
watchdog 轮询可能略有延迟，不覆盖 report 的原子更新事实。

epoch 40 稳定快照进一步核验 history 精确连续、累计 `36400` optimizer steps 与
`4658200` sample exposures、四类 RNG state、固定 revision/tree、best/last checkpoint
和报告账本一致，`problems=[]`。report、last checkpoint、best checkpoint SHA-256 分别为
`6e752e225b3aba1122e74e48e482bae7aa998ffc8a9f1613f9f371dcbdc047f4`、
`4cb2a8e75d88fc9b30f70ae2a9a570ce854a286c637bd1e5e27e2ea5415eae14` 和
`5edc8b1e24162728b2b3df155f43a295b187af3203e7c17cdcfa95e44da56b5f`。审计避开了 writer
先原子替换 `last.pt`、再替换运行中 JSON 的短暂提交窗口；该 cell 仍为 `running`，未被
计入 9 个完成 run。

epoch 45 再次执行同一稳定双读，history、账本、四类 RNG、revision/tree 与 checkpoint
全部闭合，累计 `40950` steps、`5240475` exposures，`problems=[]`。本 epoch 刷新 best
validation top-1 为 `0.024497681607418855`；report、last、best SHA-256 分别为
`731a2016de47d961c43f150c6e73768ad17d943c49d5815932caa842f89e394c`、
`f7e988dfee5a56cede55ce5671426d4fe7dd7663328accdd878a28c58fafd288`、
`9657392a340099a3f2e455c9727949a67103292645ee6d2505a930af78192354`。该中间指标不构成
科学结论，cell 仍为 `running`。

epoch 50 稳定双读同样通过，累计 `45500` steps、`5822750` exposures，best epoch 更新为
46、best validation top-1 为 `0.025502318392581144`。report、last、best SHA-256 分别为
`6db8e53a75fe9e7eb764d59f5eca10ad1ebed94b8b594252c2bd5d263f9d0207`、
`99c0d2d2b4d11f5c2c4ccc5afa36576bad056c65357d9bca22f2fbea67a03595`、
`3e3186e629b3d10379d1af1d389c997bba3e0572edc64fe7e1c67733a5531d42`，
`problems=[]`；该 cell 仍为 `running`。

epoch 55 稳定双读通过，累计 `50050` steps、`6405025` exposures，best epoch 为 55、
best validation top-1 为 `0.03539412673879443`。report、last、best SHA-256 分别为
`104d19417759724bcd9e009be874a93172ee35fe24346d8878424b76bd5c9cbb`、
`53558e5acac7622661e5f8bdac3362a361d52f0d9fb259cb3c62ed9416fbaa59`、
`21d8ca97c53b311456ec7eccaf2dccb05031d88a2561fb022630902d89569832`，
`problems=[]`；该中间指标不构成科学结论。

epoch 60 稳定双读继续通过，累计 `54600` steps、`6987300` exposures，best epoch 为 60、
best validation top-1 为 `0.04559505409582689`。report、last、best SHA-256 分别为
`73733f520a1f1a6c75cc0d18b04250c607a052702ff5d0c3dcc42fb156103132`、
`d5c14ae61bef8a538d54e1acc9183b02ed9ecee9112b2c42a80ec7bb2695eb20`、
`08fd01a71b2c7d0ef08aa2b35119e4f3418f4f4e0ebafe2fc7dbbea67598e5a1`，
`problems=[]`；该 cell 仍为 `running`，中间指标不构成科学结论。

epoch 65 稳定双读通过，history 从 1 到 65 精确连续，累计 `59150` steps、`7569575`
exposures，best epoch 为 65、best validation top-1 为 `0.05123647604327666`。report、last、
best SHA-256 分别为
`1b67f65cc4d7413213f3d4de6bee6a150db0414b5cb2d9441d44bdcd4f230e8e`、
`cc4239bdb30ae275c94f42ed541505f70445e0c086123859b7e2cc6c98839c3d`、
`a63c854c6689a020e5e9d31f1d005ea64a0278d3824e41f08b7684a8dd6f6bdd`；模型、optimizer、
scheduler、四类 RNG、固定 revision/tree、batch、目标 epoch 与报告账本全部闭合，
`problems=[]`。该 cell 仍为 `running`，中间指标不构成科学结论。

随后 watchdog 注册 epoch 66，累计全任务主矩阵进度为 `797160` steps、`102014580`
exposures；readout、recovery、watchdog 与 analyzer 均存活，NYUv2 三 split 保持完整，既有
signal 负裁决与 VOC 无监督证据 SHA 未变。该轮只证明监督链继续观察到进度，没有替代
epoch 65 的强一致 checkpoint 审计，也不构成科学结论。

epoch 70 稳定双读通过，history 从 1 到 70 精确连续，累计 `63700` steps、`8151850`
exposures；best epoch 为 69、best validation top-1 为 `0.05378670788253478`。report、last、
best SHA-256 分别为
`998333fd09924d3c884ff2a813cf5f4e6b5b6e66ecf870aabc559c1373dfe477`、
`0cf87c21f8931b464d4343afaf5d0b225d2b366a1c0b8075d1a29fe5f48dcc6b`、
`302b1e8c15047887a0f3a7fb26a64da6bd221f0428be0b1137ec7e6fdde2eece`；模型、optimizer、
scheduler、四类 RNG、固定 revision/tree、batch 与目标 epoch 全部闭合，`problems=[]`。
该 cell 仍为 `running`，中间指标不构成科学结论。

epoch 75 稳定双读继续通过，history 从 1 到 75 精确连续，累计 `68250` steps、
`8734125` exposures；best epoch 更新为 75，best validation top-1 为
`0.05579598145285935`。report、last、best SHA-256 分别为
`7f54ff490f6111148969aa0bb3386bf1b6903c4801a56fed7e0334dd28ff9619`、
`3457a607f0147ecc9ddd8df3f531064279bea424226d211e7c787f2af068f0af`、
`207d0384e9f21a2562b526098cfc46b0bd784e26184d5d128399ededa460b161`；模型、optimizer、
scheduler、四类 RNG、固定 revision/tree、batch、目标 epoch、cache 身份和语义规范化后的
config 均闭合，`problems=[]`。运行中报告按实现合约将 `last_checkpoint_sha256` 保持为
`null`，实际 last 文件 SHA 已由稳定双读独立锚定；该 cell 仍为 `running`，中间指标不构成
科学结论。

epoch 80 稳定双读通过，history 从 1 到 80 精确连续，累计 `72800` steps、
`9316400` exposures；best epoch 更新为 80，best validation top-1 为
`0.05680061823802164`。report、last、best SHA-256 分别为
`3c2b6ca71591f9f63fd87d4dabb85fbce88df7cb237f5686f862e638c6a7025f`、
`01836d0ed223a85cd048c2a0bdd6ea17f83ddcada8668626792f4a388be397bb`、
`da5ff2773bc3ffd31441999a2dd7e96a714ef27c0083db577d02c262f89b5993`；全部训练状态、
账本、四类 RNG、固定 provenance、batch/epoch 和 cache/config 合约闭合，`problems=[]`。
该 cell 仍为 `running`，中间指标不构成科学结论。

epoch 85 稳定双读通过，history 从 1 到 85 精确连续，累计 `77350` steps、
`9898675` exposures；best epoch 为 83、best validation top-1 为
`0.05811437403400309`，epoch 85 validation top-1 为 `0.0571097372488408`。report、last、
best SHA-256 分别为 `f0aa315e7ba805eaee44c4dacfb1456c0237e9073f701bfdcf2416423c67bdc0`、
`0049f3b4c9b424684e15f1917f25969c156f2c7d61534f93d5ca0e7664e5f17b`、
`1226b2cf7a78ed76fa8cfc75f27b1dd6512123b89c6a30169d3e3db0d8de8496`；全部状态与合约
继续闭合，`problems=[]`。该 cell 仍为 `running`，中间指标不构成科学结论。

`trajectory` seed 4121 随后完成 epoch 90、测试与矩阵原子注册。终态审计核验 90 个连续
history epoch、`81900` steps、`10480950` exposures、模型/optimizer/scheduler、四类 RNG、
固定 provenance、best/last checkpoint 和 cache/config 合约，`problems=[]`。best epoch 为
83，best validation top-1 为 `0.05811437403400309`；完整 5,000 样本测试 top-1/top-5 为
`0.0558/0.2036`，loss 为 `4.162533335876465`。report、last、best、test 与更新后 matrix
SHA-256 分别为 `6cddacfd5aec0291375e1519e0235a2ce68632bdf4647b1935369dfe334286f0`、
`578922187d4de72b23152dcbac07ae2bc4c5ae29af16eb753a138b70edcf4034`、
`1226b2cf7a78ed76fa8cfc75f27b1dd6512123b89c6a30169d3e3db0d8de8496`、
`3b159eeae0eb1b2bf68c32321e154bcc9b28abdc3f491b260bd93bd45509b0e8` 和
`c979be52acf4d3a828477377dac244cf7c07362ab13fd5b728e6a124ab56a57e`。聚合 test 文件不含
逐样本 predictions，因此明确不替代最终 132 份正式回放；本结果也不单独构成科学结论。

recovery 无重启地自动接续 `trajectory` seed 7319。其 epoch 2 稳定双读审计通过，累计
`1820` steps、`232910` exposures，report/last/best SHA-256 为
`239d53db3206f701b7bf89db24d5f90d89e8d550e6b53a7246341edf24dd419e`、
`cf787d65b699ca9c9646acb93a781e9cba092294117fb08c7eb44b2600f90caf`、
`1d82ada32a9bc382e9b1f156fef2e699736b2ee6deefb58698826c33eb59fb41`，`problems=[]`。

该 cell 随后推进到 epoch 10；稳定双读核验连续 history、`9100` steps、`1164550`
exposures、best/last checkpoint、模型/optimizer/scheduler、四类 RNG、固定 provenance 和
cache/config 合约，`problems=[]`。best epoch 为 10，best validation top-1 为
`0.01846986089644513`；report、last、best SHA-256 分别为
`15b525255901bfd720832427ee5ddde759907396ee3fc2bf8c8f2e4229e2c9ac`、
`45295ad406849f73843302978a88992a7659f59a0de651a09ca99bef87800b9e`、
`e8d42f52ca2149eec770a78b0456503afe19fc609d43839386e93076071f0447`。该中间指标不构成
科学结论。

epoch 15 稳定双读继续通过，累计 `13650` steps、`1746825` exposures；best epoch 为 12，
best validation top-1 为 `0.020324574961360125`，epoch 15 validation top-1 为
`0.018624420401854715`。report、last、best SHA-256 分别为
`1d5bd60b2a1810f38e520560b416304e852bfd9136360c29a18b05c9c2f5d313`、
`9eb35823b4868058aefe8bcf4200b84849ae76971ca800d2e7d0040a7ab33928`、
`0d02d8dc5d2062e65259cde4bf218aaa6cc931802c725db4f18bf7f67be0c42e`；全部状态和合约
`problems=[]`，中间指标不构成科学结论。

epoch 20 稳定双读通过，累计 `18200` steps、`2329100` exposures；best epoch 与当前 epoch
均为 20，validation top-1 为 `0.02820710973724884`。report、last、best SHA-256 分别为
`caeedd02a0b5ea1d2ea4fe806c40297b49636e4d313a2b8f09171a1f531fa994`、
`6068aa23d8a2ccd8e4baec893c55b1867a8cb28b94cf2b9522cbf40196ed52e0`、
`88d06d87b4c41db20ad79d6544f17eab42d2d55faedbcc4491db78b00cb65aad`；全部训练状态、
RNG、固定 provenance 与账本合约 `problems=[]`，中间指标不构成科学结论。

epoch 25 稳定双读通过，累计 `22750` steps、`2911375` exposures；best epoch 与当前 epoch
均为 25，validation top-1 为 `0.040880989180834625`。report、last、best SHA-256 分别为
`002e93e8fce961ec8a2fd5c9543e65137ed73c32e58800e8cc856b2f79e40ad6`、
`10ecd83868fcf7e0fa80c1e807accd4e22d071859e159bc2bfdd2c199e796f5f`、
`abea2c19455bd902362b2f2313574f20835c7ce75cb29808123d3e431e3890f3`；全部训练状态、
RNG、固定 provenance 与账本合约 `problems=[]`，中间指标不构成科学结论。

epoch 30 稳定双读通过，累计 `27300` steps、`3493650` exposures；best epoch 为 28，best
validation top-1 为 `0.04868624420401855`，epoch 30 validation top-1 为
`0.047217928902627514`。report、last、best SHA-256 分别为
`9e0568488a896fb6a10c53815e45b8be271f5019e5e555c0c56682dad103974e`、
`18d49df0dfaa27aca3213ec05db525cd68c32b01ae93a899c20fcebfad394e1e`、
`67a123fb3a0e5d8126d03a066f817ed2fb13b3fbf29c6e213a158815e71fee65`；全部训练状态、
RNG、固定 provenance 与账本合约 `problems=[]`，中间指标不构成科学结论。

epoch 35 稳定双读继续通过，history 从 1 到 35 精确连续，累计 `31850` steps、
`4075925` exposures；best epoch 为 35，best validation top-1 为
`0.05316846986089645`。report、last、best SHA-256 分别为
`e44f81d264974236ed8bc04653f7b880d8b26c3943dc3e950cf93c18df929691`、
`83d1a785251397167e20e54136504106581178dd9a0351f80bbed4b670f05bd5`、
`56ad9021367883ce4f5b063291ffa3628c49749a475ba8531a280fa1e52720d1`；模型、
optimizer/scheduler、四类 RNG、固定 provenance、batch/epoch、cache 身份与配置账本
全部闭合，`problems=[]`。该 cell 仍为 `running`，中间指标不构成科学结论。

epoch 41 稳定双读再次通过，history 从 1 到 41 精确连续，累计 `37310` steps、
`4774655` exposures；best epoch 为 41，best/current validation top-1 均为
`0.06445131375579598`。report、last、best SHA-256 分别为
`f3b5c8ac7df1b99a52aeb2f39fec9a7fe574135074a096ab647cf38178730931`、
`5f611526c59cd84bd8e8d2816119ec1b1afd580b956a95399463054a2f9fd3b0`、
`e6dfe1ae76816069105033bf656978f528c0072fcfe0ed20c3b29277293a4ab0`；模型、
optimizer/scheduler、四类 RNG、固定 revision/clean tree、cache manifest、配置语义与
账本全部闭合，`problems=[]`。该 cell 仍为 `running`，中间指标不构成科学结论。

epoch 45 稳定双读继续通过，history 从 1 到 45 精确连续，累计 `40950` steps、
`5240475` exposures；best epoch 为 44，best validation top-1 为
`0.06460587326120557`，epoch 45 validation top-1 为 `0.06282843894899537`。report、
last、best SHA-256 分别为
`3a2d664e30bb9b0d1068977d5f8c33d6e3f5b2ed83114bfa90647fec1b0fec0f`、
`b6844850c9f99c8faadc54d3a161c6c1f0ad4122be2b9363ebc432d1b04f47de`、
`4755f7d8f20c719f8b193c64f631ec8dbe8753f90ef31179a1eba392c4881a9c`；训练状态、四类
RNG、固定 revision/clean tree、cache manifest、配置语义与账本全部闭合，
`problems=[]`。该 cell 仍为 `running`，中间指标不构成科学结论。

epoch 80 稳定双读继续通过，history 从 1 到 80 精确连续，累计 `72800` steps、
`9316400` exposures；best epoch 为 78，best validation top-1 为
`0.09667697063369397`，epoch 80 validation top-1 为 `0.09103554868624421`。report、
last、best SHA-256 分别为
`c2548ee612616e17f37c0b37d07cb71d1a7d2379b45eaefe6947c4f5e3710f69`、
`ee5d02ed1fad797e3135d608a88b44cb9e7501ae06220a500c7955b95715ed89`、
`5f03d6e4279b4783bc716398a184af66ccef97ce268c0b466909248286ed00a7`；训练状态、四类
RNG、固定 revision/clean tree、cache manifest、配置语义与账本全部闭合，
`problems=[]`。该 cell 仍为 `running`，中间指标不构成科学结论。

epoch 85 稳定双读继续通过，history 从 1 到 85 精确连续，累计 `77350` steps、
`9898675` exposures；best epoch 为 78，best validation top-1 为
`0.09667697063369397`，epoch 85 validation top-1 为 `0.09636785162287481`。report、
last、best SHA-256 分别为
`ed83621125020da65460f40d117f2ecd741bdb5ce333f181902f207cd4e4a92c`、
`d5a521123bacfe198e457b4c3fdee409ffb04fec8f9802e50567514fb5692dd8`、
`5f03d6e4279b4783bc716398a184af66ccef97ce268c0b466909248286ed00a7`；训练状态、四类
RNG、固定 revision/clean tree、cache manifest、配置语义与账本全部闭合，
`problems=[]`。该 cell 仍为 `running`，中间指标不构成科学结论。

`trajectory` seed 7319 随后完成 epoch 90、完整 5,000 样本 test 与矩阵原子注册。终态
审计核验 90 个连续 history epoch、`81900` steps、`10480950` exposures、模型/
optimizer/scheduler、四类 RNG、固定 provenance、best/last checkpoint、test 与 matrix
账本，`problems=[]`。best epoch 为 78，best validation top-1 为
`0.09667697063369397`；test loss/top-1/top-5 为
`3.9312483741760254/0.0904/0.295`。report、last、best、test 与更新后 matrix SHA-256
分别为 `ace2fd3cd5c496e361c7a69b865172332400e63809a7cc5fe4f996b5c4c96a11`、
`5fed27ea84270e29992c4b614a7752059e07f035bd39878d1995fd140fca4b27`、
`5f03d6e4279b4783bc716398a184af66ccef97ce268c0b466909248286ed00a7`、
`1ed53aef6573c30e08e3ba25402741a6f7ce6731f27e665f73b7c0ecdfdf6ad9` 和
`a756c2c5a94111e0946003bd69f51fe195ee7c29a9287ac31cbc3f4c75a2b3bf`。聚合 test 不含
逐样本 predictions，明确不替代最终 132 份正式回放。

recovery 无重启地自动接续 `trajectory` seed 104729。其 epoch 2 稳定双读审计通过，
累计 `1820` steps、`232910` exposures；best epoch 为 1，best/current validation
top-1 均为 `0.010046367851622875`。report、last、best SHA-256 分别为
`0c3e2e53fc76840338bcf0d168e5c96cda47cbe149b83df854f677c89ae00db0`、
`f1771ec3fea340a08d1f152088607388367d9ffb49a77cc41e1081c9dc32477c`、
`43389a78faf30876ba467be7046f0b0993b3b06320b69b09a5eecf3280b1e6b3`；训练状态和账本
闭合，`problems=[]`，中间指标不构成科学结论。

该 cell 随后推进到 epoch 6。稳定双读核验连续 history、`5460` steps、`698730`
exposures、模型/optimizer/scheduler、四类 RNG、固定 revision/tree、cache/config 与
best/last checkpoint，`problems=[]`。best epoch 仍为 1，best validation top-1 为
`0.010046367851622875`，epoch 6 validation top-1 为 `0.00865533230293663`；report、
last、best SHA-256 分别为
`f10a9b88d2ec4a7074a1ce60cf62ef235f07f07bff60f109d166122ef2cc492b`、
`684e405b9fe0f592afc3f7c2249f28b57ee40c12d9d3e7de9ed4f086517e5cda`、
`43389a78faf30876ba467be7046f0b0993b3b06320b69b09a5eecf3280b1e6b3`。该 cell 仍为
`running`，中间指标不构成科学结论。

epoch 10 稳定双读继续通过，history 从 1 到 10 精确连续，累计 `9100` steps、
`1164550` exposures；best epoch 仍为 1，best/current validation top-1 均为
`0.010046367851622875`。report、last、best SHA-256 分别为
`84b4681432499aba14c97949c9c664e24f5904c47ff6fa3d75b67b426dd9589b`、
`ef270f24575b9b31cb4da85ab2280a8f9e916bd2dc94adbdda95a54f6a3afa27`、
`43389a78faf30876ba467be7046f0b0993b3b06320b69b09a5eecf3280b1e6b3`。模型、
optimizer/scheduler、四类 RNG、固定 provenance、cache/config 与账本全部闭合，
`problems=[]`；该 cell 仍为 `running`，chance-level 中间结果完整保留且不构成科学结论。

epoch 15 稳定双读继续通过，history 从 1 到 15 精确连续，累计 `13650` steps、
`1746825` exposures；best epoch 仍为 1，best/current validation top-1 均为
`0.010046367851622875`。report、last、best SHA-256 分别为
`0504e3fd4c833562123e2bd92284b23014eb470f3361b8be2b77807243f4fded`、
`9ceade066c026e2230090ff86a56b8fae616c239cafb1bb37c112cd16590ac07`、
`43389a78faf30876ba467be7046f0b0993b3b06320b69b09a5eecf3280b1e6b3`；训练状态、
四类 RNG、固定 provenance、cache/config 与账本审计 `problems=[]`。该 cell 仍为
`running`，当前 chance-level 中间结果按原值保留且不构成科学结论。

计划中的 epoch 20 采样时 worker 已正常推进并原子提交 epoch 21，因此里程碑按实际稳定
状态记录。history 从 1 到 21 精确连续，累计 `19110` steps、`2445555` exposures；best
epoch 仍为 1，best/current validation top-1 均为 `0.010046367851622875`。report、last、
best SHA-256 分别为 `91b25fc86efc12e695fd79a1ba913d0c4b2e0e52a91ff28b7e1081bc92e49453`、
`682113053ec0d12a3ad5764d4d3655d653e38125e681234f19fd0c22fe4b296c`、
`43389a78faf30876ba467be7046f0b0993b3b06320b69b09a5eecf3280b1e6b3`；全部训练状态、
RNG、固定 provenance、cache/config 与账本审计 `problems=[]`。训练段补采 6 次 GPU 6
均为 `100%` SM，确认此前 epoch 边界的单次 `44%` 只是验证/写盘瞬时值；该 cell 仍为
`running`，chance-level 中间结果完整保留。

计划中的 epoch 25 读取后，稳定双读时 worker 已原子提交 epoch 26，因此按实际状态
记录。history 从 1 到 26 精确连续，累计 `23660` steps、`3027830` exposures；best epoch
仍为 1，best/current validation top-1 均为 `0.010046367851622875`。report、last、best
SHA-256 分别为 `fefe4e7435c0ef617968933462c4093e17b4f10ea34f94a5307cf48506662e8d`、
`c85ab1d7f30bec7c589248990893544ce9522a5fb71cb2b76082a26fb4955758`、
`43389a78faf30876ba467be7046f0b0993b3b06320b69b09a5eecf3280b1e6b3`；全部训练状态、
四类 RNG、固定 provenance、cache/config 与账本审计 `problems=[]`。该 cell 仍为
`running`，当前 chance-level 中间结果按原值保留。

epoch 30 稳定双读通过，history 从 1 到 30 精确连续，累计 `27300` steps、`3493650`
exposures；best epoch 仍为 1，best/current validation top-1 均为
`0.010046367851622875`。report、last、best SHA-256 分别为
`cce6a5d7c910fac7693fba601bc809eb67fba8f70c0fcaca5f2077749ad715e1`、
`0b0b41ab4c2c07a9124d6a874356b3f0b1bb8274f1b0a8fdaa0be2223168f038`、
`43389a78faf30876ba467be7046f0b0993b3b06320b69b09a5eecf3280b1e6b3`；模型、
optimizer/scheduler、四类 RNG、固定 provenance、cache/config 与账本全部闭合，
`problems=[]`。该 cell 仍为 `running`，chance-level 中间结果完整保留且不构成科学结论。

epoch 35 稳定双读继续通过，history 从 1 到 35 精确连续，累计 `31850` steps、
`4075925` exposures；best epoch 仍为 1，best/current validation top-1 均为
`0.010046367851622875`。report、last、best SHA-256 分别为
`083682ee799c5fb09a9a45b424c81b2a27b670dde53a68c6f8cdc8edf7609c9b`、
`568b2b41d612bda51efc3623fc695a0152a5b9d84c47f2eafd8751318a8d7272`、
`43389a78faf30876ba467be7046f0b0993b3b06320b69b09a5eecf3280b1e6b3`；全部训练状态、
四类 RNG、固定 provenance、cache/config 与账本审计 `problems=[]`。该 cell 仍为
`running`，chance-level 中间结果按原值保留。

epoch 40 稳定双读通过，history 从 1 到 40 精确连续，累计 `36400` steps、`4658200`
exposures；best epoch 更新为 40，best/current validation top-1 均为
`0.026816074188562595`。report、last、best SHA-256 分别为
`c4ee82350708ab05ec73bf68f54a0a5659471b570d26d01c9cc2f66d7cfb4be2`、
`481dfc9e12e17fc957a109fc62ee44cb8f1c718d88e137917ce433c2ddecc6ee`、
`c4e488dcaac73c6dff203a2b4c30e33f66f4c3700fd806edb42cfc37553eb5cd`；全部训练状态、
四类 RNG、固定 provenance、cache/config 与账本审计 `problems=[]`。该 cell 仍为
`running`；该单 seed 中间指标按原值保留，不单独构成方法有效性结论。

epoch 45 稳定双读继续通过，history 从 1 到 45 精确连续，累计 `40950` steps、
`5240475` exposures；best epoch 为 43，best validation top-1 为
`0.0348531684698609`，epoch 45 validation top-1 为 `0.03330757341576507`。report、last、
best SHA-256 分别为 `80084e565e5d8936de17f731d16348f9933c393bfafd1e84ab981908c1fe9146`、
`bf5f480a903022022aa3bdfbf320ea367ed1764fe3b18e1bacbde0bb277fc4ac`、
`c282bfc8a67544064dc639ff1f3da276ce15c2e1eee10e0a2876d5e17b728e0e`；全部训练状态、
四类 RNG、固定 provenance、cache/config 与账本审计 `problems=[]`。该 cell 仍为
`running`；单 seed 中间指标完整保留，不单独构成方法有效性结论。

epoch 50 稳定双读继续通过，history 从 1 到 50 精确连续，累计 `45500` steps、
`5822750` exposures；best epoch 为 49，best validation top-1 为
`0.039335394126738796`，epoch 50 validation top-1 为 `0.038717156105100466`。report、
last、best SHA-256 分别为 `d4188211fa748c6c7e0f78d511c3a0ef1af8570e55dd39564747165489298b3f`、
`450811dabe1cd10ff6b42e9acabf8ad3c2d5e265783e4c4b739c555471df0893`、
`52c6afff6c2cd045738ce4b64aff8284a9b9cf72aa471f4929608a92f2847d83`；全部训练状态、
四类 RNG、固定 provenance、cache/config 与账本审计 `problems=[]`。该 cell 仍为
`running`；单 seed 中间指标完整保留，不单独构成方法有效性结论。

epoch 55 稳定双读继续通过，history 从 1 到 55 精确连续，累计 `50050` steps、
`6405025` exposures；best epoch 为 54，best validation top-1 为
`0.04559505409582689`，epoch 55 validation top-1 为 `0.04544049459041731`。report、
last、best SHA-256 分别为 `fb30443f17caf6f259d88cad2e900efb7e61ca842b6e3ff2513905ee13dc7667`、
`c1747001895ef69befef01746a2134b9c298ad1ea7e9f996e07c05bca9c8c815`、
`82c509e9df7caa656afc484a0579871401798fd49320fd10eac042493ebb576a`；全部训练状态、
四类 RNG、固定 provenance、cache/config 与账本审计 `problems=[]`。该 cell 仍为
`running`；单 seed 中间指标完整保留，不单独构成方法有效性结论。

epoch 61 稳定双读继续通过，history 从 1 到 61 精确连续，累计 `55510` steps、
`7103755` exposures；best epoch 为 59，best validation top-1 与 epoch 61 validation
top-1 均为 `0.04992272024729521`。report、last、best SHA-256 分别为
`be6b181e6577aa12386590c38d783ff6193987708af8da0edac0bda8f340b8c7`、
`e4a738642ff53a8c5a66148e0365adc1ed58dfe8aa206001d9ed5263fb2cb9e4`、
`9a1992b6cfa5c78b7178b60c5f418f5fb1c06660b6e78fb70700e5b9dfe88c1d`；模型、AdamW
step-state、scheduler、四类 RNG、固定 revision/clean tree、cache manifest、配置语义与
账本全部闭合，`problems=[]`。该 cell 仍为 `running`；单 seed 中间指标完整保留，
不单独构成方法有效性结论。

epoch 65 稳定双读继续通过，history 从 1 到 65 精确连续，累计 `59150` steps、
`7569575` exposures；best epoch 为 64，best validation top-1 为
`0.05131375579598145`，epoch 65 validation top-1 为 `0.05123647604327666`。report、
last、best SHA-256 分别为 `8dc933ee5d13193addaa32a24a1dfd34d54a69c55552bfc3700f02d96d1bbb2d`、
`df4abc1e2cac8befef1f195b5eb15245f8b7b9795970f94636665a1ca1de706b`、
`02e150611a6a0f27666243c4d8f508648480905fdc2a5dbfb84bdd50bf204ce4`；模型、AdamW
step-state、scheduler、四类 RNG、固定 revision/clean tree、cache manifest、配置语义与
账本全部闭合，`problems=[]`。该 cell 仍为 `running`；单 seed 中间指标完整保留，
不单独构成方法有效性结论。

等待 epoch 70 原子提交后执行的稳定双读最终落在 epoch 72：history 从 1 到 72 精确
连续，因此完整覆盖 epoch 70 里程碑；累计 `65520` steps、`8384760` exposures。best
epoch 为 70，best validation top-1 为 `0.0571870170015456`，epoch 72 validation top-1
为 `0.05540958268933539`。report、last、best SHA-256 分别为
`eb19cf5b9513af77a9d850cf8342ae74ec5468fc5af51c4b1b8906ea79c786f7`、
`9a2385e682545bed92a53495bf92a880f290e30d38b4fcb430d3c86da34589e6`、
`e1154263a07b02db9cf99782cb7d7b454f2761169de869c6bf68055f6bb7e1e7`；模型、AdamW
step-state、scheduler、四类 RNG、固定 revision/clean tree、cache manifest、配置语义与
账本全部闭合，`problems=[]`。该 cell 仍为 `running`；这些中间指标完整保留但不单独
构成方法有效性结论。

epoch 75 稳定双读继续通过，history 从 1 到 75 精确连续，累计 `68250` steps、
`8734125` exposures；best epoch 为 70，best validation top-1 为
`0.0571870170015456`，epoch 75 validation top-1 为 `0.05564142194744977`。report、
last、best SHA-256 分别为 `7786f4d9bc4d409690fc8845414ffca88c084349031bc143c9d97d299e1e6fb3`、
`2ce5fd7b8e63b25d5d0d2585b4ea18d2c9ca27611a87e70bf16f8b24ec58b493`、
`e1154263a07b02db9cf99782cb7d7b454f2761169de869c6bf68055f6bb7e1e7`；模型、AdamW
step-state、scheduler、四类 RNG、固定 revision/clean tree、cache manifest、配置语义与
账本全部闭合，`problems=[]`。该 cell 仍为 `running`；中间指标不构成科学结论。

`trajectory` seed 104729 随后完成 epoch 90、测试与矩阵第 12 run 原子注册。终态稳定
双读核验 90 个连续 history epoch、`81900` steps、`10480950` exposures、模型、AdamW、
scheduler、四类 RNG、固定 provenance、cache/config、best/last checkpoint、完整 test
和 matrix 账本，`problems=[]`。best epoch 为 84，best validation top-1 为
`0.05795981452859351`；完整 5,000 样本 test top-1/top-5 为 `0.0502/0.2112`，loss 为
`4.1640590599060054`。report、last、best、test、matrix SHA-256 分别为
`55b487463ede28792c6967714432c658c1e648675f39a2690bb3460160356a66`、
`7fe2b621cfc65dcfbccf5dc3978705c0dd8e70df24ade68184d525f008c53712`、
`46989025fa5c212f6a8605429a2dacea89ac71f5c9ce2f649e6197dc7c768491`、
`95aae9acb670ed70af1c8a9563ac0d772d23f02fa81c73e768ad3e010d9776b7` 和
`d6dfa5f2e50b8e96f8205c4a01162078113e2562afec41fb9fa312e243a350aa`。聚合 test 不含
逐样本记录，因此不替代最终 132 份正式回放；该单 cell 结果也不单独构成科学结论。

recovery 随后自动启动 `velocity` seed 4121。epoch 1 稳定双读核验 `910` steps、
`116455` exposures、训练状态/RNG、固定 provenance 与 cache/config，`problems=[]`；
validation top-1 为 `0.010046367851622875`。report、last、best SHA-256 分别为
`ac6a83ae6f15bbf015a9d2a67cc6a6802454cf8544ca812cb4c5c7e74269ae03`、
`f963e275ed6c2fe295e9c3e50a39596564bd687acf86baa55599eb5e32ba97f2` 和
`aac5296880e4a65e908efc7919fe392716088ac628eac18e394e96b6c3c0aab4`。该 epoch 1
中间指标按原值保留，不构成科学结论。

`2026-08-20T02:27:33Z` 对该 cell 再次稳定双读时已提交 epoch 11。history 从 1 到 11
精确连续，累计 `10010` steps、`1281005` exposures；best epoch 为 10、best validation
top-1 为 `0.02743431221020093`，epoch 11 validation top-1 为
`0.02457496136012365`。report、last、best SHA-256 分别为
`e356acaec92177883f539f41b35d2706547986b141256153fa20064485ff770c`、
`6e33351f2437141c6ef91d16e5eb675cdbc0103b192e146aa6276e78767f7ca4` 和
`209794575112c48bdd816ae2f038b0cf0f0ca3f8306294991c36957380fc81d0`。AdamW step-state、
scheduler、四类 RNG、固定 revision/clean tree、cache manifest、配置和账本全部闭合，
`problems=[]`；该 cell 仍为 `running`，中间指标不构成科学结论。

`2026-08-20T02:36:33Z` 的后续稳定双读覆盖 epoch 17。history 从 1 到 17 精确连续，
累计 `15470` steps、`1979735` exposures；best epoch 与当前 epoch 均为 17，validation
top-1 为 `0.02820710973724884`。report、last、best SHA-256 分别为
`156883435a312d87eee3432b0b8415dce3f8279ee24425e6a801d2d3f79a95e9`、
`2cf78f4b34b1476539402f85337a03bb5f4fc73afb11cc64cd8af1337aaabfc4` 和
`86e286a74fffb2f6b0970feacf9e0a98d1152824effe63247790533dd47438c6`。AdamW、scheduler、
四类 RNG、固定 provenance、cache/config 与账本继续闭合，`problems=[]`；该 cell 仍为
`running`，中间指标不构成科学结论。

`2026-08-20T02:42:41Z` 的 epoch 20 稳定双读继续通过，累计 `18200` steps、
`2329100` exposures；best epoch 与当前 epoch 均为 20，validation top-1 为
`0.04234930448222566`。report、last、best SHA-256 分别为
`20c3b3aa9186739b47251b1e32ca97fd1be495b4f66fb2c472aebd8bfc60a834`、
`1f8199cb80abf9546ee8d0b15c581d92d01f1f788039d1e34b17702fb97d6525` 和
`b3a55adfd08f809df3503a50d6615137cf63ffb61ccf61b7610854d3dc627a37`。全部训练状态、
RNG、固定 provenance、cache/config 和账本闭合，`problems=[]`；该 cell 仍为
`running`，中间指标不构成科学结论。

`2026-08-20T02:52:05Z` 的 epoch 25 稳定双读通过，累计 `22750` steps、
`2911375` exposures；best epoch 与当前 epoch 均为 25，validation top-1 为
`0.05479134466769706`。report、last、best SHA-256 分别为
`47b285fd9a50036d03c87b332448633fb9fe377496ac6a3f1bc56db953638faf`、
`79de965ed40bda0ea494ac3aeb2476d3fc363477cf9de9cdbf2380ee83ebc14c` 和
`c71a817031d577a5fa22bc16cf2b55a7398000fef04a6856397c9dc79d7b640d`。全部训练状态、
RNG、固定 provenance、cache/config 和账本闭合，`problems=[]`；该 cell 仍为
`running`，中间指标不构成科学结论。

`2026-08-20T03:02:13Z` 的 epoch 30 稳定双读通过，累计 `27300` steps、
`3493650` exposures；best epoch 与当前 epoch 均为 30，validation top-1 为
`0.06522411128284389`。report、last、best SHA-256 分别为
`03252f3392cbbe3e6c0446a537db36e08e66e83f066bfaa0d4ac530d1fa7c243`、
`1f2f99aa3e46fcd5fc4aa6ac68b55cee2a29142de9c23d01a26c68eb1df7c257` 和
`7cba5f4b1bd19e9285323fe5d2df15a45155f2266d7bd81c2e76709c7b3be052`。全部训练状态、
RNG、固定 provenance、cache/config 和账本闭合，`problems=[]`；该 cell 仍为
`running`，中间指标不构成科学结论。

`2026-08-20T03:11:53Z` 的 epoch 35 稳定双读继续通过，累计 `31850` steps、
`4075925` exposures；best epoch 为 34、best validation top-1 为
`0.0768160741885626`，epoch 35 validation top-1 为 `0.07472952086553324`。report、
last、best SHA-256 分别为 `71894b38e5c5987d6f8e7f9edbc22d058eda3b114bcd27c3e0586380261dd79a`、
`74e3c26092e2b9182346e98710e58f13d0c53513013c6af304cd342c4db2fc88` 和
`a79e39d1230fbf73eb9ffc29f8969ee472d9be3c3684bd1be48eac63af432b7a`。全部训练状态、
RNG、固定 provenance、cache/config 和账本闭合，`problems=[]`；该 cell 仍为
`running`，中间指标不构成科学结论。

`2026-08-20T03:21:42Z` 的 epoch 40 稳定双读通过，累计 `36400` steps、
`4658200` exposures；best epoch 为 38、best validation top-1 为
`0.08539412673879443`，epoch 40 validation top-1 为 `0.08021638330757341`。report、
last、best SHA-256 分别为 `e2f7839306386ccccbb0a04b136fe9c97a5f1658d68dc9daa7f0089937f60f59`、
`48a9395f823446dba41032a0b418723d9072374d7cba62dd4afa855195a1543b` 和
`51c313a2ef33759b1cd76fa7688f0bf178f72c44c0a27475eb6f4011923d7262`。全部训练状态、
RNG、固定 provenance、cache/config 和账本闭合，`problems=[]`；该 cell 仍为
`running`，中间指标不构成科学结论。

epoch 80 稳定双读继续通过，history 从 1 到 80 精确连续，累计 `72800` steps、
`9316400` exposures；best epoch 与当前 epoch 均为 80，validation top-1 为
`0.057882534775888714`。report、last、best SHA-256 分别为
`5ab32097ce1855d60897aa3777e235dbc043d5d7a94b0ff2fc06da74d2a91e0b`、
`ab07062d09760e502cf710d1753d75477b17fb1448ed502255463415a6ce9af9`、
`c88c6154fa2c20c17be0e32214ba91f0c8c65f033a54ff368b3174aadf38377c`；模型、AdamW
step-state、scheduler、四类 RNG、固定 revision/clean tree、cache manifest、配置语义与
账本全部闭合，`problems=[]`。该 cell 仍为 `running`；中间指标不构成科学结论。

epoch 85 稳定双读继续通过，history 从 1 到 85 精确连续，累计 `77350` steps、
`9898675` exposures；best epoch 为 84，best validation top-1 为
`0.05795981452859351`，epoch 85 validation top-1 为 `0.05734157650695518`。report、
last、best SHA-256 分别为 `933fa0b7eaaf412b8fcb38d2fe5cd388fc5be4a130a7d7c5a798d61b6ca0efd1`、
`37f5ba65c0aaff4a481bb8dc47b59fbef8bfc13e277b051bacdaa3b8ce36abac`、
`46989025fa5c212f6a8605429a2dacea89ac71f5c9ce2f649e6197dc7c768491`；模型、AdamW
step-state、scheduler、四类 RNG、固定 revision/clean tree、cache manifest、配置语义与
账本全部闭合，`problems=[]`。该 cell 仍为 `running`；中间指标不构成科学结论。

epoch 50 稳定双读继续通过，history 从 1 到 50 精确连续，累计 `45500` steps、
`5822750` exposures；best epoch 为 47，best validation top-1 为
`0.07364760432766615`，epoch 50 validation top-1 为 `0.06862442040185471`。report、
last、best SHA-256 分别为
`a6a39919b20e33e5d6b0d72d79565ef91ff63ea4d4d35828bfc15fce90559060`、
`b3a7a89cc0ab6bd5aef1f25ab2037ac8743d925ad325fda4adf1a967d8ea27bd`、
`33b08634452c4346a9a32772b1683026dcdd7e339863045ef2fa2a2b6ffb1178`；训练状态、四类
RNG、固定 revision/clean tree、cache manifest、配置语义与账本全部闭合，
`problems=[]`。该 cell 仍为 `running`，中间指标不构成科学结论。

epoch 60 稳定双读继续通过，history 从 1 到 60 精确连续，累计 `54600` steps、
`6987300` exposures；best epoch 与当前 epoch 均为 60，validation top-1 为
`0.08137557959814529`。report、last、best SHA-256 分别为
`e61d9c5c1180cc0beface0e7c2469ae4434f9901797a9332e9864e9621133a9a`、
`748202d54453ef2d779434ee8b337b25d47e7351a1608edacc536d05f245e2f9`、
`f077631136ef3c0ab2b28253d0bcc9ee51fde1e440ee87b738da47e22efdf5a7`；训练状态、四类
RNG、固定 revision/clean tree、cache manifest、配置语义与账本全部闭合，
`problems=[]`。该 cell 仍为 `running`，中间指标不构成科学结论。

epoch 65 稳定双读继续通过，history 从 1 到 65 精确连续，累计 `59150` steps、
`7569575` exposures；best epoch 为 63，best validation top-1 为
`0.09034003091190108`，epoch 65 validation top-1 为 `0.08887171561051005`。report、
last、best SHA-256 分别为
`b650db416867a447fd4595474286fbeab9080eea80f713a7a08f33b78a9d4c28`、
`2d71e4dad7c91981e656a0ce20a20cdb2536d07e2e042933695bdb5bcf2ffb2e`、
`bbc4b7eec35cbf0c639470352a22a44c3d6260aba316f6e5f3e8bb61f40ea75f`；训练状态、四类
RNG、固定 revision/clean tree、cache manifest、配置语义与账本全部闭合，
`problems=[]`。该 cell 仍为 `running`，中间指标不构成科学结论。

epoch 70 稳定双读继续通过，history 从 1 到 70 精确连续，累计 `63700` steps、
`8151850` exposures；best epoch 为 67，best validation top-1 为
`0.09234930448222566`，epoch 70 validation top-1 为 `0.09103554868624421`。report、
last、best SHA-256 分别为
`7fbbe27aafe55a2cd80aa48c160497546d94f17e85ac09bb9ab858a749ed10ff`、
`de865c95e26d39bd2fd7a313cb81b8b5cffad7b56716231d1ea88778480fd1ca`、
`96203a236cddd79fe70484788b4344aa6ea01f41dd2ed129fcfa05222c1d86c8`；训练状态、四类
RNG、固定 revision/clean tree、cache manifest、配置语义与账本全部闭合，
`problems=[]`。该 cell 仍为 `running`，中间指标不构成科学结论。

epoch 75 稳定双读继续通过，history 从 1 到 75 精确连续，累计 `68250` steps、
`8734125` exposures；best epoch 为 74，best validation top-1 为
`0.09613601236476044`，epoch 75 validation top-1 为 `0.09219474497681607`。report、
last、best SHA-256 分别为
`fee8d9e322f3e52751a8bc33f29fb5863d7d54453d543278b6d7866786f688f0`、
`5385883f7ceae520c65a8c0f6b1566b6369f647db31ce6c9152cbb8ddc550788`、
`2da33bed90c2f760614bbe9f86c399712bb3c17de96359c35eeb44a22234976d`；训练状态、四类
RNG、固定 revision/clean tree、cache manifest、配置语义与账本全部闭合，
`problems=[]`。该 cell 仍为 `running`，中间指标不构成科学结论。

## 固定证据摘要

- `matrix_report.json` SHA-256：
  `a756c2c5a94111e0946003bd69f51fe195ee7c29a9287ac31cbc3f4c75a2b3bf`；
- `z0` seed 4121 report SHA-256：
  `bc0401dd1e6f95130575bc431cbd3650b511411adb48ba94c4e2fa63453bed87`；
- `z0` seed 4121 best checkpoint SHA-256：
  `95c7f318a759664c3e8050e12d23df9098172fe0145d23bf12a19f582d3fcefd`；
- `z0` seed 4121 last checkpoint SHA-256：
  `c517af1c8134b6fc6767504ec5efd7a315d9e0c7b03736a8f29afd4de140d530`；
- `z0` seed 4121 test SHA-256：
  `06ae415abaf2e07015c3986756dde40e9520ab77f3a67dff043c06d354598fd8`；
- `z0` seed 7319 report SHA-256：
  `5c301449df12aca1e0ae7ba8f8305d339b00d31972aae2e5cd20fa9f62c6b84f`；
- `z0` seed 7319 best checkpoint SHA-256：
  `c67d94ab4d0a89a05d1cf53b9272f757087a33a96fd95d66201092b3a1150fef`；
- `z0` seed 7319 last checkpoint SHA-256：
  `3c3da3611688ab70267e29df30adaee13e462ec4272aa19a53facd2e1abd5ee7`；
- `z0` seed 7319 test SHA-256：
  `b7cc76b89af0793a87c7e27fac29e94a7889064f48d4db905ef38df34c6ed1c9`；
- `z0` seed 104729 report SHA-256：
  `38fb08763a7679c3909a0d347520617c3f8d7451d9998852c71534bd41127655`；
- `z0` seed 104729 best checkpoint SHA-256：
  `dbdb57dd60b01b58e952fb615497a258b0c87bd0d08916e5336f8af9cd13b715`；
- `z0` seed 104729 last checkpoint SHA-256：
  `31f1a8f9e257084dd97ff2015765405f3a484ed22bca05b8dc75c1ff3421c85c`；
- `z0` seed 104729 test SHA-256：
  `03d201c549290b64f09b054cb5c47543dc7b7cf99e9193cd6a2b811831c161ee`；
- `zt` seed 4121 report SHA-256：
  `4ae54f03149863643bb3ef3698471d650d55c9687ee21f0141ad2a54c69f89eb`；
- `zt` seed 4121 best checkpoint SHA-256：
  `c1e734e7ac64d9b38ac351ae92b0cb892830c6bed1cad7c86e322c065a89a7eb`；
- `zt` seed 4121 last checkpoint SHA-256：
  `9f8454fad0f60d5819517df37c5a7dfa3c8eee1b90905ee042febf4bda796dd7`；
- `zt` seed 4121 test SHA-256：
  `005720cff5e8448187ab27cd92c4630208e8f31fcddf6081c90d2afaa1d9fad2`；
- `zt` seed 7319 final report SHA-256：
  `3de7259a4b34ffddda78ff5f4f3855316a607605f9082e536ff3f2b15943606f`；
- `zt` seed 7319 best checkpoint SHA-256：
  `9339f11d195410cb1547f8d7c9b3368b04fd02f2436e902cedc3346a7ffa732e`；
- `zt` seed 7319 last checkpoint SHA-256：
  `93e44432ff39c52a1fac3dd36ba6dfdb8ec18ed611109d6f307f6bbdbec9c0a5`；
- `zt` seed 7319 test SHA-256：
  `e2514f34e39d044e716a8cca5f941f760fd63882724c64c0124e685493a51c4a`；
- `zt` seed 104729 report SHA-256：
  `eb761d99c2b8903dc5e075a8667ae09bec8cce255350b0aecee30902aad513fd`；
- `zt` seed 104729 best checkpoint SHA-256：
  `a85b6981be58fabb9727c8309d83b9714a2c39e2fc98e103b2fe9a9df51bfba7`；
- `zt` seed 104729 last checkpoint SHA-256：
  `b50f1c77792dd9feeaf9cc7127ca6cb2bfa937d0b2c7f2c6618567477c138517`；
- `zt` seed 104729 test SHA-256：
  `a035f8fea58aff2f7fa0f5f34999f4b638855d7810ce1c0bee74a73a15f27449`；
- `trajectory` seed 4121 report SHA-256：
  `6cddacfd5aec0291375e1519e0235a2ce68632bdf4647b1935369dfe334286f0`；
- `trajectory` seed 4121 best checkpoint SHA-256：
  `1226b2cf7a78ed76fa8cfc75f27b1dd6512123b89c6a30169d3e3db0d8de8496`；
- `trajectory` seed 4121 last checkpoint SHA-256：
  `578922187d4de72b23152dcbac07ae2bc4c5ae29af16eb753a138b70edcf4034`；
- `trajectory` seed 4121 test SHA-256：
  `3b159eeae0eb1b2bf68c32321e154bcc9b28abdc3f491b260bd93bd45509b0e8`；
- seed 4121 report SHA-256：
  `580e7647fd4d3c69a3f6a577d841e1ca8026449afcfbfb0663abf076e28e60bc`；
- seed 4121 test SHA-256：
  `715b20a553ce490673f003ba55a24e8ca0b33aba4d4d5f26097420afeeb50c3e`；
- seed 7319 report SHA-256：
  `e27aa7a59b74e95d70e6c843016238502549e43af3871298f28d5e2eeb7248da`；
- seed 7319 last checkpoint SHA-256：
  `4f290c6a1d2c3845905b7dc7a22f86fd91836134beeb1212e2111d1d0150fca3`；
- seed 7319 test SHA-256：
  `6dfc91a36627cab26bf3a47b8b6a10fbff8dc80789e3b24f96ee137c67e52d1d`；
- seed 104729 report SHA-256：
  `2e8d68dc61655b1407bf4d6bc75a7f516a8967b3f69e7d90ec9a4d870357e05f`；
- seed 104729 best checkpoint SHA-256：
  `d03cecc05c1a18d70c7e3de9671f60807b8031b04f2895a05744005ca0a44400`；
- seed 104729 last checkpoint SHA-256：
  `8edf179181689619ffaf332da04a40ada0fb18d1a139658b1d0416b79dde0406`；
- seed 104729 test SHA-256：
  `710f199dacde9c9fc63eff7b8a2b2dfaa2cc87d251e24a647b0d22135262ccb6`。

本次第 9 run 后的 `matrix_report.json` 阶段 SHA-256 为
`d784d7f0c6d23e43ebfe69ebc134ab30e89ff489b17a659088969f8fab35b39c`。
`matrix_report.json` 会随后续 run 原子更新，因此上述 SHA 只锚定本次阶段快照，不是最终
矩阵 SHA。

## GPU 6 实际利用率与后续 cache 预取

`2026-08-19T09:51:20Z` 将设备总利用率、正式 worker 和资源 guard 分开审计。审计前
GPU 6 有正式 readout 与 guard 两个计算 context；guard state 将 GPU 6 标为
`occupying`，其每次释放自身 matmul 后观测到的正式 readout 利用率为 `0%--15%`，低于
guard 的 25% busy threshold，因此设备侧 `100%` 主要不能归因于正式 worker。正式
readout context 当时约占 `2234 MiB`；固定 batch=128、每 epoch 910 steps、RNG、配置和
checkpoint 恢复合约均未改动。

为把 guard 原本消耗的算力用于注册工作量，同时避免与当前 ImageNet-100 输出冲突，按
正式 recovery 的相同提取路径在物理 GPU 6 并行启动下一顺位 VOC 2012 sparse cache：

```bash
cd /mnt/omni_ssd/user_workspace/wangzixi/FieldScope/recovery/worktrees/formal-020c1de
env CUDA_VISIBLE_DEVICES=6 \
  PYTHONPATH=/mnt/omni_ssd/user_workspace/wangzixi/FieldScope/recovery/worktrees/formal-020c1de/src:/mnt/omni_ssd/user_workspace/wangzixi/FieldScope/recovery/orchestration/020c1de/sitecompat \
  FIELDSCOPE_CONFIG=/mnt/omni_ssd/user_workspace/wangzixi/FieldScope/recovery/worktrees/formal-020c1de/configs/model/auraflow_v03.yaml \
  FIELDSCOPE_RUNTIME_PROFILE=/mnt/omni_ssd/user_workspace/wangzixi/FieldScope/recovery/worktrees/formal-020c1de/outputs/runtime_gate/auraflow_runtime_profile_020c1de567edd88e0eda245fd085335ffe678f47_dsw-h200.json \
  FIELDSCOPE_CACHE_TAG=auraflow_v03 FIELDSCOPE_STORAGE_POLICY=readout_sparse \
  bash scripts/eval/run_voc2012_extract.sh
```

提取 PID 为 `452591`，日志为
`$FIELDSCOPE_ROOT/logs/formal_voc2012_cache_prefetch_020c1de_gpu6.log`。tracked config、通用
提取脚本和 VOC wrapper 的 SHA-256 分别为
`1ec2ed2706ec83110c8293c26a5a1d770516ab1c38a2e011d8e22a1c03275f68`、
`21aea1f443f6b1970fd6ef07961802848b782536d7cac6cb56311c5c3d83bc01` 和
`4871de8e9687dacee4f334e91d70fd0c9837a3c779975fc77668ee653b7709e0`。
另有非 CUDA 接力 waiter PID `457846` 只等待 VOC wrapper PID `452586` 退出，随后按
recovery 顺序串行执行 `run_ade20k_extract.sh`、`run_nyuv2_extract.sh`；其 PID 与日志分别
位于 `$FIELDSCOPE_ROOT/logs/formal_future_cache_prefetch_020c1de_gpu6.{pid,log}`。ADE20K
与 NYUv2 wrapper SHA-256 分别为
`706cbdd92e41004a94a3ae602f2d6fbda626480fc50910952ad70e33fe1e0b72` 和
`7204afb379e488aef9c7728819a07c8a8caf11996d0f54dd3e4c1cc04144b590`。接力始终只允许一个
cache extractor 存活，不使用物理 GPU 0--5。

启动后的直接观测为：

- guard 将 GPU 6 切换为 `external_compute`，自身只保留约 `812 MiB` 的监控 context，
  `last_unoccupied_utilization_percent=100.0`；
- 20 次、每 2 秒一次的连续设备采样均为 `99%--100%` SM，功耗约 `659--696 W`，正式
  readout 加 VOC cache 的设备显存约 `18512--19020 MiB`；未通过空分配制造显存占用；
- 原 ImageNet worker PID `107681` 未重启，`zt` seed 7319 从 epoch 36 连续提交至 epoch
  40；epoch 37--40 的 train seconds 为 `103.006 / 112.165 / 137.761 / 127.902`，中位数
  `120.033` 秒。启动前 epoch 30--36 中位数为 `104.368` 秒；样本很少且两窗口均有明显
  波动，不能据此声称 readout 本身提速，但没有发现 checkpoint、history 或进度停滞；
- VOC train cache 已原子提交 2 个 shard、128 个样本，`status=running`、
  `complete=false`、固定 revision、`code_dirty=false`，实测写入率
  `0.3000114276` 样本/秒；该 mutable manifest 的阶段 SHA-256 为
  `b9f8a46b6370e58130c93afa265f7aad63d48a9cecd63c84bb3a9c8761efae62`。

该预取只提前执行 recovery 已注册且之后仍会逐 manifest 全审计的 cache 工作，不并发写
任何 matrix/output，不改变 20 表示 × 3 seed 主矩阵、因果门、extension 条件或最终回放
合约。VOC 三个 split 尚未全部完成，因此此处不把 VOC cache 标记为完成，也不改变科学
裁决。

## 第八个 run 与 VOC 正式 cache 完成里程碑

`2026-08-19T12:53:23Z` 对新完成产物执行内容级审计：

- ImageNet-100 `zt` seed 7319 report 为 `passed`，history 从 epoch 1 连续到 90，所有
  train/validation loss 有限，累计 `81900` optimizer steps 与 `10480950` sample
  exposures；best epoch 为 78，best validation top-1 为 `0.08918083462132921`；
- best/last checkpoint 分别锚定 epoch 78/90，history 长度、seed、representation、
  batch=128、target epochs=90、steps/epoch=910、RNG state、revision 和 clean provenance
  均与 report 一致；
- test report 为 `passed`，使用完整 ImageNet-100 test cache 5,000 样本，top-1/top-5
  为 `0.0872/0.2728`，loss 为 `3.976225350189209`；checkpoint SHA 与 matrix 注册值
  完全一致；
- `matrix_report.json` 已原子注册第 8 个 run，并精确登记 `81900` steps、`10480950`
  exposures、test metric `0.0872`；随后同一 worker 自动进入 `zt` seed 104729；
- VOC 2012 train/val/test cache 分别为 `1318/146/1449` 样本、`21/3/23` shards，三个
  manifest 均为 `passed`、`complete=true`、固定 revision、`code_dirty=false`、
  `readout_sparse`；逐 shard SHA、大小、连续覆盖、总样本数、孤儿文件和临时文件已全量
  检查，无问题；
- VOC train/val/test manifest SHA-256 分别为
  `907040640bc41384ba648d1bffcaf808d23cb75f3e98d4196bd7b341fd1d083e`、
  `3bee502654b6a7643370fb94d60b5f6b407b4a9aa069885f00828ef3d4b77b4d`、
  `940677865f95b9309651cb3df417b31ae8a18f83cf090bcaa625c34908b5e8f0`；对应 sample-ID
  SHA-256 为 `e7e152f4d1e6b0ebddb4d2b0108b13120f3b23face977cfe6afabc3775899f5c`、
  `3e2a18af250f63e1b48273d00d21b3854a296bd857c12cb8e56200af45609237`、
  `62239fdc4c3d52c5d73e63496b71e22619596f8f40f5b7115df2c299b7cb1096`。

VOC wrapper 完成后，接力按注册顺序启动 ADE20K train cache；本次快照已提交 384 样本、
6 shards，状态仍为 `running`。这证明 cache 接力执行正常，但不把 ADE20K cache 标为完成。

## 第九个 run 与 GPU 6 有效并发里程碑

`2026-08-19T14:53:52Z` 对第 9 个完成 run 和资源占用执行结构化审计：

- ImageNet-100 `zt` seed 104729 report 为 `passed`，history 从 epoch 1 连续到 90；每
  epoch 精确为 910 optimizer steps、116455 train samples，累计 `81900` steps 与
  `10480950` sample exposures；
- best/last checkpoint SHA-256 分别为
  `a85b6981be58fabb9727c8309d83b9714a2c39e2fc98e103b2fe9a9df51bfba7` 和
  `b50f1c77792dd9feeaf9cc7127ca6cb2bfa937d0b2c7f2c6618567477c138517`；
- test report 使用完整 5,000 样本，top-1/top-5 为 `0.01/0.05`。这是 chance-level 负
  结果，保留原值且不改变科学裁决；
- `matrix_report.json` 已原子注册第 9 个 run；同一 worker 未重启，随后自动进入
  `trajectory` seed 4121，观测时已连续提交 epoch 1--2；
- 为提高有效吞吐，在保持原 ADE20K train extractor 的同时，初次新增 ADE20K val/test
  与 NYUv2 train/val/test 五个独立 extractor。PID 分别为 `72691/72701`、
  `72695/72696/72699`；六个 extractor 均只使用物理 GPU 6，输出目录两两独立；
- GPU 6 实测为 `100%` utilization、`95772/143771 MiB`、`682.25/700 W`、56 摄氏度；
  未使用空张量或其他人工显存分配，并保留 `47999 MiB` 波动余量；
- 原 ADE20K train 已提交 2,368 样本、37 shards，累计速率仍为
  `0.281339749990354` 样本/秒；新增五路在本快照时仍处于首 shard，因此不提前声称总
  吞吐提升。后续以完整 manifest 和逐 shard 审计为准。

随后首批 manifest 暴露出初次手动 launch 漏传 `FIELDSCOPE_RUNTIME_PROFILE`，导致五个
cache 的 `runtime_profile=null`。虽然实际 image/probe batch 与权威 profile 选择相同，
这些产物仍不满足正式 provenance 合约，不能事后补字段冒充正式结果。五个任务在累计写出
6 个 shards 后全部停止；目录完整移入
`$FIELDSCOPE_ROOT/recovery/quarantine/2026-08-19_missing_runtime_profile_parallel_cache`，
没有删除或计入正式 cache。独立机器可读 quarantine 清单为
`artifacts/reports/2026-08-19_missing_runtime_profile_parallel_cache_quarantine.json`，
它记录 11 个文件、`447852409` bytes 及逐文件 SHA-256；清单 SHA-256 为
`9fdbf8ea2affa1a70148cb00b7ea9e0ed7938aaf988bb2d743f9da377e0ce671`，
`formal_cache_eligibility=false`。

`2026-08-19T15:20:44Z` 从空 canonical 输出目录重启五路，PID 为
`83442/83444/83446/83448/83450`。逐 PID 环境审计确认：

- `CUDA_VISIBLE_DEVICES=6`，未使用物理 GPU 0--5；
- `FIELDSCOPE_EXPECTED_REVISION=020c1de567edd88e0eda245fd085335ffe678f47`；
- `FIELDSCOPE_RUNTIME_PROFILE` 指向 formal worktree 的 H200 runtime profile，其
  SHA-256 为
  `8e36dc71a2110b48ee05e05b12e8fe38bd88105a992d923ead053c37e4d5e37f`；
- `FIELDSCOPE_STORAGE_POLICY=readout_sparse`，并保留 `--resume`；
- `PYTHONPATH` 同时包含固定 revision `src` 与已审计 `sitecompat`。

本次并发只提前执行 recovery 已注册的正式 cache split，不并发写任何 matrix 输出，也不
触发因果或 extension 分支。原接力 waiter 之后以 `--resume` 遇到完整 split 时只复用并
审计。只有纠正后重启的 canonical cache 才能进入后续正式审计；初次无 profile 产物作为
失败 provenance 完整保留，`changes_scientific_verdict=false`。

## NYUv2 val 正式 cache 完成里程碑

`2026-08-19T15:53:05Z`，纠正重启后的 NYUv2 val 完成并通过固定 revision 代码中的
cache validator：

- manifest 为 `status=passed`、`complete=true`，精确包含 80 样本、2 shards，覆盖
  `[0,64)` 与 `[64,80)`；
- manifest SHA-256 为
  `3038eebaf0276ac39a5125c6591cebddc321a69285628e2a1ac749109d5013d2`，
  sample-ID SHA-256 为
  `86d332465face6cac11e8aad209200fd47ee1ed44284fceaa5d68fb7b58a1a7b`；
- runtime profile SHA-256 为
  `8e36dc71a2110b48ee05e05b12e8fe38bd88105a992d923ead053c37e4d5e37f`，
  固定 revision/tree、`code_dirty=false`、AuraFlow frozen backend、batch/probe
  profile 和 `readout_sparse` 均匹配；
- 逐 shard 文件大小、SHA、tensor 样本数、grid、feature dtype/dimension、baseline/
  graph 字段、depth target shape/dtype、metadata、sample IDs 及其重算摘要全部通过；
  无孤儿文件或临时文件；
- cache 共 `130588882` bytes，实测写入率
  `0.05253758759072356` 样本/秒。

这只完成 NYUv2 的 val split；train/test 仍在运行，因此不把 NYUv2 三 split 或主矩阵
标记为完成。同期 ImageNet-100 `trajectory` seed 4121 已连续到 epoch 20，最新 train
time 为 `120.82060451060534` 秒；原 worker PID 未重启。

## Analyzer GPU 6 绑定纠正

`2026-08-19T16:35:32Z` 的进程环境复核发现，等待 final decision 的 analyzer PID
30573 含 `FIELDSCOPE_GPU_INDEX=6`，但没有 `CUDA_VISIBLE_DEVICES=6`。该 waiter 尚未
开始 132 份回放、没有 analyzer CUDA worker，也未写入任何 summary、registry 或
completion audit；因此这是未来资源边界缺口，不是已经发生的跨卡执行。

外部 watchdog 的 `start_analyzer` 启动环境只增加
`CUDA_VISIBLE_DEVICES="$gpu_index"`：

- 修改前 watchdog SHA-256：
  `f8c46f0abbdf950a23b32ab4bccaced7a674f59832af1cab98b94fcfb0a04390`；
- 原件备份：
  `$FIELDSCOPE_ROOT/recovery/orchestration/020c1de/watchdog_h200.sh.f8c46f0abbdf950a.pre-analyzer-gpu6-binding`；
- 修改后 watchdog SHA-256：
  `23a2a22c6fa6a28c368094a9536989c38ad574178a4736e11f66660aea691963`；
- `bash -n`：passed；两个固定 worktree 仍 detached、clean、revision 未变；
- 受控重启后 watchdog PID 为 `106745`、analyzer PID 为 `105935`，analyzer 环境已直接
  核验 `CUDA_VISIBLE_DEVICES=6`；`/proc` 精确 argv/cwd 扫描各只有一个实例；
- 正式 readout PID `107681`、recovery PID `92034` 和五个 cache worker 均连续存活，
  没有重启或修改其 cache/checkpoint/report；GPU 6 继续保持正式负载；
- 最新 watchdog state 为 `status=active`、`analyzer.alive=true`、
  `changes_scientific_verdict=false`。

该纠正只约束未来 secondary replay 的物理设备可见性，不改变固定 scientific code、
训练预算、已完成结果、main/causal/extension/final 规则或科学 verdict。

## GPU 6 饱和度与占卡 guard 回避核验

`2026-08-19T16:48:15Z` 复核用户指出的显存与利用率归属。GPU 6 上仍是一个正式
ImageNet-100 readout 与五个互不重叠的正式 cache extractor；后者分别处理 ADE20K
train/val/test 和 NYUv2 train/test，NYUv2 val 已完成。逐 PID 环境再次确认所有正式 CUDA
进程仅见物理 GPU 6，固定 revision 与 cache runtime profile 均未改变。

设备端每 2 秒采样一次，共得到 14 个有效点：SM 利用率全部为 `100%`，功耗为
`678--692/700 W`，memory-controller 利用率为 `33%--35%`，显存恒为
`80320/143771 MiB`。GPU process accounting 中五个 cache context 各约 `15446 MiB`，
readout 约 `2234 MiB`。资源 guard 自身状态文件明确为 `external_compute`，记录
`external_memory_mib=79763`、`last_unoccupied_utilization_percent=100`；它只保留
`525 MiB` context 和 `32 MiB` reserve，occupier 已停止。因此当前 `100%` 计算利用率来自
正式 workload，不能归因于占卡矩阵乘法。

未再启动进程或分配空张量，理由是：五个仍可独立写入的正式 cache split 已全部在运行；
runtime gate 中更大的注册 cache batch 不能保持逐字段精确等价；readout gate 选择单 seed，
因为 2/3 seed 候选相对串行分别慢约 `1.10%` 和 `76.22%`。继续占用剩余显存不会增加
算力或有效吞吐，反而会重复 cache、引入 matrix writer 竞争或违反权威 runtime profile。
本次只确认资源饱和与 guard 回避，不改变任何科学结果，
`changes_scientific_verdict=false`。


`2026-08-21T09:38:11Z` 再次观察到非 FieldScope 的 M3Call/OmniCall worker PID
`1817990` 使用物理 GPU 6，命令范围为 `stage1_h200_gpu67_train_r2 --stop-after-step
256`，当时占用约 `62716 MiB`。该外部进程自 `2026-08-21T09:21:57Z` 启动；FieldScope
worker PID `102910` 仍存活并保持 `CUDA_VISIBLE_DEVICES=6`。没有终止外部进程、重启
正式 worker 或修改 batch/cache/runtime/scientific contract。重叠结束时间尚未观察到，
因此只保留为活动资源 provenance，不推断科学影响，`changes_scientific_verdict=false`。

ImageNet-100 `velocity / seed 4121` 于 `2026-08-21T09:42:09.246936811Z` 原子提交
epoch 81，并于 `2026-08-21T09:46:09.212285Z` 通过语义强审计。epoch 80→81 新增
`910` optimizer steps 与 `116455` sample exposures，累计为 `73710` 与 `9432855`；
history `1..81` 连续，report/checkpoint history 经 JSON list/tuple 归一化后相等，
AdamW step `73710`，scheduler `last_epoch=81/T_max=90`，四类 RNG、cache/config/
control/coverage、注册 runtime profile `seed_workers=1`、两个固定 revision clean worktree
与稳定双读均通过，`problems=[]`。validation top-1/top-5 为
`0.13809891808346214 / 0.37302936630602784`；best 仍为 epoch 75 的
`0.14049459041731066`。report、last、best checkpoint SHA-256 分别为
`60e38f05385df22fa9a01cdb73824d0c6c928fe30e22896c7c1e99a4b8ddfb9c`、
`ff315f38c69d89137a8711da5220d87097da36d567e25cde3b2109a8e8202966`、
`ea5c9d87d458f29db3effbd9ddfbb1d577b1849b66f56406e836739e9406e4f3`；审计日志
SHA-256 为 `4789f9bc702da3cede2fbc926867ccfdcbe95eed16829a1011e161b82aa99a73`。
该提交发生在上述外部 GPU 6 重叠期间；强审计证明原子状态一致，但不证明资源重叠
没有时序或科学影响。该 cell 仍为中间证据，`execution_complete=false`、
`method_effectiveness_conclusion=null`、`changes_scientific_verdict=false`。

`2026-08-21T09:50:07.656826Z` 的 watchdog 状态确认 epoch 81 已注册，worker 自动
进入 epoch 82；主矩阵仍为 `12/60`，已完成 optimizer steps `1056510`、sample exposures
`135204255`（包含此前 12 个完整 cell 与当前 cell 的 epoch 81）。GPU 6 当时为
`100%`、约 `66239 MiB` 总占用；外部 M3Call PID `1817990` 仍在运行。该运行态只作
provenance，未把未提交的 epoch 82 计入正式 ledger，`execution_complete=false`、
`method_effectiveness_conclusion=null`、`changes_scientific_verdict=false`。

外部 M3Call/OmniCall 于 `2026-08-21T10:25:34.036650037Z` 写完 step 256 checkpoint，
并于 `2026-08-21T10:31:27.925662705Z` 写完对应 validation；`2026-08-21T10:33:20Z`
复核时 PID `1817990` 已退出，GPU 6 总显存从约 `66509 MiB` 回落至 `3515 MiB`。
FieldScope worker PID `102910` 保持存活并独立达到约 `99%` GPU utilization，未被重启，
正式 batch/cache/runtime/scientific contract 均未改变。该资源重叠现标记为已结束；结束
事实不用于推断资源重叠有无科学影响。epoch 82 此时仍未原子提交，因此首个重叠后原子
证据仍待后续强审计，`changes_scientific_verdict=false`。

ImageNet-100 `velocity / seed 4121` 于 `2026-08-21T10:36:32.610554199Z` 原子提交
epoch 82，并于 `2026-08-21T10:37:45.947129Z` 通过语义强审计。这是上述外部重叠
结束后的首个正式原子提交。epoch 81→82 新增 `910` optimizer steps 与 `116455`
sample exposures，累计 `74620` 与 `9549310`；history `1..82` 连续，report/checkpoint
history 经 JSON list/tuple 归一化后相等，AdamW step `74620`，scheduler
`last_epoch=82/T_max=90`，四类 RNG、cache/config/control/coverage、注册 runtime profile
`seed_workers=1`、两个固定 revision clean worktree 与稳定双读均通过，`problems=[]`。
validation top-1/top-5 为 `0.1376352395672334 / 0.37364760432766614`；best 仍为
epoch 75 的 `0.14049459041731066`。report、last、best checkpoint SHA-256 分别为
`d48b6ec9ac30c92ea480deaa43cf83eb4fe868d6500f25417f13e77bcbf941cd`、
`b4a9eb39d08b4046acd6dbf2ea99ab5192e9c0a2cb88baacbb9fd447d07443c1`、
`ea5c9d87d458f29db3effbd9ddfbb1d577b1849b66f56406e836739e9406e4f3`；审计日志
SHA-256 为 `abb3d65ece32b3d4f7ba6945acccb0e9f95845a310a18f4b25e1668ee8436149`。
该审计证明重叠后的原子状态一致，但不证明此前资源重叠没有时序或科学影响。该 cell
仍为中间证据，`execution_complete=false`、`method_effectiveness_conclusion=null`、
`changes_scientific_verdict=false`。

`2026-08-21T11:00:34Z` 又观察到新的非 FieldScope M3Call/OmniCall worker PID
`3336415` 使用物理 GPU 6，约占用 `62714 MiB`。该进程自
`2026-08-21T10:36:45Z` 启动，命令范围为恢复 `checkpoints/step-00000256` 并运行至
`--stop-after-step 512`。FieldScope worker PID `102910` 仍存活并保持
`CUDA_VISIBLE_DEVICES=6`，最新权威原子边界为 epoch 82。没有终止外部进程、重启
正式 worker 或改变 batch/cache/runtime/scientific contract；该新重叠保持活动资源
provenance，尚未观察结束时间，不推断科学影响，`changes_scientific_verdict=false`。

ImageNet-100 `velocity / seed 4121` 于 `2026-08-21T11:35:56.810681427Z` 原子提交
epoch 83，并于 `2026-08-21T11:36:51.158318Z` 通过语义强审计。epoch 82→83 新增
`910` optimizer steps 与 `116455` sample exposures，累计 `75530` 与 `9665765`；
history `1..83` 连续，report/checkpoint history 经 JSON list/tuple 归一化后相等，
AdamW step `75530`，scheduler `last_epoch=83/T_max=90`，四类 RNG、cache/config/
control/coverage、注册 runtime profile `seed_workers=1`、两个固定 revision clean worktree
与稳定双读均通过，`problems=[]`。validation top-1/top-5 为
`0.1375579598145286 / 0.37812982998454403`；best 仍为 epoch 75 的
`0.14049459041731066`。report、last、best checkpoint SHA-256 分别为
`ca93063502ffc13783c990be78d20bfe2cc8c6b2eb028fdb348a258535e708c4`、
`78f43805fdc6893944d21bc0962d48e4460e677e36298286c59cbf1575bf3521`、
`ea5c9d87d458f29db3effbd9ddfbb1d577b1849b66f56406e836739e9406e4f3`；审计日志
SHA-256 为 `d8b5ced33119fe64a82d54e690bac7fface6d6241a8286d43bb9cb4e75d51446`。
该提交发生在第二次外部 GPU 6 重叠期间；强审计证明原子状态一致，但不证明并发外部
任务没有时序或科学影响，相关 provenance 已保留。该 cell 仍为中间证据，
`execution_complete=false`、`method_effectiveness_conclusion=null`、
`changes_scientific_verdict=false`。

第二次外部 M3Call/OmniCall 重叠随后闭合：step 512 checkpoint 于
`2026-08-21T11:36:21.554406038Z` 完成，对应 validation 于
`2026-08-21T11:42:24.652518470Z` 写出；`2026-08-21T11:44:19Z` 复核时 PID
`3336415` 已退出，GPU 6 总显存回落至 `3515 MiB`，FieldScope 独立达到 `100%`
utilization。正式 worker 未重启，batch/cache/runtime/scientific contract 未变化。
epoch 83 是本次重叠期间的正式原子提交并已通过强审计；该一致性结果仍不用于推断并发
任务有无科学影响。重叠结束事实已保留，`changes_scientific_verdict=false`。

`2026-08-21T05:17:25Z` 复核确认外部 M3Call AVA indexed evaluator PID `1076690` 已结束，GPU 6 显存回落至约
`2993 MiB`；FieldScope worker `102910` 未重启并继续正式执行。资源并发结束事实已保留为 provenance，
未修改正式条件，`changes_scientific_verdict=false`。
`2026-08-21T05:09:19Z`，ImageNet-100 `velocity / seed 4121` 原子提交 epoch 74，并完成强审计。
首次审计脚本曾因手写累计 exposures 预期值少写 `10` 而产生非权威 false failure；该操作未修改任何 artifact，
已按正确的 `8501215 + 116455 = 8617670` 重跑并通过。epoch 73→74 新增 `910` optimizer steps、
`116455` sample exposures，累计为 `67340` 和 `8617670`；train/validation 耗时
`1841.3976269075647 / 205.7817157646641` 秒，validation top-1/top-5 为
`0.13724884080370942 / 0.3731066460587326`，best 更新为 epoch 74（best primary metric `0.13724884080370942`）。
history 1→74 连续，report/checkpoint history 经 JSON list/tuple 语义归一化后相等，AdamW state step `67340`，
scheduler `last_epoch=74`、`T_max=90`，四类 RNG、cache/config/control/coverage、固定 revision clean tree 与稳定双读
全部通过，`problems=[]`。report/last/best SHA-256 分别为
`e0964a80915f3051caa33c5d7b784b8e1c94397a24c0d59a7d3ee84b8c0a4d91`、
`f8648d499cd65c4260a58808dfda43ab8390685685bbc2f867781e2d39146241`、
`b6150662ba78f0e0a259f4aaa818dbf6de4dfadd0afdf33cedd014d88791b2ed`。该结果仍仅为单 cell 中间证据，
不构成主矩阵或最终科学裁决，`changes_scientific_verdict=false`。
`2026-08-21T05:12:13Z` 观察到非 FieldScope M3Call AVA indexed evaluator PID `1076690`，从 `05:03:09Z`
启动，占用 GPU 6 约 `62018 MiB`。FieldScope worker `102910` 未中断；未终止外部进程、未修改正式
batch/cache/runtime/scientific contract。该并发仅作资源 provenance，未推断科学影响，
`changes_scientific_verdict=false`。
`2026-08-21T04:39:19.324935Z` 的 watchdog 快照已独立纳入 epoch 73：ImageNet-100 完整 run 仍为 `12`，
active cell 为 `velocity / 4121 / 73`，cell 累计 `66430` steps、`8501215` exposures；全局累计
`1049230` optimizer steps、`134272615` training sample exposures。`execution_complete=false`、
`method_effectiveness_conclusion=null`、`changes_scientific_verdict=false`。
`2026-08-21T04:40:32Z` 复核确认外部 M3Call AVA evaluator PID `917204` 已结束，GPU 6 显存回落至约
`3515 MiB`；FieldScope worker `102910` 未重启并继续正式执行。该资源并发结束事实已保留为 provenance，
未修改正式条件，`changes_scientific_verdict=false`。
`2026-08-21T04:45:49Z` 再次观察到非 FieldScope M3Call AVA natural-512 evaluator PID `1010686`，
从 `04:41:38Z` 启动，占用 GPU 6 约 `61860 MiB`。FieldScope worker `102910` 未中断；未终止外部进程，
未修改正式 batch/cache/runtime/scientific contract。该并发仅作资源 provenance，未推断科学影响，
`changes_scientific_verdict=false`。
`2026-08-21T04:35:17Z`，ImageNet-100 `velocity / seed 4121` 原子提交 epoch 73，并完成强审计。
epoch 72→73 新增 `910` optimizer steps、`116455` sample exposures，累计为 `66430` 和 `8501215`；
train/validation 耗时 `1852.6515919109806 / 205.51331823784858` 秒，validation top-1/top-5 为
`0.1357032457496136 / 0.3690880989180835`，best 更新为 epoch 73（best primary metric `0.1357032457496136`）。
history 1→73 连续，report/checkpoint history 经 JSON list/tuple 语义归一化后相等，AdamW state step `66430`，
scheduler `last_epoch=73`、`T_max=90`，四类 RNG、cache/config/control/coverage、固定 revision clean tree 与稳定双读
全部通过，`problems=[]`。report/last/best SHA-256 分别为
`bd8032cc77e80705c986cb7827d2e98e88a47bc9ec8a289cfcc6c469be52089e`、
`5541203fd406ce3221fbdc292fe270db552ad65d486c33dfe9c6724983643b30`、
`e2e27e1ff0bbbd299a5dca35554121188a01e635394899a2316373cda542ac04`。该结果仍仅为单 cell 中间证据，
不构成主矩阵或最终科学裁决，`changes_scientific_verdict=false`。
`2026-08-21T04:37:00Z` 观察到新的非 FieldScope M3Call AVA evaluator PID `917204`，从 `04:34:15Z` 启动，
占用 GPU 6 约 `60920 MiB`；FieldScope worker `102910` 继续在 GPU 6 上运行。未终止外部进程，未修改
正式 batch/cache/runtime/scientific contract；该并发仅作资源 provenance，未推断科学影响，
`changes_scientific_verdict=false`。
`2026-08-21T04:01:05Z`，ImageNet-100 `velocity / seed 4121` 原子提交 epoch 72，并完成强审计。
epoch 71→72 新增 `910` optimizer steps、`116455` sample exposures，累计为 `65520` 和 `8384760`；
train/validation 耗时 `1839.696630849503 / 215.08878369722515` 秒，validation top-1/top-5 为
`0.13176197836166925 / 0.36020092735703246`，best 更新为 epoch 72（best primary metric `0.13176197836166925`）。
history 1→72 连续，report/checkpoint history 经 JSON list/tuple 语义归一化后相等，AdamW state step `65520`，
scheduler `last_epoch=72`、`T_max=90`，四类 RNG、cache/config/control/coverage、固定 revision clean tree 与稳定双读
全部通过，`problems=[]`。report/last/best SHA-256 分别为
`ef21dc7d1114e823dce20b256390d9c8856ec856b7c220071089ffcb147f2a7b`、
`b090b25368e02a2b1a9bebe896dc795da3bf1d37c9e8741b401e61f8edaa8af7`、
`74b4516b9b9263acf94b37bd32cb756600a1de7ec0c9e5746c02554dd10bd81e`。该结果仍仅为单 cell 中间证据，
不构成主矩阵或最终科学裁决，`changes_scientific_verdict=false`。
`2026-08-21T04:09:17.104396Z` 的 watchdog 快照已独立收敛到 epoch 72：ImageNet-100 完整 run
仍为 `12`，active cell 为 `velocity / 4121 / 72`，cell 累计 `65520` steps、`8384760` exposures；
全局累计 `1048320` optimizer steps、`134156160` training sample exposures。main/causal decision 均不存在，
watchdog `status=active`、`execution_complete=false`、`method_effectiveness_conclusion=null`、
`changes_scientific_verdict=false`。
`2026-08-21T02:52:28Z`，ImageNet-100 `velocity / seed 4121` 原子提交 epoch 70，并完成强审计。
epoch 69→70 新增 `910` optimizer steps、`116455` sample exposures，累计分别为 `63700` 和 `8151850`；
本 epoch train/validation 耗时 `2109.1764528434724 / 204.47672473080456` 秒，validation top-1/top-5 为
`0.1241112828438949 / 0.35115919629057185`，best 仍为 epoch 65 的 `0.13060278207109738`。
history 1→70 连续，report/checkpoint history 经 JSON list/tuple 语义归一化后相等，AdamW state step 为 `63700`，
scheduler `last_epoch=70`、`T_max=90`，四类 RNG、cache/config/control/coverage contract、固定 revision/clean worktree
和稳定双读全部通过，`problems=[]`。report/last/best SHA-256 分别为
`394f863a0fb3421bda2cf3d8b8676c24d5347babe26d041ff40f55d7aa429031`、
`c96710e4d76d0a50c801b92a79294c579df15ca87837091a79ce8f07b692e512`、
`7fbca4b974cadf6c9d280485f190496bcc9c35ee3f3004a3f7bead9fd8e8ecd3`。该结果仍仅是单 cell 中间证据，
不构成主矩阵或最终科学裁决，`changes_scientific_verdict=false`。
`2026-08-21T03:29:14.565372Z` 的 watchdog 快照已独立收敛到 epoch 71：ImageNet-100 完整 run
仍为 `12`，active cell 为 `velocity / 4121 / 71`，cell 累计 `64610` steps、`8268305` exposures；
全局累计 `1047410` optimizer steps、`134039705` training sample exposures。main decision 仍不存在，
watchdog `status=active`、`execution_complete=false`、`method_effectiveness_conclusion=null`、
`changes_scientific_verdict=false`。
`2026-08-21T02:59:11.995356Z` 的下一轮 watchdog 快照已独立收敛到 epoch 70：ImageNet-100 完整 run
仍为 `12`，active cell 为 `velocity / 4121 / 70`，该 cell 为 `63700` steps、`8151850` exposures；
全局已观察到 `1046500` optimizer steps、`133923250` training sample exposures。watchdog `status=active`，
`execution_complete=false`、`method_effectiveness_conclusion=null`、`changes_scientific_verdict=false`。
`2026-08-21T03:26:48Z` 左右，ImageNet-100 `velocity / seed 4121` 原子提交 epoch 71，并完成强审计。
epoch 70→71 新增 `910` optimizer steps、`116455` sample exposures，累计为 `64610` 和 `8268305`；
train/validation 耗时 `1837.0867910599336 / 204.03959820140153` 秒，validation top-1/top-5 为
`0.13013910355486863 / 0.3601236476043277`，best 仍为 epoch 65 的 `0.13060278207109738`。
history 1→71 连续，report/checkpoint history 经 JSON list/tuple 语义归一化后相等，AdamW state step `64610`，
scheduler `last_epoch=71`、`T_max=90`，四类 RNG、cache/config/control/coverage、固定 revision clean tree 与稳定双读
全部通过，`problems=[]`。report/last/best SHA-256 分别为
`5edd39a9cc9ebfc1ab11a264a24f6dd8f18afd328e69f7c11e9bccc00c88d199`、
`fa9d27c5cdb3bf941110604e6c41314c80e6dd15957c51b5f9a02a8a7879ecec`、
`7fbca4b974cadf6c9d280485f190496bcc9c35ee3f3004a3f7bead9fd8e8ecd3`。该结果仍仅为单 cell 中间证据，
不构成主矩阵或最终科学裁决，`changes_scientific_verdict=false`。

NYUv2 val/test 相继完成后，活动 cache extractor 从五个降为四个。`2026-08-19T18:28:03Z`
再次连续采样 10 次、间隔 2 秒：SM 全部为 `100%`，功耗 `679--692/700 W`，其中 9 次
报告 power violation active；显存为 `64868/143771 MiB`，memory-controller 为
`33%--35%`。正式 context 为一个 readout（约 `2234 MiB`）与四个 cache extractor（各约
`15446 MiB`）；guard 仍为 `external_compute`，occupier 未恢复。剩余显存不是当前瓶颈，
在 SM 与功耗已饱和时增加重复 extractor 或第二个 matrix writer 只会违反权威串行合约并
争抢现有吞吐，因此未启动；也未通过空张量伪造显存占用。

NYUv2 train 完成后只剩 ADE20K train/val/test 三个互不重叠的 extractor。约
`2026-08-19T19:02:08Z` 再连续采样 12 秒：GPU 6 的 SM 每次均为 `100%`，功耗为
`671--690/700 W`，其中 11 次报告 power violation active；显存为
`48599/143771 MiB`，memory-controller 为 `33%--35%`。当前四个 CUDA context 均来自正式
工作：一个 readout 约 `2234 MiB`，三个 extractor 各约 `15446 MiB`；未观察到 guard
occupier。显存尚有余量是计算受功耗墙限制的结果，不代表存在可转化为有效吞吐的空闲
算力；cache/readout runtime profile 继续保持不变，没有启动重复 writer 或人工显存占用。

ADE20K val/test 完成后，GPU 6 上只剩一个正式 readout 与 ADE20K train extractor。
`2026-08-19T23:24:59Z` 再连续采样 10 次、间隔 1 秒：SM 为 `99%--100%`，其中 9 次为
`100%`；功耗中位数为 `687.865/700 W`，memory-controller 为 `33%--36%`，显存恒为
`17694/143771 MiB`。同一时刻物理 GPU 0--5 各有约 `125019--126155 MiB` 的大显存占用但
SM 均为 `0%`，而 GPU 6 未出现该类大额 guard 分配，说明占卡负载已回避 GPU 6。当前
低显存占用不等于低吞吐：正式计算已经占满 SM 并接近功耗上限。继续塞空张量只会占容量，
不会提高已注册工作吞吐；readout batch 128、cache image/probe batch 2/8 与 worker 数均
保持权威 profile，不启动重复 writer，`changes_scientific_verdict=false`。

`2026-08-20T02:28:56Z` 又连续采样 5 秒：GPU 6 的 SM 为 `99%--100%`，其中 4 次为
`100%`；功耗 `638--688/700 W`，memory-controller 为 `33%--36%`，显存恒为
`17694/143771 MiB`。GPU 上仍只有一个正式 readout（约 `2234 MiB`）和一个 ADE20K
train extractor（约 `15446 MiB`）。资源 guard 的 GPU monitor 此时因其自身
`AttributeError` 已停止，未运行 occupier；所以当前 `99%--100%` 利用率全部来自
FieldScope 正式负载。剩余显存不是吞吐瓶颈，保持已审计的 batch/profile，不用空张量
填充显存，也不启动会争抢同一 SM 的重复 writer。

## 活动 cache 新增前缀增量审计

`2026-08-19T16:52:43Z`，五个活动 extractor 各原子提交了一个新 shard。固定 revision
代码逐份重新加载新增 payload 并验证 feature fingerprint、目标字段、tensor/sample
数量、sample IDs、文件大小与 SHA-256；同时检查完整 manifest 注册范围连续、provenance、
runtime profile、storage policy、孤儿文件和临时文件。结果 `status=passed`、
`problems=[]`：

| cache | 已验证前缀 | manifest SHA-256 | 新 shard SHA-256 |
|---|---:|---|---|
| ADE20K train | 2816 | `c69ce9c6ddd012a45d9a23915b002a06f99ab756927bb35715cffd6dae10c5e6` | `0b07af48007eebe91e50e7b4fc3c11a5ec60a50c28186d0a3c5500620be2721a` |
| ADE20K val | 320 | `a497c5ef5211be89040e78d847c58d3e2bc2abd8aad0ec3321a645748ca2491e` | `121834c53b6e5f2be26b4eb72ea592cc16e1e2110fc3cd1bef8e7dd2b583ee3b` |
| ADE20K test | 320 | `7bd8cf693b775e1490696c34d537e1d8639cffdde999d7f2e9a943d0dbc0427f` | `636ae635d6d067b0c4f1796f17c1495ca4577f5e723aaee84309c007470fed45` |
| NYUv2 train | 320 | `f7475c194866227b2e5635336f121d95461e5611f8d957035ab634307ab22683` | `8937b62dab575b0a91a05f044fb1a3aec34869d8b1ecce8762431371a53a4a48` |
| NYUv2 test | 320 | `fd58becbe8ae8a4ca3d393dc641de5197ae86939ebdce19cb9e1fbe85618cbcb` | `3d7e2120e4a2ef310c93f46bb2fdc1463f4f7144d6acd854ff66311efc3f566d` |

这些仍是 mutable 前缀；除已完成的 NYUv2 val 外，不将任何 split 标记为完整，也不改变
主矩阵、因果门或科学裁决。

`2026-08-19T17:08:14Z` 的第二次增量审计继续从上述边界验证两个新提交：ADE20K train
从 2816 到 2880，manifest/new-shard SHA-256 为
`dd711906be870f7f7d49bc651b46d5b7c947f7e39c02abf394f8d2e5b7dd799c` / 
`76331f3ee34b12a6b973f6f504f599992955919f3f703d48d87a81e9451d4786`；NYUv2 train
从 320 到 384，对应 SHA-256 为
`313f72eb718551c03b91d5b73fff6c2958330c61963420d807694e08cfbf74e8` / 
`79259d5a2035c23b7f3fd42c6ee4c948183df09fdc3f6bf5a47ff2959678e28b`。
两份 payload 的 fingerprint、目标、样本 ID、范围、bytes、runtime provenance 和文件集
均通过，`problems=[]`；二者仍为 mutable 前缀。

`2026-08-19T17:12:43Z` 的第三次增量审计验证另外三个从 320 到 384 的新提交：

| cache | manifest SHA-256 | 新 shard SHA-256 |
|---|---|---|
| ADE20K val | `6e8d938ecf18bd96b18c47ffcb5635dbe2f0d6a053eeaea7d551418d51d77c43` | `7c9caecdb105cad8902c2d47a415c96cce8a5389e63019760e9b17aa2603826b` |
| ADE20K test | `57333e29e63e50da986ece8b051c3eb887af9ea9d29be4a91be6f406f762744d` | `4b064327e8d6fcaf2da2c1115f06df64d9295bc7a06b8a98c1e2ef5d4124d49e` |
| NYUv2 test | `aca381b9bce06f81b4cfad911346f1fcb2bcb8cc366a6cbb28211f841edc6534` | `4ed845ee70a9377a1262cc1550da11fe9fbe5fb3b73429e2c8a739fd4c900209` |

三份新增 payload 的内容、provenance 与文件集均通过，`problems=[]`；三个 split 仍为
mutable 前缀，未标记完整。

最新可覆盖前缀检查点为 `2026-08-19T18:17:19Z`：ADE20K train 到 3136，ADE20K
val/test 与 NYUv2 train 均到 640；四个新 shard 的 manifest/new-shard SHA-256 已写入
对应机器可读报告，完整增量检查 `problems=[]`。后续普通 shard 进度覆盖该检查点；只有
完整 split 才追加永久终态审计。

`2026-08-19T18:42:00Z` 的后续增量审计把 ADE20K train 的覆盖边界从 3136 推进到
3264，val/test 均从 640 推进到 768；共验证 6 个新增 shard。三份快照 manifest SHA-256
分别为 `5f6573dcd9906cbc95b426462a0c64ccd7ed27cd6533404909203ea4792c7afc`、
`9ab962bf2a889f3a4fd2b782d92a417a5b163f7f288867ebbc328a6aa0593eb6`、
`288135da64e026b0bc1657be79088ce57b6b99d0574a28e867bc6464b0648fab`。所有新 shard 的
SHA、payload、segmentation target、sample IDs、fingerprint、随机性 metadata、连续范围
和文件集均通过，`problems=[]`；三个 split 仍为 mutable 前缀，不提前标记完整。

`2026-08-19T18:58:57Z` 的下一次增量审计把 ADE20K train 从 3264 推进到 3392，
val/test 均从 768 推进到 832，共验证 4 个新增 shard。train/val/test manifest SHA-256
分别为 `30ee146817a922ede38f32f8c3561647b606f9acb494aa46fde43fdf5b6a11cd`、
`15ac3015e53455adcb88edc36dc24542128108b06e86ecfb6b30fa5f7372e8e8`、
`b79b140f36894482909deba093d130ef6b3a9bd8c5c3d8c32dea7bffb5b0b6cb`；所有新增 payload
检查通过，`problems=[]`。三个 split 仍在运行，不提前宣称完整。

`2026-08-19T19:13:41Z` 的增量内容审计继续把 ADE20K train 从 3392 推进到 3456，
val/test 均从 832 推进到 960，共验证 5 个新增 shard。train/val/test manifest SHA-256
分别为 `26650aa005a9623c4f442dc5b68d7c2192c292cd88939f5a382d75272b086278`、
`0dd94ec543723c9b49be99ab6befffe6fbb1d76b7fd89c12b2805ec9f56dea38`、
`6ee97b7a42585f6c67c42e087752022d6308c728ecec87f42dc2fb8ae0c7645e`。稳定 manifest
双读、完整注册范围与文件集、新 shard SHA/大小、feature/target 样本数、tensor 有限值、
segmentation target、sample IDs、随机性、runtime profile 和固定 provenance 全部通过，
`problems=[]`；三个 split 仍为 mutable 前缀，不提前宣称完整。

`2026-08-19T19:24:46Z` 的下一次增量审计验证 ADE20K train/val/test 各一个新 shard，
前缀分别推进到 `3520/1024/1024`。三份 manifest SHA-256 为
`bcd50898fc39e83507c50ac8db99659858d7d0855486f37a48e80643c4fd15a3`、
`8d2a3395156e64cf71121865eb29f22a972eac2f1870b509cb775f5b09b94db7`、
`e11530a5fc12ec42ff85b863eb87871564e41a8e7e3001b5447f9750d0c38cf8`；新增 payload、
SHA、有限值、segmentation target、sample IDs、文件集和 provenance 全部通过，
`problems=[]`。三个 split 仍未完成。

`2026-08-19T19:37:33Z` 的增量审计把 ADE20K train 从 3520 推进到 3648，val/test 均从
1024 推进到 1088，共验证 4 个新增 shard。train/val/test manifest SHA-256 分别为
`e738447f63e9f3d6c7aa17dbd87f5ed93f8cbc66b10d1b362b1b460d5b832150`、
`160ada5072cae91e894d187a5f603cef3ace00d028996745d90f19b689e4a676`、
`7f342bc2442974667c860d517e774a79023a0b07f23fd4e66056c1f2ece903ff`；内容与 provenance
检查均通过，`problems=[]`。三个 split 仍为运行中前缀。

`2026-08-19T19:55:43Z` 的后续审计把 ADE20K train 从 3648 推进到 3712，val/test 均从
1088 推进到 1216，共验证 5 个新增 shard。train/val/test manifest SHA-256 分别为
`426747d9796bccf2b95bd7a0b541e9c3e4f5112bf4c0e6ff97e0160d295caaf0`、
`93e67c384023edd4d156b83f388ef2a2950b720a8a043530f7cdda4667f829dc`、
`2a94149bb610ece0c46e0be8c5a9b999dcb50648bca9e09264ebcc77afb390b2`；内容审计
`problems=[]`。三个 split 仍为运行中前缀。

`2026-08-19T20:13:09Z` 的下一次审计把 ADE20K train 从 3712 推进到 3840，val/test 均从
1216 推进到 1280，共验证 4 个新增 shard。train/val/test manifest SHA-256 分别为
`e1f8d188072b702eb50c27aab7689ab328490b4063bfc301f5672ae63a70063a`、
`beff4c3753b3ce78679ccbdb466e3d0888412e49a93a3ef8046bd6bcb87aa0ff`、
`1d89d9cce94b358bd8b8ed635ca65dbd26fe6b77300af070b1d1e78627175e46`；内容与 provenance
审计 `problems=[]`。三个 split 仍未完成。

`2026-08-19T20:25:34Z` 的增量审计把 ADE20K train 从 3840 推进到 3904，val/test 均从
1280 推进到 1408，共验证 5 个新 shard。train/val/test manifest SHA-256 分别为
`01538b509af7c74c48837bef66dad2fcb8a4f1996b464f738e147a9f2866b69d`、
`cff2aa0ca19bc595f8970537680c2b6422180fc134b0c052aa2cca8e9de5789d`、
`26479acc938fcc2e70a5a298497e406476e46306753d6af23fbb48bf79b2c065`；内容与 provenance
审计 `problems=[]`，三个 split 仍为运行中前缀。

`2026-08-19T20:37:47Z` 的增量审计把 ADE20K train 从 3904 推进到 4032，val/test 均从
1408 推进到 1472，共验证 4 个新 shard。train/val/test manifest SHA-256 分别为
`84afb8eb67b65fe2b76e6339ffb77ad575d3ca431c501a99ce9af35e1c948a9b`、
`f71c69a2574c07f786a22cb14d2eb7042644cff326cfa750c458a6d3dcc59ace`、
`ea8bf48907a24c1696af4efd5d960e564f090fb588d436bb93069663f64ef70e`；内容与 provenance
审计 `problems=[]`，三个 split 仍未完成。

`2026-08-19T20:50:42Z` 的增量审计把 ADE20K train 从 4032 推进到 4096，val/test 均从
1472 推进到 1536，共验证 3 个新 shard。train/val/test manifest SHA-256 分别为
`a849923e59b717de6ccc2f5119d6cddb3eb9e526f0d1f7633d5cba5b749d6914`、
`3e9e6b5fe978125fdd2fa2745c7b7130f6881d3c46923b361026eed1636ca4c6`、
`ee73333c6362aeb1844b7342af62c33ee7ca0ce0bfb3fa43eb6c12790ef82305`；内容与 provenance
审计 `problems=[]`，三个 split 仍未完成。

`2026-08-19T21:03:40Z` 的增量审计验证 ADE20K train/val/test 各一个新 shard，前缀推进到
`4160/1600/1600`。三份 manifest SHA-256 分别为
`62403475f00b10111552727399c1e03a209e7727c2457b547d80c81662b4e51a`、
`4b9617f5fd9df11db138e8d837f22515b947dd854fc342de5e4dae098918a08c`、
`d87886353bb3efb667c8420a3b67455600bb644231b664568a909948cbd41256`；内容与 provenance
审计 `problems=[]`，三个 split 仍为运行中前缀。

`2026-08-19T21:17:42Z` 的下一次增量审计把 ADE20K train 从 4160 推进到 4224，val/test
均从 1600 推进到 1728，共验证 5 个新 shard。train/val/test manifest SHA-256 分别为
`b625df8276c5e8f4c2524c23d4af130fc846b815b6a8c046afcf18af4bcf2cf3`、
`c857096bead3e2673ec99ec4911852dd56ff83eacc2592c0ac51cd2375d31072`、
`4d633d325d0edb8ddaf5ff6154a0a088bb33de0a4a122fad400826d8c779850a`；新增 shard 内容、
连续范围、runtime profile 与固定 revision 审计 `problems=[]`。三个 split 仍为运行中
前缀，不提前标记完整。

`2026-08-19T21:35:35Z` 的后续增量审计把 ADE20K train 从 4224 推进到 4352、val 从
1728 推进到 1856、test 从 1728 推进到 1792，共验证 5 个新 shard。train/val/test
manifest SHA-256 分别为
`d2d6dcd71af485cf16c774902a47b7e5e6df4723cf1f8bdc4476bac8626c43e8`、
`057b4efbdd35605b60c8adeef741ddddf8d3a883e72c1b0aec0f027be7a7f98a`、
`590164f62a8ff4385a0fa981a0d8516aecc67eacd8f03fdb73b61281d3c62439`；逐文件 SHA、
payload feature/target 数量、sample IDs、连续范围、runtime profile 与固定 provenance
审计 `problems=[]`。三个 split 仍为运行中前缀，不提前标记完整。

`2026-08-19T21:53:33Z` 的下一次增量审计把 ADE20K train 从 4352 推进到 4480、val 从
1856 推进到 1920、test 从 1792 推进到 1920，共验证 5 个新 shard。train/val/test
manifest SHA-256 分别为
`e3fdb98805df68cf0d825d1511b91fcbf6e1283b24102907c3d1d736c50e7f0a`、
`ebfe80780d052262a199ba3bc4672e2dc3b55f9b774bed27e92b0327d488758d`、
`8b4e0fbf0688e5ff1b0440b14532cf1fa1f3d3c12b2b43339105f2ba0478dcd5`；逐文件 SHA、
payload 内容、连续范围、runtime profile 与固定 provenance 审计 `problems=[]`。三个
split 仍为运行中前缀，不提前标记完整。

`2026-08-19T22:01:05Z`，ADE20K test 完成 `2000/2000` 样本和 32 个连续 shard。固定
revision 分析逐 shard 复算全部 SHA-256、加载全部 payload，并核验 `segmentation`
target、feature 数量、fingerprint、sample IDs、连续范围、孤儿与临时文件；结果
`status=passed`、`problems=[]`。manifest SHA-256 为
`a4695f688dd38178839b390bfbf22fb144bea1a521ef85562e69b45d251da37c`，重算 sample-ID
SHA-256 为 `999c035837c84033247f0c7a7945806ec6181965ac4c4a7846a4e016c4c477f9`，总 cache
大小 `1692544416` bytes，累计写入速率 `0.0841238143634852` 样本/秒。ADE20K train/val
仍在运行，因此此处不宣称 ADE20K 三 split 全部完成。

`2026-08-19T22:04:53Z`，ADE20K val 随后完成 `2021/2021` 样本和 32 个连续 shard。
同一固定 revision 内容审计复算全部 shard SHA、加载全部 payload，并核验分割 target、
feature 数量、fingerprint、sample IDs、连续范围、孤儿与临时文件；结果
`status=passed`、`problems=[]`。manifest SHA-256 为
`592af67413a3f4d85bb176c59f3eccccc8ebfda4a371479c07ef6ceea1369bfe`，重算 sample-ID
SHA-256 为 `902144de2a8b7a2fa3e23d8194650900f8988f3ea17e62d4260ad207bc549c80`，总 cache
大小 `1710198880` bytes，累计写入速率 `0.08458116855289505` 样本/秒。ADE20K 只剩
train split 尚未完成，因此仍不宣称三 split 全部完成。

`2026-08-19T22:14:15Z` 的 train-only 增量审计把 ADE20K train 从 4480 推进到 4800，
共验证 5 个新 shard。manifest SHA-256 为
`a8332b730000acc8c1b014a662f6db00731bf75b7e5baed4dc094cd04f518a7c`；逐文件 SHA、
payload feature/target 数量、sample IDs、连续范围、runtime profile 与固定 provenance
审计 `problems=[]`。train 仍为运行中前缀，不提前标记完整。

`2026-08-19T22:24:25Z` 的下一次 train-only 增量审计把 ADE20K train 从 4800 推进到
4992，共验证 3 个新 shard。manifest SHA-256 为
`438a1231076a81840cb94db44d0b470f65d0823cd4cd3ee4efe50eecef513766`；逐文件 SHA、
payload 内容、连续范围、runtime profile 与固定 provenance 审计 `problems=[]`。train
仍为运行中前缀，不提前标记完整。

`2026-08-19T22:34:00Z` 的 train-only 增量审计把 ADE20K train 从 4992 推进到 5120，
共验证 2 个新 shard。manifest SHA-256 为
`4975a47af3c5f41e81cf20db52e0fcbc69a8a97c01a56147f25020fe9ff99910`；逐文件 SHA、
payload 内容、连续范围、runtime profile 与固定 provenance 审计 `problems=[]`。train
仍为运行中前缀，不提前标记完整。

`2026-08-19T22:44:15Z` 的 train-only 增量审计把 ADE20K train 从 5120 推进到 5312，
共验证 3 个新 shard。manifest SHA-256 为
`8d48d4c3d39bce27c1003f5f7b330f8d8b6d2b4a30622d98d463b9b9751d2a6d`；逐文件 SHA、
payload 内容、连续范围、runtime profile 与固定 provenance 审计 `problems=[]`。train
仍为运行中前缀，不提前标记完整。

`2026-08-19T22:54:12Z` 的 train-only 增量审计把 ADE20K train 从 5312 推进到 5504，
共验证 3 个新 shard。manifest SHA-256 为
`83981d2748c03f30473f0c085d8d1068d7f034b14e2c00a7992882378c53a30d`；逐文件 SHA、
payload 内容、连续范围、runtime profile 与固定 provenance 审计 `problems=[]`。train
仍为运行中前缀，不提前标记完整。

`2026-08-19T23:04:03Z` 的 train-only 增量审计把 ADE20K train 从 5504 推进到 5696，
共验证 3 个新 shard。manifest SHA-256 为
`883d0c64e54ecd0921ab74d94e4463ab9df3e34759e948756f8e9e422ba92518`；逐文件 SHA、
payload 内容、连续范围、runtime profile 与固定 provenance 审计 `problems=[]`。train
仍为运行中前缀，不提前标记完整。

`2026-08-19T23:16:25Z` 的 train-only 增量审计把 ADE20K train 从 5696 推进到 5952，
共验证 4 个新 shard。manifest SHA-256 为
`0045c488a03cb782c0c79833c1091d4a24088fd1a0000820a90923b099dc295a`；逐文件 SHA、
payload 内容、连续范围、runtime profile 与固定 provenance 审计 `problems=[]`。train
仍为运行中前缀，不提前标记完整。

`2026-08-19T23:24:59Z` 的下一次 train-only 增量审计把 ADE20K train 从 5952 推进到
6080，共验证 2 个新 shard。manifest SHA-256 为
`1ebc22ea8cc46aecfe22fedb5bce92d601f9a0592c972f38395dbec8a0f736ea`；两份 shard 的
SHA-256 分别为 `548bdd2969b7c6a31829ef29bdfacd1853c1be223a8911375a75eb1fcee196e6`
和 `c54e51e4e3f2069e78af4307259082d13429049ec985415a0d44732453d4be23`。逐份重载验证
feature/segmentation target 数量、sample IDs、fingerprint 与连续范围，`problems=[]`。
train 仍为运行中前缀，不提前标记完整。

`2026-08-19T23:32:59Z` 的后续 train-only 增量审计把 ADE20K train 从 6080 推进到
6272，共验证 3 个新 shard。manifest SHA-256 为
`1175f37a7d3f8a102405fb9ee39023e067a44ded278da565fe463cb6118e33a7`；新增文件 SHA-256
分别为 `92f107daf0fc6faf17921d36b6f35901f043d956055fb96bcf6b35efc6995b10`、
`a4bb26ac5bed9986a4c58505adb85354dc70fe8036afd20e94682cc610875b2d` 和
`d7dcd69ce37014caab4d29f5e4460090e97e3c497d8c76a4e82300eefd53ff17`。逐份 payload、
segmentation target、sample IDs、fingerprint 与连续范围审计 `problems=[]`；train 仍为
运行中前缀，不提前标记完整。

`2026-08-19T23:42:49Z` 的下一次 train-only 增量审计把 ADE20K train 从 6272 推进到
6400，共验证 2 个新 shard。manifest SHA-256 为
`7f250e1886db189ef550f36c372227d948c35d3067c59b9a7003d03dcaf0d370`；新增文件 SHA-256
分别为 `057492f231c198121a40e5fd6db0edf47c2f6e91372a0791edc3061aac33ceff` 和
`c924b9fc79a4ed83792877b54d23629733b7708be5d7eef16725b406b9866707`。逐份 payload、
segmentation target、sample IDs、fingerprint 与连续范围审计 `problems=[]`；train 仍为
运行中前缀，不提前标记完整。

`2026-08-19T23:55:01Z` 的后续 train-only 增量审计把 ADE20K train 从 6400 推进到
6656，共验证 4 个新 shard。manifest SHA-256 为
`7e8849352842d320361e908abffcf8b025f75839ed74738e2f12bcd32e782ff4`；新增文件 SHA-256
分别为 `8be60205771bdb1f32d0106eeb55643b7e6bdfc9c7a373c5527d29957afd0ee8`、
`f389354f796508bf5d4efd2b6a6ad8101d11a05c1cb35f2a1d81a305c6d0f8ed`、
`608ba12a54d9016e1ca98d7939a5b3a896c6e5af603e5c4451e6c9757e98b66a` 和
`c98f9460604e5a4ee6d3bc4fbf6188e59338a4a1c98600f78837dae882e3652f`。逐份 payload、
segmentation target、sample IDs、fingerprint 与连续范围审计 `problems=[]`；train 仍为
运行中前缀，不提前标记完整。

`2026-08-20T00:04:12Z` 的下一次 train-only 增量审计把 ADE20K train 从 6656 推进到
6848，共验证 3 个新 shard。manifest SHA-256 为
`1e8adb6f1bc9591d369c279354aff0cfe20e71cfcb12ef734fb25fa86d24e4c2`；新增文件 SHA-256
分别为 `8b04cf9c56b4b434ae4ae757d8bec0f5de478e37861f30ed9fbc991e74b15b11`、
`c1c51cd0e86d480a8386d33e07ff9b070a2ea8f6e82294ed8b9e03adc057f9a9` 和
`0cbc1fc4523574eb7c09f65bd1dec4b37319f1aa52dd1a03b25547d2e5c3e08b`。逐份 payload、
segmentation target、sample IDs、fingerprint 与连续范围审计 `problems=[]`；train 仍为
运行中前缀，不提前标记完整。

`2026-08-20T00:12:17Z` 的后续 train-only 增量审计把 ADE20K train 从 6848 推进到
6976，共验证 2 个新 shard。manifest SHA-256 为
`a689fbfd18ee444053188997614ebc59f15abe8fcb6b1d22ab4548301e61b8cd`；新增文件 SHA-256
分别为 `4011b172bd17d1a1f2c2819bc6f7f3c0f775108d16f90d913e5e31a29df75d71` 和
`d98162eea51f192d0384738910c63d96a13c32dd778ea5707b94a6b52968bd57`。逐份 payload、
segmentation target、sample IDs、fingerprint 与连续范围审计 `problems=[]`；train 仍为
运行中前缀，不提前标记完整。

`2026-08-20T00:22:43Z` 的下一次 train-only 增量审计把 ADE20K train 从 6976 推进到
7168，共验证 3 个新 shard。manifest SHA-256 为
`6cd175fa17b32672a33d24f2c94775562f143b7912eb9928c6e46c2ad9b0634f`；新增文件 SHA-256
分别为 `ff987ff0db0ace116bc6174a93026b471a887aa60f02ac0e58f3402376c25307`、
`f8e3b54e36e1bc6feaeb71b462bf30271b3fe1361b9c0a0a8d2d7fba8df61459` 和
`da771a1c23ec105fe3dce72beaaea68468698065c2d04f6b6e00d77429e534ec`。逐份 payload、
segmentation target、sample IDs、fingerprint 与连续范围审计 `problems=[]`；train 仍为
运行中前缀，不提前标记完整。

`2026-08-20T00:32:41Z` 的后续 train-only 增量审计把 ADE20K train 从 7168 推进到
7360，共验证 3 个新 shard。manifest SHA-256 为
`b936a596f84add21efad8935288b86acf2a5f3ca1c6da42c772cf490e4d75349`；新增文件 SHA-256
分别为 `bd78f8fc9bb19d40424f07449df513b7c7cb81e076d9304c57c100487a11f440`、
`7f048f248b0a798a8a575ff60f120e7c686aedfbefbf99312902b2db3d2c8e2b` 和
`bfae9399a6da19924a6847d10fa4378b0dc3cc8102a46166ce65e62d4131218a`。逐份 payload、
segmentation target、sample IDs、fingerprint 与连续范围审计 `problems=[]`；train 仍为
运行中前缀，不提前标记完整。

`2026-08-20T00:43:07Z` 的下一次 train-only 增量审计把 ADE20K train 从 7360 推进到
7552，共验证 3 个新 shard。manifest SHA-256 为
`5398a1a1ad816dc70b8d101c8337f98c335c7805adb979e70296d113ec7f27fa`；新增文件 SHA-256
分别为 `a857e7fdecaa40e280d6b09d3d5cb4f85c6fe9263c9d2c738dbc79164e4f4ca8`、
`8692faa5be3cc69321f54401b1ac6d4df3c9f2c171cbbb23a9f0e7cd527a2388` 和
`0beb7fe6e1727d91a9861bc32d97e8d5980f8a749e1091a90afbe9b6267620fb`。逐份 payload、
segmentation target、sample IDs、fingerprint 与连续范围审计 `problems=[]`；train 仍为
运行中前缀，不提前标记完整。

`2026-08-20T00:52:25Z` 的后续 train-only 增量审计把 ADE20K train 从 7552 推进到
7744，共验证 3 个新 shard。manifest SHA-256 为
`89f5c39248d5f7083147eb88fe7b51769ee9685072d101ee77f112894cfe01cb`；新增文件 SHA-256
分别为 `29dc19b9f9560f0be1fcbf2eaa5569884079c3d88a15ae64b7bdb6f9d2f2b19a`、
`ebb82e8f1979825213c72e74bb467f2576964593ef4283b9432f17caf34f9d69` 和
`e282d432e7710172033f4fe54e96681ff68dc7cd1149818cca74dcb73b023238`。逐份 payload、
segmentation target、sample IDs、fingerprint 与连续范围审计 `problems=[]`；train 仍为
运行中前缀，不提前标记完整。

`2026-08-20T01:02:50Z` 的下一次 train-only 增量审计把 ADE20K train 从 7744 推进到
7936，共验证 3 个新 shard。manifest SHA-256 为
`eeb3ba2ff44cef1e36f6e25cf72fc511945e54356c68c1d72c961eecb39f5c7a`；新增文件 SHA-256
分别为 `792f20446cd08e88891fe230c57b54a3cf7dfc0820d0e4f046abcc8b394a8f3f`、
`a4920288f8c178b2257a0bf3827d284a2347ad410f1eb6ce57b4d26e71929cc5` 和
`08a248d6a06d7f53fda37106927909d6f1f7542b6994fb999823e0bda0471942`。逐份 payload、
segmentation target、sample IDs、fingerprint 与连续范围审计 `problems=[]`；train 仍为
运行中前缀，不提前标记完整。

`2026-08-20T01:16:09Z` 的后续 train-only 增量审计把 ADE20K train 从 7936 推进到
8192，共验证 4 个新 shard。manifest SHA-256 为
`3cdd3db844f1a29e346e4f4f55fc3f2dabce2d9165623fd350767218b53b7e63`，已提交前缀的
sample-ID SHA-256 为 `ab77974c1b069807ff48c95aa72f718904a266d2cb0363cb6955add3f0ae56a9`；
新增文件 SHA-256 分别为 `c21493b49e381ca49f86c84b4ef8c462d8ac1c560a9e06dfac4776493b1eb71a`、
`89b159c418172e2d237f16d7eec942d126ab9bd6cad258f78a1ca6177e4832c8`、
`47f448ebd38338476bb45322fbd6ae3cddb6bbe1a0f77df9dc9e1ffe544bb18e` 和
`e5958c459198e3f47c661e2727f1fe8546449a62ed85b8dca6314134b5d646d9`。逐份 payload、
segmentation target、源数据 sample IDs、fingerprint、全局连续范围与文件集审计
`problems=[]`；train 仍为运行中前缀，不提前标记完整。

`2026-08-20T01:22:55Z` 的下一次 train-only 增量审计把 ADE20K train 从 8192 推进到
8320，共验证 2 个新 shard。manifest SHA-256 为
`1bcbddb04bef058c145e6b2ac8b3ccac767fe93de85055f3f55504f6a8e9d0af`，已提交前缀的
sample-ID SHA-256 为 `ff076d2053a27d12c2f12768f7fcd64aba5294b3cb01199dfe3c5c85ec5e7c43`；
新增文件 SHA-256 分别为 `8711ef8eec52e2ca5efdf075fc7a1a9fa7739ce7fdd0031a6ed9732f0bab5864`
和 `acb613a12c5a60265c88042c3c34d32296cd9152b12c0f47abbfc213e3b7382d`。逐份 payload、
segmentation target、源数据 sample IDs、fingerprint、全局连续范围与文件集审计
`problems=[]`；train 仍为运行中前缀，不提前标记完整。

`2026-08-20T01:32:48Z` 的后续 train-only 增量审计把 ADE20K train 从 8320 推进到
8512，共验证 3 个新 shard。manifest SHA-256 为
`30b492ec1d2f7ed78acb1095ca14240b0a9008e5db71ba666b89f7150f66c3d1`，已提交前缀的
sample-ID SHA-256 为 `97724af62584b5aa5dbf94b295116929169c6e5a96ed3d188e7ef198e9db3a25`；
新增文件 SHA-256 分别为 `515d5e195fd51c39821b81861bc022f5a1cd9e7da4087a06e41bb0268be9660a`、
`5a9036ceec7d72a2c1c9e64e5ce8198f4c1a032591340adb4dec7c05af21ebce` 和
`7b4d2b3463fedf0554451c3ca7d49ff93fa2697818cc369326f6881bd1a4fed6`。逐份 payload、
segmentation target、源数据 sample IDs、fingerprint、全局连续范围与文件集审计
`problems=[]`；train 仍为运行中前缀，不提前标记完整。

`2026-08-20T01:40:04Z` 的下一次 train-only 增量审计把 ADE20K train 从 8512 推进到
8640，共验证 2 个新 shard。manifest SHA-256 为
`b1a935d9b1bd175e777c0c4974d161918c28ea0851cf6eebcd3a43e0ecf63e88`，已提交前缀的
sample-ID SHA-256 为 `5952821206be25b3f08b35c51cb7f54e9c5f3daae413fba9cea33eea9ceff1d5`；
新增文件 SHA-256 分别为 `c8ededef75003a508783e1bd346f2ca51d840f7e8dec6dd06f16be577eed29c2`
和 `21f0c84bc72823c5d9ef1a24123bdc49d4349c99c380b5b808dc0521dc47923e`。逐份 payload、
segmentation target、源数据 sample IDs、fingerprint、全局连续范围与文件集审计
`problems=[]`；train 仍为运行中前缀，不提前标记完整。

`2026-08-20T01:50:16Z` 的后续 train-only 增量审计把 ADE20K train 从 8640 推进到
8832，共验证 3 个新 shard。manifest SHA-256 为
`8d7451ecfaae2a70cabeb9ed59d85f3717f9f1b875e3941d1f7135bc1cfcbe94`，已提交前缀的
sample-ID SHA-256 为 `85bcad91212a078db317aa00e1873537a030de465e15fa7ade7a3854d688a5ee`；
新增文件 SHA-256 分别为 `1a739db006dd498e75326c3a4d871f0f8a5e19c64d4d55b6394168af6c5bdb3d`、
`592ca152f44c21c48accc302f69eaadad65b0f7248dc64b1945f5777396beab3` 和
`1edf4642a8175bca46790edd7997824c8cab85471bb5183e0ad0f9905df1fc1e`。逐份 payload、
segmentation target、源数据 sample IDs、fingerprint、全局连续范围与文件集审计
`problems=[]`；train 仍为运行中前缀，不提前标记完整。

`2026-08-20T02:00:00Z` 的下一次 train-only 增量审计把 ADE20K train 从 8832 推进到
9024，共验证 3 个新 shard。manifest SHA-256 为
`c6f6e8943f3e76fc64a1102e3b5d6fca1c2517372fedad5f1653e4cdca752692`，已提交前缀的
sample-ID SHA-256 为 `9b860a18279e32a4e9095fdd95a3944664f9723b61ae490e8f9f63d603d66ed5`；
新增文件 SHA-256 分别为 `a2d48379a7dc93e78b3c573cd361cf90976ddf217581c18b470cb513254832af`、
`1e65d8015088097075f968ab43cad32ca13506ff3c21928df697514c8d7c5545` 和
`b1fcab3aa2bbc977d4cbc31337fb3ad18465214895ad008243fdf0ce3825d7b9`。逐份 payload、
segmentation target、源数据 sample IDs、fingerprint、全局连续范围与文件集审计
`problems=[]`；train 仍为运行中前缀，不提前标记完整。

`2026-08-20T02:10:09Z` 的 terminal 同期 train-only 增量审计把 ADE20K train 从 9024
推进到 9216，共验证 3 个新 shard。manifest SHA-256 为
`f381dfeb8678501293bcad88d097e364ecbf31f3a06845de82afa58046ef812a`，已提交前缀的
sample-ID SHA-256 为 `1f82a35afb38a092dee7b601b0a218c359aa9413f7925923286e09dff5e09867`；
新增文件 SHA-256 分别为 `cfec9a43e57eb42235bc23e5c58a69b4d1ce0ae5c449ce57cb1b5e17606c2c3e`、
`0964b337be9c4ddef6c59ea05a2499be9f77d7359b2009612f17b9bedb971a27` 和
`e140cb10d15f684ed93b4d5cd0dee33913ecb548bb71d6f6d6d735e72fedd624`。逐份 payload、
segmentation target、源数据 sample IDs、fingerprint、全局连续范围与文件集审计
`problems=[]`；train 仍为运行中前缀，不提前标记完整。

`2026-08-20T02:24Z` 的后续增量内容审计从 9216 推进到 9472，共验证 4 个新 shard。
manifest SHA-256 为 `9645375173b2c2f2ed5763a3ea1a63e55b79935df68d7a85deaaa5d69a837e42`，
已提交前缀的 sample-ID SHA-256 为
`5fb4505bdfbc8f990d1204944b7eb43d15abded08c3497b6c7d97385c5570da7`；新增文件 SHA-256
分别为 `bf60c30053b5a86930f145380c2271cab1938e74018fd05d9c4ce50df3a90789`、
`6778f786a10b6765bc43a9fae5bf7f04610625cf7ee0286fe5d0c48c19388ab5`、
`0b4dad7a7dc0458e446ee099d22d7bbb2d5f498adf62df29e66153d49f84ad9e` 和
`b7055cefe3eb2fd7c269417e9c633f8f6238d0fec6a301c10ecf2a522054f88b`。逐份 payload、
segmentation target、源数据 sample IDs、fingerprint、全局连续范围与文件集审计均通过，
`problems=[]`；ADE20K train 仍为 `9472/18189` 的运行中前缀，不提前标记完整。

`2026-08-20T02:37:35Z` 的下一次增量内容审计从 9472 推进到 9664，共验证 3 个新 shard。
manifest SHA-256 为 `ccbaa1501f03505060086e36209fec6278f4655317c400bf508c405b5bdcdf11`，
已提交前缀的 sample-ID SHA-256 为
`e0f06228b7457311e6a91a2ca1ed68ea11e9b295679c1ed9df086e57d1895822`；新增文件 SHA-256
分别为 `ac7fba99057101537675ca1b9e368cebbf885e1ae193df369bd2aa66a5947d0f`、
`cb5dea4f0839d38d349c5167bb99f9b5f3a490d869ee8b1ab62052221a0566dc` 和
`d76245e8e2f0dea3acd8eab799ac973dd6de8b8ffd489dc3853ee4a0e5094ad2`。逐份 payload、
segmentation target、源数据 sample IDs、fingerprint、全局连续范围与文件集审计均通过，
`problems=[]`；ADE20K train 仍为 `9664/18189` 的运行中前缀，不提前标记完整。

`2026-08-20T02:43:13Z` 的下一次增量内容审计从 9664 推进到 9792，共验证 2 个新 shard。
manifest SHA-256 为 `3a595b894437f9db2483e7c905f5ecf1a8144b4e12fee762597203b8ad6a4ea0`，
已提交前缀的 sample-ID SHA-256 为
`568062e4c7c7e2c7d85485f4823cbe3ee990ae4ccc29208fa120d7116399d53b`；新增文件 SHA-256
分别为 `6ef33f30347c48671478733c02eac020b3293e210a9a89cc6c14025c673c7d5b` 和
`dfe917e78a89f098909bf1f69dfc9cefcb14ab7027336d424e845bff0fb29622`。逐份 payload、
segmentation target、源数据 sample IDs、fingerprint、全局连续范围与文件集审计均通过，
`problems=[]`；ADE20K train 仍为 `9792/18189` 的运行中前缀，不提前标记完整。

`2026-08-20T02:52:49Z` 的后续增量内容审计从 9792 推进到 9984，共验证 3 个新 shard。
manifest SHA-256 为 `d09e4609931f9b68eb3f2da171407534259de67079bf9e5570f59d9f3699778e`，
已提交前缀的 sample-ID SHA-256 为
`87621ff499174e4aad3a40311f79863eaf0772c6a60accbb7d6cc86edb2efe13`；新增文件 SHA-256
分别为 `dcc23fd2e3957abd729ef12df282e29e7410f0d58d859ca9a72f41c9f0a81a2a`、
`1a313c25568ea0ec3648c31a5a8558ee9a381800a7280f6e67cb86a4c373f0a0` 和
`4e3d2a129c1e3aa052afcab4b27ee573ece1bafd1947dbc31674622d5eedabc5`。逐份 payload、
segmentation target、源数据 sample IDs、fingerprint、全局连续范围与文件集审计均通过，
`problems=[]`；ADE20K train 仍为 `9984/18189` 的运行中前缀，不提前标记完整。

`2026-08-20T03:02:49Z` 的后续增量内容审计从 9984 推进到 10112，共验证 2 个新 shard。
manifest SHA-256 为 `393516fbfe74171cdbaa5cfacd37be2e0d506ef9e12079d43d9b30950c2c0d81`，
已提交前缀的 sample-ID SHA-256 为
`cb0f77ff168f603bbc7171bebf33fae53abfb724ebf599496b9c3127c634364e`；新增文件 SHA-256
分别为 `2332625762948097e4be0a683b803fc2a954d5b65164d8dc556b62a2860280a1` 和
`2ceb88af9b33611e0d1a552b9b20ce38a1832cd09569f72db9201803e675d050`。逐份 payload、
segmentation target、源数据 sample IDs、fingerprint、全局连续范围与文件集审计均通过，
`problems=[]`；ADE20K train 仍为 `10112/18189` 的运行中前缀，不提前标记完整。

`2026-08-20T03:12:29Z` 的后续增量内容审计从 10112 推进到 10304，共验证 3 个新 shard。
manifest SHA-256 为 `95673922f391086684b1fa27b42457c55bb01556d1e2fea02a593c814c55f53e`，
已提交前缀的 sample-ID SHA-256 为
`30b0f51d28a5e56a199e1453a4bf188d4b09b0d4318eeeeb590b7606365e0866`；新增文件 SHA-256
分别为 `729f8ab293858c37f9284e13aad6925a9ff551071cda073446b0f4ec9fc421eb`、
`f84ba74f7ca1425ec14a7c06c22eaf9f2a2ea014dbc2a4785d04bee3bf752087` 和
`f42312149ceb022db98050facaf2e2fe1534f9a07dae9d046773adeec0b1d5f7`。逐份 payload、
segmentation target、源数据 sample IDs、fingerprint、全局连续范围与文件集审计均通过，
`problems=[]`；ADE20K train 仍为 `10304/18189` 的运行中前缀，不提前标记完整。

`2026-08-20T03:22:22Z` 的后续增量内容审计从 10304 推进到 10432，共验证 2 个新 shard。
manifest SHA-256 为 `4dab1bb079ac8508267c49806c0d6fcdaadf66b74c95a7ecea0d8ab8c8532fde`，
已提交前缀的 sample-ID SHA-256 为
`9cefb5a13fb70ed6caff91388a08eb6193a78539c550eb1f674be5655b19b245`；新增文件 SHA-256
分别为 `85e372e61f9bdfaebb8c3ad1e389bc477f9aa08be03ab4204f1c060841fca06f` 和
`df563c79553cec93ed6d6663cfc0c85d3ef29e1ba8e1a63331fc2ffb6f0e958d`。逐份 payload、
segmentation target、源数据 sample IDs、fingerprint、全局连续范围与文件集审计均通过，
`problems=[]`；ADE20K train 仍为 `10432/18189` 的运行中前缀，不提前标记完整。

`2026-08-19T18:19:56Z`，NYUv2 test 完成 `654/654` 样本和 11 个连续 shard。固定
revision 分析代码逐 shard 复算 SHA-256、加载 payload，并核验 feature/target 数量、
fingerprint、`depth` target、随机性 metadata、sample IDs、连续范围、孤儿与临时文件；
结果 `status=passed`、`problems=[]`。manifest SHA-256 为
`66adc764339848d1d7d9e981f0f24f06708c750bb4e1dfaf511412e160dc24d7`，重算 sample-ID
SHA-256 为 `caa100e50132e01e4c6ef84a6cf4b44e99010df7c614e55f612cc7c67969d547`，总 cache
大小 `1067549731` bytes。NYUv2 train 尚未完成，因此此处不宣称 NYUv2 三 split 全部完成。

`2026-08-19T18:32:37.665914Z`，NYUv2 train 随后完成 `715/715` 样本和 12 个连续
shard。全量内容审计以相同固定 revision 分析代码复算所有 shard SHA、加载所有 payload，
并核验 `depth` target、feature 数量、fingerprint、随机性 metadata、sample IDs、范围与
文件集；结果 `status=passed`、`problems=[]`。manifest SHA-256 为
`302dc8da8cc7d87655d8c644dfb65773c922f4c58061d8bdba5152efcbc428d9`，重算 sample-ID
SHA-256 为 `72eb9607ff0117d370c8d9d151ce0a3f3de287acbe9bec6dd31774ae6fbbb209`，cache 大小
`1167135212` bytes。至此 NYUv2 train/val/test 分别为 `715/80/654` 样本、`12/2/11`
shards，三个正式 cache 均完整通过，合计 `1449` 样本、`2365273825` bytes。

NYUv2 train extractor 退出后，GPU 6 显存按预期降到 `49417/143771 MiB`，但 SM 仍为
`100%`、功耗 `691.32/700 W`；guard 仍为 `external_compute`。当前一个 readout 与 ADE20K
train/val/test 三个剩余正式 extractor 已覆盖所有仍可独立写入的注册 cache 输出，未启动
重复 writer 或触碰物理 GPU 0--5。

## 后续门控链审计

`2026-08-18T16:58:59Z` 对实际 H200 recovery、final、extension wrapper 和固定 revision
分析脚本执行了 `bash -n`，均通过。内容寻址 SHA-256 为：

- H200 readout recovery：
  `fef17407a2b261335183ae17f9dd23dedf4446aa36eb41f32f3776a22d21c251`；
- final recovery：
  `8e918f0053768e50025ddb7d281b329c1de2e71288547bbc574a6af1db95e9b1`；
- extension recovery：
  `ebd52059cc65de0562b2446847c0e4bce82212251102730695f281a105280233`；
- 固定 revision supervised error-analysis waiter：
  `920184d3083389fd138ed464a007dd0df111714dc2f0b84e48acf96074a606cb`；
- 固定 revision completion auditor：
  `b12b1187195f38ac8dd7337a0b30722b306def5f7ba00f098120a5fa304da021`。

审计确认：

- 四任务 cache/matrix 完成后才生成 main `evidence_decision.json`；
- 四项 VOC 因果/条件控制在正、负 main 分支都会执行并进入 causal decision；
- extension wrapper 自身及其调用方都要求 main verdict 严格等于
  `main_tasks_supported_pending_causal_audits`，负 main 不会触发 extension；
- final auditor 接受完整的正结果或负结果，负结果不会因流程门被丢弃；
- analyzer 接受四种完整 final verdict，包括 `limited_or_negative`，随后对四任务各处理
  33 份输入报告（总计 132），并要求每任务 summary `status=passed`、
  `changes_main_verdict=false`；
- analyzer 的实际 `FIELDSCOPE_ERROR_ANALYSIS_OUTPUT_ROOT` 为
  `$FIELDSCOPE_ROOT/recovery/orchestration/020c1de/analysis-output/formal-supervised-errors-020c1de`，
  没有使用固定脚本中的仓库外默认路径；
- `registry.json` 完成后才运行 `audit-research-completion`，最终 wrapper 明确要求
  `status=passed`、`execution_complete=true`、`changes_scientific_verdict=false`。

这只证明后续门控已经注册且当前 waiter 正确等待，并不证明尚未执行的控制、回放或最终
审计已经完成。

## 后续任务资产与容量复核

`2026-08-18T17:01:05Z` 在 H200 上重新读取当前文件系统与固定 revision preflight：

- 项目所在共享盘总量 `841126395248640` bytes，当前可用
  `35506410749952` bytes（约 35.5 TB）；全盘使用率为 96%，需持续监测，但当前不是
  FieldScope 容量阻塞；
- 已存在的 `auraflow_v03` 正式 feature cache 占用 `78449670738` bytes；
- 注册的 sparse cache 预计总量 `117206194777` bytes；包含主 dense VOC 诊断、四项
  causal/conditional dense VOC 控制和 10 GiB reserve 后，注册总需求为
  `159171409492` bytes；
- 当前可用空间约为注册总需求的 223 倍。预算 JSON 中的旧 pro6000 绝对路径与当时
  free-space 数字仅作为历史 provenance 保留；本节只使用其注册 workload 大小，容量
  判断采用本次 H200 `df -B1` 实测值。

三个尚未生成正式 cache 的任务 split audit 均为 `passed`、固定 revision、
`code_dirty=false`，且每个 split 内无重复、任意两个 split 间无重叠：

| 任务 | train | val | test | split audit SHA-256 |
|---|---:|---:|---:|---|
| VOC 2012 | 1318 | 146 | 1449 | `9e127cce0d6fe0336f0f6af673174213ace53b8b2b74d6fb4b13d372706e460d` |
| ADE20K | 18189 | 2021 | 2000 | `74670458ac2de76ec234fa9bf8bc79e1da45e08ee9b77d6d22de712ee4d11d83` |
| NYUv2 | 715 | 80 | 654 | `06aff8fddfc538980cc515f82d5fa59a24722fb66c1882cdbb0f635fa98b73f6` |

预算证据 SHA-256：sparse cache budget
`bfd9795a8f80d678f263879d5fb6bfb1173f2fb20a9a4cf627e009ce746fde6c`；combined cache
budget `879463ec309c6ac41fc9eabf1bab93e5d4de175dd9cf66f4825d78fd93dbbe9e`。

因此 ImageNet-100 矩阵之后的三任务提取目前没有已知资产或容量阻塞；实际提取仍须由
正式 recovery 顺序启动并生成完整 manifest 后才能算完成。

## 已完成 cell 的聚合测试证据边界

`2026-08-18T20:09:52Z` 对 `random_feature_local` 已完成的三个 seed 逐项核对：

- 三份 test report 均为 `status=passed`、固定 revision、`code_dirty=false`，evaluation
  都覆盖完整 `5000` 个 ImageNet-100 test 样本，loss 为有限值；
- 三份报告登记的 test cache manifest SHA-256 均为
  `e92f56bbdaa4941e3693f185a8548351824dfffae622de9835c0e7088a9791e7`，sample ID SHA-256
  均为 `6817c39f4c6319b0c5e2199f1c325cfa108ef51c25d09404180b059884401ef1`；
- seed 4121 test 使用的 best checkpoint SHA-256 为
  `8fc2430c7a30815e2305b5124617ba66156b15a2ef4bafcab453920a199ba1ab`；seed 7319 为
  `f02ccf556b7cf34d54775c09a4703e93ffd727855ab3a6fc9ee9af062fcbbf1b`；seed 104729 为
  `d03cecc05c1a18d70c7e3de9671f60807b8031b04f2895a05744005ca0a44400`；三者均与
  `matrix_report.json` 注册值一致；
- 三份报告的 top-1 均为 `0.01`，top-5 均为 `0.05`，且 top-1 与 matrix test metric
  一致；这是完整测试集上的随机特征负对照结果。

正式 `*_test.json` 只保存聚合 evaluation，不含 `per_sample`、`samples` 或
`predictions` 数组。因此上述 5,000 样本覆盖与 checkpoint/cache 身份审计不能替代最终
注册的 132 份逐样本回放；逐样本证据只有 final decision 完成后由独立 analyzer 生成并
通过 `registry.json`/`completion_audit.json` 审核，当前仍不存在。

## 全量工作量分母独立推导

`2026-08-18T17:07:43Z` 直接从固定 revision 的四个 `train_*_readout.sh`、
`run_readout_matrix.sh` 和四任务 split audit 重新推导 watchdog 的最终分母。固定注册表为
20 个表示、3 个 seed，即每任务 60 个 run。每任务 optimizer steps 使用
`ceil(train_samples / batch_size) × epochs × 60`，sample exposures 使用
`train_samples × epochs × 60`：

| 任务 | train | epochs | batch | steps/epoch | 60 runs steps | 60 runs exposures |
|---|---:|---:|---:|---:|---:|---:|
| ImageNet-100 | 116455 | 90 | 128 | 910 | 4,914,000 | 628,857,000 |
| VOC 2012 | 1318 | 80 | 4 | 330 | 1,584,000 | 6,326,400 |
| ADE20K | 18189 | 80 | 2 | 9095 | 43,656,000 | 87,307,200 |
| NYUv2 | 715 | 80 | 4 | 179 | 859,200 | 3,432,000 |
| **总计** |  |  |  |  | **51,013,200** | **725,922,600** |

独立推导值与 watchdog 的 `required_optimizer_steps` 和
`required_training_sample_exposures` 完全一致。ADE20K 因 batch size 为 2，占全量 optimizer
steps 的约 85.6%，因此在 ImageNet-100 阶段 sample-exposure fraction 高于 optimizer-step
fraction 是注册工作量构成导致的正常现象，不是进度统计错误。

## 官方 completion contract 反向审计

`2026-08-18T17:10:42Z` 从固定 revision analyzer 的 `fieldscope/completion.py` 提取正式
`audit-research-completion` 合约。该检查不会重新解释科学 verdict，并且明确接受完整的
正结果或负结果。最终 `status=passed` / `execution_complete=true` 需要同时满足：

- final decision verdict 属于四个注册终态之一，status 为 `passed` 或 `failed`，
  `problems=[]`，formal source revision/tree/clean worktree 均可复核；
- main decision 精确包含 ImageNet-100、VOC 2012、ADE20K、NYUv2，每任务恰好 60 runs，
  每任务及 aggregate 的 reported steps/exposures 精确等于 required 值，并包含 1449 张
  VOC 无监督证据；
- causal decision 精确包含 `empty_prompt`、`random_flow`、
  `spatially_shuffled_probe`、`neutral_prompt`、`unrelated_prompt` 五个 source，每个 report
  与 cache 目录均存在；
- 仅当 main 为正时必须存在 extension：ImageNet-1k 恰好 60 runs，高成本 ablation
  恰好 15 个 source 和 15 个 comparison；负 main 反而必须不存在 extension path；
- error-analysis registry 必须为 `passed`、`changes_main_verdict=false`，精确包含四任务；
  每任务 summary 必须有 33 个 input report、12 个 comparison、正确 task type、formal
  source revision、analyzer provenance 和内容 SHA；
- registry 中记录的 final-decision path/SHA 必须与实际 final artifact 完全一致；最终
  analyzer worktree 也必须 clean。

当前 source/analyzer checkout、固定 revision/tree、既有 VOC 1449 无监督报告和 analyzer
waiter 已就绪；四任务主矩阵、main/causal/final、条件 extension、四份 secondary summary、
`registry.json` 和正式 `completion_audit.json` 尚未完成。因此这里只记录 completion
contract 与缺失项，不提前运行或写入官方 completion 输出路径。

此外，官方 completion auditor 不单独读取早期 signal-gate decision；用户要求保留的
signal `stop_or_redesign` 负结果仍由 watchdog、阶段记录及最终外部要求审计单独固定 SHA，
不能仅凭将来的 `completion_audit.json` 认定该要求满足。

## 服务器实例重启后的 H200 无损恢复（2026-08-20）

`2026-08-20T09:29:46Z` 重新连入 H200 时，目标实例已变为
`interactive-eouhw7dc45t3-9576ccc4b-stg7v`。旧实例的 formal worker、recovery、watchdog、
analyzer 和 cache extractor PID 均不存在，但共享盘上的正式输出、checkpoint、cache、日志
和 stale PID/lock 文件仍在。GPU 6 当时只有资源 guard，约 `1335 MiB`，设备侧 `100%`
utilization/约 `695 W` 不能单独解释为 FieldScope 工作量。

启动前按 readiness 记录完成复核：两份 fixed worktree 均为 clean detached HEAD
`020c1de567edd88e0eda245fd085335ffe678f47`；readiness JSON 为 `passed`；H200 recovery
wrapper、watchdog 和 RNG sitecompat SHA-256 仍分别为
`fef17407a2b261335183ae17f9dd23dedf4446aa36eb41f32f3776a22d21c251`、
`23a2a22c6fa6a28c368094a9536989c38ad574178a4736e11f66660aea691963` 和
`2f834fc37d66f66d7bd47104d56bc203071b50d91d2b6aba865a0755a2531c89`。
ImageNet-100 `velocity` seed 4121 的 report/best/last checkpoint 双读稳定，history 连续到
epoch 43，累计 `39130` optimizer steps、`5007565` sample exposures；AdamW 参数状态、
CosineAnnealing scheduler epoch 43、Python/NumPy/CPU CUDA RNG 四类状态、train/validation
cache 身份和固定 revision/tree 均通过，`problems=[]`。当时 best/last SHA-256 分别为
`689c7553b60f778110d50df7f842d4043fbb7add945925000ff299cc9e71f95c` 和
`ca9b3a2bac3596e0695ea1dddc0d3bb0d6428147857433f4e2c117fcd923dca0`。

`2026-08-20T09:37:39Z` 使用 readiness 中的权威命令重新启动 watchdog；新 watchdog、
analyzer waiter 和 recovery PID 分别为 `41096/41355/41610`，analyzer 和所有正式 CUDA
进程均显式绑定 `CUDA_VISIBLE_DEVICES=6`。recovery 完成两轮、每轮五次 GPU 6 空闲显存
门，并对 ImageNet-100 三个正式 cache 逐 shard 重算 SHA 后，于
`2026-08-20T09:50:44Z` 启动唯一 readout matrix worker PID `102910`。命令仍为每任务
20 表示 × 3 seeds、ImageNet-100 batch 128、`readout_memory_cache_gib=0`，没有使用物理
GPU 0--5。

ADE20K train 的重启前前缀已先全量复核到 `10560/18189`、165 个连续 shard，manifest
SHA-256 为 `2871ab56c65f6b69ce05157293fb032492974626092fe27ec64a10fb606574bd`，
sample-ID SHA-256 为
`1ed3739030eb20e181dc6a25b9e7c789a36d7fbb501e4ce99fc111575d8767b4`。其中从 10432
新增的两片 SHA-256 为
`0089330183479efdce7cedf04f7d524372f8410d632804421b89af6744b3e908` 和
`ac73a344ec63d105a1953ef62acc8195d60593b5a856a5af1cfa2271b402fffe`；payload、
segmentation target、sample IDs、fingerprint、范围、孤儿和临时文件检查均通过。

cache prefetch 重启后，固定 revision 的 `--resume` 会从第 0 片开始重新加载旧 shard，核验
source target/sample IDs/SHA，并逐片原子重建 mutable manifest；因此 manifest 的阶段
`num_samples` 会暂时从 10560 变成较小的已复核前缀，但旧 165 片并未删除或重算。首次观察
该行为时曾在 576 样本阶段停止一次 extractor，随后从同一 165 片重新启动。该操作中断了
一次只读复核进程，没有删除或改写任何正式 shard；中断阶段 manifest 已保存在
`$FIELDSCOPE_ROOT/recovery/quarantine/2026-08-20_ade20k_resume_verification_interrupted/`
下，SHA-256 为
`f0276016a16dffca5b0c062390debb84a427de5355d2f71017cb8b83d61d815f`。这是保留的负面
操作 provenance，`changes_scientific_verdict=false`。重新启动后 165 片均标为 `reused`，
再继续写新片。

`2026-08-20T10:49:01Z`，同一 readout PID 无损提交 epoch 44：history 43→44，optimizer
steps `39130→40040`（+910），sample exposures `5007565→5124020`（+116455）。服务器
重启后的首个 epoch 因 64 GiB train 与 7.1 GiB validation cache 冷读，train/validation
时间分别为 `3234.9534472450614 / 218.65098006557673` 秒；这只记录执行性能，不改变科学
条件。epoch 44 validation top-1 为 `0.08941267387944359`，best 仍为 epoch 43 的
`0.09327666151468315`，继续保留为负轨迹。强审计 report SHA-256 为
`fcbc6e0a27e42551f6a5a4e0e7aa9f4727baf89cd1d3ae7e48e371409a31a658`，last checkpoint
SHA-256 为 `f6f5b96ff706e9476ceea467a228883796feae5ae94ea4f21afc62dc353f0f85`；
AdamW step、scheduler=44、四类 RNG、cache/config identity 和双读均通过，`problems=[]`。

同一时刻 ADE20K train 首个重启后新片 `10560--10624` 已写出，SHA-256 为
`3c20f1e0003f5e79afe14614a7c06a99ce13db52fdede70106da2e11dedb0aea`，manifest SHA-256
为 `879f0fc85712deafb502aae2148b7e24b0d1ac2dd87872b233fb251291cf95f7`，sample-ID
SHA-256 为 `8531265ec837e8ea9f03dbc445cf07b6e0b28fedcd69ea7dbe72fd377a9d7fe3`；
payload、segmentation target、sample IDs、fingerprint、连续范围、文件集和临时文件检查
均通过。`2026-08-20T10:57:53Z` watchdog 的随后实时快照为 ImageNet-100 12 个完成 run、
当前 epoch 44，累计 `1022840` optimizer steps / `130895420` sample exposures；ADE20K
train 已继续到 10752。后者是实时执行前缀，本文的最新强内容审计边界仍为 10624。

signal `stop_or_redesign` 和 VOC unsupervised 两份既有证据 SHA-256 仍分别为
`0378867d09a99c99b69ec7179a8ac7d20523106cc3f7b53236177e37a0e76a1e` 与
`4382c9e53be7fa4a77b5d5a6409011cd8ce21ff4dcb3f4422bb4f668c6aa840e`。main、causal、
extension、final、132 份回放、`registry.json` 和 `completion_audit.json` 仍不存在，
`method_effectiveness_conclusion=null`、`execution_complete=false`。

`2026-08-20T11:32:09Z`，同一 worker 随后提交 epoch 45：history 连续到 45，optimizer
steps `40040→40950`（+910），sample exposures `5124020→5240475`（+116455）。epoch 45
train/validation 时间为 `2330.1604290418327 / 248.15331494435668` 秒；validation top-1
为 `0.09961360123647604`，best 同步更新为 epoch 45。该值只是当前单个未完成 cell 的
中间轨迹，不构成 main verdict 或论文结论。强审计 report、last 和 best checkpoint
SHA-256 分别为
`f278220f3ac4e443f9803a64f33cde45f1df15c338ee0703aa54c4360a9a31a7`、
`cab18ee3694fa9ef51c354f02c3ff0c83391474656ae0138f532eab8387f3c72` 和
`da1670bb894187a1deda8e9478315e3bc099662d54ffbcfd83d551c8c6ab78a5`。history、累计
ledger、AdamW step 40950、scheduler epoch 45、四类 RNG、双读和 running-report 合约
均通过，`problems=[]`。

同次 ADE20K train 强审计覆盖 10624→11136 的 8 个新 shard，manifest SHA-256 为
`ba6df20e61c81ec2e2a26dc9955b330450f0eb7e3014014c6ed780d68b726632`，sample-ID
SHA-256 为 `f5245b9a316930556e6aa18a0af38f37222b53c35abc3fd49c887813ff127027`。
8 份 shard SHA-256 依次为：
`7cde552b45422e5e69b2d5725661dc5e672aec54646524a6bf96ff8bb4501d9c`、
`d1564f97f39fd7316eedfb0ab1677cffb5a6f6d83d6f3e98adcf454ca5914975`、
`4c1fd6a7d74d3b67916b4566545e5f97281a8898a2283e5d8708c5a3b0086afc`、
`cf6a2e0671a00cc595152d1ae25055fc0dbb5f4449979c0461e12a9eada2f4e7`、
`df0394dbc71c127fa17ac18c62a1f83de7d8a05ec5c731103685d684d33f2735`、
`e02105524002cebe2dc2c8efe649fb6ff9d83f892611e799b02409ce86d3b92e`、
`e769445c3215da202b8fc03277b648ea65936cd043e1a48b782de0ddf4dc70e1` 和
`c50fb8b98464ffe252325183b3fa7733e4912ecd833e9778508218785a27e311`。逐片 bytes/SHA、
state/response shape、segmentation target、sample IDs、fingerprint、连续范围、孤儿和
临时文件检查均通过，`problems=[]`；cache 仍为运行中前缀，不提前标记完整。

`2026-08-20T11:36:37Z` 在服务器继续运行后复核：watchdog、analyzer waiter、recovery、
唯一 ImageNet-100 readout worker 和唯一 ADE20K train extractor 仍分别为 PID
`41096/41355/41610/102910/123989`，两份 worktree 继续保持 clean detached HEAD
`020c1de567edd88e0eda245fd085335ffe678f47`。所有适用进程环境仍包含
`CUDA_VISIBLE_DEVICES=6`。GPU 6 连续 12 次一秒采样均为 `100%` SM，显存恒为
`18971 MiB`，功耗为 `682.22--692.14 W`；GPU 0--5 未被 FieldScope 使用。readout report
仍稳定提交到 epoch 45，worker 正在计算 epoch 46，因此没有把未提交的内存中进度计入
正式 ledger。

同次增量 cache 审计首次读取到 `11200` 前缀后，运行中的 extractor 在审计加载 source
targets 期间原子提交了下一片 `11200--11264`。首版一次性脚本错误地把提交后的最新 shard
套用到固定的旧范围 `11136--11200`，因而产生非权威 `failed` 输出；脚本只读，没有写入
报告、manifest 或 shard，也没有停止任何正式进程。该失败明确保留为审计快照竞争的负面
操作 provenance，不解释为 cache 内容失败，`changes_scientific_verdict=false`。

随后改用“起始 manifest 快照 + entry 自带动态范围”重跑内容审计，快照覆盖
`11264/18189`、176 个连续 shard，manifest SHA-256 为
`231e2d996cde7cfda89569251abd0f846253bfef7c9a536109be6aa826b101bb`，sample-ID
SHA-256 为 `740dfbaf0375a06a2d04b833cf64d8c46baecc62eaca6df57d754cf60044131f`。
新增两片 `11136--11200` 与 `11200--11264` 的 SHA-256 分别为
`700ec2a6d4b93a6a3a1270c8697b29ac982776bcb05d5383f36301a70c38f810` 和
`5dfaa942dc5e0b228d1b9858820f34b39d0976b336cd9090220bc0168a0f08d1`；逐片 bytes、
state/response shape、全部 128 个 source segmentation target、sample IDs、fingerprint、
连续覆盖、已注册文件存在性和临时文件检查均通过，`problems=[]`。ADE20K train 仍是运行中
前缀，不提前标记完整。

`2026-08-20T11:45:05Z` 的后续快照确认上述五个正式进程和两份 clean fixed-revision
worktree 均连续存活；GPU 6 再连续采样 6 秒，每次仍为 `100%` SM，显存为
`18459 MiB`。ADE20K train 随后原子推进到 `11328/18189`、177 个连续 shard；新增
`11264--11328` shard 大小 `54149193` bytes，SHA-256 为
`f1998c18e312427b7379675883f3b584032a9c8e0accb4b9f4c6d2e8a15de899`。动态范围审计重新
核验该片全部 64 个 source segmentation target、sample IDs、state/response shape、
finiteness、fingerprint 和 SHA；快照 manifest SHA-256 为
`b09389cd5303d8d38292c7cccb124470c84778ca64f2071b014a68ef4180ae1e`，sample-ID
SHA-256 为 `8640a3faaf3fa4ce9f1045556b4c29b717c18bf832a58f4e908835b1f3f3cbdb`，
`problems=[]`。watchdog 的 600 秒 state 轮询可暂时落后于 extractor 的原子 manifest，
因此强审计边界以本次稳定快照为准。

`2026-08-20T11:49:44Z`，ADE20K train 再推进到 `11392/18189`、178 个连续 shard。
新增 `11328--11392` shard 大小 `54168713` bytes，SHA-256 为
`42ce03dbf267e74e5f304fd8e6537583233201c97a6c07f2447f450522a3ef07`；其全部 64 个
source segmentation target、sample IDs、state/response shape、finiteness、fingerprint、
SHA 与文件集均通过动态范围强审计。该快照 manifest SHA-256 为
`842329a8ec85ae7987e82f0798354af44d7517dbda77ea19a1542bfdc17d2136`，sample-ID
SHA-256 为 `a77155bfc83f725726e4654865476608fd97de6eb181204bf0944edf76bf5ab6`，
`problems=[]`。同一时刻 readout 仍在计算 epoch 46，最后原子提交仍为 epoch 45；五次
GPU 6 采样均为 `100%` SM，未把未提交进度写入正式 ledger。

下一次原子提交将 ADE20K train 推进到 `11456/18189`、179 个连续 shard。新增
`11392--11456` shard 大小 `54161865` bytes，SHA-256 为
`dc23e1e95730438a884f85abadfdcc5ecd24126f3d59bbbf45a13d227fbc13a7`；全部 64 个
source segmentation target、sample IDs、特征 shape/finiteness、fingerprint、SHA、连续
覆盖和临时文件检查通过，`problems=[]`。对应 manifest SHA-256 为
`a5f34e9860cb99354121263da53df8a2ec669c61612f7f87b2763d6c46c990fb`，sample-ID
SHA-256 为 `64c3bd94f9609015d23462c57aeb86aaa7279806423d85e59fbc2f5215faf4ba`。
ImageNet-100 epoch 46 仍在计算，GPU 6 继续为 `100%`，没有新的 readout 原子提交。

随后 ADE20K train 原子推进到 `11520/18189`、180 个连续 shard。新增
`11456--11520` shard 大小 `54170121` bytes，SHA-256 为
`1b5ab23529564efd8ad47aa896325eede9969fef9b7fe196ecd582f4b496ec21`；全部 64 个
source segmentation target、sample IDs、特征 shape/finiteness、fingerprint、SHA、连续
覆盖和临时文件检查均通过，`problems=[]`。对应 manifest SHA-256 为
`6ce0bd95f2e08e8d00130c7fc51b3c6dcddb32a11c0b5756fbe099fb60287be7`，sample-ID
SHA-256 为 `a8347fb342a5db09d5e5a505cdd0b7b8e3a41ea1104e1b6352bfaa679aa51934`。

`2026-08-20T12:02:53Z` 对服务器重启后的资源归属做了逐进程纠正审计。单独读取
`nvidia-smi` 的整卡 `100%` 不能始终归因于 FieldScope：一次连续 6 秒 `pmon` 采样中，
guard PID `1813` 占 GPU 6 的 `97%--99%` SM，而 readout PID `102910` 与 extractor PID
`123989` 当时正在 CPU/磁盘阶段，只保留各自约 `2172/15452 MiB` context。guard 的状态
文件同时把 GPU 6 标为 `occupying`，monitor context/reserve 为 `527/544 MiB`，正式进程
外部显存为 `17900 MiB`。

这不是 guard 长期抢占正式 CUDA 工作。guard 日志逐次记录：当正式进程的无 guard 探针
达到 `57%--100%` 时，GPU 6 occupier 会释放并转为 `external_compute`；当正式进程进入
数据读取或 CPU 阶段、探针低于 25% 时，occupier 会再次启动。因此资源事实应表述为
“guard 按阈值在正式 GPU kernel 活跃时回避、在正式 CPU/I/O 间隙占用”，而不能表述为
“guard 始终不在 GPU 6 上计算”或“整卡 100% 始终来自 FieldScope”。没有修改、暂停或
终止外部 guard；readout/cache batch、worker 数、RNG、输出与科学合同均未改变，
`changes_scientific_verdict=false`。

后续 ADE20K train 原子推进到 `11584/18189`、181 个连续 shard。新增
`11520--11584` shard 大小 `54151753` bytes，SHA-256 为
`18edfdfdd07e576c9100dd82df971cc049e5578cdf3791ab4d48172220931031`；全部 64 个
source segmentation target、sample IDs、特征 shape/finiteness、fingerprint、SHA、连续
覆盖和临时文件检查通过，`problems=[]`。对应 manifest SHA-256 为
`ed8b0b432e68b6e2a335a0c0fb151c5460bfe130c0b4f4c1a7d41b2a1a1ac536`，sample-ID
SHA-256 为 `e9d0b5543b683443f37c73da0f16c8d7bcd0623ace0335c9f22d44eac372a1f9`。
提交快照时 guard 状态为 `external_compute`，其无 guard 探针为 `100%`；该项只说明此
瞬间正式 kernel 在运行，不外推为整个 epoch 的资源归属。

随后 ADE20K train 原子推进到 `11648/18189`、182 个连续 shard。新增
`11584--11648` shard 大小 `54167113` bytes，SHA-256 为
`0fc1fad8a1593a70a249d071d6a13256ee3e4b8f20712305fbf92b89876dbe5e`；全部 64 个
source segmentation target、sample IDs、特征 shape/finiteness、fingerprint、SHA、连续
覆盖和临时文件检查通过，`problems=[]`。对应 manifest SHA-256 为
`7e574a1b3bbcfb02cd270c1836be1d5e2d7340f37d69913ee017c0f812ffa701`，sample-ID
SHA-256 为 `6e12afee7f8536cf84e4dd0dadf25a6851e45fcf0c4dcd02b957465b85fed7c2`。

随后 ADE20K train 原子推进到 `11712/18189`、183 个连续 shard。新增
`11648--11712` shard 大小 `54155209` bytes，SHA-256 为
`3fc532891269fe31fd9904ea49a12d1a4e9ebab4c04db21b6fbe603d356817f0`；全部 64 个
source segmentation target、sample IDs、特征 shape/finiteness、fingerprint、SHA、连续
覆盖和临时文件检查通过，`problems=[]`。对应 manifest SHA-256 为
`97409d276af18460502e8462514dbf069596018b9809deae9ec148e85870ea39`，sample-ID
SHA-256 为 `60075c32685c31bc3eb6c92d89ad2a3a9de8647c0d24ef0ab8eb2eb9f196985e`。

ImageNet-100 `velocity` seed 4121 随后原子提交 epoch 46。history 45→46，累计 optimizer
steps `40950→41860`（+910），sample exposures `5240475→5356930`（+116455）。epoch 46
train/validation 时间为 `2224.4102109689265 / 286.9798591276631` 秒，validation top-1 为
`0.09343122102009274`；低于 epoch 45，best 继续保持 epoch 45 的
`0.09961360123647604`。report、last 与 best checkpoint SHA-256 分别为
`f44a48c1366fac8de8cab46cd14beefe8c4e15fd98da1b349ddd0ae38e83ef7b`、
`46e7f7ee80a930e1766e0fa1f42284fb97da87ca8065642e76664e5faa101756` 和
`da1670bb894187a1deda8e9478315e3bc099662d54ffbcfd83d551c8c6ab78a5`。history、累计与
逐 epoch ledger、AdamW step 41860、scheduler epoch 46、Python/NumPy/CPU CUDA 四类 RNG、
cache/config/control contract、固定 revision/clean worktree 和稳定双读均通过，
`problems=[]`。该单 cell 中间轨迹不构成 main verdict。

ADE20K train 随后原子推进到 `11776/18189`、184 个连续 shard。新增
`11712--11776` shard 大小 `54164105` bytes，SHA-256 为
`9c56c80d701e7fb1298285683a0cd7a696e3de35ea4b175d768816fd22af0b7a`；全部 64 个
source segmentation target、sample IDs、特征 shape/finiteness、fingerprint、SHA、连续
覆盖和临时文件检查通过，`problems=[]`。对应 manifest SHA-256 为
`76db34224f5c56ce4d07094cb55d99ee404e545c1e5a0b4ca098fd2ddeae853d`，sample-ID
SHA-256 为 `f51cc81c555f0d74a73b13935c7173588fb967163aa6988f366dbbd696e7644b`。

ADE20K train 随后原子推进到 `11840/18189`、185 个连续 shard。新增
`11776--11840` shard 大小 `54151689` bytes，SHA-256 为
`806f420f6da3edcc950ccf352a7b3c14639cc0931e741f9389b770cce205665c`；全部 64 个
source segmentation target、sample IDs、特征 shape/finiteness、fingerprint、SHA、连续
覆盖和临时文件检查通过，`problems=[]`。对应 manifest SHA-256 为
`6b40cd056886afa38eaf7c500cae0b6b29db404826d65a4fc50e4f4fe86c01e4`，sample-ID
SHA-256 为 `744de4c078ebe24331d51057bc301745d24ca8a2d64bde921a821031c14354bc`。

ADE20K train 随后原子推进到 `11904/18189`、186 个连续 shard。新增
`11840--11904` shard 大小 `54153545` bytes，SHA-256 为
`402680611087e331281584943e393d34c0c5331fa2eb380d52af9f6207195697`；全部 64 个
source segmentation target、sample IDs、特征 shape/finiteness、fingerprint、SHA、连续
覆盖和临时文件检查通过，`problems=[]`。对应 manifest SHA-256 为
`bc7d140365e40907f7aaa323807b205375b74612cc54073562df655a309887d9`，sample-ID
SHA-256 为 `9792b48e472e901fb8b484a4ee9bc7f9657d3631d5f282917842b89da137b831`。

ADE20K train 随后原子推进到 `11968/18189`、187 个连续 shard。新增
`11904--11968` shard 大小 `54149641` bytes，SHA-256 为
`36cb9cb51631c6650e90bc209875fd85ac68d5869571bff42b4c1c6903098a66`；全部 64 个
source segmentation target、sample IDs、特征 shape/finiteness、fingerprint、SHA、连续
覆盖和临时文件检查通过，`problems=[]`。对应 manifest SHA-256 为
`f894a1c2bdde1b3a846ec2b61c5e2692237011d44b757eb26e7952e693790e9a`，sample-ID
SHA-256 为 `10dd6f2d2b176765f8ede77e093227cf3f4591f759c5cee0ebc5a41b360f5d6d`。

`2026-08-20T12:42:23Z` 服务器继续运行期间复核：watchdog、analyzer waiter、recovery、
唯一 ImageNet-100 readout worker 与唯一 ADE20K train extractor 仍存活，正式 CUDA
进程继续显式绑定 `CUDA_VISIBLE_DEVICES=6`；两个 worktree 仍为 clean detached HEAD
`020c1de567edd88e0eda245fd085335ffe678f47`。ImageNet-100 `velocity` seed 4121 的
readout report 稳定双读仍为已提交 epoch 46，worker 正在计算 epoch 47，未把未提交进度
计入 ledger；watchdog 的全局 optimizer steps 与 12 个已完成 run 加当前 46 epoch 精确
一致。所有 main/causal/extension/final decision、`registry.json` 和
`completion_audit.json` 仍不存在，没有越过注册门提前生成。

同次持久化 cache 审计确认 ADE20K train 原子推进到 `12032/18189`、188 个连续 shard。
新增 `11968--12032` shard 大小为 `54157449` bytes，SHA-256 为
`d641c3797805018b66b4161b2cb9bb589fa8427e0bb7df231b099b9581622a9f`；全部 64 个
source segmentation target、sample IDs、state/response shape、dtype、finiteness、
fingerprint、SHA、全局连续范围、已注册文件存在性和临时/孤儿文件检查均通过，
`problems=[]`。对应 manifest SHA-256 为
`2f9e5d17d7a32ec5daa40c4c105ddb7dbf7b3fdc07cd58859ef847070f6d5c79`，sample-ID
SHA-256 为 `af3d2d757fe69bbdc726a2b5277a8261340eb100fcdbe26f14817255e9374f5e`；train
仍是运行中前缀，不提前标记完整，`changes_scientific_verdict=false`。

ImageNet-100 `velocity` seed 4121 于 `2026-08-20T12:55:05Z` 原子提交 epoch 47。
history 46→47，累计 optimizer steps `41860→42770`（+910），sample exposures
`5356930→5473385`（+116455）。epoch 47 train/validation 时间为
`2283.523969382979 / 210.93769166897982` 秒，validation top-1/top-5 为
`0.09358578052550232 / 0.30370942812983`；best 仍为 epoch 45 的
`0.09961360123647604`。report、last 与 best checkpoint SHA-256 分别为
`8dbd28bbe0c967a95792ffe0acdcc6d7593e981a5b0700f458d9ecdc5b63bf4d`、
`a3e244f311c40905f1cee113926102f5370a78c14b2f493093dfc4d73998dfa6` 与
`da1670bb894187a1deda8e9478315e3bc099662d54ffbcfd83d551c8c6ab78a5`。
history、逐 epoch ledger、AdamW step 42770、scheduler epoch 47、四类 RNG、cache、
config/control contract、固定 revision/clean worktree 与稳定双读均通过，
`problems=[]`；该单 cell 中间轨迹不构成 main verdict。

第一次 epoch 47 只读审计曾要求 report 与 checkpoint 的 Python `config` 对象逐类型
相等，因此把 JSON report 中的 list 与 checkpoint 中语义相同的 tuple 判成
`report/checkpoint config` 失败。差异只限 `config.probe.graph_grid` 与
`config.probe.times`；没有修改任何正式文件或进程。随后按固定 revision 的 JSON
序列化语义归一化 list/tuple 后重审通过。该首次失败保留为非权威负操作 provenance，
不是训练或配置内容失败，`changes_scientific_verdict=false`。

`2026-08-20T12:59:11Z` 的 ADE20K train 稳定快照已推进到 `12288/18189`、192 个
连续 shard。相对上次强审计新增 `12032--12288` 共 4 片、256 个样本；逐片 SHA-256
分别为 `f22bd1ff2d8471ec31a8b021608fea588b969986f0627cdeeb2ed9f50adce5ef`、
`f579a50ee6af4d6b076f00a29521c5d45d25391323445ad963987fc77c9edaa4`、
`df63b57c857eca525961e0ca5655212a941bccf50a5f2463c39f37c2d0f40379` 与
`1dfca740438ef539b0ddae556f75d57ae1e4394e3834f0e059a5418c7e6d2b75`。
全部 256 个 source segmentation target、sample IDs、state/response shape、dtype、
finiteness、fingerprint、全局连续性、文件集和临时文件检查均通过，`problems=[]`。
manifest SHA-256 为 `2239d81c2281f4b0f23dcb49bda9563aa5dab4ce4868b681d63a3a58ab0211d9`，
sample-ID SHA-256 为 `be9d98f5e91ad24b9f8c63c6f30f54ecf1ad8d96b879980c70c97916462351a9`；
train 仍是运行中前缀，不提前标记完整。

`2026-08-20T13:03:13Z` 的下一次 ADE20K train 原子快照推进到
`12352/18189`、193 个连续 shard。新增 `12288--12352` shard 大小为
`54168073` bytes，SHA-256 为
`d2268ecb9081ebf2fe25b3167cdad0c178b7a1c16bd782dd8042af2cc9d64d42`；全部 64 个
source segmentation target、sample IDs、state/response shape、dtype、finiteness、
fingerprint、全局连续性、文件集与临时文件检查均通过，`problems=[]`。对应 manifest
SHA-256 为 `4a1e1057802e204c8549be4950ba806de0bf7ba87024da5261f8535759f03527`，
sample-ID SHA-256 为 `31021d2a447ebcc83e305417cd89bcc87d077d909bd93c99877ca06eb0b7d7af`；
train 仍为运行中前缀，未提前标记完整。

`2026-08-20T13:06:32Z` 的后续 ADE20K train 原子快照推进到
`12416/18189`、194 个连续 shard。新增 `12352--12416` shard 大小为
`54158473` bytes，SHA-256 为
`847fb79754ef33d85c35a984fb12522084035e037855fa9963f4ea602011effd`；全部 64 个
source segmentation target、sample IDs、state/response shape、dtype、finiteness、
fingerprint、全局连续性、文件集与临时文件检查均通过，`problems=[]`。对应 manifest
SHA-256 为 `2e600bcb1bbcb983cc5cb90ed9ba52738f0fd20c72a5ca3e0574e2978b71dd99`，
sample-ID SHA-256 为 `55762eab08514b43f016018c4c5931e9b73df847e3221c801665d1bd904575af`；
ImageNet-100 epoch 48 此时仍在计算，没有把未提交进度计入正式 ledger。

`2026-08-20T13:10:26Z` 的下一份 ADE20K train 原子快照推进到
`12480/18189`、195 个连续 shard。新增 `12416--12480` shard 大小为
`54165513` bytes，SHA-256 为
`07d551d1a5d41c6efb3fd59ad43c419cd26bc25562025bcb5beb77e4d2176f7f`；全部 64 个
source segmentation target、sample IDs、state/response shape、dtype、finiteness、
fingerprint、全局连续性、文件集及临时文件检查均通过，`problems=[]`。对应 manifest
SHA-256 为 `6a553194d0d79ad9b318a04d3eb3436f194116e4383b21aa068d39703d3c9338`，
sample-ID SHA-256 为 `c74c9f58b4e2c35527effbbdacf173af827d7b161718468131f461b8c3a429a4`；
ImageNet-100 epoch 48 尚未原子提交。

`2026-08-20T13:15:29Z` 的 ADE20K train 原子快照推进到
`12544/18189`、196 个连续 shard。新增 `12480--12544` shard 大小为
`54144329` bytes，SHA-256 为
`e55ee9a92a0b404d79eb5488aedd8ea1960b1e5c377a64884e70c3ee07bb43cf`；全部 64 个
source segmentation target、sample IDs、state/response shape、dtype、finiteness、
fingerprint、全局连续性、文件集及临时文件检查均通过，`problems=[]`。对应 manifest
SHA-256 为 `105b9d7413864f5a07dae7020a61df24b66a27ecd190bb43e933f51d004949f9`，
sample-ID SHA-256 为 `c3062aa6d4e31e6121638215b5d927c0f7d4d7e4730731535ec09182fdcebefe`。
同一窗口逐进程 GPU 6 采样观测 extractor SM 从 `70%`、`77%` 上升到 `98%`，而整卡
一次 `0%` 快照落在 CPU/I/O 边界；这再次说明资源归属只能按进程和时间窗口条件描述。
未改变 batch、worker、guard 或任何科学合同，ImageNet-100 epoch 48 仍未原子提交。

`2026-08-20T13:20:32Z` 的 ADE20K train 原子快照推进到
`12608/18189`、197 个连续 shard。新增 `12544--12608` shard 大小为
`54156105` bytes，SHA-256 为
`06713ee8cf61382beaf5991548885fca9bd95e4ef6e5fef2e8eef5aa3b31be10`；全部 64 个
source segmentation target、sample IDs、state/response shape、dtype、finiteness、
fingerprint、全局连续性、文件集及临时文件检查均通过，`problems=[]`。对应 manifest
SHA-256 为 `96912ebd9389202edba191936dd0cb970fb7a2ba2df9367153b5abecb8e2a75e`，
sample-ID SHA-256 为 `56330ead201088ddfdd77750305fbf05723472b844c94f6d593f588f699089e0`；
ImageNet-100 epoch 48 仍在运行中，未计入正式 ledger。

`2026-08-20T13:25:07Z` 的 ADE20K train 原子快照推进到
`12672/18189`、198 个连续 shard。新增 `12608--12672` shard 大小为
`54152009` bytes，SHA-256 为
`b0272e20ccc12e11fdf669262bc104b902d0d83c355e6658af715b97a858b80f`；全部 64 个
source segmentation target、sample IDs、state/response shape、dtype、finiteness、
fingerprint、全局连续性、文件集及临时文件检查均通过，`problems=[]`。对应 manifest
SHA-256 为 `6554a2d2258cd7a63ca1b3e82adaac14ce59643cfbecfccf9c44c976128e69eb`，
sample-ID SHA-256 为 `b97b7b9d6a7cb5ee1bfa0a7f37d71838c58876598f46535f75c20d1f7e47f98d`；
main/causal/extension/final 与 analyzer 产物仍未出现，ImageNet-100 epoch 48 仍未提交。

`2026-08-20T13:28:30Z` 的 ADE20K train 原子快照推进到
`12736/18189`、199 个连续 shard。新增 `12672--12736` shard 大小为
`54162633` bytes，SHA-256 为
`32dc5c72c0e16b941fbc682c9870ac5137211f5a8fa150d271dfb099e73e2e4e`；全部 64 个
source segmentation target、sample IDs、state/response shape、dtype、finiteness、
fingerprint、全局连续性、文件集及临时文件检查均通过，`problems=[]`。对应 manifest
SHA-256 为 `b5a2c03523ea7af9fec301c9547fd7335b8dfca5369bf5c58c0f7753475c5af7`，
sample-ID SHA-256 为 `5634142839473414b8658e25f72110a920c154f6b8382ee7d31a7af037de32df`；
ImageNet-100 epoch 48 继续运行且未计入正式 ledger。

ImageNet-100 `velocity` seed 4121 于 `2026-08-20T13:31:59Z` 原子提交 epoch 48。
history 47→48，累计 optimizer steps `42770→43680`（+910），sample exposures
`5473385→5589840`（+116455）。epoch 48 train/validation 时间为
`1992.0745074013248 / 219.1614428376779` 秒，validation top-1/top-5 为
`0.09165378670788253 / 0.2918083462132921`；best 仍为 epoch 45 的
`0.09961360123647604`。report、last 与 best checkpoint SHA-256 分别为
`3d1b0a2a273f56f361e9efb44815c1edfbc12821d61149123363010e8276c5ee`、
`a791b7f69934e04a86ae14b53a9f72c1f77bdfb2174ebade41e453d972069baa` 与
`da1670bb894187a1deda8e9478315e3bc099662d54ffbcfd83d551c8c6ab78a5`。
history、逐 epoch ledger、AdamW step 43680、scheduler epoch 48、四类 RNG、cache、
config/control contract、固定 revision/clean worktree 与稳定双读均通过，
`problems=[]`；该单 cell 中间结果不构成 main verdict。

`2026-08-20T13:34:13Z` 的 ADE20K train 原子快照同时推进到
`12800/18189`、200 个连续 shard。新增 `12736--12800` shard 大小为
`54145481` bytes，SHA-256 为
`6d7e0c78f7189a95e460a298e14ce74a4c29bc698d82558bd6e535dcd4954547`；全部 64 个
source segmentation target、sample IDs、state/response shape、dtype、finiteness、
fingerprint、全局连续性、文件集及临时文件检查均通过，`problems=[]`。对应 manifest
SHA-256 为 `47fd46d2f4ef0451e7e41e48639e23ea9521a8f833148cecb9c6c748e958b78a`，
sample-ID SHA-256 为 `e3b9e466df029f7854fb9c707c5f890007b61d02f04fd07cdc1483435e963a48`。

`2026-08-20T13:37:53Z` 的 ADE20K train 原子快照推进到
`12864/18189`、201 个连续 shard。新增 `12800--12864` shard 大小为
`54166281` bytes，SHA-256 为
`652ab017c38817018dcb77912db5c41960cab67da59f279dc4870e0edb9f078b`；全部 64 个
source segmentation target、sample IDs、state/response shape、dtype、finiteness、
fingerprint、全局连续性、文件集及临时文件检查均通过，`problems=[]`。对应 manifest
SHA-256 为 `b7ae8241ecd3d2481186698f067bb19b75818ade7bdab9e80036f267e188c59b`，
sample-ID SHA-256 为 `b4ff5169916a86342d0c8446511b3a05d9b84ae350598d3214adb81dcd80eeb8`；
ImageNet-100 同一 worker 已连续进入 epoch 49 计算，main 及所有下游 decision 仍不存在。

`2026-08-20T13:42:40Z` 的 ADE20K train 原子快照推进到
`12928/18189`、202 个连续 shard。新增 `12864--12928` shard 大小为
`54144329` bytes，SHA-256 为
`ced0ccddfe34515e7db4d3799df9033390ef9f3d9a902fcf70b7130247d69a4b`；全部 64 个
source segmentation target、sample IDs、state/response shape、dtype、finiteness、
fingerprint、全局连续性、文件集及临时文件检查均通过，`problems=[]`。对应 manifest
SHA-256 为 `b91eed7ab97c5dc05791950dd4017dd2fd93c5c5575024a2da17c7bee3e01b68`，
sample-ID SHA-256 为 `092d1335a5e47cb860f6606cc9bc99c2109aef055a47a1180307328af566404a`；
ImageNet-100 epoch 49 持续运行，未提交的进度未计入 ledger。

`2026-08-20T13:47:22Z` 的 ADE20K train 原子快照推进到
`12992/18189`、203 个连续 shard。新增 `12928--12992` shard 大小为
`54161417` bytes，SHA-256 为
`765cb0b8b3665efe4ba4ae5f1257c423f4c33d342b4978103c84919fc79d6e64`；全部 64 个
source segmentation target、sample IDs、state/response shape、dtype、finiteness、
fingerprint、全局连续性、文件集及临时文件检查均通过，`problems=[]`。对应 manifest
SHA-256 为 `5dd09033cfe6d2750d1a9e9a1285d4cf12a896cb6cbef077f97a58f4929f6385`，
sample-ID SHA-256 为 `8c2e4561a26bc5c4f4c2e8bf32c1b5ea2b35adb28740e169f887141a580c9d12`；
ImageNet-100 epoch 49 持续运行，未提交进度未计入 ledger。

`2026-08-20T13:52:10Z` 的 ADE20K train 原子快照推进到
`13056/18189`、204 个连续 shard。新增 `12992--13056` shard 大小为
`54157257` bytes，SHA-256 为
`e2d389a6cd017200edb55f734380c96f7489c6a6d6f626fb1f03198f4c8bcc2f`；全部 64 个
source segmentation target、sample IDs、state/response shape、dtype、finiteness、
fingerprint、全局连续性、文件集及临时文件检查均通过，`problems=[]`。对应 manifest
SHA-256 为 `e0427518dd7aadc053d7176db67c6eaa641dd8fe36da799a222c145725885fb4`，
sample-ID SHA-256 为 `d714b8d296141103f608cf00d852ef66620af7361a69c42ecbbed74811169156`；
ImageNet-100 epoch 49 持续运行，正式进程与 GPU 6 资源状态正常。

`2026-08-20T13:58:49Z` 的 ADE20K train 原子快照推进到
`13120/18189`、205 个连续 shard。新增 `13056--13120` shard 大小为
`54158921` bytes，SHA-256 为
`2835093fccefadad0da0bc38e858bd0187688e15652b40f9d2686284559e66a3`；全部 64 个
source segmentation target、sample IDs、state/response shape、dtype、finiteness、
fingerprint、全局连续性、文件集及临时文件检查均通过，`problems=[]`。对应 manifest
SHA-256 为 `d4cc70425b40ca660d517e56feb32cfc5c77e6cc1ff625903c446acf92c8716d`，
sample-ID SHA-256 为 `e10497fcc7574c084c5e4cb4752b9ae1b177a7fae2ec3e0b66af6f1e66a6d20d`；
ImageNet-100 epoch 49 持续运行，未提交进度未计入 ledger。

`2026-08-20T14:03:36Z` 的 ADE20K train 原子快照推进到
`13184/18189`、206 个连续 shard。新增 `13120--13184` shard 大小为
`54161801` bytes，SHA-256 为
`0e57c22acb17a07b9c85ed52725f7a8be31619b90cc81d121cd59646dc3fa564`；全部 64 个
source segmentation target、sample IDs、state/response shape、dtype、finiteness、
fingerprint、全局连续性、文件集及临时文件检查均通过，`problems=[]`。对应 manifest
SHA-256 为 `4a3c4414799efc20ef9088dee4f16c94bf7b90a43408099f9b1bafd560e191c3`，
sample-ID SHA-256 为 `52053005cae0e2ae988dda85bc01d41bede203448b0a601ba0d16c16eec676ff`；
ImageNet-100 epoch 49 持续运行，GPU 6 仍由正式工作与阈值 guard 按阶段共享。

`2026-08-20T14:06:54Z` 的 ADE20K train 原子快照推进到
`13248/18189`、207 个连续 shard。新增 `13184--13248` shard 大小为
`54163081` bytes，SHA-256 为
`1c0468eeb4a80840b65d68247c5e9896e8108f576a7fce8a55ca1356f05a04be`；全部 64 个
source segmentation target、sample IDs、state/response shape、dtype、finiteness、
fingerprint、全局连续性、文件集及临时文件检查均通过，`problems=[]`。对应 manifest
SHA-256 为 `27d36017241721e4b01dd7f38cb1d3063cbc5f522aa44209dc9668d85bfa02d5`，
sample-ID SHA-256 为 `d3e4e003a536225d8987d84558f72d6803c3d1afa64e5db0831f6562e24603a4`；
ImageNet-100 epoch 49 仍在运行中，未提交进度未计入 ledger。

ImageNet-100 `velocity` seed 4121 于 `2026-08-20T14:08:37Z` 原子提交 epoch 49。
history 48→49，累计 optimizer steps `43680→44590`（+910），sample exposures
`5589840→5706295`（+116455）。epoch 49 train/validation 时间为
`1977.4528920715675 / 225.51653966959566` 秒，validation top-1/top-5 为
`0.10911901081916538 / 0.32047913446676973`；该 cell best 更新为 epoch 49 的
`0.10911901081916538`。report、last 与 best checkpoint SHA-256 分别为
`bf874d6e080c95bc38ccbf3669f0789768d83958b167422d04fb1cd953a6a739`、
`79ef4eae45acbfdccac1e75785e364ea1e6f38d3692f612deba9bdf82cfa6987` 与
`5c68aa986d8efdd4ef65168fa675791cc608e6eeb269573edc257587a1b3db22`。
history、逐 epoch ledger、AdamW step 44590、scheduler epoch 49、四类 RNG、cache、
config/control contract、固定 revision/clean worktree 与稳定双读均通过，
`problems=[]`；该单 cell 中间结果不构成 main verdict。

`2026-08-20T14:10:57Z` 的 ADE20K train 原子快照同时推进到
`13312/18189`、208 个连续 shard。新增 `13248--13312` shard 大小为
`54141257` bytes，SHA-256 为
`4d9ab2213221caededd47624e5eeeb9a33b4d7ebf9eb03e8c08852d2040ed478`；全部 64 个
source segmentation target、sample IDs、state/response shape、dtype、finiteness、
fingerprint、全局连续性、文件集及临时文件检查均通过，`problems=[]`。对应 manifest
SHA-256 为 `785391bceb0fb046c5113ede9eff140107c569a4d156905796439237df877d3f`，
sample-ID SHA-256 为 `e980a20f203b468f1618d568fb6d6a7eec0bdec357fa7937e35b409787a1f8f7`。

服务器重新开放后的接管复核确认既有运行拓扑没有丢失：watchdog PID `41096`、
analyzer waiter PID `41355`、recovery PID `41610`、ImageNet-100 readout PID
`102910` 与 ADE20K train extractor PID `123989` 均继续存活；两个正式 CUDA worker
均显式绑定 `CUDA_VISIBLE_DEVICES=6`，两个固定 worktree 仍为 clean detached HEAD
`020c1de567edd88e0eda245fd085335ffe678f47`。`2026-08-20T14:14:44Z` 的 GPU 6
快照为 `100%`、约 `684.15/700 W`、总/已用显存 `143771/18973 MiB`；其中正式
ImageNet readout 与 ADE extractor 分别占约 `2174/15452 MiB`，阈值 guard 占约
`1326 MiB`。未修改 batch、memory cache、worker 数、guard 或科学合同。

同轮第一次 ADE20K 只读审计错误地逐项解码整个已提交 source 前缀；为避免无必要的
共享盘 I/O 干扰正式 worker，在产生裁决输出前由操作员中断。该操作没有写入 cache、
没有修改正式进程，也没有形成权威结果；随后改用固定 revision 的
`dataset_sample_ids()` 无图像解码路径重跑。此中断保留为非权威负操作 provenance，
`changes_scientific_verdict=false`。

优化后的强审计于 `2026-08-20T14:20:56Z` 稳定覆盖 ADE20K train
`13312--13440` 两个新增 shard，共 128 个样本；快照为 `13440/18189`、210 个连续
shard。manifest 稳定双读 SHA-256 为
`482a7758803fe6a4d5fdaf27cf8a3adee0dc793459d784c73f08ebcfe74450b9`，source 前缀与
manifest 的 sample-ID SHA-256 均为
`0395da969cc0de3e97d9a0430137558348cd74137c3d0b60b3e476669a46d44c`。
`13312--13376` 与 `13376--13440` shard 大小分别为 `54147273 / 54137417` bytes，
SHA-256 分别为
`57dbae47025abf2606f64c74268a90a0696100d3ac515c4dd4661f6c9f58d141` 与
`94484b3df6b5549de484e6de6b501c0ce0cdc217f9f9d134930afa735e5ccb7f`。全部 128 个
source segmentation target 与 sample ID 精确相等；state `[64,256,14]`、response
`[64,256,192]`、segmentation `[64,512,512]` 的 shape/dtype/finiteness、完整 baseline/
graph key、fingerprint、`readout_sparse` policy、全局范围、文件集和稳定双读均通过，
`problems=[]`。ImageNet-100 `velocity` seed 4121 仍在计算 epoch 50，尚无新原子提交。

`2026-08-20T14:24:38Z` 的下一次 ADE20K train 稳定快照推进到
`13504/18189`、211 个连续 shard。新增 `13440--13504` shard 大小为
`54174409` bytes，SHA-256 为
`64f8c16559d1d5efca16a4e60b1ba15a8ea1a0db9975b300eb1e8f17f5d4643f`；全部 64 个
source segmentation target、sample ID、state/response/target shape、dtype、finiteness、
baseline/graph key、fingerprint、storage policy、全局范围、文件集与稳定双读均通过，
`problems=[]`。manifest SHA-256 为
`317b295882002663eade00827142036b1c0331050d66c4c4e70d5f5f99aba98d`，sample-ID
SHA-256 为 `17b373833a37e82d830f76cf00794a7fa4d7e4bff9624eedf38e4880ca20c51f`。

`2026-08-20T14:56:37Z` 的批量强审计将 ADE20K train 已验证边界从 13504 推进到
`13888/18189`、217 个连续 shard。新增 `13504--13888` 共 6 片、384 个样本；逐片
SHA-256 为 `377f5838161669edba883846e4fd39097872b64d29c1be60e087a31d48e5d285`、
`74818f9f6b6e08b5fe57c5f60b5a32912a7a135bce5799c281ad7e9618dfc1c6`、
`a32920d90b9a3ad77829f9c1ca78af7fdb362ec6a895dcbd0ebae30db0be0866`、
`6593775e42562283563fe25922cbe36f69aa519abb830b3646e91bf67fe213eb`、
`9be6c20967d8a8c924b6f55cca360ec254fe46a0d26f68725d2466582d07e055` 与
`6b2dc7edc649842a90ed245fdf5912a3ef87f45e564747963fbb6b0d09e4690c`。全部 384 个
source segmentation target 与 sample ID、特征/目标合同、finiteness、fingerprint、
storage policy、连续范围、文件集及稳定双读通过，`problems=[]`。manifest SHA-256
为 `19c2cc1095677e62be4a8a8b54930afe271851104c68d736b772b57b932460e1`，source 与
manifest sample-ID SHA-256 均为
`7174bad8e5a74de3d4784dba8486f1ef30b879c65e3d8dfe25600d0361020a65`。

因 ImageNet-100 epoch 50 的原子提交间隔明显长于前几轮，随后进行了不改变进程的活性
诊断。`102910` 在 15 秒窗口内用户态 CPU ticks `101041183→101167398`、`rchar`
`691405851953→692255184397`，RSS 约 `2.12 GiB`，进程状态持续为 `Rl`；这证明 readout
仍在处理 cache，而非挂死。同窗 ADE extractor PID `123989` 的进程级 GPU SM 为
`99%`、显存约 `15452 MiB`，整卡约 `100% / 693.50 W`；readout 保留约 `2174 MiB`
GPU 上下文，但当时主要处于 CPU/cache 阶段。没有重启 worker 或改变合同，epoch 50
仍只视为未提交运行中状态。

`2026-08-20T15:00:38Z` 的下一份 ADE20K train 原子快照推进到
`13952/18189`、218 个连续 shard。新增 `13888--13952` shard 大小为
`54168841` bytes，SHA-256 为
`9b9055f07784030e3457cb95ab248809af4547cfeff09b32d86f33305b1aa45f`；全部 64 个
source segmentation target、sample ID、state/response/target shape、dtype、finiteness、
baseline/graph key、fingerprint、`readout_sparse` policy、全局连续范围、文件集及稳定
双读均通过，`problems=[]`。manifest SHA-256 为
`199dc20c9ad8c1e03ea20813d53eb3a211f6b90f72387a74fc9a162b829cb2ce`，source 与
manifest sample-ID SHA-256 均为
`7666bb1f6eaf0b925185aa4d782823e5223a9270ee88adcdec6e3fa8b69fd80b`。

ImageNet-100 `velocity` seed 4121 于 `2026-08-20T15:02:44Z` 原子提交 epoch 50，
并于 `2026-08-20T15:03:52Z` 完成强审计。history 49→50，累计 optimizer steps
`44590→45500`（+910），sample exposures `5706295→5822750`（+116455）。epoch 50
train/validation 时间为 `3030.6487432196736 / 218.67831818945706` 秒，validation
top-1/top-5 为 `0.09636785162287481 / 0.301854714064915`；best 仍为 epoch 49 的
`0.10911901081916538`。report、last 与 best checkpoint SHA-256 分别为
`eaf4adf4d78ed379c480f2edcc55eaf0ca1427ce2b4dcb46c50b82b116a5b5c6`、
`306c84f390773f9084296bcf3219c9dc0d7e272ba7e5f2a370271abc155dd865` 与
`5c68aa986d8efdd4ef65168fa675791cc608e6eeb269573edc257587a1b3db22`。history、逐
epoch ledger、AdamW step 45500、scheduler epoch 50、四类 RNG、cache/config/control
语义身份、固定 revision/clean worktree 与稳定双读均通过，`problems=[]`；该单 cell
中间结果不构成 main verdict。

`2026-08-20T15:13:35Z` 的 ADE20K train 捕获快照为 `14080/18189`、220 个连续
shard；新增 `13952--14016` 与 `14016--14080` 两片大小分别为
`54171337 / 54169865` bytes，SHA-256 分别为
`8ca056b76c4f60d6376997aec38bcbab43690711d3a1006c6199c21e0332205c` 与
`815876d37779981b42bb0a04a37de20d1aa8f7b5d0e37db5774daea1a3cabc7c`。全部 128 个
source segmentation target 与 sample ID、特征/目标合同、finiteness、fingerprint、
storage policy、连续性、文件集与稳定双读均通过，`problems=[]`。对应 manifest 与
sample-ID SHA-256 分别为
`83dedde7b9084ac72542717dfca10a1b7000cbfcb4d819b3905787947c6858f0` 与
`385f9e33f54492d41f84f1705435028a355d6898f233699659106117a088cbff`。审计期间 manifest
原子推进到 14144，但已捕获的 14080 前缀保持稳定。

随后的 `2026-08-20T15:14:24Z` 强审计覆盖新片 `14080--14144`，使已验证边界推进到
`14144/18189`、221 个连续 shard。该片大小 `54153993` bytes，SHA-256 为
`a47c65619009bf6a618850699444ef7b25c589d2415998035c7cb4f65af3f012`；全部 64 个 source
target/sample ID 与其余 cache 合同检查通过，`problems=[]`。manifest SHA-256 为
`cdbaa746ce523d5ad2e4bd6202af45ed85301443415510e5e9ca0fcba1e83915`，source 与 manifest
sample-ID SHA-256 均为
`1b4bb05f490124804c4f92a795922b08b208e22aa2894245cd2bf829efde1017`。

`2026-08-20T15:27:31Z` 的后续 ADE20K train 强审计将已验证边界推进到
`14272/18189`、223 个连续 shard。新增 `14144--14208` 与 `14208--14272` 两片大小
分别为 `54168713 / 54162121` bytes，SHA-256 分别为
`57a507dd76c1ea271cda35081f6e56db086f956f08e1a7d46a90eab5911727e5` 与
`7f628d9309ec4c431e20ad07e5d69645d6d5660965fd40ca35fb6afcfea946e3`。全部 128 个 source
segmentation target 与 sample ID、特征/目标合同、finiteness、fingerprint、storage
policy、连续范围、文件集与稳定双读通过，`problems=[]`。manifest SHA-256 为
`30c618c8683b69d4cff37ebcbc461296da6b17aa7aa8ecdb71cbb78be6e6d0ea`，source 与 manifest
sample-ID SHA-256 均为
`f83a356abaa977a38713af272e281486e2faba6c16a788368b102ae0e223f6eb`。

`2026-08-20T15:29:41Z` 的下一片 ADE20K train 强审计将边界推进到
`14336/18189`、224 个连续 shard。新增 `14272--14336` shard 大小为
`54138761` bytes，SHA-256 为
`445f457d9cbfb004b0d58ec2ed1c65517a6a40255bc261c37de97189ea416cc4`；全部 64 个 source
segmentation target、sample ID、特征/目标合同、finiteness、fingerprint、storage
policy、连续范围、文件集与稳定双读通过，`problems=[]`。manifest SHA-256 为
`449e6b7b4c90ba928aa97ff0a0d576ea56ba6cfdee31fff30484f327d29ee0d1`，source 与 manifest
sample-ID SHA-256 均为
`4141f5d28188be56f42c3f2e4e7589d938d060faabe7f24dba2dbb823c16805c`。

`2026-08-20T15:39:24Z` 的后续 ADE20K train 强审计将已验证边界推进到
`14464/18189`、226 个连续 shard。新增 `14336--14400` 与 `14400--14464` 两片大小
分别为 `54173641 / 54185609` bytes，SHA-256 分别为
`f3090a8234818e13de13b3fb592cc7c33d8aebdc246e152223dc0b8ae1a29feb` 与
`a5e6519e9fe7fbe742de420effb1566e71ffa8be2fa59e8c0740d385ed700576`。全部 128 个 source
segmentation target 与 sample ID、特征/目标合同、finiteness、fingerprint、storage
policy、连续范围、文件集与稳定双读通过，`problems=[]`。manifest SHA-256 为
`075e02da5bfa38143443900e3630fe43f8e28e6bf1861a9c1effe6d034ceb8fc`，source 与 manifest
sample-ID SHA-256 均为
`66b16813551c8e2e99bc45a168b07fc5a33409368361605d24ac51887be21064`。

ImageNet-100 `velocity` seed 4121 于 `2026-08-20T15:40:25Z` 原子提交 epoch 51，
并于 `2026-08-20T15:41:01Z` 完成强审计。history 50→51，累计 optimizer steps
`45500→46410`（+910），sample exposures `5822750→5939205`（+116455）。epoch 51
train/validation 时间为 `1943.6198410382494 / 214.30400398746133` 秒，validation
top-1/top-5 为 `0.10316846986089645 / 0.3185471406491499`；best 仍为 epoch 49 的
`0.10911901081916538`。report、last 与 best checkpoint SHA-256 分别为
`acf5d1c947248020ec188b6f3160c818d012bd048a28e9dc0cc53ff26b8134ca`、
`a6a22193650c9fa72abbb2076ecf672f8d3e33ccab17a4a82a45c30d2ce73e1f` 与
`5c68aa986d8efdd4ef65168fa675791cc608e6eeb269573edc257587a1b3db22`。history、逐
epoch ledger、AdamW step 46410、scheduler epoch 51、四类 RNG、cache/config/control、
固定 revision/clean worktree 与稳定双读均通过，`problems=[]`；仍仅为单 cell 中间证据。

`2026-08-20T15:51:40Z` 的 ADE20K train 强审计将已验证边界推进到
`14592/18189`、228 个连续 shard。新增 `14464--14528` 与 `14528--14592` 两片大小
分别为 `54178697 / 54152393` bytes，SHA-256 分别为
`5c268108c36b04e716e34d51a0f9ed6adfbb71bdd0040ef2c7137872c13e28db` 与
`4bb523f6e01b1d7c473439553ac721b903598f5e30581d8558a871501e1dde3b`。全部 128 个 source
segmentation target 与 sample ID、特征/目标合同、finiteness、fingerprint、storage
policy、连续范围、文件集与稳定双读通过，`problems=[]`。manifest SHA-256 为
`b463f2b145f0c490fef88bf275441dce2bc356ba90e812d57d24a0f1932cb681`，source 与 manifest
sample-ID SHA-256 均为
`3f599107ed5b2e6d2d6b3cdaed0b7be355c98df5d32cace45af80d966d81a297`。

`2026-08-20T15:53:46Z` 的下一片 ADE20K train 强审计将已验证边界推进到
`14656/18189`、229 个连续 shard。新增 `14592--14656` shard 大小为
`54169481` bytes，SHA-256 为
`d8fbfbbc9e049a109125de99ff420db05e73be9ccc553e31e000ac1278de962e`；全部 64 个 source
segmentation target、sample ID、特征/目标合同、finiteness、fingerprint、storage
policy、连续范围、文件集与稳定双读通过，`problems=[]`。manifest SHA-256 为
`6ea49a0a2b676e13c809738455f74c8f32adc3f4de40751dc3c81c8cd424c536`，source 与 manifest
sample-ID SHA-256 均为
`5cf322ea9792ac01a61c09479218432ec08881d5825ad655485e4f937c3ed412`。

`2026-08-20T16:11:18Z` 的批量 ADE20K train 强审计将已验证边界推进到
`14848/18189`、232 个连续 shard。新增 `14656--14848` 共 3 片、192 个样本，逐片
SHA-256 为 `2c08944515f7bd6aa114be8619352c49537210ee2b39555065a978762be12e67`、
`32f8a329a06a0e63a280a1a1326d4cc123725ec6c3f0a36b6b054336d938b66d` 与
`914862911748999c26d4ce7e5059a3a369ebc1484251d0cc2c162e99436fb6fe`。全部 192 个 source
segmentation target 与 sample ID、特征/目标合同、finiteness、fingerprint、storage
policy、连续范围、文件集与稳定双读通过，`problems=[]`。manifest SHA-256 为
`0fca6d0463830d0160a0738fa866283f2fe31c35f284251521ad4ad90c6d225a`，source 与 manifest
sample-ID SHA-256 均为
`674c567a370d62b67e74b57cf47d800f38afd8af69f7fbe335bb81be02d7ccad`。

`2026-08-20T16:13:26Z` 的下一片 ADE20K train 强审计将已验证边界推进到
`14912/18189`、233 个连续 shard。新增 `14848--14912` shard 大小为
`54137353` bytes，SHA-256 为
`9cf332c12d878d81a8be5fddce0c1efd59c4bee7e0502aa8445c7e9a2d391213`；全部 64 个 source
segmentation target、sample ID、特征/目标合同、finiteness、fingerprint、storage
policy、连续范围、文件集与稳定双读通过，`problems=[]`。manifest SHA-256 为
`39eb0dcd948c276c2258e5da7977ecde60e1d92ef01fead8d272e17cc1917daf`，source 与 manifest
sample-ID SHA-256 均为
`8153238968da28162455b97045f5c4c9df82594227ade8bac17fd49fad4ff8bc`。

`2026-08-21T06:01:09Z` 的 epoch 75 首次审计被记录为非权威操作失败：审计脚本误把报告字段结构当成旧假设（`readout_memory_cache` 实际为零预算对象、`seed_workers` 在注册 runtime profile、报告不内嵌 `last_checkpoint_sha256`）。没有修改科学产物；随后按注册 schema 重跑。

`2026-08-21T06:02:25Z`，ImageNet-100 `velocity / seed 4121` 的 epoch 75 语义强审计通过：epoch 74→75，新增 `910` optimizer steps 与 `116455` exposures，累计 `68250` steps、`8734125` exposures；history `1..75` 连续，report/checkpoint history 在 JSON list/tuple 归一化后相等，AdamW step `68250`，scheduler `last_epoch=75/T_max=90`，四类 RNG、cache/config/control/coverage、runtime profile seed_workers=1、固定 revision/clean worktree 与稳定双读均通过，`problems=[]`。validation top-1/top-5 为 `0.14049459041731066 / 0.38601236476043277`，best epoch 为 75。report、last、best checkpoint SHA-256 分别为 `6976e17c594f66dded93ea2a4951b6d03a3dfde6373e2f48c55251ed0bc3285f`、`db75e705eebe2d514f8e6d72b027b12360a8de68f31abbf01e14676cd8471117`、`ea5c9d87d458f29db3effbd9ddfbb1d577b1849b66f56406e836739e9406e4f3`。这仍只是中间 cell 证据，不构成 main/causal/final 科学裁决，`changes_scientific_verdict=false`。

`2026-08-21T06:12:31Z` 的 watchdog 运行态记录显示 epoch 75 之后 worker 仍在 `imagenet100 / velocity / seed 4121` 上运行，CPU 约 `6511%`，GPU 6 `100% / 691.76 W / 3515 MiB`，进程 PID `102910`，report 最近一次原子提交为 `05:53:33Z`。主矩阵仍为 `12/60` 个已完成 cell，`execution_complete=false`；该观察不构成科学裁决，`changes_scientific_verdict=false`。

`2026-08-21T06:34:03Z`，ImageNet-100 `velocity / seed 4121` 的 epoch 76 语义强审计通过。epoch 75→76 新增 `910` optimizer steps、`116455` exposures，累计 `69160` steps、`8850580` exposures；history `1..76` 连续，report/checkpoint history 经 JSON list/tuple 归一化后相等，AdamW step `69160`，scheduler `last_epoch=76/T_max=90`，四类 RNG、cache/config/control/coverage、注册 runtime profile seed_workers=1、固定 revision/clean worktree 和稳定双读全部通过，`problems=[]`。本 epoch validation top-1/top-5 为 `0.132225656877898 / 0.364451313755796`；best 仍为 epoch 75 的 `0.14049459041731066`。report、last、best checkpoint SHA-256 分别为 `b2c15e927ed1865c74738710c0622ed65f3afd9f7653ece78a699a4ea67b7aeb`、`49cfd322e66793bddeffe4a5806f52d8ea81b728473743df6e38865785f27abc`、`ea5c9d87d458f29db3effbd9ddfbb1d577b1849b66f56406e836739e9406e4f3`；审计日志 SHA-256 为 `797491935c70fc5fb37a236ae7d8815199db989daa65705b1d4bcae4394dfa53`。这仍是单 cell 中间证据，`changes_scientific_verdict=false`。

`2026-08-21T06:44:02Z` 的 watchdog 运行态记录显示 worker 已从 epoch 76 自动继续 epoch 77，主矩阵仍为 `12/60`。PID `102910` 运行态，GPU 6 为 `100% / 691.92 W / 3515 MiB`，CPU 约 `6527%`；最新原子提交为 epoch 76。该记录仅为运行 provenance，不构成 main/causal/final 科学裁决，`execution_complete=false`、`changes_scientific_verdict=false`。

## 未完成项与科学边界

- ImageNet-100 仍有 48 个主矩阵 run 未完成（含当前运行中的 `velocity` seed 4121）；
  当前 cell 完成后还剩 47 个。
- VOC 2012 与 NYUv2 三个正式 cache split 均已完整；ADE20K val/test 已完整通过，仅
  train 仍在预取；三个任务的主矩阵尚未完成。
- 规定的因果/条件控制尚未执行。
- main、causal、extension、final decision 均不存在；extension 不得提前触发。
- 132 份逐样本回放、`registry.json` 与 `completion_audit.json` 尚不存在。
- 既有 signal gate 负结果继续保留：`status=failed`、
  `verdict=stop_or_redesign`、SHA-256
  `0378867d09a99c99b69ec7179a8ac7d20523106cc3f7b53236177e37a0e76a1e`。
- VOC 无监督报告继续保留，SHA-256
  `4382c9e53be7fa4a77b5d5a6409011cd8ce21ff4dcb3f4422bb4f668c6aa840e`。

因此 `method_effectiveness_conclusion=null`，正式验证仍必须保持 active。

`2026-08-21T01:32:19Z` 的 ADE20K train 全尾段强审计已完成并通过：从已验证边界
`14912` 覆盖到完整 `18189/18189`，共审计尾段 `3277` 个样本、完整 manifest 的
`285` 个连续 shard。manifest SHA-256 为
`e83276167077f8e1fe7e1703ea0e472ecaee77ccd8517b1317f645a41edb1f81`；manifest 与
source sample-ID SHA-256 均为
`548bfcd1d06f8d909f8665d2f695c7862ef9cfe00be6143e221adf56d3fd8258`。缺失、孤立、
临时文件均为空；尾段每个 shard 的 bytes/SHA、feature/target shape-dtype、有限性、
source segmentation target、sample ID、baseline/graph key、fingerprint、
`readout_sparse` policy 和稳定双读全部通过，`problems=[]`。完整审计原始日志位于
`/mnt/omni_ssd/user_workspace/wangzixi/FieldScope/logs/ade20k_train_tail_audit_14912_18189_20260821.json`，
SHA-256 为 `51d1605004a7eca04606fe65199b6b6576b6be23134e18920999dd4a6c1ec1d7`，耗时
`747.0582060813904` 秒。该结果只将 ADE20K train cache 标记为完成，不构成科学结论。

同一轮累计 ImageNet-100 `velocity / seed 4121` 强审计已从旧记录的 epoch 51 更新到
epoch 67：history 连续、report/checkpoint ledger、AdamW step `60970`、scheduler
epoch `67`、四类 RNG、cache/config/control、固定 revision/clean worktree 与稳定双读
全部通过，`problems=[]`。累计 sample exposures 为 `7802485`；best epoch `65`，best
validation top-1 `0.13060278207109738`。本 cell 仍为 `running` 中间证据，不能替代
四任务主矩阵或最终科学裁决。

服务器重启后另有非 FieldScope 的 M3Call 训练占用 GPU 6 约 62 GiB；FieldScope
worker 仍显式 `CUDA_VISIBLE_DEVICES=6`，没有杀掉外部进程、修改 batch/cache 或注入
人工显存。该并发资源状态作为运行 provenance 保留，`changes_scientific_verdict=false`。

`2026-08-21T01:42:01Z`，ImageNet-100 `velocity / seed 4121` 原子提交 epoch 68，
并完成强审计。epoch 67→68 新增 `910` optimizer steps 与 `116455` sample exposures；
累计分别为 `61880` 与 `7918940`。history 1→68 连续，report/checkpoint history 在
JSON list/tuple 语义归一化后相等；AdamW state step 为 `61880`，scheduler `last_epoch=68`、
`T_max=90`，四类 RNG、cache/config/control/coverage contract、固定 revision/clean
worktree 和稳定双读全部通过，`problems=[]`。report SHA-256 为
`0bf07c9ed819e42d662af7a5365254abe815ddc9a772c9c839f44d1f3b41f2db`，last checkpoint
SHA-256 为 `3ab943272f75a4d2c70d9cafd7624d43c33d60953510343323fca9602f3f9517`；best
epoch 为 65，best validation top-1 为 `0.13060278207109738`。该 cell 仍是 running
中间证据，不构成主矩阵或科学裁决。

同一 epoch 的第一次原始审计曾因把 JSON list 与 checkpoint tuple 直接比较而报出
`config` 假失败；随后按语义归一化规则重跑并通过。该非权威失败操作已单独保留，
未修改任何科学 artifact，`changes_scientific_verdict=false`。

`2026-08-21T02:18:06Z`，ImageNet-100 `velocity / seed 4121` 又提交 epoch 69，
并完成强审计。epoch 68→69 新增 `910` optimizer steps、`116455` sample exposures；
累计 `62790` steps、`8035395` exposures。history 1→69 连续，report/checkpoint history
经 JSON list/tuple 语义归一化后相等；AdamW step `62790`，scheduler `last_epoch=69`、
`T_max=90`，四类 RNG、cache/config/control/coverage contract、固定 revision/clean
worktree 与稳定双读全部通过，`problems=[]`。report SHA-256 为
`a4c335cd10b0db1495dc483edc53a4b4bacbba541654778d51a396cf4569b97a`，last checkpoint
SHA-256 为 `80aefa3aa40911ca8f2529df43f29508778cc3d6d42303fa1a64a7bb986ad035`；best
epoch 仍为 65，best validation top-1 `0.13060278207109738`。该 cell 仍是 running
中间证据，不构成主矩阵或科学裁决。

此前占用 GPU 6/7 的外部 M3Call 训练于 `2026-08-21T02:12:32Z` 结束，GPU 6 显存降至
约 `3515 MiB`；FieldScope worker 保持 PID `102910`、`CUDA_VISIBLE_DEVICES=6`，未改动
正式 batch/cache 或其它科学条件。资源竞争结束事实已作为 provenance 记录。
`2026-08-21T02:35:57Z` 再次观察到非 FieldScope M3Call baseline trainer PID `791164` 占用 GPU 6，
从 `2026-08-21T02:19:20Z` 启动；显存约 `64788 MiB`、GPU utilization `100%`。FieldScope worker
`102910` 保持 `CUDA_VISIBLE_DEVICES=6` 并继续运行；未终止外部进程、未修改 batch/cache/runtime/
scientific contract。该并发仅作为资源 provenance 保留，未推断科学影响，`changes_scientific_verdict=false`。
`2026-08-21T02:38:09Z` 复核确认外部 PID `791164` 已结束，GPU 6 显存回落至约 `3515 MiB`；FieldScope worker
`102910` 仍在 `CUDA_VISIBLE_DEVICES=6` 上运行。未重启 worker 或修改正式条件，该资源并发结束事实已保留为 provenance。
截至 `2026-08-21T02:49:41Z`，`imagenet100 / velocity / seed 4121` 仍为 `committed_epoch=69/90`。
报告自 `02:13:54Z` 后尚未原子更新，但 worker PID `102910` 仍存活，累计 CPU 约 `6505%`，
并且 `/proc/102910/io` 的 `rchar` 持续增长至约 `2.72 TB`。该观察仅记录缓存读取运行态，未推断挂死或科学影响，
`changes_scientific_verdict=false`。


`2026-08-21T12:43:35.981723Z`，ImageNet-100 `velocity / seed 4121` 的 epoch 84 语义强审计通过。epoch 83→84 新增 `910` optimizer steps 与 `116455` training-sample exposures；累计分别为 `76440` 与 `9782220`，history `1..84` 连续，AdamW step `76440`，scheduler `last_epoch=84/T_max=90`，四类 RNG、cache/config/control/coverage、注册 runtime profile `seed_workers=1`、readout batch `128`、cache batch `2`、readout memory cache `0 GiB`、固定 revision/clean worktree 与稳定双读全部通过，`problems=[]`。本 epoch validation top-1/top-5 为 `0.13786707882534777 / 0.3744204018547141`；best 仍为 epoch 75 的 `0.14049459041731066`。report、last、best checkpoint SHA-256 分别为 `2b48bc82b64dcb5c2b9f7b6dc5113003d28b99bc8651b68f1a98333d1cccc435`、`497eaacff7a56d8bfbb501062c3bc257ca02724e6152cd64029dfe9c99e8160c`、`ea5c9d87d458f29db3effbd9ddfbb1d577b1849b66f56406e836739e9406e4f3`；审计日志 SHA-256 为 `222052bfd5de2f7a0b0aad4fac3e11d028217f6bb9675634c91488991b4586e6`。GPU 6 上另有资源 guard 进程，FieldScope worker 仍显式绑定 GPU 6；未终止外部进程、未修改 batch/cache/runtime/scientific contract，资源重叠科学影响保持 `not_inferred`。这仍是单 cell 中间证据，不构成 main/causal/final 科学裁决，`execution_complete=false`、`method_effectiveness_conclusion=null`、`changes_scientific_verdict=false`。worker 继续向 epoch 85 运行。


`2026-08-21T12:56:04.964Z`–`12:56:24.969Z` 连续性审计：epoch 84 报告未被覆盖，worker PID `102910` 保持存活；`rchar` 增长 `499521816` bytes，GPU 6 保持约 `99% / 3515 MiB / 694.10 W`。这表明正式 worker 仍在有效读取/训练，未推断挂死或科学影响；`execution_complete=false`、`method_effectiveness_conclusion=null`、`changes_scientific_verdict=false`。


`2026-08-21T13:00:18.967848Z`–`13:00:48.997887Z` 连续性审计：epoch 84 报告仍未被覆盖，worker PID `102910` 存活；`rchar` 增长 `799235840` bytes。该时段 GPU 6 采样约 `100% / 3515 MiB / 691.67 W`，紧邻的 `nvidia-smi dmon` 连续采样为 SM `92–100%`、约 `690–694 W`。判定为有效慢运行，不触发恢复、不修改 batch/cache/runtime/scientific contract；`execution_complete=false`、`method_effectiveness_conclusion=null`、`changes_scientific_verdict=false`。

`2026-08-21T16:01:44Z`，ImageNet-100 `velocity / seed 4121` 原子提交 epoch 87。更正后的语义强审计通过：history `1..87` 连续，累计 optimizer steps `79170`、training sample exposures `10131585`，AdamW state step `79170`，scheduler `last_epoch=87/T_max=90`，四类 RNG、report/checkpoint history、cache/config/control/coverage、readout batch `128`、cache batch `2`、memory cache `0 GiB`、注册 runtime profile seed workers `1` 以及两个固定 revision clean worktree 全部通过，`problems=[]`。本 epoch validation top-1/top-5 为 `0.13786707882534777 / 0.37812982998454403`，best 仍为 epoch 75 的 `0.14049459041731066`。report、last、best checkpoint SHA-256 分别为 `d5f9a85860e484c4c9936832cc18c681d398ed792277d60f72de570bcb573aef`、`1649020101981dfcd1629630ac92a8bb7928fd72c1139c4db8d18f47045df706`、`ea5c9d87d458f29db3effbd9ddfbb1d577b1849b66f56406e836739e9406e4f3`。首次审计因误把运行中 report 的合法 `last_checkpoint_sha256=null` 当成摘要不一致而非权威失败；该失败日志以 SHA-256 `4012254ae886931a6e81468a70b23ad7ead82fe9a91621ea4c299b0c0d1a5730` 完整保留，未改变科学产物。该 cell 仍在向 epoch 88 自动运行，`execution_complete=false`、`method_effectiveness_conclusion=null`、`changes_scientific_verdict=false`。

`2026-08-21T16:16:20Z`，对 ImageNet-100 矩阵中已完成的前 12 个 cell 进行了逐-cell 强审计并通过，`problems=[]`。覆盖 `random_feature_local/z0/zt/trajectory × seeds 4121/7319/104729` 的注册顺序，总计 `982800` optimizer steps、`125771400` training-sample exposures。12 份 90-epoch terminal training report 的连续 history、last/best checkpoint SHA、12 份 held-out test report、checkpoint identity、test metric 回写、batch/revision/cache 合同及稳定读取全部一致。矩阵仍为 `status=running`，审计时 SHA-256 为 `d6dfa5f2e50b8e96f8205c4a01162078113e2562afec41fb9fa312e243a350aa`。第一次批量审计仅因把固定 revision 使用的原始相对 checkpoint 路径错误解析成绝对路径后比较而产生 12 个非权威假失败；失败审计以 SHA-256 `ef319fa527bc5874a95cf6e5998b266a869838ff3862ce65330bf7ae900e0b0d` 保留，更正审计 SHA-256 为 `15a4292524a94738c0e867ec5a3cca6389e1ac17a30af1a2b1ff053a3e2afe77`。该证据仍只是主矩阵中间子集，`execution_complete=false`、`method_effectiveness_conclusion=null`、`changes_scientific_verdict=false`。

`2026-08-21T17:03:49Z`，ImageNet-100 `velocity / seed 4121` 原子提交 epoch 88；强审计通过，`problems=[]`。history `1..88` 连续，累计 optimizer steps/exposures 为 `80080 / 10248040`，AdamW step `80080`，scheduler `last_epoch=88/T_max=90`，四类 RNG、report/checkpoint history、cache/config/control/coverage、readout batch `128`、cache batch `2`、memory cache `0 GiB`、注册 seed workers `1` 及两个 clean 固定 worktree 全部通过。本 epoch train/validation 为 `3352.447165149264 / 372.19575950317085` 秒，validation top-1/top-5 为 `0.13717156105100464 / 0.37534775888717153`；best 仍为 epoch 75。report、last、best checkpoint SHA-256 分别为 `235a917551adf799d9277b9935dc6872a1220d1ddc709eff9ad88da3de9ee95e`、`880867e19b75ac5f789fa7e7122b5958c21c710c1e228a62436304915e0377c6`、`ea5c9d87d458f29db3effbd9ddfbb1d577b1849b66f56406e836739e9406e4f3`。worker 已自动进入 epoch 89；`execution_complete=false`、`method_effectiveness_conclusion=null`、`changes_scientific_verdict=false`。

`2026-08-21T18:23:59Z`，同一 cell 原子提交 epoch 89；强审计 `passed`、`problems=[]`。history `1..89`、累计 optimizer steps/exposures `80990 / 10364495`、AdamW step、scheduler `89/90`、RNG、report/checkpoint、cache/config/control/coverage、固定 batch/runtime 以及 clean worktree 全部一致。本 epoch train/validation 为 `4430.68719429709 / 378.47353957686573` 秒，validation top-1/top-5 `0.13709428129829984 / 0.375193199381762`，best 仍为 epoch 75。report/last/best SHA-256 为 `eccdae654ddc4c00686931b792f4ab623d7be2e55f403d19fae0f8dcdcb72b05`、`ae03f4f801f3e9f96e0e079989e4f9bcb317a0f490a16c16aa6d995e767ce13b`、`ea5c9d87d458f29db3effbd9ddfbb1d577b1849b66f56406e836739e9406e4f3`。worker 已进入 terminal epoch 90；全局状态仍为 incomplete，`changes_scientific_verdict=false`。

`2026-08-21T19:28:52Z`，`velocity / seed 4121` terminal epoch 90 训练报告原子提交为 `passed`；训练账本强审计通过，`problems=[]`。90 个 epoch 连续，累计 optimizer steps/exposures `81900 / 10480950`，AdamW step `81900`，scheduler `90/90`，terminal report 已正式填入并匹配 last-checkpoint SHA。最后一轮 train/validation 为 `3501.412796163 / 391.5171627141535` 秒，validation top-1/top-5 `0.13709428129829984 / 0.3750386398763524`。training report/last/best SHA-256 为 `9191693ba9bc6a0a53f2cfa82cad72f529e60cac2585546a663dd331e829e9ee`、`694d808bcdb6b074b62d9ad819fc6153911fff707d82b097261909978fe66643`、`ea5c9d87d458f29db3effbd9ddfbb1d577b1849b66f56406e836739e9406e4f3`。

随后 held-out official test 与 matrix registration 完成：5000 个 test 样本，top-1/top-5 为 `0.1384 / 0.3706`，test report SHA-256 `c94a08f9b088641751d10ef7f068c591267346a59f7fa1c5a26cf0f4019057e7`。ImageNet-100 matrix 从 12 增至 13 个 completed runs，累计 `1064700` steps、`136252350` exposures，matrix SHA-256 `cf26e15d07864c41d5ba8b12ddb053614c808831e2d3a4688c512fd9f6febf38`；终态 training/test/checkpoint/cache/顺序/指标回写审计 `passed`、`problems=[]`。同一 PID `102910` 已自动进入 `velocity / seed 7319`，审计时尚无首个 epoch commit。全局仍 `execution_complete=false`、`method_effectiveness_conclusion=null`、`changes_scientific_verdict=false`。

`2026-08-21T20:30:19Z`，ImageNet-100 `velocity / seed 7319` 首次原子提交 epoch 1；强审计 `passed`、`problems=[]`。history `[1]`、optimizer steps/exposures `910 / 116455`、AdamW step `910`、scheduler `last_epoch=1/T_max=90`、四类 RNG、report/checkpoint、cache/config/control/coverage、readout batch `128`、cache batch `2`、memory cache `0 GiB`、注册 seed workers `1` 及两个 clean 固定 worktree 全部通过。train/validation 为 `3192.316484350711 / 345.9408985329792` 秒，validation top-1/top-5 `0.010046367851622875 / 0.05023183925811438`。report/last/best SHA-256 为 `60e5875608fa6e517d33a5c02169102b3cfe449332cc4f4d6d695f1c5193d78f`、`9d42747e48118a43529e4f8306673a9d0023213b80edb6f65fd58d77a3424d3e`、`ee2c47e02764fa58d2c63493584b2780056d456b96583f6a7e0f03adfa404a98`。这是早期中间值，不构成科学判定；worker 已进入 epoch 2，`changes_scientific_verdict=false`。

`2026-08-21T21:29:51Z`，同一 cell 原子提交 epoch 2；强审计 `passed`、`problems=[]`。history `1..2` 连续，累计 optimizer steps/exposures `1820 / 232910`，相对 epoch 1 增量严格为 `910 / 116455`；AdamW step `1820`、scheduler `last_epoch=2/T_max=90`、四类 RNG、report/checkpoint、cache/config/control/coverage、固定 batch/runtime 与两个 clean worktree 全部通过。train/validation 为 `3214.2878253739327 / 357.2727726493031` 秒，validation top-1/top-5 仍为 `0.010046367851622875 / 0.05023183925811438`。report/last/best SHA-256 为 `b10b9302c5b7fdd1083367b7098f55a431e0b89e40b74b83f2e2122ab938c656`、`76d167d90ef7d5ba1729c8071ca5edf19d7aefd284054c4063d7834e1de233ba`、`ee2c47e02764fa58d2c63493584b2780056d456b96583f6a7e0f03adfa404a98`。worker 已进入 epoch 3；仍不构成科学结论，`changes_scientific_verdict=false`。

`2026-08-21T22:29:27Z`，ImageNet-100 `velocity / seed 7319` 原子提交 epoch 3；语义强审计 `passed`、`problems=[]`。history `1..3` 连续，累计 optimizer steps/exposures 为 `2730 / 349365`，相对 epoch 2 增量严格为 `910 / 116455`；AdamW step `2730`，scheduler `last_epoch=3/T_max=90`，四类 RNG、report/checkpoint history、cache/config/control/coverage、readout batch `128`、cache batch `2`、memory cache `0 GiB`、注册 runtime profile seed workers `1`、两个固定 revision clean worktree 和稳定双读全部通过。该 epoch train/validation 为 `3197.5665762815624 / 377.99071024917066` 秒，validation top-1/top-5 为 `0.010046367851622875 / 0.05023183925811438`，best 仍为 epoch 1。report/last/best SHA-256 为 `6edaa6f5e1d26421259a92aa967dd18afe74a4f13b8242c53dd289dae512f3bd`、`2551f0ade42c2d85f4b3492707f453e6f3351a4b53929edd252c0902cfd5a8b0`、`ee2c47e02764fa58d2c63493584b2780056d456b96583f6a7e0f03adfa404a98`；审计 artifact SHA-256 为 `e1a8b1587781dd92d8d007104826eaf5274129ff882b0d1eeff5d138a236127d`。审计时 GPU 6 仅有 FieldScope worker 与已暂停的 resource guard，未观察到外部 M3Call。该结果仍只是单 cell 早期中间证据，`execution_complete=false`、`method_effectiveness_conclusion=null`、`changes_scientific_verdict=false`；worker 已自动进入 epoch 4。

`2026-08-21T23:31:28Z`，同一 cell 原子提交 epoch 4；语义强审计继续 `passed`、`problems=[]`。history `1..4` 连续，累计 optimizer steps/exposures `3640 / 465820`，epoch 3→4 增量严格为 `910 / 116455`；AdamW step `3640`、scheduler `last_epoch=4/T_max=90`，四类 RNG、report/checkpoint history、cache/config/control/coverage、固定 batch/runtime profile、两个 clean 固定 worktree与稳定双读全部闭合。epoch 4 train/validation 为 `3363.7035988532007 / 357.19826459512115` 秒，validation top-1/top-5 仍为 `0.010046367851622875 / 0.05023183925811438`，best 仍为 epoch 1。report/last/best SHA-256 为 `b53cb3f4d5a6c959c23918da8ecaa0f13f5def7aca71be6e07393f52f3143d16`、`16985f4e7e9c52f7afdb894d645dfbb76def9c3d4e42a609b805fe8b4ccf33fa`、`ee2c47e02764fa58d2c63493584b2780056d456b96583f6a7e0f03adfa404a98`；审计 artifact SHA-256 为 `0abaea12ad52258ceb81b3c2ac23930120a8d9ac515433ae10aabffe2ed613d3`。审计时无外部 M3Call，资源重叠科学效果保持 `not_inferred`。该中间结果不构成主矩阵或科学裁决；worker 已自动进入 epoch 5，`execution_complete=false`、`method_effectiveness_conclusion=null`、`changes_scientific_verdict=false`。

`2026-08-22T00:25:39Z`，同一 cell 原子提交 epoch 5；强审计 `passed`、`problems=[]`。history `1..5` 连续，累计 optimizer steps/exposures `4550 / 582275`，epoch 4→5 增量严格为 `910 / 116455`；AdamW step `4550`、scheduler `last_epoch=5/T_max=90`，四类 RNG、report/checkpoint history、cache/config/control/coverage、固定 batch/runtime profile、两个 clean 固定 worktree与稳定双读全部通过。该 epoch train/validation 为 `2919.9021246302873 / 330.90664293430746` 秒；validation top-1/top-5 仍为 `0.010046367851622875 / 0.05023183925811438`，作为完整早期负结果保留，不解释为最终科学结论。report/last/best SHA-256 为 `d6eda2e444f167947637d4378a2ff29f441961aee8293de9191c582361ce5c46`、`23198a383ffcd97f03dc34a13d553ad8786b50b62595c29f832c33874917f6cf`、`ee2c47e02764fa58d2c63493584b2780056d456b96583f6a7e0f03adfa404a98`；审计 artifact SHA-256 为 `2564ceda2973951e8a58019e8aaf7bd0390eb2c16fe793578cadd26322772e9a`。审计时无外部 M3Call，资源重叠科学效果保持 `not_inferred`。worker 已自动进入 epoch 6；全局仍为 `execution_complete=false`、`method_effectiveness_conclusion=null`、`changes_scientific_verdict=false`。

`2026-08-22T01:19:59Z`，同一 cell 原子提交 epoch 6；强审计 `passed`、`problems=[]`。history `1..6` 连续，累计 optimizer steps/exposures `5460 / 698730`，epoch 5→6 增量严格为 `910 / 116455`；AdamW step `5460`、scheduler `last_epoch=6/T_max=90`，四类 RNG、report/checkpoint history、cache/config/control/coverage、固定 batch/runtime profile、两个 clean 固定 worktree与稳定双读全部通过。该 epoch train/validation 为 `2952.946419845335 / 307.0158624397591` 秒；validation top-1/top-5 仍为 `0.010046367851622875 / 0.05023183925811438`，继续完整保留为早期负结果。report/last/best SHA-256 为 `a8c2de9f88180ce5fb917b92de31bc16aa17ada9a151452a4d0289fb1c1ac1c5`、`e1f80d2cf090977799ba26a29177881315dd2c230c2900bfbdecf095c3e95df0`、`ee2c47e02764fa58d2c63493584b2780056d456b96583f6a7e0f03adfa404a98`；审计 artifact SHA-256 为 `edb36ba481a99fc300dfa080ff856d431387cef9552bb008a80866254850764b`。审计时无外部 M3Call，资源重叠科学效果保持 `not_inferred`。worker 已自动进入 epoch 7；全局仍为 `execution_complete=false`、`method_effectiveness_conclusion=null`、`changes_scientific_verdict=false`。

`2026-08-22T02:13:02Z`，同一 cell 原子提交 epoch 7；强审计 `passed`、`problems=[]`。history `1..7` 连续，累计 optimizer steps/exposures `6370 / 815185`，epoch 6→7 增量严格为 `910 / 116455`；AdamW step `6370`、scheduler `last_epoch=7/T_max=90`，四类 RNG、report/checkpoint history、cache/config/control/coverage、固定 batch/runtime profile、两个 clean 固定 worktree与稳定双读全部通过。本 epoch train/validation 为 `2831.149357547052 / 352.2701987242326` 秒；validation top-1/top-5 仍为 `0.010046367851622875 / 0.05023183925811438`，继续作为完整早期负结果保留。report/last/best SHA-256 为 `ee8c003b73c4e2a8607a1ecd60fd68c528e7fedf9197da4caec92e95b3226eea`、`199c518ef09ac2108cb1512c32e96947f529891c459b19dbfcd16209f3923337`、`ee2c47e02764fa58d2c63493584b2780056d456b96583f6a7e0f03adfa404a98`；审计 artifact SHA-256 为 `cc2c1cb7ba0fb261b1dbc674a061ae5cc6fcc2028bcaf63367f48cc83f7a889a`。审计时无外部 M3Call，资源重叠科学效果保持 `not_inferred`。worker 已自动进入 epoch 8；全局仍为 `execution_complete=false`、`method_effectiveness_conclusion=null`、`changes_scientific_verdict=false`。

`2026-08-22T05:28:48Z` 恢复 SSH 权威访问后确认：两 worktree 仍 clean 且固定在 `020c1de567edd88e0eda245fd085335ffe678f47`，worker PID `102910`、watchdog PID `41096`、recovery PID `41610` 与 analyzer PID `41355` 均持续存活，GPU 6 guard pause 文件仍存在，未重启任何正式进程。断联期间 ImageNet-100 `velocity / seed 7319` 已原子提交到 epoch 10；强审计 `passed`、`problems=[]`，history `1..10` 连续，累计 optimizer steps/exposures `9100 / 1164550`，相对 epoch 7 增量 `2730 / 349365`，AdamW step `9100`、scheduler `last_epoch=10/T_max=90`、四类 RNG、report/checkpoint history、cache/config/control/coverage、readout batch `128`、cache batch `2`、memory cache `0 GiB`、注册 profile `selected_profile.seed_workers=1`、稳定双读与两个 clean worktree 全部通过。epoch 10 train/validation 为 `2916.47917692177 / 315.47849005740136` 秒，validation top-1/top-5 仍为 `0.010046367851622875 / 0.05023183925811438`，best 仍为 epoch 1。report/last/best SHA-256 为 `6bbc3d2092a943d59d244dcc4e532a9bb84d74ddc86063262b939f169be462e3`、`15101ebeae8c8a8c2fac667432d5aa28472f5a75eb4ee78e7bf6c1cdfbbc7aed`、`ee2c47e02764fa58d2c63493584b2780056d456b96583f6a7e0f03adfa404a98`；权威审计 SHA-256 为 `25c0e48395af190f71b429f1fbb70b50d42911194c548e5284ec6d6b11f48387`。

同次审计在 GPU 6 观察到外部 `m3call_baselines` PID `1368198` 占用约 `61806 MiB`；未终止该外部进程，未修改 FieldScope batch/cache/runtime/scientific contract，资源重叠科学效果保持 `not_inferred`。首次 epoch-10 审计因读取 runtime profile 的 `seed_workers` JSON 路径错误而生成非权威 `failed` artifact，SHA-256 `c35e374c6da05a5173f4c7ad0c4682f4ae57c96b607b19bff9338a92309115c1`，已完整保留；修正只涉及审计读取路径，不涉及正式训练。当前 ImageNet-100 仍为 `13/60` terminal cells，其余三任务为 `0/60`；全局正式 optimizer steps/exposures 更新为 `1073800 / 137416900`，且仍为 `execution_complete=false`、`method_effectiveness_conclusion=null`、`changes_scientific_verdict=false`。

`2026-08-22T06:12:27Z`，按用户新增 GPU 6/7 授权启动跨任务并行：原 GPU 6 ImageNet-100 worker/recovery/watchdog/analyzer 均未重启；GPU 7 新增 VOC2012 segmentation worker，父 PID `3925620`、CLI PID `3925626`。VOC 合同仍为固定 revision、20 representations、3 seeds、80 epochs、batch 4、readout memory cache 0 GiB、注册 `selected_profile.seed_workers=1`，输出根与 ImageNet-100 不重叠。GPU 6/7 guard pause 均已建立，独立锁防止重复 VOC worker，安全 sidecar PID `3961934` 在主 recovery PID `41610` 退出时终止 GPU 7 父 worker，以免遮蔽 watchdog 的主链恢复判断。启动强审计 `passed`、`problems=[]`，SHA-256 `fde06902b3be7a0023ce54d442fcb8224e08e5c61f4a7d5e6884220b92cb7286`。首次 sidecar 设置因 PID 文件转义成字面 `3925620n` 而 fail-closed；VOC 训练已成功启动且未受影响，PID 文件随后原子修正，失败 artifact SHA-256 `296bf56ea72ebfa8b9a4b49094371ecd79584b5d3fd8d2c54474e431c1d9a43e` 完整保留。外部 M3Call 未终止，资源重叠科学效果保持 `not_inferred`；`execution_complete=false`、`method_effectiveness_conclusion=null`、`changes_scientific_verdict=false`。

`2026-08-23T10:27:08Z`，在不重启或修改 VOC 训练的情况下完成 GPU-7 安全 sidecar 加固。旧 PID `3961934` 仅向父 shell 发送 `TERM`，存在子 CLI 成为 orphan 的操作风险；其已由 PID `2925950` 替换。新 sidecar 已核验 VOC 专属进程组 `3921096` 的成员仅为 `3925620/3925624/3925625/3925626`，当主 recovery PID `41610` 退出时对整个组发送 `TERM`，15 秒后若 CLI 仍存活则升级为 `KILL`。加固审计 `status=passed`、`problems=[]`，SHA-256 `b0ff4a87a42e9b866d372f9eebc9694034229ba3d7877db539d81051f3d727b1`；`training_restarted=false`、`training_outputs_modified=false`。本次仅改变操作安全边界，`execution_complete=false`、`method_effectiveness_conclusion=null`、`changes_scientific_verdict=false`。

`2026-08-23T10:37:12Z`，按用户“显存足够时继续跨任务并行”的授权，在不重启现有 ImageNet-100/VOC2012 worker 的情况下，于物理 GPU 7 新增 ADE20K segmentation worker：父 PID/PGID `2969678`、CLI PID `2969687`、安全 monitor PID `2969870`。正式合同保持 80 epochs、batch 2、150 classes、20 representations × 3 seeds、`selected_profile.seed_workers=1`、readout memory cache 0 GiB；ADE train/val/test manifest `18189/2021/2000` 全部 `passed`、complete、clean 且绑定固定 revision。独立锁、输出根隔离、GPU-7 guard pause、仅 ADE20K 的进程组成员和 recovery 退出后的整组 `TERM`→15 秒→`KILL` 联动均通过。权威启动审计 SHA-256 为 `e5a82171b85833351447023dfe2429b3fa7eda855a78d6d03f13dd2be882970b`。启动后的只读日志 tail 因 PowerShell→SSH 管道残留回车而失败，但 worker/锁/monitor 已成功建立且训练没有失败或重启；该非权威失败 artifact SHA-256 `315b9ff1b248da8046b049d0609b2b80d1ebaec0bce91fa74ac29a9b2313d5fe` 已完整保留。三 worker 合计 CPU 约 96 核，接近 100 核 cgroup 上限，因此 NYUv2 暂以首个 ADE 原子 epoch 和整体吞吐为门，不因显存空闲直接过度订阅 CPU。科学状态不变：`execution_complete=false`、`method_effectiveness_conclusion=null`、`changes_scientific_verdict=false`。

随后根据用户明确要求以可用显存继续跨任务并行，并确认 ADE20K 持续存活、无 OOM、无锁冲突后，于物理 GPU 6 启动 NYUv2 depth worker：父 PID/PGID `2992173`、CLI PID `2992183`、安全 monitor PID `2992347`。至此四个注册主任务同时执行：GPU 6 为 ImageNet-100 + NYUv2，GPU 7 为 VOC2012 + ADE20K；每任务仍仅 `seed_workers=1`，四个输出根及锁相互隔离。NYUv2 合同为 80 epochs、batch 4、20 representations × 3 seeds；train/val/test manifest `715/80/654` 全部 `passed`、complete、clean 且绑定固定 revision。进程组 exclusivity、GPU-6 guard pause、主 recovery 退出后的整组 `TERM`→15 秒→`KILL` 联动均通过；权威启动审计 SHA-256 `25fe195222381d8fab7a8672a68246de85e62f56deeae59d39b32b0fd25706f3`。审计文件写成后 SSH 末尾执行了孤立回车命令，未影响训练；非权威失败 artifact SHA-256 `401e5b6e46d77e1df1a1b4a3950ed6dda0b2c331ed014a68882803d9c6bcd61e` 已保留。CPU 超配风险明确记录，后续以四 worker 的原子 epoch 实测总吞吐决定是否维持；科学状态仍为 `execution_complete=false`、`method_effectiveness_conclusion=null`、`changes_scientific_verdict=false`。

四路并行后的首批原子证据表明原 recovery 链仍在推进。ImageNet-100 `velocity / seed 7319` 已提交 epoch 32：累计 `29120` optimizer steps、`3726560` sample exposures，本 epoch train/validation 为 `4295.673937787302 / 525.0698408698663` 秒，validation top-1/top-5 为 `0.02990726429675425 / 0.12998454404945906`，best 更新为 epoch 32。强审计覆盖稳定双读、history `1..32`、report/checkpoint history、AdamW step `29120`、scheduler `32/90`、四类 RNG、cache/control/coverage/config、固定 revision/tree、两个 clean worktree 与 `seed_workers=1`，`status=passed`、`problems=[]`；artifact SHA-256 `452d8442b12b5be0078bd6bbb0c78df1a7966fbf6eb2e06f54a13d342c450877`。该中间值不构成科学结论。

VOC2012 `random_feature_local / seed 104729` 的 epoch 33 同样通过强审计，累计 `10890` steps、`43494` exposures；本 epoch train/validation 为 `464.7219314686954 / 111.73075172770768` 秒，validation mean-IoU/pixel accuracy 为 `0.034048039466142654 / 0.648590624332428`，best 仍为 epoch 28 的 `0.03518079221248627`。report/checkpoint/RNG/optimizer/scheduler/cache/control/coverage/revision 账本全部闭合，artifact SHA-256 `ba14a0b1f5e40779d7353168aeade7c4869135a91fd21e46510426da6b5403bf`。

同时完成 VOC2012 前两个 terminal cells 的终态强审计。seed 4121 与 7319 均覆盖连续 80 epoch、`26400` steps、`105440` exposures、last/best checkpoint、完整 held-out test、matrix 单一注册、test metric 回写、固定 cache/control/revision 和 `seed_workers=1`，均为 `status=passed`、`problems=[]`。两 seed 的 held-out test mean-IoU 分别为 `0.036541689187288284` 与 `0.0365767702460289`，均作为完整低结果保留，不因数值低而省略或改写；审计 SHA-256 分别为 `cf67968124399ca5ddcdc1ac7b3ebc4d5cbe628460de45e708b0a93d679c190c`、`8b403df96397ded2aa1b26a053c711676a4c8c66fb3242cd73b28b66e94efa15`。

NYUv2 `random_feature_local / seed 4121` 已在四路并行下产生首个可恢复原子 commit。epoch 1 为 `179` optimizer steps、`715` sample exposures，train/validation `499.51651001535356 / 59.95765916723758` 秒，当前 best primary metric `0.36730616837739943`。固定 revision/tree、history、last/best checkpoint、优化器、scheduler `1/80`、四类 RNG、depth cache/control/coverage 与两个 clean worktree均通过，强审计 SHA-256 `2644a003a8988fcae2a0a9bc6a8775e0b904cb28e1ec2c3ef6b009916111320c`，`problems=[]`。ADE20K 此时仍在首个较长 epoch，尚无原子报告；不得把进程存活替代为完成证据。按最新原子报告计算，全局已观察 `1157689 / 51013200` optimizer steps 与 `140233999 / 725922600` training-sample exposures。最终 main/causal/extension/final、132 replays、`registry.json` 与 `completion_audit.json` 仍不存在且不得提前生成；`execution_complete=false`、`method_effectiveness_conclusion=null`、`changes_scientific_verdict=false`。

VOC2012 随后原子提交 epoch 34，并再次通过同一强审计合同：history `1..34`、`11220` steps、`44812` exposures、AdamW/scheduler `34/80`、四类 RNG、report/checkpoint、cache/control/coverage 与固定 revision/clean tree 全部闭合，`problems=[]`。epoch 34 train/validation 为 `826.9531709142029 / 148.7837290968746` 秒，validation mean-IoU/pixel accuracy `0.03402479737997055 / 0.6665673851966858`，best 仍为 epoch 28 的 `0.03518079221248627`；审计 SHA-256 `11b35336fb3eccbab9db684837489179dc99417a4259b2b05083e748bb921f0d`。相较 epoch 33 的 `464.7219314686954 / 111.73075172770768` 秒，单任务速度出现 CPU 超配下的可测下降；但新增 NYUv2 已同时完成首个 epoch、ADE20K 也在并行积累首 epoch。按用户明确授权暂维持四路，并继续以多个完整 epoch 的全局总 steps/exposures 增量判断净吞吐，而不是仅凭某一个任务的 wall time 调整。最新全局原子总量为 `1158019 / 51013200` steps 与 `140235317 / 725922600` exposures；科学状态不变。

为持续捕获四路 worker 的原子恢复点，启动外部只读强审计 watcher PID `3047103`，以 `nice=19`、idle I/O、600 秒间隔运行；脚本不在固定 worktree 中，不改训练报告/checkpoint/matrix，只读取已原子提交的 `running` report 并输出独立 audit JSON，且在主 recovery PID `41610` 退出时自动停止。watcher/helper SHA-256 分别为 `da1d2149038e41c9eb87380600f40f58b2be35c50f92ecc0566fc2b12b1f5604` 与 `fa668e47ce879d903cd17036961fbdbca901c8fc96d4eda7975a574e50318ae5`。首轮 `failures=[]`，并捕获 NYUv2 epoch 2：`358` steps、`1430` exposures，本 epoch train/validation `479.0122753372416 / 57.20299460925162` 秒，validation abs-rel/delta1/delta2/delta3/MAE/RMSE 为 `0.3514666877686977 / 0.3307942390441895 / 0.6341113567352294 / 0.8406130790710449 / 1.117336204648018 / 1.5425260309468134`。强审计 `passed`、`problems=[]`，SHA-256 `7fefd030ed9903639ff908ee32a243b3d7167a7bedc90746bac18edeb0b04545`。最新全局原子总量更新为 `1158198 / 51013200` steps 与 `140236032 / 725922600` exposures；仍不构成科学结论。

随后将只读 watcher 无损升级为同时自动审计 terminal cells：旧 PID `3047103` 正常退出，新 PID `3062994` 保留原 running-epoch state 后接管；新版 SHA-256 `ae99087a7fb34519ed8938df02b8f65150908b198e76342c95e47c6576317648`，terminal helper SHA-256 `d67d3c61f3a78d0ad8dcec63b97519282928f4f018c294479b98c9d61894c34b`。terminal 审计覆盖完整训练 history、last/best checkpoint、optimizer/scheduler/RNG、held-out test、test checkpoint、cache/control/config、matrix 唯一注册及 metric/steps/exposures 回写。首轮对 ImageNet-100 现有 13 个与 VOC2012 现有 2 个 terminal cells 共 `15/15` 全部 `passed`，`failures=[]`；manifest SHA-256 `0e462eda6606af4302a5788af62695981132143c5f2edc72c8454bc2da16c76a`。升级未修改或重启训练，不改变科学裁决。

`2026-08-23T11:11:58Z`，ADE20K 首个正式 cell
`random_feature_local / seed 4121` 在 epoch 1 内失败，未产生任何原子 epoch、
training report、checkpoint 或 matrix entry。不可变日志记录非有限 training loss，
随后在 `loss.backward()` 报出 `CUBLAS_STATUS_EXECUTION_FAILED`；日志 SHA-256 为
`519edea4a325e5fb24ccd5d3939cd538b283d065303d159f5eaf9428490d029f`。
父/CLI/monitor `2969678/2969687/2969870` 均已退出，独立锁已释放，ADE 输出根为空，
因此 ADE 正式进度仍严格为 0。两个固定 worktree 继续 clean 且绑定
`020c1de567edd88e0eda245fd085335ffe678f47`，三个 cache manifest 继续 passed/complete；
cgroup OOM 计数全为 0，磁盘仍有 `43121211604992` bytes 可用。随后 GPU 7 出现一条
报告为其他瞬时 PID `1067227` 的 `Xid 43`，与失败 ADE PID 不同，因果归属保持
`not_inferred`。机器可读失败审计为
`artifacts/reports/ade20k_gpu7_epoch1_failure_audit_20260823.json`。未更改 LR、batch、
seed、representation、cache 或科学合同；失败与后续原条件恢复均须完整保留。
全局已提交总量不因本次未提交 epoch 增加，科学状态仍为
`execution_complete=false`、`method_effectiveness_conclusion=null`、
`changes_scientific_verdict=false`。

完成失败审计并通过 GPU 7 固定 UUID、确定性 CUBLAS、有限值及 ECC/driver 健康门后，
于 `2026-08-23T11:23:37Z` 启动 ADE20K 同条件 retry 1：父/PGID `3232226`、CLI
`3232250`、安全 monitor `3232368`。80 epochs、batch 2、LR `0.001`、weight decay
`0.0001`、150 classes、20 representations × 3 seeds、`seed_workers=1`、0-GiB readout
memory cache、三个 cache manifest 与固定 revision 均未改变，且未添加
`strict_host_sync`。重启前 ADE 输出根仍为空，旧失败日志未覆盖；独立锁已由 retry 1
持有，进程组只包含 ADE parent/task/matrix/CLI，主 recovery 退出联动继续有效。
机器可读启动审计为
`artifacts/reports/ade20k_gpu7_retry1_launch_audit_20260823.json`（SHA-256
`cd0849e3c7d2d7f78f32d900f58ffcbd7bb0c617465f0a0e7e9ac73cdf46a27c`），诊断脚本 UUID
格式/JSON 序列化与 PowerShell→SSH 回车造成的非权威只读失败另存为
`artifacts/reports/ade20k_gpu7_retry1_launch_non_authoritative_failed_20260823.json`
（SHA-256 `dec80bdaf0af3bbb17db0782bb726d87dfb7bf4f7f0866a90b4886aa7b0dfa03`）。
retry 1 尚无原子 epoch，故 ADE 与全局正式已提交总量均不增加；首次失败不会被重试
覆盖或从最终负结果中删除，科学状态不变。

随后完成 ADE20K train 全 18,189 标签的低优先级只读扫描：16 个样本全部像素均为
ignore 255，且不存在越界的非 ignore label。注册的 seed 4121 epoch-1 sampler 在固定
batch 2 下恰好把 cache index `5336/5328`（`ADE_train_00005913/00005905`）组成
zero-based batch 1697；PyTorch 2.7.1 的 mean cross-entropy 对整批无有效像素返回 NaN，
从而确定性触发固定 revision 的 finite-loss gate。完整 scan SHA-256 为
`ef4d7b5a0a9474fbe8692de8ca3c87191a80b8335155ce292719cb28fcf83788`。

在证明 retry 1 必然复现同一失败后，于其到达该 batch 前对 ADE 专属进程组发送 TERM；
其无正式输出、无原子 epoch、无进度计入，日志仅记录 `Terminated`，SHA-256
`b4a6c06672677cf0e25ec72bec83beae33305cee4b8087f168962014373b02bb`；退出证据 SHA-256
`aad83fc4aae8b5c5d67c5e0fffe9576ee07d04f28955a2712953ea417a779703`。ImageNet-100、
VOC2012、NYUv2 worker 未重启并继续运行。

按既有固定 revision 外部内容寻址兼容层先例，锁定 ADE all-ignore recovery protocol：
保留全部样本，不改 batch/sampler/seed/LR/weight decay/epochs/representation/step/exposure；
只在整批无任何有效监督像素时返回可微零 loss/零梯度并仍执行注册 optimizer step，任何
含有效像素的 batch 继续调用原 PyTorch cross-entropy。最终 ADE-only shim SHA-256
`f95bbb0ed70ebbc127e6e08cd8d47c7bb18e6d394c2233a5677dc9cdfbf7bac4`，Ruff、
`py_compile` 与 GPU-7 逐值/逐梯度自测均 passed；self-test SHA-256
`1aa00970165abc2ace956ba60259df0d5cc1b845acfe42f9a678d0162c291cc8`。固定 formal/
analysis worktree 未修改。协议与机器可读 gate 分别为
`docs/experiment_plans/2026-08-23_ade20k_all_ignore_recovery_protocol.md` 和
`artifacts/reports/ade20k_all_ignore_recovery_gate_20260823.json`。该恢复只补全结构上未定义
的空监督 batch，不改变科学裁决，且原失败继续完整保留。

`2026-08-23T11:49:51Z`，在重新核验两个 clean 固定 worktree、主 recovery PID
`41610`、GPU-7 guard pause、空 ADE 输出根、独占锁、无重复 ADE writer，以及 recovery
gate/full scan/self-test/final shim 的精确 SHA 后，启动 ADE retry 2：父/PGID `3634680`、
CLI `3634699`、安全 monitor `3635008`、低优先级退出证据 monitor `3639896`。CLI 环境
直接核验 `CUDA_VISIBLE_DEVICES=7`、最终 shim 路径与 SHA identity；进程组仅含
ADE parent/task/matrix/CLI 并持有独立锁。机器可读启动审计为
`artifacts/reports/ade20k_gpu7_retry2_launch_audit_20260823.json`。

retry 2 启动审计时仍无 ADE 原子 epoch 或正式输出，因此 ADE 进度严格保持 0；同一时刻
其它三个未重启 worker 已继续推进到 ImageNet-100 active epoch 32、VOC2012 active epoch
37、NYUv2 active epoch 8。下一项决定性 ADE 证据必须是跨过 zero-based batch 1697 后的
原子 epoch-1 report；进程存活本身仍不是进度或科学证据。

自动强审计 watcher 随后确认三条既有 worker 在 ADE 恢复期间继续闭合账本且
`failures=[]`：VOC2012 `random_feature_local / seed 104729` 已审计通过 epoch
35/36/37，最新审计 SHA-256
`65fab362cf4c9e5e6d39807aba20bfcefa3197d161c47d8e7e3ed624a685094d`；NYUv2
`random_feature_local / seed 4121` 已审计通过 epoch 4/5/6/8，最新审计 SHA-256
`38c0adcd78ab473d3e45146dea66067ebf85bcf0432ab28c012f15e0c2cc1c9c`。NYUv2 epoch 8
累计 `1432` optimizer steps、`5720` exposures，validation abs-rel
`0.36146524250507356`；VOC 与 NYUv2 的这些值均只是中间轨迹，不构成科学裁决。

ADE 全固定训练调度枚举进一步确认，三个 seeds × 80 epochs 共含 22 个 all-ignore
training batches：seed `4121/7319/104729` 分别 `7/10/5` 个，占注册
`43,656,000` optimizer steps 的 `22` 步。因此 ADE-only shim 必须贯穿完整矩阵，不能在
epoch 1 后卸载。validation 的 2,021 样本中有 1 个全-ignore sample，但固定 batch 2 下
无全-ignore batch；test 2,000 样本中没有此类样本。训练调度与 val/test scan SHA-256
分别为 `ace13a3ba06dd20f28e45a29a9e692e30ff657a7c91e2a84412f1a50de0571fd` 和
`e60d7cbd259baff88986b17cbae2118e89510674e80101d7b08e7c847e9c0e78`；合并覆盖审计为
`artifacts/reports/ade20k_all_ignore_coverage_audit_20260823.json`。

为保证 ADE 首个原子 report 不仅通过普通 checkpoint/history 审计，还能绑定外部兼容层
身份，已启动一次性只读增强 monitor PID `3678237`（`nice=19`、60 秒轮询）；脚本
SHA-256 `d56bd39ac8bb06d701da731c8d547418e247df506d31f20697ea50b0032b1c6a`，Ruff 与
`py_compile` passed。它在 epoch 1 出现后同时核验标准强审计、固定 worktree、CLI 实际
环境、shim/gate/scan SHA 与 all-ignore batch 身份；若 retry 2 先退出则固化失败。启动
审计为 `artifacts/reports/ade20k_retry2_epoch1_bound_audit_monitor_launch_20260823.json`。

`2026-08-23T12:08:50Z` 的自动只读强审计继续返回 `failures=[]`。VOC2012
`random_feature_local / seed 104729` 已原子提交并审计通过 epoch 38：连续 history
`1..38`、累计 `12540` optimizer steps、`50084` sample exposures、AdamW step 与
scheduler `38/80`、四类 RNG、report/checkpoint/cache/control/coverage、固定 revision
及两个 clean worktree 均闭合，`status=passed`、`problems=[]`；artifact SHA-256
`cf4629b7d7e8d25e135c03424deff36cee761b076bfd9dfeddafdb3322a47edb`。该 epoch
validation mean-IoU/pixel accuracy 为 `0.03484974429011345 / 0.6505610346794128`，
best 仍为 epoch 28 的 `0.03518079221248627`，低值按原样保留且不构成科学裁决。

NYUv2 `random_feature_local / seed 4121` 的 epoch 9/10 也分别通过同一强审计合同，
artifact SHA-256 为
`332116c228274c34398c887224bc82e0d107127b64b95b681782fc003d962cb1` 与
`7d8743c0ad5122ff868b90cc3788ddcb2482863b2da1ac0f0b6202b64195e616`。
epoch 10 累计 `1790` optimizer steps、`7150` exposures，scheduler `10/80`，
validation abs-rel/delta1 为 `0.42928059548139574 / 0.42122464179992675`；best 仍为
epoch 3 的 abs-rel `0.34919863343238833`。最新全局原子总量因此更新为
`1160950 / 51013200` optimizer steps 与 `140247024 / 725922600` training-sample
exposures。ADE20K retry 2 此时仍在首个较长 epoch、尚无原子 report，故其正式进度仍为
0；进程存活和 GPU 活跃不能代替该证据。

`2026-08-23T12:11:17Z` 的资源快照观察到外部 M3Call `stage1_trainer.py` 两个进程
PID `3660411/3660412` 自 `11:53:15Z` 起分别占用物理 GPU 6/7 约
`61808/61522 MiB`；其环境可见 GPU `6,7`，命令带 `--stop-after-step 256`。同一快照中
两卡仍分别剩余约 `77407/79025 MiB`，四个 FieldScope worker 全部存活：GPU 6 为
ImageNet-100 + NYUv2，GPU 7 为 VOC2012 + ADE20K。未终止外部进程、未重启正式训练、
未修改 batch/cache/runtime/seed 合同，也未启动重复 dataset/cell writer；机器可读记录为
`artifacts/reports/gpu67_m3call_overlap_20260823.json`。资源重叠的科学效果保持
`not_inferred`；四路跨任务并行继续，每任务 `seed_workers=1`。整体状态仍为
`status=active`、`execution_complete=false`、`method_effectiveness_conclusion=null`、
`changes_scientific_verdict=false`。

自动 watcher 的下一轮于 `2026-08-23T12:19:10Z` 再次得到 `failures=[]`：NYUv2
已审计通过 epoch 11（`1969` steps、`7865` exposures，artifact SHA-256
`19df2cf14d944da4282b8bd6aaa99b311bde7547756b12ee3c94da0562cbcba9`），VOC2012
已审计通过 epoch 39（`12870` steps、`51402` exposures，artifact SHA-256
`21823e4ef2819cf6b2de473350bbd13fc1e74190921d2d8bc009b444c6c7f2f6`）。两份均为
`status=passed`、`problems=[]`，且保持 `resource_overlap_scientific_effect=not_inferred`。
全局原子总量更新为 `1161459 / 51013200` optimizer steps 与
`140249057 / 725922600` exposures；ADE retry 2 仍未产生首个原子 epoch。

`2026-08-23T12:29:29Z`，watcher 下一轮仍为 `failures=[]`，并闭合两个新原子点。
ImageNet-100 `velocity / seed 7319` epoch 33 通过全部强审计：history `1..33`、
`30030` optimizer steps、`3843015` sample exposures、AdamW step、scheduler `33/90`、
四类 RNG、report/last/best checkpoint、cache/config/control/coverage、固定 revision 与
两个 clean worktree 均匹配，`status=passed`、`problems=[]`；artifact SHA-256
`4254d04c096991549af800510e213bc62d48a20ca12adda074653204e47e760a`。本 epoch
train/validation 为 `5037.084784193896 / 641.5599445179105` 秒，validation top-1/top-5
为 `0.03253477588871716 / 0.14760432766615147`，best 更新为 epoch 33；低中间结果按原样
保留，不构成 main verdict。

NYUv2 `random_feature_local / seed 4121` epoch 12 同样通过全部强审计，累计
`2148` steps、`8580` exposures，scheduler `12/80`，artifact SHA-256
`b43876b4c5c2dc2b038efe5924a3b19e8040161dd61685d280268f60872eac68`；validation
abs-rel/delta1 为 `0.37221103683114054 / 0.392230224609375`，best 仍为 epoch 3 的
`0.34919863343238833`。同步的 watcher state SHA-256 为
`25c12edb429e6c0111d5b4f862a0bb282a0bb115106ecdf72b9a057872fca73a`。全局原子总量
更新为 `1162548 / 51013200` optimizer steps 与 `140366227 / 725922600` exposures。
ADE retry 2、shim-bound monitor、其它三任务 worker、主 recovery/watchdog 继续存活；
ADE 仍无 epoch-1 原子 report，故正式进度保持 0。

`2026-08-23T12:39:45Z` 的后续 watcher 轮次继续为 `failures=[]`。VOC2012
`random_feature_local / seed 104729` epoch 40 通过强审计，累计 `13200` optimizer
steps、`52720` exposures、scheduler `40/80`，artifact SHA-256
`4b3175414aca971a1cb9fbb65200e0e171a60fc19efce2eaae71cb3675bfbc0a`；validation
mean-IoU/pixel accuracy 为 `0.033954840153455734 / 0.637732744216919`，best 仍为
epoch 28 的 `0.03518079221248627`。NYUv2 `random_feature_local / seed 4121`
epoch 13 也通过强审计，累计 `2327` steps、`9295` exposures，artifact SHA-256
`2396e61dd980a344238eea1e557178d3d143b19cd0df58cb00d9a27a7deb5c9b`；validation
abs-rel/delta1 为 `0.37457831799983976 / 0.37751035690307616`，best 仍为 epoch 3。
两份均 `status=passed`、`problems=[]`，负/低中间值完整保留。watcher state SHA-256
更新为 `789628cacd3dafcd26129a580f18e491cbf52fe2053b3135409b8e01be6f67ff`；全局原子
账本更新为 `1163057 / 51013200` optimizer steps 与
`140368260 / 725922600` exposures。ADE retry 2 继续存活但仍无原子输出。

`2026-08-23T12:49:55Z`，NYUv2 `random_feature_local / seed 4121` 又原子提交并强审计
通过 epoch 14：history `1..14`、`2506` optimizer steps、`10010` exposures、scheduler
`14/80`、optimizer/RNG/report/checkpoint/cache/control/revision/clean worktree 全部闭合，
`status=passed`、`problems=[]`；artifact SHA-256
`3f09b0541bbc0da39caf0a1b176adb78ec3d4ba3375823122363fcd3bea9fa10`。validation
abs-rel 为 `0.4073479138314724`，best 仍为 epoch 3 的 `0.34919863343238833`，负向波动
按原样保留。watcher state SHA-256 更新为
`0b2c31cf3cd1eb49f6f1a428902ba215ab0f402d9844e235123b48cb169c6a99`；全局原子账本为
`1163236 / 51013200` optimizer steps 与 `140368975 / 725922600` exposures。ADE retry 2
仍处于首个 epoch 内，无原子报告或退出证据。

`2026-08-23T13:00:14Z`，watcher 下一轮仍为 `failures=[]`。NYUv2
`random_feature_local / seed 4121` epoch 15 通过强审计，累计 `2685` optimizer steps、
`10725` exposures，artifact SHA-256
`da595c56fd0e1ec5e34c346d0edcbe3a36681c592d45df89013dc19449314fab`；validation
abs-rel/delta1 为 `0.37124239206314086 / 0.2967079639434814`，best 仍为 epoch 3。
VOC2012 `random_feature_local / seed 104729` epoch 41 也通过强审计，累计 `13530`
steps、`54038` exposures，artifact SHA-256
`d22d2c8f09610f92637ce0edc94ea99693911f4b42bc044d24f207206922e5d1`；validation
mean-IoU 为 `0.03398339822888374`，best 仍为 epoch 28。两份均 `status=passed`、
`problems=[]`，全部低结果与中间波动保留。watcher state SHA-256 更新为
`f7f124042184060f055053a5c00866280c8171a57314f019060e9348217269c4`；全局原子账本
更新为 `1163745 / 51013200` optimizer steps 与 `140371008 / 725922600` exposures。
ADE retry 2 继续存活但仍无原子输出。

`2026-08-23T13:10:35Z`，watcher 继续返回 `failures=[]`。NYUv2 在单个 watcher 周期内
从 epoch 15 推进到 epoch 17；最新 epoch 17 强审计验证连续 history `1..17`，因此同时
覆盖未单独捕获的 epoch 16 账本连续性。累计 `3043` optimizer steps、`12155`
exposures，artifact SHA-256
`00f68cfeade3f3736552613502c5cd6e7c743d2dd5e100c0187d7c1abb2c5130`；validation
abs-rel/delta1 为 `0.3916629523038864 / 0.24304871559143065`，best 仍为 epoch 3。
VOC2012 epoch 42 也通过强审计，累计 `13860` steps、`55356` exposures，artifact
SHA-256 `7a7ca4618931d1aab85c05cbe516acbb9d75bd652f118103f74f8c9be0fe4ede`；本 epoch
validation mean-IoU `0.036858487874269485` 成为该 cell 新 best。两份均
`status=passed`、`problems=[]`，仍仅为中间轨迹。watcher state SHA-256 更新为
`f7ac36a7fd1b8ca2990a583d7f84de0d2c325c7ae7fcbe19a80698f3c0233e35`；全局原子账本
更新为 `1164433 / 51013200` optimizer steps 与 `140373756 / 725922600` exposures。
ADE retry 2 仍无 epoch-1 原子报告或退出证据。

`2026-08-23T13:20:44Z`，NYUv2 `random_feature_local / seed 4121` epoch 18 通过下一轮
强审计：连续 history `1..18`、`3222` optimizer steps、`12870` exposures、scheduler
`18/80`、optimizer/RNG/report/checkpoint/cache/control/revision/clean worktree 均闭合，
`status=passed`、`problems=[]`；artifact SHA-256
`05e8af828da12d6bf26a07bfd7b6f2c6fa68f84f09e690134cdda19231d72821`。validation
abs-rel/delta1 为 `0.3727941647171974 / 0.30799560546875`，best 仍为 epoch 3，负向
中间轨迹完整保留。watcher state SHA-256 更新为
`f6daabeeecea53c923820bfa5b40de139df2be989558662d6381685c1ce90aee`；全局原子账本为
`1164612 / 51013200` optimizer steps 与 `140374471 / 725922600` exposures。ADE retry 2
继续存活，仍无原子输出。

`2026-08-23T13:31:02Z`，watcher 下一轮保持 `failures=[]`。NYUv2
`random_feature_local / seed 4121` epoch 19 通过强审计，累计 `3401` optimizer steps、
`13585` exposures，artifact SHA-256
`0329325de7d40295f03353b25f7752a53a839a8c9751d7c09d314c120ee749e9`；validation
abs-rel/delta1 为 `0.3841987505555153 / 0.28440561294555666`，best 仍为 epoch 3。
VOC2012 `random_feature_local / seed 104729` epoch 43 同样通过强审计，累计 `14190`
steps、`56674` exposures，artifact SHA-256
`8a3b64cc73850f272af26a55e1a68dca95c60ed7a54967e23bc2791fd5865e9f`；validation
mean-IoU `0.036461539566516876`，best 保持 epoch 42。两份均 `status=passed`、
`problems=[]`。watcher state SHA-256 更新为
`d55412ed7fa26abfa0ddffa744e092605cb85c4449fa6515aa17d0f8baff9929`；全局原子账本
更新为 `1165121 / 51013200` optimizer steps 与 `140376504 / 725922600` exposures。
ADE retry 2 仍无 epoch-1 原子输出或退出证据。

`2026-08-23T13:41:10Z`，NYUv2 `random_feature_local / seed 4121` epoch 20 通过强审计：
连续 history `1..20`、`3580` optimizer steps、`14300` exposures、scheduler `20/80`、
optimizer/RNG/report/checkpoint/cache/control/revision/clean worktree 全部闭合，
`status=passed`、`problems=[]`；artifact SHA-256
`24755a203c1ef00253ac2ad148a646925969fe72240d4266d2acc11ab31d1a1c`。validation
abs-rel/delta1 为 `0.3893373392522335 / 0.3692410945892334`，best 仍为 epoch 3。
watcher state SHA-256 更新为
`d59fe43d7d6c6b700e4b4dda765dce2f199cc6be2817265a61583f2ff9903201`；全局原子账本为
`1165300 / 51013200` optimizer steps 与 `140377219 / 725922600` exposures。ADE retry 2
继续存活，无原子输出或退出证据。

`2026-08-23T13:51:29Z`，watcher 保持 `failures=[]`。NYUv2 epoch 21 通过强审计，累计
`3759` optimizer steps、`15015` exposures，artifact SHA-256
`10a229c118d49da0505e6fb2a28187ea5beee7f7a5ce768a015596057b097136`；validation
abs-rel/delta1 为 `0.3888220891356468 / 0.3479002475738525`，best 仍为 epoch 3。
VOC2012 epoch 44 同样通过强审计，累计 `14520` steps、`57992` exposures，artifact
SHA-256 `b97e69fe177cb993b70dd5d44a745d149257d69d8ff8f52b2ca66a66dd7642de`；validation
mean-IoU 为 `0.03637664020061493`，best 仍为 epoch 42。两份均 `status=passed`、
`problems=[]`。watcher state SHA-256 更新为
`77ecc4bcbc8f2cc65ef31802e8fb0eab70054c32e2ce4de1ed596740aaaa2963`；全局原子账本
更新为 `1165809 / 51013200` optimizer steps 与 `140379252 / 725922600` exposures。
ADE retry 2 仍无 epoch-1 原子输出或退出证据。

`2026-08-23T14:01:40Z`，NYUv2 epoch 22 通过下一轮强审计，累计 `3938` optimizer
steps、`15730` exposures，artifact SHA-256
`53d57c119828615244b4693b69a283c55d487babf26e070a37b0433314781085`；validation
abs-rel/delta1 为 `0.38224650770425794 / 0.2997328758239746`，best 仍为 epoch 3。
全部 history、optimizer、scheduler、RNG、report/checkpoint、cache/control、固定 revision
与 clean worktree 检查通过，`status=passed`、`problems=[]`。watcher state SHA-256 更新为
`fc388e91175d7e8a44c5c93522e4b5f2823cfec9ea308d7a74c9a50b24ed56ee`；全局原子账本为
`1165988 / 51013200` optimizer steps 与 `140379967 / 725922600` exposures。ADE retry 2
继续存活，无原子输出或退出证据。

`2026-08-23T14:12:04Z`，watcher 轮次同时闭合三个任务的新原子点，继续
`failures=[]`。ImageNet-100 `velocity / seed 7319` epoch 34 通过强审计，累计 `30940`
optimizer steps、`3959470` exposures，artifact SHA-256
`b086ddecf1156909487fecf974cd99b6993b0043a89ff878cabfdcacd2899211`；validation top-1/
top-5 为 `0.03508500772797527 / 0.15239567233384854`，best 更新为 epoch 34。
NYUv2 epoch 23 通过强审计，累计 `4117` steps、`16445` exposures，artifact SHA-256
`5f42ab28ddf7dd10a54e0a50e312450f8bf04691deb34c5b0d1b41b2e21f9511`；validation
abs-rel `0.3852349519729614`，best 仍为 epoch 3。VOC2012 epoch 45 通过强审计，累计
`14850` steps、`59310` exposures，artifact SHA-256
`a5e90e1311160cdc2095ad5119c25e81d88313bd71176aa889ef9941f15c6f6f`；validation
mean-IoU `0.038343317806720734`，更新为该 cell 新 best。三份均 `status=passed`、
`problems=[]`，仍只构成中间证据。watcher state SHA-256 更新为
`6b53b250a4c4f5171745382c55b451248ac9a356e86a03d31e0cb339f9eedc46`；全局原子账本
更新为 `1167407 / 51013200` optimizer steps 与 `140498455 / 725922600` exposures。
ADE retry 2 仍无 epoch-1 原子输出或退出证据。

`2026-08-23T14:17:38Z`，ADE20K retry 2 首个决定性原子门通过。`random_feature_local /
seed 4121` epoch 1 已提交 report、last 与 best checkpoint；标准强审计 SHA-256
`cf40806701b71423e4a27ce751784f1580f63e460614ac75df9ed73b6813f4f1`，ADE shim-bound
增强审计 SHA-256
`d99c553ac9aa0b66fc32bff4ac734d873cffcfdcd796f0e5a80d6e73424e4782`，二者均
`status=passed`、`problems=[]`。账本精确为 `9095` optimizer steps、`18189` sample
exposures、scheduler `1/80`，train loss `3.269691086987271` 有限；train/validation 时间
为 `7209.330173842609 / 1636.0241944994777` 秒，validation mean-IoU/pixel accuracy 为
`0.0033956333063542843 / 0.1695859432220459`。该极低中间结果按原样保留，不构成方法
有效性裁决。

增强审计同时绑定 fixed revision、两个 clean worktree、物理 GPU 7、CLI 实际
`PYTHONPATH` 中的最终 ADE-only shim、shim SHA-256
`f95bbb0ed70ebbc127e6e08cd8d47c7bb18e6d394c2233a5677dc9cdfbf7bac4`、recovery gate
SHA-256 `fe59df4ce274fb809dd12acb3f143872cdade242b102a7ce7ee2c0b8d703625d`、完整 label scan
SHA-256 `ef4d7b5a0a9474fbe8692de8ca3c87191a80b8335155ce292719cb28fcf83788`，以及注册
all-ignore batch 1697 的 cache indices `5336/5328`。这证明 retry 2 已在不改变样本、
batch、sampler、seed、LR、epoch、optimizer step 或 exposure 的条件下跨过原确定性 NaN
位置；首次失败和 retry-1 停止证据继续保留。ADE 正式进度由 0 更新为 epoch 1，但 terminal
cell 仍为 0。全局原子账本更新为 `1176502 / 51013200` optimizer steps 与
`140516644 / 725922600` exposures；`execution_complete=false`、
`method_effectiveness_conclusion=null`、`changes_scientific_verdict=false`。

`2026-08-23T14:22:22Z`，自动 watcher 的下一轮保持 `failures=[]`，并新增两份通过的
强审计。ADE20K `random_feature_local / seed 4121` epoch 1 的自动审计 artifact SHA-256
为 `10ab8c6e2ae619a941e16ebf9e9407fd16fe9c39c51dab5acc49129937d494c5`，再次得到
`status=passed`、`problems=[]`；它与前述标准及 shim-bound 审计指向同一已计入的
`9095` steps、`18189` exposures，因此作为重复独立证据归档，账本不二次累计。

NYUv2 `random_feature_local / seed 4121` epoch 24 的强审计 artifact SHA-256 为
`5d89cdfe7c1119f360fcc9127180ea122f363e4d1223e74e1f7b05f58040c97c`，连续 history
累计 `4296` steps、`17160` exposures，`status=passed`、`problems=[]`。validation
abs-rel/delta1 为 `0.5526431165635586 / 0.3891017436981201`；该明显变差的中间值按原样
保留，不构成方法有效性裁决。watcher state SHA-256 更新为
`083f25fc6f6710cf2c3a45239aa510634652509d309c2b0e1861f9666e1fa013`。全局原子账本仅
增加 NYUv2 epoch 24 相对 epoch 23 的 `179` steps 与 `715` exposures，更新为
`1176681 / 51013200` optimizer steps 和 `140517359 / 725922600` exposures。
四任务 worker、主 recovery、watchdog 和 watcher 均继续存活；四任务 terminal cells
仍为 ImageNet-100 `13/60`、VOC2012 `2/60`、NYUv2 `0/60`、ADE20K `0/60`。
正式状态保持 `status=active`、`execution_complete=false`、
`method_effectiveness_conclusion=null`、`changes_scientific_verdict=false`。

`2026-08-23T14:32:41Z`，自动 watcher 再次返回 `failures=[]`。NYUv2
`random_feature_local / seed 4121` epoch 25 通过强审计，累计 `4475` optimizer steps、
`17875` exposures，artifact SHA-256
`ab12401d57a0dac8cbebcd11506d28394e4a6d6d7c1154e54058140b7c19ccdb`；validation
abs-rel/delta1 为 `0.3950044304132462 / 0.279897403717041`，best 仍为 epoch 3 的
`0.34919863343238833`。VOC2012 `random_feature_local / seed 104729` epoch 46 同样通过
强审计，累计 `15180` steps、`60628` exposures，artifact SHA-256
`1e8839823999776e1a4da47b9be090ff886aecec6b7bb2216f9f3102a13f0ec0`；validation
mean-IoU/pixel accuracy 为 `0.038020022213459015 / 0.6466395258903503`，best 保持
epoch 45。两份均 `status=passed`、`problems=[]`，低/波动中间结果完整保留。
watcher state SHA-256 更新为
`bf65cb782a1b44c30aa05b044a059dff59b1ef1891220a07049221092d4615e1`；全局原子账本
更新为 `1177190 / 51013200` optimizer steps 与
`140519392 / 725922600` exposures。ADE epoch 1 的重复自动审计没有被二次计数；所有
四路任务继续运行，科学状态不变。

同一观测点另做了不依赖人工增量的四任务账本重建。ImageNet-100 与 VOC2012 的
`matrix_report.json` 分别给出 `13` 与 `2` 个 terminal cells，合计 `1117500` steps、
`136463230` exposures；四个唯一 active cell 的最新已强审计报告合计 `59690` steps、
`4056162` exposures。两者相加精确得到 `1177190` steps 与 `140519392` exposures，
与进度报告一致，且 ADE epoch 1 的三份审计只计一次。机器可读重建报告为
`artifacts/reports/four_task_atomic_ledger_reconstruction_20260823.json`，SHA-256
`7ac066ccc961d947486f4fc63655dd6dc3366a2d010c39ccb0a1717307100372`，
`status=passed`、`problems=[]`。该核对只证明执行账本闭合，不构成科学结论。

`2026-08-23T14:42:54Z`，NYUv2 `random_feature_local / seed 4121` epoch 26 通过自动
强审计：连续 history 累计 `4654` optimizer steps、`18590` exposures，artifact
SHA-256 `943bbff9a81a80bdb6e4063ccb83cdb4de71a6e1e88b76f16f7d1891f94fd51c`，
`status=passed`、`problems=[]`。validation abs-rel/delta1 为
`0.39618089273571966 / 0.3038174629211426`，best 仍为 epoch 3；中间波动按原样保留。
watcher state SHA-256 更新为
`a8359e67f1cd25dfdef6c98a6516e8050f71855ba99beabad8e453b8ce203ed9`，全局原子账本
更新为 `1177369 / 51013200` optimizer steps 与
`140520107 / 725922600` exposures。科学状态保持 active 且未形成结论。

`2026-08-23T14:52:53Z` 的 watcher 轮次继续为 `failures=[]`。NYUv2 epoch 27 通过强
审计，累计 `4833` steps、`19305` exposures，artifact SHA-256
`c6b1b90c339703cd863bfc09778fbc35842e983a5dc73b55aeda1c696a62d8f8`；validation
abs-rel/delta1 为 `0.43093058094382286 / 0.3762960910797119`，best 仍为 epoch 3。
VOC2012 epoch 47 同样通过，累计 `15510` steps、`61946` exposures，artifact SHA-256
`9f5d3bfae75895797007ef59efd61b6f4cfd1670c4b93262289cbe89437de4be`；本 epoch
validation mean-IoU `0.04328776150941849` 更新为该 cell 新 best。两份均
`status=passed`、`problems=[]`。watcher state SHA-256 更新为
`e36ce7c65a24810e7a41704ccc490aa5ee3e206ab8bc5c6c02f45502037455d9`，全局原子账本
为 `1177878 / 51013200` optimizer steps 与
`140522140 / 725922600` exposures。

同一时段完成了 GPU 6/7 侧 worker 的 recovery-handoff 加固。三个训练 worker 的
PID/PGID/start-ticks 在替换前后完全不变；仅旧的主-PID绑定 monitor 被替换为绑定权威
watchdog、固定 clean checkout 与精确 worker 身份的内容寻址 monitor。外部 H200 recovery
wrapper 也只为未来 ADE20K 恢复增加已审计 all-ignore shim 的 fail-closed SHA 绑定，当前
recovery 未重启。完整记录见
`docs/records/2026-08-23_dsw_h200_gpu67_recovery_handoff_hardening.md`。这项运维加固未改
训练或科学合同，`changes_scientific_verdict=false`。

`2026-08-23T15:03:12Z`，自动 watcher 新一轮继续 `failures=[]`。NYUv2 epoch 28 通过
强审计，累计 `5012` steps、`20020` exposures，artifact SHA-256
`882db745199c059dc9987ed543695621f216661d7425cc4f4e947e734d3f74b3`；validation
abs-rel/delta1 为 `0.38464670777320864 / 0.2998485565185547`，best 仍为 epoch 3。
VOC2012 epoch 48 也通过，累计 `15840` steps、`63264` exposures，artifact SHA-256
`b42de4b5988c497ef2dba5ae04ede5ba5b145fdabc1ed0e60b678c66a494d6aa`；validation
mean-IoU `0.03942684829235077`，best 仍为 epoch 47。两份均 `status=passed`、
`problems=[]`，中间低值和波动完整保留。watcher state SHA-256 更新为
`d847d37d907cf4738df64e08d0425700892a624969daa2ed9250b306727177d6`；全局原子账本
更新为 `1178387 / 51013200` optimizer steps 与
`140524173 / 725922600` exposures。

下游链同时执行了 pre-trigger 审计：在四任务主矩阵未完整前，main/causal/extension/final
decision、`registry.json` 与 `completion_audit.json` 均不存在；因此没有提前消费条件扩展
或回放。固定 revision 接续链仍注册为：完整 positive 或 negative main 后均运行四项因果
控制；仅 positive main 运行 ImageNet-1k 与 15 项高成本扩展；final decision 后四任务各
运行 33 份、共 132 份逐样本回放，随后才生成 registry 与 completion audit。机器可读审计
为 `artifacts/reports/downstream_gate_pretrigger_audit_20260823.json`。该审计证明门控顺序，
不证明方法有效性或执行完成。

`2026-08-23T15:04:17Z`，此前接受的三项 recovery-handoff monitor 暴露路径解析错误：
watchdog 实际仍为 PID `41096`、argv `bash watchdog_h200.sh`，但 monitor 将相对脚本路径
错误地按自身 cwd 解析，因而在 grace period 后将 VOC2012、NYUv2 与 ADE20K 三个已注册
进程组误判为 `terminated_watchdog_absent` 并终止。三份 exit artifact、旧 monitor SHA-256
`9317f0e108c769c1d7533e56a80c1e97c92d3aa8cc94ef57dc4d96885d417b3f` 与旧 registry
SHA-256 `50bc7d697ae4fc152431d6f9eecd1c634d271285e64a9f9b7c6be49936d2e02d`
均原样保留。已接受的原子 report/checkpoint 未被改写，但三个任务各自未提交的当前 epoch
工作丢失；该负面运维结果已显式记录，未被隐藏。

三个任务于 `2026-08-23T15:09:34Z` 从最后接受的 VOC epoch 48、NYUv2 epoch 28、ADE
epoch 1 checkpoint 恢复，仍严格保持每任务 `seed_workers=1` 且无重复 dataset/cell writer。
修复 monitor SHA-256 为
`e0677f25e01dccb4ccaf723a7f188efecd5cc760aa423877bce5a843161b8960`，当前 registry
SHA-256 为 `486c21301c656fedf2ea662a0240e2c14fe43708e195a2dd178ca6ded0aea189`。
三任务精确 self-test、短时 live timeout test 与完整轮询周期均通过；新 monitor PID 为
`1615506/1615510/1615515`。原 hardening 报告已标记 `superseded_failed`，更正审计见
`artifacts/reports/gpu67_parallel_monitor_false_watchdog_recovery_20260823.json`。

`2026-08-23T15:23:31Z`，自动 watcher 接受恢复后的首个新原子点：NYUv2
`random_feature_local / seed 4121` epoch 29 强审计 `status=passed`、`problems=[]`，累计
`5191` optimizer steps 与 `20735` exposures，artifact SHA-256
`eba1af4eb0a4f16bc1de1b40f48bf2460ccb1b17293effb8a124cd67040cab8b`；validation
abs-rel/delta1 为 `0.4006128191947937 / 0.3375543117523193`，best 仍为 epoch 3。
watcher state SHA-256 为
`e1b616b0c208329be0393e0ddcd0fa873c68b489b15d22ec01265d2eafddbee8`，`failures=[]`。
全局原子账本更新为 `1178566 / 51013200` optimizer steps 与
`140524888 / 725922600` exposures；四任务 terminal cells 仍为 `13/60`、`2/60`、
`0/60`、`0/60`。科学状态保持 `active`、`execution_complete=false`、
`method_effectiveness_conclusion=null`、`changes_scientific_verdict=false`。

`2026-08-23T15:33:41Z`，watcher 同时观察到 VOC2012 epoch 49 与 NYUv2 epoch 30。
VOC epoch 49 强审计正常通过，累计 `16170` optimizer steps、`64582` exposures，artifact
SHA-256 `c0f8508e16a5f37e71074638e81098edf6e38466383b1af2b202a5241a9fdbe2`。
NYUv2 的 epoch-30 审计则发生可复现的选择—读取 race：watcher 选定 expected epoch 30
后，训练在 helper 读取前已原子推进到 epoch 31，故 requested epoch/step/exposure/scheduler
检查失败。失败 artifact SHA-256
`e360b37cb1ffc612215e4c554032dc570aa87999449e1d0a815266b5e82ead60`
完整保留，不计入通过账本，也不解释为训练或科学失败。

稳定双读后，epoch 31 由同一 helper 独立重审两次并均通过：手动重审 SHA-256
`2b5300348ee1d48e1e694abcad37a0408c1cded8be79a60e978ab3bab98f0d43`，升级后 watcher
标准审计 SHA-256 `17c026af12f0993a3dc9dec623924c53603b7cfc47e0454b99fd4607c83cd388`。
二者绑定相同 report SHA-256
`49e5b90afe27a514889289bd26e416be4f3985e8e8a3f8cadacc956f4d9b1ca6`、
`5549` steps 与 `22165` exposures。watcher 由 PID/SHA
`3062994/ae99087a7fb34519ed8938df02b8f65150908b198e76342c95e47c6576317648`
升级为 `1870499/e02012a02729ebbba6f0a3c2b85f81bb0770a68df251219e8aa1c442a0057c55`；
失败证据移入 `transient_audit_races` 而非删除，当前 `failures=[]`。机器可读加固审计见
`artifacts/reports/atomic_audit_watcher_epoch_race_hardening_20260823.json`。

按 VOC epoch 49 与 NYUv2 epoch 31 的最新接受累计值重建后，全局原子账本为
`1179254 / 51013200` optimizer steps、`140527636 / 725922600` exposures。四任务
terminal cells 仍为 ImageNet-100 `13/60`、VOC2012 `2/60`、NYUv2 `0/60`、ADE20K
`0/60`；下游证据门仍未触发，科学状态不变。

同一 `2026-08-23T15:40:30Z` watcher 轮次还接受了 VOC2012
`random_feature_local / seed 104729` epoch 50：累计 `16500` optimizer steps、`65900`
exposures，validation mean-IoU/pixel accuracy 为
`0.04268353432416916 / 0.6245777606964111`，best 仍为 epoch 47。强审计 SHA-256
`4962d021280858bffe1817faaa4784d93817cb88948c8b4773246712c064802c`，
`status=passed`、`problems=[]`。全局原子账本据此更新为
`1179584 / 51013200` steps 与 `140528954 / 725922600` exposures；中间低值原样保留，
不构成方法有效性结论。

`2026-08-23T15:50:42Z`，升级后的 watcher 下一轮保持 `failures=[]`、
`transient_audit_races=1`，并接受两个新原子点。ImageNet-100
`velocity / seed 7319` epoch 35 强审计 SHA-256
`8caf453654b66cd29c84e10ae64a38707b90f65613614338cf3e5bd5a14ef003`，累计
`31850` steps、`4075925` exposures；validation top-1/top-5 为
`0.03740340030911901 / 0.16321483771251932`。NYUv2 epoch 32 强审计 SHA-256
`649a7457cdc9a1921a1bf4e731e138f44062a08aecf82173f7c2e6b7de943bc4`，累计
`5728` steps、`22880` exposures；validation abs-rel/delta1 为
`0.4763966672122478 / 0.38733634948730467`。两份均 `status=passed`、
`problems=[]`，低/波动中间结果完整保留。watcher state SHA-256 更新为
`3ba34d0dddf06f8f62646d539b700330f8ff5ecaa8212f303e368fb6d8d984ae`。
全局原子账本更新为 `1180673 / 51013200` steps、
`140646124 / 725922600` exposures；terminal cell 数与科学状态不变。

`2026-08-23T16:00:51Z`（北京时间 `2026-08-24T00:00:51+08:00`），watcher
保持 `failures=[]`、`transient_audit_races=1`。VOC2012 epoch 51 强审计 SHA-256
`e8fa585e3407ba21558df0ed390adb8f43a44766a5673cfec2b9f6c22184da4d`，累计
`16830` steps、`67218` exposures；validation mean-IoU/pixel accuracy 为
`0.036488670855760574 / 0.6294333338737488`。NYUv2 epoch 34 强审计 SHA-256
`90a744b4186c9625aad541752bfac648ea5587675d5ca8dcbea02f6e6c7321ee`，连续 history
覆盖至 epoch 34，累计 `6086` steps、`24310` exposures；validation abs-rel/delta1 为
`0.40698774755001066 / 0.35341243743896483`。两份均 `status=passed`、
`problems=[]`。epoch 34 的连续累计覆盖未单独采样的 NYUv2 epoch 33，不漏记也不重复计数。
watcher state SHA-256 更新为
`4bd2806ae51054d9bc7ae4ca059b8f6e08e732eab4d9ec3e9b8636fed9f07112`；全局原子账本
更新为 `1181361 / 51013200` steps、`140648872 / 725922600` exposures。SSH 等待连接
随后被远端关闭，但立即重连确认所有 worker/monitor PID 连续存活，因此不构成训练失败。

`2026-08-23T16:11:02Z`，watcher 继续 `failures=[]`，接受 NYUv2
`random_feature_local / seed 4121` epoch 35。强审计 SHA-256
`6a2c6d98c28a839f059fcae5294fc31a67dafaaf70c0b7d1ea81dd83ea37f51b`，累计
`6265` optimizer steps、`25025` exposures；validation abs-rel/delta1 为
`0.4623841144144535 / 0.3889937400817871`。该波动中间值按原样保留；审计
`status=passed`、`problems=[]`。watcher state SHA-256 更新为
`caef17ba824492be4d7195ee734aab74ee17aeeddd9425dd4bfbb7b72842fdc5`，全局原子账本
更新为 `1181540 / 51013200` steps、`140649587 / 725922600` exposures。科学状态不变。

`2026-08-23T16:21:06Z`，watcher 保持 `failures=[]`、`transient_audit_races=1`，接受
VOC2012 epoch 52 与 NYUv2 epoch 36。VOC 强审计 SHA-256
`5546dcffef3ea98a53e3ee1c05b1e8456c2eda8dced370c8091090c2a09f51ba`，累计
`17160` steps、`68536` exposures，validation mean-IoU/pixel accuracy 为
`0.04196411371231079 / 0.6335916519165039`。NYUv2 强审计 SHA-256
`a162242a6831290c682826f014fbd6a8b42511bdf74ff0c65874829480b3ccd6`，累计
`6444` steps、`25740` exposures，validation abs-rel/delta1 为
`0.4237875215709209 / 0.3689557075500488`。两份均 `status=passed`、
`problems=[]`。watcher state SHA-256 为
`4ede33e157153092f48d727f9cede62705336cc6ab3e9d845fca3dd1a0e577f9`，全局原子账本
更新为 `1182049 / 51013200` steps、`140651620 / 725922600` exposures。

同一时段直接读取 ADE20K worker PID `1571772` 的 `/proc/io`：5 秒内 `rchar` 从
`8033756405` 增至 `8100496968`，增长 `66740563` bytes，且进程约 `1984% CPU`，证明
epoch 2 仍有实际数据读取与计算活动；这只是运行活性证据，不计入正式原子账本。

`2026-08-23T16:31:15Z`，watcher 继续 `failures=[]`，接受 VOC2012 epoch 53 与
NYUv2 epoch 37。VOC 强审计 SHA-256
`c599272960b6184223f6ce0e5681e4f81ec3c11829f4e6e386266e3d8c8f1cf2`，累计
`17490` steps、`69854` exposures，validation mean-IoU/pixel accuracy 为
`0.04434805363416672 / 0.6187670826911926`。NYUv2 强审计 SHA-256
`7139a4eb050053627b4bbc19d040d525a4ee1674f0262f9cc5c0f81c5367e07d`，累计
`6623` steps、`26455` exposures，validation abs-rel/delta1 为
`0.41094927191734315 / 0.3169434070587158`。两份均 `status=passed`、
`problems=[]`，中间波动完整保留。watcher state SHA-256 更新为
`9823178108daa71ea7a8d5db590f1c1ec80a419df6a95a5288f5e28f5410cb4a`；全局原子账本
更新为 `1182558 / 51013200` steps、`140653653 / 725922600` exposures。

`2026-08-23T16:41:32Z`，watcher 保持 `failures=[]`，接受 NYUv2
`random_feature_local / seed 4121` epoch 39。连续 history 覆盖未单独采样的 epoch 38，
累计 `6981` optimizer steps、`27885` exposures；强审计 SHA-256
`be489f4653211c429c7a42fbd8fd81af28ce0fa356bcaa7a2059df1ca5db8ddd`，
`status=passed`、`problems=[]`。validation abs-rel/delta1 为
`0.48464830294251443 / 0.382401180267334`，波动与退化值原样保留。watcher state
SHA-256 更新为 `6190b4b14e2187108817aae300dde1a601f896a2fcb17b0ee1af829e89707b7d`；
全局原子账本更新为 `1182916 / 51013200` steps、`140655083 / 725922600` exposures。

`2026-08-23T16:51:37Z`，watcher 保持 `failures=[]`，接受 VOC2012 epoch 54 与
NYUv2 epoch 40。VOC 强审计 SHA-256
`22c26935b714fafa184ab62a6bfcb7e71c34a9d3a32fc824dfe1d2719309f15f`，累计
`17820` steps、`71172` exposures，validation mean-IoU/pixel accuracy 为
`0.04276473820209503 / 0.6223744750022888`。NYUv2 强审计 SHA-256
`d817279c9e04495c7f1e7cdd100809797cd82d8d195368dd582b62fd4ca0f53d`，累计
`7160` steps、`28600` exposures，validation abs-rel/delta1 为
`0.4137936390936375 / 0.3309149742126465`。两份均 `status=passed`、
`problems=[]`；中间值继续原样保留。watcher state SHA-256 更新为
`86e13f2771dcfdb09f218867ef0b1cbdcaa29e1ba22220cdb0257670469623a6`；全局原子账本
更新为 `1183425 / 51013200` steps、`140657116 / 725922600` exposures。

`2026-08-23T17:01:46Z`，watcher 保持 `failures=[]`、`transient_audit_races=1`，接受
VOC2012 `random_feature_local / seed 104729` epoch 55 与 NYUv2
`random_feature_local / seed 4121` epoch 41。VOC 强审计 SHA-256
`411768382e04cf70ec96e360582332b5159ecb16bd50f519c44d525d8e7fca86`，累计
`18150` optimizer steps、`72490` exposures，validation mean-IoU/pixel accuracy 为
`0.03792376071214676 / 0.6246172785758972`，best 仍为 epoch 53 的
`0.04434805363416672`。NYUv2 强审计 SHA-256
`d8135095dec013d114709ffd8cd6df1b8eff39dcffb464dd356ded38555356f3`，累计
`7339` optimizer steps、`29315` exposures，validation abs-rel/delta1 为
`0.48699565827846525 / 0.36911783218383787`，best 仍为 epoch 3 的
`0.34919863343238833`。两份均 `status=passed`、`problems=[]`；低值与退化波动完整
保留，不解释为科学结论。watcher state SHA-256 更新为
`45fac44d7ff2b797370fcfd7b01fce594536ab536f0af17ad2e37835104cf0ed`；四个唯一
active cell 合计 `66434` steps、`4195919` exposures，全局原子账本更新为
`1183934 / 51013200` steps、`140659149 / 725922600` exposures。terminal cells 仍为
ImageNet-100 `13/60`、VOC2012 `2/60`、NYUv2 `0/60`、ADE20K `0/60`，科学状态不变。

`2026-08-23T17:11:56Z`，watcher 新一轮仍为 `failures=[]`、
`transient_audit_races=1`，接受 NYUv2 `random_feature_local / seed 4121` epoch 42。
强审计 SHA-256
`4f4e911ed1258f81b4589b789354ff586bd5d9f67694ed266ac509ee2def5ab8`，连续 history
累计 `7518` optimizer steps、`30030` exposures；validation abs-rel/delta1 为
`0.457818241417408 / 0.3634984016418457`，best 仍为 epoch 3 的
`0.34919863343238833`。该波动中间值原样保留，审计 `status=passed`、`problems=[]`。
watcher state SHA-256 更新为
`d6dc6423072ef7de24c5533b54bbd9b4be084eda92bdf9e038d242f091207c5e`；四个唯一
active cell 合计 `66613` steps、`4196634` exposures，全局原子账本更新为
`1184113 / 51013200` steps、`140659864 / 725922600` exposures。terminal cells 与科学
状态均不变。

`2026-08-23T17:22:06Z`，watcher 保持 `failures=[]`、`transient_audit_races=1`，接受
VOC2012 `random_feature_local / seed 104729` epoch 56 与 NYUv2
`random_feature_local / seed 4121` epoch 43。VOC 强审计 SHA-256
`a6b7ecd14d278a8a838c6ac6c9348afa666cd61bd6c86bc6071c669b9758778b`，累计
`18480` optimizer steps、`73808` exposures；validation mean-IoU/pixel accuracy 为
`0.039493586868047714 / 0.6274118423461914`，best 仍为 epoch 53 的
`0.04434805363416672`。NYUv2 强审计 SHA-256
`8dfe333611beec49ada94c11d907a02b77e2a89f1bf58dcc9cab029d1255eeb0`，累计
`7697` optimizer steps、`30745` exposures；validation abs-rel/delta1 为
`0.4185707435011864 / 0.29657444953918455`，best 仍为 epoch 3 的
`0.34919863343238833`。两份均 `status=passed`、`problems=[]`，低值与波动按原样
保留。watcher state SHA-256 更新为
`f0ad7aaf7d42c7330bf24630716e02396c2add528f1e7cdf5cc9a4c8ffe66c05`；四个唯一
active cell 合计 `67122` steps、`4198667` exposures，全局原子账本更新为
`1184622 / 51013200` steps、`140661897 / 725922600` exposures。terminal cells 与科学
状态均不变。

`2026-08-23T17:26:20Z` 再次记录 GPU 6/7 资源重叠 provenance。四个 FieldScope
worker 分别占用约 `2174/938/880/862` MiB，GPU 6/7 上额外各约 `61974/61954` MiB
来自外部 M3Call `omnicall-stage1-h200-gpu67-train-20260823-r6`，其两个 child PID 为
`1815680/1815681`，命令从 step 256 恢复并计划在 step 512 停止。resource guard PID
`1813` 的 GPU 6 与 GPU 7 均保持 documented device pause；pause 文件均存在，state
snapshot SHA-256 为
`f8ed09df0059aff4f9833dbb092a27a5fcd7bd2c91bcc5c16f20a6f0e002217d`。FieldScope
没有终止或修改外部进程，没有重启 worker、改变 batch/runtime 合同或新增重复 writer；
四任务继续运行。该重叠的科学影响保持 `not_inferred`，不据此声称有影响或无影响。
机器可读审计为
`artifacts/reports/gpu67_m3call_r6_overlap_20260823.json`。

`2026-08-23T17:32:29Z`，watcher 继续 `failures=[]`、`transient_audit_races=1`，接受
VOC2012 `random_feature_local / seed 104729` epoch 57 与 NYUv2
`random_feature_local / seed 4121` epoch 45。VOC 强审计 SHA-256
`b2399cc8de69cd95e642c105c9243273b823e2a295aa08647f0cc16ebb042208`，累计
`18810` optimizer steps、`75126` exposures；validation mean-IoU/pixel accuracy 为
`0.04984709247946739 / 0.611583411693573`，epoch 57 更新为该 cell 新 best。NYUv2
强审计 SHA-256
`07a8c7eedb039887381fe4081b27b1b4e6ec06b461738c8fcd80ee8353a64e9f`，连续 history
覆盖未单独采样的 epoch 44 并累计 `8055` optimizer steps、`32175` exposures；
validation abs-rel/delta1 为 `0.44258485436439515 / 0.3530094146728516`，best 仍为
epoch 3 的 `0.34919863343238833`。两份均 `status=passed`、`problems=[]`，指标按原样
保留。watcher state SHA-256 更新为
`2cefeaaabe31a22382487003353964a64bfdb112e1e5481049e9974b3104bc25`；四个唯一
active cell 合计 `67810` steps、`4201415` exposures，全局原子账本更新为
`1185310 / 51013200` steps、`140664645 / 725922600` exposures。terminal cells 与科学
状态均不变。

`2026-08-23T17:42:52Z`，watcher 保持 `failures=[]`、`transient_audit_races=1`，接受
VOC2012 `random_feature_local / seed 104729` epoch 58 与 NYUv2
`random_feature_local / seed 4121` epoch 46。VOC 强审计 SHA-256
`8dba91c6104ad89555dbd61f7ffdec889cbe036c295971f11749180d908c2070`，累计
`19140` optimizer steps、`76444` exposures；validation mean-IoU/pixel accuracy 为
`0.043059904128313065 / 0.6221145391464233`，best 仍为 epoch 57 的
`0.04984709247946739`。NYUv2 强审计 SHA-256
`fdb060203095ec74e18f60805a51742544d6cf52bf400d108665efb3195676c2`，累计
`8234` optimizer steps、`32890` exposures；validation abs-rel/delta1 为
`0.44236420542001725 / 0.35317177772521974`，best 仍为 epoch 3 的
`0.34919863343238833`。两份均 `status=passed`、`problems=[]`、全部 checks 为 true；
低值与波动原样保留。watcher state SHA-256 更新为
`a764918654f8b5a6f163ffdfd85d863ab6c392c6fcabd3c7a0b73264c3361921`；四个唯一
active cell 合计 `68319` steps、`4203448` exposures，全局原子账本更新为
`1185819 / 51013200` steps、`140666678 / 725922600` exposures。terminal cells 与科学
状态均不变。

`2026-08-23T17:53:15Z`，watcher 新一轮仍为 `failures=[]`、
`transient_audit_races=1`，本轮仅接受 NYUv2 `random_feature_local / seed 4121`
epoch 47。强审计 SHA-256
`c558e66df39a5f5fc75e9e5415ec92c1cd46f276bfd8b096ccde09c157e020cd`，累计
`8413` optimizer steps、`33605` exposures；validation abs-rel/delta1 为
`0.4579716116189957 / 0.35987038612365724`，best 仍为 epoch 3 的
`0.34919863343238833`。审计 `status=passed`、`problems=[]`、全部 checks 为 true，
波动值按原样保留。VOC2012 本轮没有比 epoch 58 更新的强审计，故其账本不变。
watcher state SHA-256 更新为
`dc0fac2dbcb5765a4ee93b2f15c49bc2847501165eb0d7020eb8d106a409c677`；四个唯一
active cell 合计 `68498` steps、`4204163` exposures，全局原子账本更新为
`1185998 / 51013200` steps、`140667393 / 725922600` exposures。terminal cells 与科学
状态均不变。

`2026-08-23T18:03:26Z`，watcher 保持 `failures=[]`、`transient_audit_races=1`，同时
接受三个新原子点。ImageNet-100 `velocity / seed 7319` epoch 36 强审计 SHA-256
`aedd3d33ca511c6dc07187fa7f80a842ec34e1e9e5ea8f816d2dd4563915195f`，累计
`32760` optimizer steps、`4192380` exposures；validation top-1/top-5 为
`0.03817619783616692 / 0.1598145285935085`，epoch 36 更新为该 cell 当前 best。
VOC2012 `random_feature_local / seed 104729` epoch 59 强审计 SHA-256
`e6c721908947cc934cb2b5d570f180c816c7f18785a6a7a13ebd8f1ecdb3af63`，累计
`19470` steps、`77762` exposures；validation mean-IoU/pixel accuracy 为
`0.04444348067045212 / 0.6134964823722839`，best 仍为 epoch 57。NYUv2
`random_feature_local / seed 4121` epoch 48 强审计 SHA-256
`39e3f19d809a0fa8e75dd986851f605768f98b1dc0d92bfc49cb6b3242e0aea8`，累计
`8592` steps、`34320` exposures；validation abs-rel/delta1 为
`0.4273382924497128 / 0.3390836238861084`，best 仍为 epoch 3。三份均
`status=passed`、`problems=[]`、全部 checks 为 true；低指标与波动按原样保留。
watcher state SHA-256 更新为
`04a73b3b7e5b1de66e1b997097ab7a0d8c61209710b3477dec31c5dcd8263142`；四个唯一
active cell 合计 `69917` steps、`4322651` exposures，全局原子账本更新为
`1187417 / 51013200` steps、`140785881 / 725922600` exposures。terminal cells 与科学
状态均不变。

`2026-08-23T18:13:59Z`，watcher 保持 `failures=[]`、`transient_audit_races=1`，接受
ADE20K、VOC2012 与 NYUv2 三个新原子点。ADE20K retry 2 首次从 epoch 1 正式推进至
epoch 2；强审计 SHA-256
`ba4766166a601807ee69cb0c5e0c7e900fb03e2de8fe8df6129bf8fa51e29c92`，连续 history
累计 `18190` optimizer steps、`36378` exposures，validation mean-IoU/pixel accuracy
为 `0.003214373020455241 / 0.16373369097709656`，best 仍为 epoch 1 的
`0.0033956333063542843`。该极低结果按原样保留；全部强审计 checks 为 true，证明 retry 2
在固定 revision、cache、batch、seed、RNG、scheduler 与 checkpoint 账本下完成了第二个
原子 epoch，不改变此前 all-ignore 失败与 shim 恢复 provenance。

同一轮 VOC2012 `random_feature_local / seed 104729` epoch 60 强审计 SHA-256
`dbf0e8bd9b275c6b88ea87cadc4a02be4d919c60b2515ce8ed70e4c64ff54f1d`，累计
`19800` steps、`79080` exposures；validation mean-IoU/pixel accuracy 为
`0.04492133483290672 / 0.6205296516418457`，best 仍为 epoch 57。NYUv2
`random_feature_local / seed 4121` epoch 50 强审计 SHA-256
`959abf0567ead46218961b91b1f558a138e6d081838dc58a23f3c370f972b165`，连续 history
覆盖未单独采样的 epoch 49，累计 `8950` steps、`35750` exposures；validation
abs-rel/delta1 为 `0.4220243752002716 / 0.3208939552307129`，best 仍为 epoch 3。
三份均 `status=passed`、`problems=[]`、全部 checks 为 true。watcher state SHA-256
更新为 `f646b15e8a576022c41dbb4185badffb7d01523e42af611d345be8c1b47eeb02`；四个唯一
active cell 合计 `79700` steps、`4343588` exposures，全局原子账本更新为
`1197200 / 51013200` steps、`140806818 / 725922600` exposures。terminal cells 与科学
状态均不变。

`2026-08-23T18:18:19Z`，此前记录的 M3Call r6 双卡训练子进程 `1815680/1815681`
及 launcher `1815248` 均已自然退出；其 canary launcher PID `3659582` 随后启动仅使用
GPU 6 的 step-512 checkpoint reload/evaluation PID `3528022`，占用约 `60854` MiB。
GPU 7 已不再存在外部 M3Call compute process，只保留 FieldScope VOC2012、ADE20K 与
paused guard context。FieldScope 没有终止或修改外部进程，没有重启 worker、改变正式
batch/runtime 合同或新增重复 writer。该阶段转换的科学影响仍为 `not_inferred`。机器可读
记录为 `artifacts/reports/gpu6_m3call_r6_step512_reload_overlap_20260823.json`。

`2026-08-23T18:24:33Z`，watcher 保持 `failures=[]`、`transient_audit_races=1`，本轮
仅接受 NYUv2 `random_feature_local / seed 4121` epoch 51。强审计 SHA-256
`ecfcd336ab1b668db4df4f6b8692c7d5378d315cf7a1b43f4049fe4bad4e49ca`，累计
`9129` optimizer steps、`36465` exposures；validation abs-rel/delta1 为
`0.4544716127216816 / 0.3449239253997803`，best 仍为 epoch 3 的
`0.34919863343238833`。审计 `status=passed`、`problems=[]`、全部 checks 为 true；
退化波动值按原样保留。其它三任务本轮无更新强审计，故账本保持各自上一个接受点。
watcher state SHA-256 更新为
`d9437e6573ee58beac4bc29c76f043442890033a528a81c7a64c56a3ec888ac9`；四个唯一
active cell 合计 `79879` steps、`4344303` exposures，全局原子账本更新为
`1197379 / 51013200` steps、`140807533 / 725922600` exposures。terminal cells 与科学
状态均不变。

`2026-08-23T18:34:43Z`，watcher 保持 `failures=[]`、`transient_audit_races=1`，接受
VOC2012 `random_feature_local / seed 104729` epoch 61 与 NYUv2
`random_feature_local / seed 4121` epoch 52。VOC 强审计 SHA-256
`f0213b36813d0e401494e539a5d00a3af5733dc48a1383565c653b0912fb28f1`，累计
`20130` optimizer steps、`80398` exposures；validation mean-IoU/pixel accuracy 为
`0.04702213406562805 / 0.616513192653656`，best 仍为 epoch 57。NYUv2 强审计
SHA-256 `a0e1d19f100e0436cc163ea8ab49b5a0d857c708222610a9000accd0127d060d`，累计
`9308` steps、`37180` exposures；validation abs-rel/delta1 为
`0.43015885204076765 / 0.3265073299407959`，best 仍为 epoch 3。两份均
`status=passed`、`problems=[]`、全部 checks 为 true；低值和波动原样保留。
watcher state SHA-256 更新为
`56bbf561217f559bbb61be29ea0cdff2ea1b5905d0427e55de878eb4991ed17e`；四个唯一
active cell 合计 `80388` steps、`4346336` exposures，全局原子账本更新为
`1197888 / 51013200` steps、`140809566 / 725922600` exposures。terminal cells 与科学
状态均不变。

`2026-08-23T18:45:04Z`，watcher 保持 `failures=[]`、`transient_audit_races=1`，接受
VOC2012 `random_feature_local / seed 104729` epoch 62 与 NYUv2
`random_feature_local / seed 4121` epoch 53。VOC 强审计 SHA-256
`5375ab6d0ef8253dc02c09d5d13dd9952a196638db75fb42cc097546c9f28a45`，累计
`20460` optimizer steps、`81716` exposures；validation mean-IoU/pixel accuracy 为
`0.04415399581193924 / 0.6300187706947327`，best 仍为 epoch 57。NYUv2 强审计
SHA-256 `c770bf9d53e20bffc8ff531b2990f93583ed7a4d3adbcc89765914ab546fb713`，累计
`9487` steps、`37895` exposures；validation abs-rel/delta1 为
`0.430114221572876 / 0.33948240280151365`，best 仍为 epoch 3。两份均
`status=passed`、`problems=[]`、全部 checks 为 true；低值与波动原样保留。
watcher state SHA-256 更新为
`946a98baa91bfdb5a260baada8b2cc1b34608902da381d153a3f7fce36a84ca9`；四个唯一
active cell 合计 `80897` steps、`4348369` exposures，全局原子账本更新为
`1198397 / 51013200` steps、`140811599 / 725922600` exposures。terminal cells 与科学
状态均不变。

`2026-08-23T18:55:17Z`，watcher 保持 `failures=[]`、`transient_audit_races=1`，接受
VOC2012 `random_feature_local / seed 104729` epoch 63 与 NYUv2
`random_feature_local / seed 4121` epoch 55。VOC 强审计 SHA-256
`82d07ca7813e0feb2d8b028eef8052ba22b9f5d5adae7926be7981917f7ac184`，累计
`20790` optimizer steps、`83034` exposures；validation mean-IoU/pixel accuracy 为
`0.04634128883481026 / 0.6200064420700073`，best 仍为 epoch 57。NYUv2 强审计
SHA-256 `a28e5460d15f18c17ff2bf77879babd908d7880e7a9e9a57b68ca6ad242387ef`，连续
history 覆盖未单独采样的 epoch 54 并累计 `9845` steps、`39325` exposures；validation
abs-rel/delta1 为 `0.46670693829655646 / 0.3528579235076904`，best 仍为 epoch 3。
两份均 `status=passed`、`problems=[]`、全部 checks 为 true；低值与波动原样保留。

`2026-08-23T19:05:27Z`，watcher 再接受 NYUv2
`random_feature_local / seed 4121` epoch 56；强审计 SHA-256
`faacde324847a89dcbb23aebbee390bf4a34376ab17f8e33c8ad94927c96eaf3`，累计
`10024` optimizer steps、`40040` exposures，validation abs-rel/delta1 为
`0.4413928486406803 / 0.3529170513153076`，best 仍为 epoch 3。审计
`status=passed`、`problems=[]`、全部 checks 为 true。watcher 运行态 SHA-256 更新为
`d15f3268b9baa988ab97320003d9f3d2911aeb1b6c1b388b1009a66113d39d93`；四个唯一
active cell 合计 `81764` steps、`4351832` exposures，全局原子账本更新为
`1199264 / 51013200` steps、`140815062 / 725922600` exposures。terminal cells、下游门
与科学状态均不变。

`2026-08-23T19:15:33Z`，watcher 保持 `failures=[]`、`transient_audit_races=1`，接受
VOC2012 `random_feature_local / seed 104729` epoch 64 与 NYUv2
`random_feature_local / seed 4121` epoch 57。VOC 强审计 SHA-256
`720ba931bd27afcc968fceb64a2299a4aafc3bb18bb2aa32a67ef8dd486f1a33`，累计
`21120` optimizer steps、`84352` exposures；validation mean-IoU/pixel accuracy 为
`0.043480999767780304 / 0.6247711777687073`，best 仍为 epoch 57。NYUv2 强审计
SHA-256 `08cdf6a54dc7eb4d914a825fd8a5eb5bfeb4d575ce3454602155f6c3c7f946c9`，累计
`10203` steps、`40755` exposures；validation abs-rel/delta1 为
`0.45145426541566847 / 0.3501879692077637`，best 仍为 epoch 3。两份均
`status=passed`、`problems=[]`、全部 checks 为 true；低值与波动原样保留。watcher
state SHA-256 更新为
`2957f8ae68f6b65b9944274f4e9c633b40035d6374772a5bbcf84398e5cf51f8`；四个唯一
active cell 合计 `82273` steps、`4353865` exposures，全局原子账本更新为
`1199773 / 51013200` steps、`140817095 / 725922600` exposures。terminal cells、下游门
与科学状态均不变。

`2026-08-23T19:18:35Z` 复核下游预触发门：四个存活 matrix writer 分别且唯一对应
ImageNet-100、VOC2012、NYUv2 与 ADE20K，每任务仍为 `seed_workers=1`；main、causal、
extension、final evidence，以及监督回放 `registry.json`、`completion_audit.json` 均不存在，
因此没有提前触发后置产物。更新后的机器可读预触发审计 SHA-256 为
`cba6bcf7a0375160e5015dc08a3dc03e84a17204aea71c2eac23f0b529657ee4`，状态
`passed`、`problems=[]`、`execution_complete=false`、`changes_scientific_verdict=false`。

`2026-08-23T19:25:47Z`，watcher 保持 `failures=[]`、`transient_audit_races=1`，接受
VOC2012 `random_feature_local / seed 104729` epoch 65 与 NYUv2
`random_feature_local / seed 4121` epoch 58。VOC 强审计 SHA-256
`8b42dc2b6bcefb30d417d556b8d5d892e9580c90a1806e121c2109c0eebb5c2c`，累计
`21450` optimizer steps、`85670` exposures；validation mean-IoU/pixel accuracy 为
`0.04563377797603607 / 0.6195284128189087`，best 仍为 epoch 57。NYUv2 强审计
SHA-256 `f79ff82c69ae027a2490cb60a50d25b04af78def46260625989d9a959c1bbe5b`，累计
`10382` steps、`41470` exposures；validation abs-rel/delta1 为
`0.4392715536057949 / 0.3516525745391846`，best 仍为 epoch 3。两份均
`status=passed`、`problems=[]`、全部 checks 为 true；低值与波动原样保留。watcher
state SHA-256 更新为
`f0d3bec8adfaf29223061416e819850331b46b0ecddad7c3fb09041ddca7cd53`；四个唯一
active cell 合计 `82782` steps、`4355898` exposures，全局原子账本更新为
`1200282 / 51013200` steps、`140819128 / 725922600` exposures。terminal cells、下游门
与科学状态均不变。

`2026-08-23T19:36:12Z`，watcher 保持 `failures=[]`、`transient_audit_races=1`，本轮
仅接受 NYUv2 `random_feature_local / seed 4121` epoch 60。强审计 SHA-256
`61fd12aa3be46a166d89ca5294d8a34321943f9c074c71f72d0599fede3f3bbb`，连续 history
覆盖未单独采样的 epoch 59，累计 `10740` optimizer steps、`42900` exposures；validation
abs-rel/delta1 为 `0.4648809053003788 / 0.35581283569335936`，best 仍为 epoch 3。
审计 `status=passed`、`problems=[]`、全部 checks 为 true；退化波动原样保留。watcher
state SHA-256 更新为
`6e2ab61d51c104c802cacf2ec266e7c8ec1e81fbf269d52c70f1c8fcb738f00b`；四个唯一
active cell 合计 `83140` steps、`4357328` exposures，全局原子账本更新为
`1200640 / 51013200` steps、`140820558 / 725922600` exposures。terminal cells、下游门
与科学状态均不变。

`2026-08-23T19:46:18Z`，watcher 保持 `failures=[]`、`transient_audit_races=1`，接受
VOC2012 `random_feature_local / seed 104729` epoch 66 与 NYUv2
`random_feature_local / seed 4121` epoch 61。VOC 强审计 SHA-256
`acc85fe07c2ce10a522a2eedb8345a9b6ff2d44f340ea1d790b184e9a913fb61`，累计
`21780` optimizer steps、`86988` exposures；validation mean-IoU/pixel accuracy 为
`0.042677681893110275 / 0.6204013824462891`，best 仍为 epoch 57。NYUv2 强审计
SHA-256 `40f37c447a10002787026d0986c41c628d246e0a2bdc8e23739a09b02530e55a`，累计
`10919` steps、`43615` exposures；validation abs-rel/delta1 为
`0.4498134762048721 / 0.3497323989868164`，best 仍为 epoch 3。两份均
`status=passed`、`problems=[]`、全部 checks 为 true；低值与波动原样保留。watcher
state SHA-256 更新为
`14dce4014761981ed62fac0ec26730e10a2d62e03f2a8012f26c3336e5f77484`；四个唯一
active cell 合计 `83649` steps、`4359361` exposures，全局原子账本更新为
`1201149 / 51013200` steps、`140822591 / 725922600` exposures。terminal cells、下游门
与科学状态均不变。

`2026-08-23T19:56:29Z`，watcher 保持 `failures=[]`、`transient_audit_races=1`，接受
VOC2012 `random_feature_local / seed 104729` epoch 67 与 NYUv2
`random_feature_local / seed 4121` epoch 62。VOC 强审计 SHA-256
`ad7257c85f4bc44411822bb8fa7c2698eadc630c90e2b12971f338d414e3ba17`，累计
`22110` optimizer steps、`88306` exposures；validation mean-IoU/pixel accuracy 为
`0.04588232934474945 / 0.6180673837661743`，best 仍为 epoch 57。NYUv2 强审计
SHA-256 `5fa7fe23bf08bbc59b3bfb11c9df95d77639a1bfb107d57994c8e1888440103f`，累计
`11098` steps、`44330` exposures；validation abs-rel/delta1 为
`0.45007050409913063 / 0.34882268905639646`，best 仍为 epoch 3。两份均
`status=passed`、`problems=[]`、全部 checks 为 true；低值与波动原样保留。watcher
state SHA-256 更新为
`301885edfa3dd99175d455deb58992dd91b5fe9c75a5db38bafc9a0e391f78e6`；四个唯一
active cell 合计 `84158` steps、`4361394` exposures，全局原子账本更新为
`1201658 / 51013200` steps、`140824624 / 725922600` exposures。terminal cells、下游门
与科学状态均不变。

`2026-08-23T20:06:39Z`，watcher 保持 `failures=[]`、`transient_audit_races=1`，接受
VOC2012 `random_feature_local / seed 104729` epoch 68 与 NYUv2
`random_feature_local / seed 4121` epoch 63。VOC 强审计 SHA-256
`300cf0bee574b4f7cf57f71903427b297f447c315c5c3d0a2966264618ce9f1d`，累计
`22440` optimizer steps、`89624` exposures；validation mean-IoU/pixel accuracy 为
`0.045143645256757736 / 0.6173548698425293`，best 仍为 epoch 57。NYUv2 强审计
SHA-256 `49f1aea4562f9e015dd3cd180d79baa3b39c4ff8d0a5e5f8c9a516b638773e80`，累计
`11277` steps、`45045` exposures；validation abs-rel/delta1 为
`0.4625768393278122 / 0.34198970794677735`，best 仍为 epoch 3。两份均
`status=passed`、`problems=[]`、全部 checks 为 true；退化值与波动原样保留。watcher
state SHA-256 更新为
`8917314ed84ada065c9cbecd8c8998c670b4affd8f367d27271f6218ad5b4c25`；四个唯一
active cell 合计 `84667` steps、`4363427` exposures，全局原子账本更新为
`1202167 / 51013200` steps、`140826657 / 725922600` exposures。terminal cells、下游门
与科学状态均不变。

`2026-08-23T20:16:50Z`，watcher 保持 `failures=[]`、`transient_audit_races=1`，接受
ImageNet-100 `velocity / seed 7319` epoch 37 与 NYUv2
`random_feature_local / seed 4121` epoch 64。ImageNet 强审计 SHA-256
`f414820f581a3f7b482041f4330dbfb6689e1c2ff797338ed6001a2180cda173`，累计
`33670` optimizer steps、`4308835` exposures；validation top-1/top-5 为
`0.03616692426584235 / 0.15873261205564143`，best 仍为 epoch 36 的
`0.03817619783616692`。NYUv2 强审计 SHA-256
`605753d11de4a741441df746636283685c232e05c9318b5001a9318b06809320`，累计
`11456` steps、`45760` exposures；validation abs-rel/delta1 为
`0.45802064090967176 / 0.3499192237854004`，best 仍为 epoch 3。两份均
`status=passed`、`problems=[]`、全部 checks 为 true；低值与退化波动原样保留。
watcher state SHA-256 更新为
`7b587842966252ba82073ce797b868743c0fce1c57a4e0973ae2a33fd9152dca`；四个唯一
active cell 合计 `85756` steps、`4480597` exposures，全局原子账本更新为
`1203256 / 51013200` steps、`140943827 / 725922600` exposures。terminal cells、下游门
与科学状态均不变。

`2026-08-23T20:27:15Z`，watcher 保持 `failures=[]`、`transient_audit_races=1`，接受
VOC2012 `random_feature_local / seed 104729` epoch 69 与 NYUv2
`random_feature_local / seed 4121` epoch 66。VOC 强审计 SHA-256
`472fceef664fdd7c633740ba700598c70ac6ebb3b52b73086d764210e3083654`，累计
`22770` optimizer steps、`90942` exposures；validation mean-IoU/pixel accuracy 为
`0.04114404320716858 / 0.613889217376709`，best 仍为 epoch 57。NYUv2 强审计
SHA-256 `1372b38369a14917b405e16645daebbc8a27e7dc5c54587951a17cb0ec16f139`，连续
history 覆盖未单独采样的 epoch 65，累计 `11814` steps、`47190` exposures；validation
abs-rel/delta1 为 `0.4748153954744339 / 0.3516340732574463`，best 仍为 epoch 3。
两份均 `status=passed`、`problems=[]`、全部 checks 为 true；明显退化与低值原样保留。
watcher state SHA-256 更新为
`25c9407ca2ee586be1fd467c481e2da012f3c1d477f3ff06a48cf46354911cbb`；四个唯一
active cell 合计 `86444` steps、`4483345` exposures，全局原子账本更新为
`1203944 / 51013200` steps、`140946575 / 725922600` exposures。terminal cells、下游门
与科学状态均不变。

`2026-08-23T20:37:38Z`，watcher 保持 `failures=[]`、`transient_audit_races=1`，本轮
仅接受 NYUv2 `random_feature_local / seed 4121` epoch 67。强审计 SHA-256
`c384fbb027317c732d6d504a8a5b5999d03638f306db45d24a18227dd67dc4c2`，累计
`11993` optimizer steps、`47905` exposures；validation abs-rel/delta1 为
`0.4505818672478199 / 0.34454913139343263`，best 仍为 epoch 3。审计
`status=passed`、`problems=[]`、全部 checks 为 true；退化波动原样保留。watcher
state SHA-256 更新为
`45cb5d8198c189a8fa6dfe9eeb6b4b566350989953b6eb1caa3c0e3cce4c5442`；四个唯一
active cell 合计 `86623` steps、`4484060` exposures，全局原子账本更新为
`1204123 / 51013200` steps、`140947290 / 725922600` exposures。terminal cells、下游门
与科学状态均不变。

`2026-08-23T20:47:48Z`，watcher 保持 `failures=[]`、`transient_audit_races=1`，接受
VOC2012 `random_feature_local / seed 104729` epoch 70 与 NYUv2
`random_feature_local / seed 4121` epoch 68。VOC 强审计 SHA-256
`adca5340cac325bfd1cd5b4f1bf316179140ee97fe6454ddf417527a83e488ca`，累计
`23100` optimizer steps、`92260` exposures；validation mean-IoU/pixel accuracy 为
`0.04499000310897827 / 0.6162062287330627`，best 仍为 epoch 57。NYUv2 强审计
SHA-256 `e7f4963f103d10f2cc70faa4fa1c8f72b9174e7ee918b781479bdf8c4b911316`，累计
`12172` steps、`48620` exposures；validation abs-rel/delta1 为
`0.4558731496334076 / 0.3446944236755371`，best 仍为 epoch 3。两份均
`status=passed`、`problems=[]`、全部 checks 为 true；低值与退化波动原样保留。
watcher state SHA-256 更新为
`f4f258c7a12a142421bb0c4b5f09067fd379eb283c8fca77bd3e8f5f06bb4cee`；四个唯一
active cell 合计 `87132` steps、`4486093` exposures，全局原子账本更新为
`1204632 / 51013200` steps、`140949323 / 725922600` exposures。terminal cells、下游门
与科学状态均不变。

`2026-08-23T20:58:07Z`，watcher 保持 `failures=[]`、`transient_audit_races=1`，接受
VOC2012 `random_feature_local / seed 104729` epoch 71 与 NYUv2
`random_feature_local / seed 4121` epoch 69。VOC 强审计 SHA-256
`f1b4f25db6172972dc8d406208046888bbc42b938c4cfe8e71a1613db4a90090`，累计
`23430` optimizer steps、`93578` exposures；validation mean-IoU/pixel accuracy 为
`0.042507827281951904 / 0.6117920279502869`，best 仍为 epoch 57。NYUv2 强审计
SHA-256 `ca66afa1ab861f2b8510ff0d8cf845adbbf8894f9b113398ead9dd117dca61a6`，累计
`12351` steps、`49335` exposures；validation abs-rel/delta1 为
`0.46049114465713503 / 0.3448500633239746`，best 仍为 epoch 3。两份均
`status=passed`、`problems=[]`、全部 checks 为 true；低值与退化波动原样保留。
watcher state SHA-256 更新为
`a9242cac53f7493576397db4a8498c4109296f232e7f139daf10ec88620e9014`；四个唯一
active cell 合计 `87641` steps、`4488126` exposures，全局原子账本更新为
`1205141 / 51013200` steps、`140951356 / 725922600` exposures。terminal cells、下游门
与科学状态均不变。

`2026-08-23T21:08:30Z`，watcher 保持 `status=active`、`failures=[]`、
`transient_audit_races=1`，本轮接受 NYUv2 `random_feature_local / seed 4121`
epoch 70。强审计 SHA-256
`6eb07539f04e4b1c23dea09647265bdad137c0c7326be7ca2d0e409ea07bc444`，
`status=passed`、`problems=[]`、全部 checks 为 true；累计 `12530` optimizer
steps、`50050` exposures，validation abs-rel/delta1 为
`0.45781762450933455 / 0.3430696964263916`，best 仍为 epoch 3。低值与退化波动
原样保留。watcher state SHA-256 更新为
`bb416e38c206c34bd0fafe7eacad1a5928fb5387fd332d5da3255000116ca0d7`；
四个唯一 active cell 合计 `87820` steps、`4488841` exposures，全局原子账本更新为
`1205320 / 51013200` steps、`140952071 / 725922600` exposures。terminal cells、下游门
与科学状态均不变。GPU 6/7 各自继续承载两个不同任务的唯一 writer；本轮 12 秒
`nvidia-smi dmon` 显示两卡瞬时 SM 利用率接近 0%，但四个 worker 均保持高 CPU 活跃，
且最近 20 分钟有原子 report/checkpoint 写入，判定为固定 readout 的 CPU/指标计算阶段，
不据此改动已注册 batch、runtime 条件或增加同任务并发 writer。

`2026-08-23T21:18:42Z`，watcher 保持 `status=active`、`failures=[]`、
`transient_audit_races=1`，同轮接受三个正式原子点。VOC2012
`random_feature_local / seed 104729` epoch 72 强审计 SHA-256 为
`4a89fd2f90b5270fa97722facd4e6b65a9eace4012fae73b4bf8ce2038c5eaea`，
累计 `23760` steps、`94896` exposures，validation mean-IoU/pixel accuracy 为
`0.04605509713292122 / 0.6160385608673096`，best 仍为 epoch 57。NYUv2
`random_feature_local / seed 4121` epoch 71 强审计 SHA-256 为
`025b1318b5965562c2e5b7070256298fe92303e0d68bc0815acbee11c816bddb`，
累计 `12709` steps、`50765` exposures，validation abs-rel/delta1 为
`0.4557947784662247 / 0.3423459529876709`，best 仍为 epoch 3。ADE20K retry 2
`random_feature_local / seed 4121` epoch 3 强审计 SHA-256 为
`ee27fbfb46e055141371834c9035acf497f711d4df93fc6d8aedb09f7f2a3a4f`，
累计 `27285` steps、`54567` exposures，validation mean-IoU/pixel accuracy 为
`0.003720963839441538 / 0.19527418911457062`，best 更新为 epoch 3。三份均
`status=passed`、`problems=[]`、全部 checks 为 true；ADE 的低 mIoU 与既往
all-ignore/CUBLAS 失败证据均原样保留。watcher state SHA-256 更新为
`64334b27c93efe4a2166420740d71985c8aaa748bb817de603fa317d91745003`；
四个唯一 active cell 合计 `97424` steps、`4509063` exposures，全局原子账本为
`1214924 / 51013200` steps、`140972293 / 725922600` exposures。terminal cells 仍为
`15/240`，尚未触发 causal、extension、final evidence 或 132 份监督回放。

`2026-08-24T00:54:47Z` 合并审计：自上一记录点后 watcher 新接纳 `38` 份审计，
逐份复核均为 `returncode=0`、`status=passed`、`problems=[]`、全部 checks 为 true。
其中 NYUv2 `random_feature_local / seed 4121` 已完成 80 epoch，terminal 强审计
SHA-256 为 `7f24a86bb2c0b3622964d1f24f436cc8fe0e502c961555534aaaf4ea69c75152`，
test abs-rel 为 `0.3321781563102652`；VOC2012
`random_feature_local / seed 104729` 也已完成 80 epoch，terminal 强审计 SHA-256 为
`fa5ac86afc7282bdc4e76dc67ff0afab035762ce8271a0b5e1394561ef6b8eae`，
test mIoU 为 `0.03384625166654587`。两项低值均原样保留，不解释为正向科学结论。
两任务均由原有唯一 writer 自动接续，VOC2012 当前为 `z0 / seed 4121 / epoch 27`，
NYUv2 当前为 `random_feature_local / seed 7319 / epoch 15`；未出现重复 dataset/cell
writer。ImageNet-100 当前 `velocity / seed 7319 / epoch 39`，ADE20K retry 2 当前
`random_feature_local / seed 4121 / epoch 4`。四个 active cell 的最新强审计 SHA-256
依次为 `c3d4b515b4e26bb158c4bc1a4202d8d1112fb823936fd42a7cc9975b9095b611`、
`3b7e6c4484ac5e3523486db1ddb035fbee5c97e5de1074ad8d8ef7e4ef7241be`、
`a10f2d0fa0980d4a8341cce7a1798ab3767da9f1c82d36aebcc3439ccbc7bd04`、
`3ffcf24f3af356ed805e96b4b8e00ed691501a74b786474a9aeaacf99578ef90`。

terminal cells 更新为 ImageNet-100 `13/60`、VOC2012 `3/60`、NYUv2 `1/60`、
ADE20K `0/60`，合计 `17/240`。terminal subtotal 为 `1158220` optimizer steps、
`136625870` exposures；四个唯一 active cell subtotal 为 `83465` steps、`4660812`
exposures；全局原子账本为 `1241685 / 51013200` steps、
`141286682 / 725922600` exposures。watcher state SHA-256 为
`8ea70f2072e5485a718e6c48f5679eece055e242d6aa7b49aa3fe022ff576261`，
`status=active`、`failures=[]`、`transient_audit_races=1`。两个固定 revision worktree
仍 clean；更新后的下游预触发审计 SHA-256 为
`c164ec80741854c1bf142bd80ca91f6e42e5ec50166bb54fb584e49a14e3af5d`，确认 main、
causal、extension、final evidence、`registry.json` 和 `completion_audit.json` 均未提前生成。

`2026-08-24T01:05:06Z`，watcher 继续接受 NYUv2
`random_feature_local / seed 7319` epoch 16 与 VOC2012 `z0 / seed 4121` epoch 30。
NYUv2 强审计 SHA-256 为
`5ca1b8e9cfefcb063833cf0422a41d317b4a2caa1eb2e199bd551aea2aef77c5`，累计
`2864` steps、`11440` exposures，validation abs-rel/delta1 为
`0.3629045233130455 / 0.29566106796264646`；VOC 强审计 SHA-256 为
`82faa67cbcb5b4b6c1bee6396fcb35575ced7661b96301fb2ab39404623e26d6`，累计
`9900` steps、`39540` exposures，validation mean-IoU/pixel accuracy 为
`0.05460420250892639 / 0.6709406971931458`。两份均 `status=passed`、
`problems=[]`、全部 checks 为 true。合并后四个 active cell subtotal 为 `84634`
steps、`4665481` exposures，全局原子账本为 `1242854 / 51013200` steps、
`141291351 / 725922600` exposures；terminal cells 仍为 `17/240`。watcher state
SHA-256 更新为 `3821a0131e676cb7fc32df466868ac984143be1d8b8811fa2b23bc53aa8d6792`，
`failures=[]`；下游门仍未触发。

`2026-08-24T01:15:20Z`，自动接续链再次稳定提交：NYUv2
`random_feature_local / seed 7319` epoch 17 与 VOC2012 `z0 / seed 4121` epoch 33。
强审计 SHA-256 分别为
`599660d8682328a47e9f3e4feaa9e20023e3716a72ee1e252ad21e9fca1a3dd7` 和
`b3cb73a641a84800a1a302147729e5eb99fbc8b12f661f03b764ca026a506c71`；两份均
`status=passed`、`problems=[]`、全部 checks 为 true。NYUv2 累计 `3043` steps、
`12155` exposures，validation abs-rel/delta1 为
`0.36906701922416685 / 0.3334028244018555`；VOC 累计 `10890` steps、`43494`
exposures，validation mean-IoU/pixel accuracy 为
`0.05880025401711464 / 0.6674947738647461`。全局原子账本更新为
`1244023 / 51013200` steps、`141296020 / 725922600` exposures，terminal cells 仍为
`17/240`。watcher state SHA-256 为
`596b96df934a4d6e5d0aa82e05e61c4dff971134a954e6f824676bb1960c20f6`，
`status=active`、`failures=[]`；下游门继续保持未触发。

`2026-08-24T01:25:34Z`，watcher 接受 NYUv2
`random_feature_local / seed 7319` epoch 18 与 VOC2012 `z0 / seed 4121` epoch 36。
强审计 SHA-256 分别为
`f5650fc3752617a069523e939cb526b8977283942f73a63c926f507e132abf9f` 与
`5236750df8e74d7465a82fc1ce40eb95cda930299b741fc7260bb0cd095d2faa`；均为
`status=passed`、`problems=[]`、全部 checks 为 true。NYUv2 累计 `3222` steps、
`12870` exposures，validation abs-rel/delta1 为
`0.38666961491107943 / 0.3560675621032715`；VOC 累计 `11880` steps、`47448`
exposures，validation mean-IoU/pixel accuracy 为
`0.06309057772159576 / 0.6721590757369995`。四个 active cell subtotal 更新为
`86972` steps、`4674819` exposures，全局原子账本为
`1245192 / 51013200` steps、`141300689 / 725922600` exposures；terminal cells 仍为
`17/240`。watcher state SHA-256 为
`104bbf67eee216c2600b17506ffa1cc75bf7e54372391852d81a7a29c690b873`，
`failures=[]`；下游门未触发。

`2026-08-24T01:35:43Z`，watcher 接受 NYUv2
`random_feature_local / seed 7319` epoch 20 与 VOC2012 `z0 / seed 4121` epoch 39。
强审计 SHA-256 分别为
`d094275685b362b766ca2965e0395939aa32298d29cb1cb62969f3b815913729` 与
`10a63004891fe4a8adc009fffc72ef891d93a7f2f5cb567441fea9e5f4558f09`；均为
`status=passed`、`problems=[]`、全部 checks 为 true。NYUv2 累计 `3580` steps、
`14300` exposures，validation abs-rel/delta1 为
`0.44292999505996705 / 0.3911698818206787`；VOC 累计 `12870` steps、`51402`
exposures，validation mean-IoU/pixel accuracy 为
`0.058345988392829895 / 0.6682742238044739`。四个 active cell subtotal 为
`88320` steps、`4680203` exposures，全局原子账本为
`1246540 / 51013200` steps、`141306073 / 725922600` exposures；terminal cells 仍为
`17/240`。watcher state SHA-256 为
`ec48930a007c6f665a6ce60070c5539174e297be7db0630c089cc664489188cc`，
`status=active`、`failures=[]`；下游门未触发。

`2026-08-24T01:45:54Z`，watcher 接受 NYUv2
`random_feature_local / seed 7319` epoch 21 与 VOC2012 `z0 / seed 4121` epoch 42。
强审计 SHA-256 分别为
`965bba7cb0e4b69b6d084a0abe9aa1c1e88e1ed0e20708f88b2c69b00e01cbaf` 与
`8b80bb36ac17eff6b18decd7b3aa554dd5ae5323244d2b16bf6a8958df3757e1`；均为
`status=passed`、`problems=[]`、全部 checks 为 true。NYUv2 累计 `3759` steps、
`15015` exposures，validation abs-rel/delta1 为
`0.3842214301228523 / 0.36094322204589846`；VOC 累计 `13860` steps、`55356`
exposures，validation mean-IoU/pixel accuracy 为
`0.07238198816776276 / 0.6682637929916382`。四个 active cell subtotal 为
`89489` steps、`4684872` exposures，全局原子账本为
`1247709 / 51013200` steps、`141310742 / 725922600` exposures；terminal cells 仍为
`17/240`。watcher state SHA-256 为
`82bd244481f27df85d2b4f660526b5dbd4c6f59e546fac23c8d06f21b6626561`，
`status=active`、`failures=[]`；下游门未触发。

`2026-08-24T01:56:05Z`，用户再次授权在 GPU 6/7 显存足够时继续跨任务并行。现场复核确认四个正式任务已经全部并行：GPU 6 上为 ImageNet-100 与 NYUv2，GPU 7 上为 VOC2012 与 ADE20K；四任务各自只有一个正式 matrix writer，注册条件仍为每任务 `seed_workers=1`。因此未启动第五个 writer，也未修改正式 batch、sampler、LR 或 runtime profile。两张卡各约有 `77–78 GiB` 可见空闲显存，但其中各约 `62 GiB` 由外部 `m3call_baselines` 占卡/回避进程持有；该外部进程未被修改或终止。30 秒 `nvidia-smi dmon` 观察到 FieldScope 计算呈间歇突发，GPU 6 SM 峰值 `100%`、GPU 7 峰值 `54%`；低利用率采样期间四个 worker 均存活并继续产生原子报告，不据此改变已注册科学条件。

同一轮 watcher 新接纳 NYUv2 `random_feature_local / seed 7319` epoch 22 与 VOC2012 `z0 / seed 4121` epoch 45。两份强审计 SHA-256 分别为 `4a547a7febd7d1bc732f1e7dd3ff00ddb6e8806bb475d78e23916e2f4d21b58e` 和 `970c64b20407556171d839ed5ecfab76946a5cac9cccaa9736591c2251efd3dd`，均为 `returncode=0`、`status=passed`、`problems=[]` 且全部 checks 为 true。NYUv2 累计 `3938` optimizer steps、`15730` exposures，validation abs-rel/delta1 为 `0.3847302719950676 / 0.27360963821411133`；VOC 累计 `14850` steps、`59310` exposures，validation mean-IoU/pixel accuracy 为 `0.07690302282571793 / 0.6548343300819397`。指标原样保留，不形成方法有效性结论。

该轮 watcher state SHA-256 为 `32d1fc59cfe4dfe525f4b5dace9f262b38a5dc1c1421cb7d2c821aeed66b54b9`，`status=active`、`failures=[]`、`transient_audit_races=1`，共接纳 `155` 份中间审计和 `17` 份 terminal 审计。四个唯一 active cell subtotal 为 `90658` steps、`4689541` exposures；全局原子账本为 `1248878 / 51013200` steps、`141315411 / 725922600` exposures，terminal cells 仍为 `17/240`。main、causal、extension、final evidence 以及 `registry.json`、`completion_audit.json` 均保持未生成；`execution_complete=false`、`method_effectiveness_conclusion=null`、`changes_scientific_verdict=false`。

`2026-08-24T02:06:18Z`，watcher 再接纳 NYUv2 `random_feature_local / seed 7319` epoch 23 与 VOC2012 `z0 / seed 4121` epoch 48。强审计 SHA-256 分别为 `8430e14142c2e3231b16ef89db58564c3c701395d64f68352d97643c4420bd9a` 和 `34a17d0274260ef0961e8fc7363015540dc69f625a08380f1a65c3d1bc074040`；两份均为 `returncode=0`、`status=passed`、`problems=[]` 且全部 checks 为 true。NYUv2 累计 `4117` steps、`16445` exposures，validation abs-rel/delta1 为 `0.4086731106042862 / 0.3613119602203369`；VOC 累计 `15840` steps、`63264` exposures，validation mean-IoU/pixel accuracy 为 `0.08351431041955948 / 0.6674757599830627`。这些仍是单 cell 中间指标，原样保留且不形成科学裁决。

最新 watcher state SHA-256 更新为 `d6205ca3082adf458b3de44f69c579c5d1f1d059a146d22374fc8c3f41436817`，`status=active`、`failures=[]`、`transient_audit_races=1`，累计 `157` 份中间审计、`17` 份 terminal 审计。四个唯一 active cell subtotal 为 `91827` steps、`4694210` exposures；全局原子账本为 `1250047 / 51013200` steps、`141320080 / 725922600` exposures。四任务仍各一个 writer，terminal cells 仍为 `17/240`；六类下游/终局产物均不存在，未触发 causal、extension、final 或逐样本回放。

`2026-08-24T02:16:28Z`，watcher 接纳 NYUv2 `random_feature_local / seed 7319` epoch 24 与 VOC2012 `z0 / seed 4121` epoch 51。强审计 SHA-256 分别为 `5053e5f3eec792f25f759a05dfb863130cfb6b1318256aa221a97e0fdf5b3f7d` 和 `ada649c8132e7f7396dea3432f7d3571079bf14d0e8de5c9f3da69aa669cc443`；均为 `returncode=0`、`status=passed`、`problems=[]` 且全部 checks 为 true。NYUv2 累计 `4296` steps、`17160` exposures，validation abs-rel/delta1 为 `0.392384247481823 / 0.2709263801574707`；VOC 累计 `16830` steps、`67218` exposures，validation mean-IoU/pixel accuracy 为 `0.08543980121612549 / 0.6698160767555237`。指标继续原样保留，不推导主方法结论。

watcher state SHA-256 更新为 `5d405bee165f8a35b8b94b8b530e94fd3094bcbb02baadf2959c1607209e49fa`，累计 `159` 份中间审计与 `17` 份 terminal 审计，`status=active`、`failures=[]`、`transient_audit_races=1`。四个唯一 active cell subtotal 为 `92996` steps、`4698879` exposures；全局原子账本为 `1251216 / 51013200` steps、`141324749 / 725922600` exposures。ImageNet-100 与 ADE20K 未出现新的原子提交，仍分别按 epoch 39 和 retry 2 epoch 4 计账；不得用进程存活替代其原子进度。terminal cells 和全部下游门状态不变。

`2026-08-24T02:26:39Z`，watcher 接纳 NYUv2 `random_feature_local / seed 7319` epoch 26 与 VOC2012 `z0 / seed 4121` epoch 54。NYUv2 强审计 SHA-256 为 `842717616ca781f91072fa930bb0cc8a494890a4217394ccc01756662cd69082`，连续 history 包含未单独采样的 epoch 25，累计 `4654` steps、`18590` exposures，validation abs-rel/delta1 为 `0.39002257138490676 / 0.3683329105377197`。VOC 强审计 SHA-256 为 `d47e03a9eeadc5fbaf7cda4e5c52d624881023e6ae67728cf2621e8f22f033c9`，累计 `17820` steps、`71172` exposures，validation mean-IoU/pixel accuracy 为 `0.08742199838161469 / 0.6727145910263062`。两份均为 `returncode=0`、`status=passed`、`problems=[]` 且全部 checks 为 true。

watcher state SHA-256 更新为 `1f6cb39243e2a9572042d0834c3160b1255572c7d07e9df6b6aab4c330f1a6fb`，累计 `161` 份中间审计、`17` 份 terminal 审计，`failures=[]`。四个唯一 active cell subtotal 为 `94344` steps、`4704263` exposures；全局原子账本为 `1252564 / 51013200` steps、`141330133 / 725922600` exposures。ImageNet-100 与 ADE20K 仍无新原子提交，四任务 writer 和下游门状态不变；所有中间指标继续作为原始证据保留。

`2026-08-24T02:36:50Z`，watcher 接纳 NYUv2 `random_feature_local / seed 7319` epoch 27 与 VOC2012 `z0 / seed 4121` epoch 57。两份强审计 SHA-256 分别为 `d52b123058eec14031ec77b4b98a48d657ce0c984c4a39c29e719fc6a9677552` 与 `e1f18cc8064c053efdcf2a57bfa08f8263921214c0610eda77c29e5fdbec5f96`，均为 `returncode=0`、`status=passed`、`problems=[]` 且全部 checks 为 true。NYUv2 累计 `4833` steps、`19305` exposures，validation abs-rel/delta1 为 `0.4135191634297371 / 0.35849647521972655`；VOC 累计 `18810` steps、`75126` exposures，validation mean-IoU/pixel accuracy 为 `0.09358499944210052 / 0.668360710144043`。数值原样保留，不形成总体结论。

watcher state SHA-256 更新为 `20a05a2e57d557f1f488670421f70a6f14fde7fa782cf50c55e5ead7aa0fdce5`，累计 `163` 份中间审计与 `17` 份 terminal 审计，`status=active`、`failures=[]`。四个唯一 active cell subtotal 为 `95513` steps、`4708932` exposures；全局原子账本为 `1253733 / 51013200` steps、`141334802 / 725922600` exposures。四个 writer、固定 revision/clean worktree 与下游预触发门均保持合规；ImageNet-100 和 ADE20K 仍按最近原子 epoch 39/4 计账。

`2026-08-24T02:47:13Z`，watcher 接纳 NYUv2 `random_feature_local / seed 7319` epoch 28 与 VOC2012 `z0 / seed 4121` epoch 60。强审计 SHA-256 分别为 `564b7c5f0b8df4c4ea8eccb65d0e1e55c0e20d970a20714a14b0e1b98fd4b8d4` 和 `f95878544af00c5319ecec1d04c71d8208583a94e4fd1d595be011db0c5cc7db`，均为 `returncode=0`、`status=passed`、`problems=[]` 且全部 checks 为 true。NYUv2 累计 `5012` steps、`20020` exposures，validation abs-rel/delta1 为 `0.4116827681660652 / 0.36803436279296875`；VOC 累计 `19800` steps、`79080` exposures，validation mean-IoU/pixel accuracy 为 `0.09196818619966507 / 0.6662207245826721`。这些中间数值继续原样保留。

watcher state SHA-256 更新为 `960cb804cde3ca3a5b51e81f1fe638aea7e2d0282f5e443795ec1bf59c8aa99e`，累计 `165` 份中间审计、`17` 份 terminal 审计，`status=active`、`failures=[]`。四个唯一 active cell subtotal 为 `96682` steps、`4713601` exposures；全局原子账本为 `1254902 / 51013200` steps、`141339471 / 725922600` exposures。GPU 6/7 采样曾分别达到约 `35%/100%` SM，外部占卡进程按负载回避；未修改正式 batch/runtime 条件。ImageNet-100 epoch 40 尚未原子落盘，仍按 epoch 39 计账；下游门状态不变。

`2026-08-24T02:57:36Z`，watcher 接纳 NYUv2 `random_feature_local / seed 7319` epoch 29 与 VOC2012 `z0 / seed 4121` epoch 63。强审计 SHA-256 分别为 `c0e8a9eb345870cfd5686a7fb54a1210453ac5562c86fd851afaef5cc1a90d61` 和 `f4021014a64e0326ff7fe356b57885a456a1b6bcce6b6995cee1b1621efda1a6`，均为 `returncode=0`、`status=passed`、`problems=[]` 且全部 checks 为 true。NYUv2 累计 `5191` steps、`20735` exposures，validation abs-rel/delta1 为 `0.3976718448102474 / 0.32610416412353516`；VOC 累计 `20790` steps、`83034` exposures，validation mean-IoU/pixel accuracy 为 `0.08922883868217468 / 0.6612809300422668`。指标原样保留。

watcher state SHA-256 更新为 `0586b0c6616edf38780b46e094013a4fac4ab20578bf11d9a7cee97dd1bbd308`，累计 `167` 份中间审计与 `17` 份 terminal 审计，`status=active`、`failures=[]`。四个唯一 active cell subtotal 为 `97851` steps、`4718270` exposures；全局原子账本为 `1256071 / 51013200` steps、`141344140 / 725922600` exposures。ImageNet-100 和 ADE20K 继续运行但没有新原子点；四任务唯一 writer、下游门和终局文件缺席状态不变。

`2026-08-24T03:08:03Z`，watcher 接纳 NYUv2 `random_feature_local / seed 7319` epoch 31 与 VOC2012 `z0 / seed 4121` epoch 65。NYUv2 连续 history 覆盖未单独采样的 epoch 30；强审计 SHA-256 为 `b25b043737b609b304fdab8e8b5587f9977fd8fbf27636ccf308895b07c8fcb9`，累计 `5549` steps、`22165` exposures，validation abs-rel/delta1 为 `0.38642340824007987 / 0.27260074615478513`。VOC 强审计 SHA-256 为 `2bb460f136a0679bfda46c04f4a40e682c1afe582f03a97268eb02e4f0c834be`，累计 `21450` steps、`85670` exposures，validation mean-IoU/pixel accuracy 为 `0.09970813989639282 / 0.6651194095611572`。两份均为 `returncode=0`、`status=passed`、`problems=[]` 且全部 checks 为 true。

watcher state SHA-256 更新为 `e34403d5f5f763cfe0b277b9c2557c4c8c90eeaa18944644e20bf3fe18d224fe`，累计 `169` 份中间审计、`17` 份 terminal 审计，`status=active`、`failures=[]`。四个唯一 active cell subtotal 为 `98869` steps、`4722336` exposures；全局原子账本为 `1257089 / 51013200` steps、`141348206 / 725922600` exposures。运行中 report 直接复核也确认 ImageNet-100 仍为 39 条 history、ADE20K 为 4 条，与 watcher 一致；不存在漏接纳完成 epoch。下游门状态不变。

`2026-08-24T03:18:29Z`，四个正式任务在同一 watcher 轮次全部产生新原子点，且四份强审计均为 `returncode=0`、`status=passed`、`problems=[]`、全部 checks 为 true。ImageNet-100 `velocity / seed 7319` epoch 40 强审计 SHA-256 为 `e1fa64b1d495f108989c960e4ae9797a2fb947807dd933b7840db05500ced8ee`，累计 `36400` steps、`4658200` exposures，validation top-1/top-5 为 `0.04095826893353941 / 0.17156105100463678`。VOC2012 `z0 / seed 4121` epoch 69 强审计 SHA-256 为 `11734b9728b7d6420c66b46bc8530644222a8ea320c9c0c49614f34394c6eed0`，累计 `22770` steps、`90942` exposures，validation mean-IoU/pixel accuracy 为 `0.09620590507984161 / 0.6677539944648743`。

同轮 NYUv2 `random_feature_local / seed 7319` epoch 32 强审计 SHA-256 为 `15f73c2676495bc059a04e5d98b03b2034ac5217e3fee9b1468718ea2e83d32c`，累计 `5728` steps、`22880` exposures，validation abs-rel/delta1 为 `0.3950006291270256 / 0.29871330261230467`。ADE20K retry 2 `random_feature_local / seed 4121` epoch 5 强审计 SHA-256 为 `6c500fe0330104642235241396aa19e31f69795596bd52824da570649b6773ae`，累计 `45475` steps、`90945` exposures，validation mean-IoU/pixel accuracy 为 `0.004229159560054541 / 0.1959560513496399`。ADE 的极低 mIoU 与既有 all-ignore/CUBLAS 失败证据继续原样保留，不作正向解释。

watcher state SHA-256 更新为 `5927ad8298ed4858efe09f7b54572084b07bcfce4fef00b952d9aeaf11f74d5d`，累计 `173` 份中间审计、`17` 份 terminal 审计，`status=active`、`failures=[]`、`transient_audit_races=1`。四个唯一 active cell subtotal 为 `110373` steps、`4862967` exposures；全局原子账本为 `1268593 / 51013200` steps、`141488837 / 725922600` exposures。四个 writer、固定 revision/clean worktree 与下游门保持合规；terminal cells 与科学状态不变。

`2026-08-24T03:29:22Z`，watcher 接纳 NYUv2 `random_feature_local / seed 7319` epoch 33 与 VOC2012 `z0 / seed 4121` epoch 71。NYUv2 强审计 SHA-256 为 `1eba05d8b6f80261be531e47b908cea18911606b4ef61e68689e53e79329fc1e`，累计 `5907` steps、`23595` exposures，validation abs-rel/delta1 为 `0.4020660273730755 / 0.35291213989257814`。VOC 强审计 SHA-256 为 `0eceace8a1d2162ddccb499daa5c4fcdadc561062ceba7c61dab4eef19980843`，累计 `23430` steps、`93578` exposures，validation mean-IoU/pixel accuracy 为 `0.09258504956960678 / 0.6654554605484009`。两份均为 `returncode=0`、`status=passed`、`problems=[]` 且全部 checks 为 true；指标作为中间原始证据保留，不形成方法有效性结论。

watcher state SHA-256 更新为 `5b97d8ef7df33920f712005819f59e7015ff6c7f3fd4a47a5824186fcd9ba204`，累计 `175` 份中间审计、`17` 份 terminal 审计，`status=active`、`failures=[]`、`transient_audit_races=1`。四个唯一 active cell subtotal 为 `111212` steps、`4866318` exposures；全局原子账本为 `1269432 / 51013200` steps、`141492188 / 725922600` exposures。GPU 6 上 ImageNet-100 与 NYUv2、GPU 7 上 VOC2012 与 ADE20K 已覆盖全部四任务，各任务保持唯一 writer 与 `seed_workers=1`；即使仍有可见空闲显存，也不启动会重叠写同一 dataset/cell 的第五个 writer，不修改正式 batch、sampler、LR 或 runtime profile。`2026-08-24T03:32:32Z` 复核两个固定 revision worktree 仍 clean，main、causal、extension、final evidence、`registry.json` 与 `completion_audit.json` 均未提前生成；`execution_complete=false`、`method_effectiveness_conclusion=null`、`changes_scientific_verdict=false`。

`2026-08-24T03:39:48Z`，watcher 接纳 NYUv2 `random_feature_local / seed 7319` epoch 35 与 VOC2012 `z0 / seed 4121` epoch 74；NYUv2 连续 history 包含未单独采样的 epoch 34。NYUv2 强审计 SHA-256 为 `e8496e71fbdeec1d8b3a95e452b94ddf6b7f98360f1c7ad14b58cdabf7c257ab`，累计 `6265` steps、`25025` exposures，validation abs-rel/delta1 为 `0.4058659166097641 / 0.3413184642791748`。VOC 强审计 SHA-256 为 `435ac8269838db6ce4cc789034aa3b8ede808313deae166c393f093c96598df5`，累计 `24420` steps、`97532` exposures，validation mean-IoU/pixel accuracy 为 `0.09814269095659256 / 0.6660163402557373`。两份均为 `returncode=0`、`status=passed`、`problems=[]` 且全部 checks 为 true；这些仍为中间 cell 结果，原样保留且不形成总体科学结论。

watcher state SHA-256 更新为 `d23e4c5283aba35efa4e707db0047522514fde91b0800984d260badb31253f5f`，累计 `177` 份中间审计、`17` 份 terminal 审计，`status=active`、`failures=[]`、`transient_audit_races=1`。四个唯一 active cell subtotal 为 `112560` steps、`4871702` exposures；全局原子账本为 `1270780 / 51013200` steps、`141497572 / 725922600` exposures。`2026-08-24T03:41:08Z` 复核正式与分析 worktree 均 clean，六类下游/终局产物继续缺席；四任务唯一 writer、`seed_workers=1` 与门顺序保持合规，goal 继续 active。

`2026-08-24T03:50:10Z`，watcher 接纳 NYUv2 `random_feature_local / seed 7319` epoch 36 与 VOC2012 `z0 / seed 4121` epoch 77。强审计 SHA-256 分别为 `9d0dfa8f3b74bdc54a5c0371fe2ae0c4c512fa40a15f02f38d57f485c33e4ef3` 和 `3b55ac58e46bc9b83a2a0da4054fc62375d988f94fca9d405194e91009fca64e`；两份均为 `status=passed`、`problems=[]` 且 37 项 checks 全为 true。NYUv2 累计 `6444` steps、`25740` exposures，validation abs-rel/delta1 为 `0.39665661156177523 / 0.3172856330871582`；VOC 累计 `25410` steps、`101486` exposures，validation mean-IoU/pixel accuracy 为 `0.09858237206935883 / 0.6664003729820251`。中间指标原样保留。

`2026-08-24T04:00:21Z`，watcher 继续接纳 NYUv2 epoch 37；强审计 SHA-256 为 `5a397d04f52ba9ac638db1f624661788c99855a2445489054f6f81ce3e29b394`，累计 `6623` steps、`26455` exposures，validation abs-rel/delta1 为 `0.4272005997598171 / 0.3749685287475586`，全部 checks 为 true。同期只读双读确认 VOC `z0 / seed 4121` 已稳定完成 80-epoch training report（`26400` steps、`105440` exposures），但尚无 test 字段，因此未提前计作 terminal。

`2026-08-24T04:10:25Z`，VOC2012 `z0 / seed 4121` terminal 强审计正式通过，SHA-256 为 `d8f203be551dccb2a56203b6e96072ecf3fd4187171f48b1fa5136e361f742bd`，43 项 checks 全为 true，`problems=[]`、`cell_execution_complete=true`、`full_execution_complete=false`。test mean-IoU 为 `0.08926922082901001`，test 样本数 `1449`；该低结果原样保留，不作正向解释。同轮 NYUv2 epoch 38 强审计 SHA-256 为 `d2da0511a5b83ebc7c906c0546ad0f7a853e331c8c281b32346ac139f7de8c6e`，累计 `6802` steps、`27170` exposures，validation abs-rel/delta1 为 `0.42202609553933146 / 0.37037076950073244`，37 项 checks 全为 true。

watcher state SHA-256 更新为 `b06fc8f79f0aa8b45c9eecedf258dac83d5b4b538c2fde3b09c9bba9aa8ee2e7`，累计 `181` 份中间审计、`18` 份 terminal 审计，terminal 分布为 ImageNet-100 `13/60`、VOC2012 `4/60`、NYUv2 `1/60`、ADE20K `0/60`。terminal subtotal 为 `1184620` steps、`136731310` exposures；已强审计的三个 active cells subtotal 为 `88677` steps、`4776315` exposures；全局原子账本为 `1273297 / 51013200` steps、`141507625 / 725922600` exposures。VOC 同一唯一 writer 已自动接续至 `z0 / seed 7319`，`2026-08-24T04:14:15Z` 的稳定运行中 report 为 epoch 1、`330` steps、`1318` exposures；因 watcher 尚未强审计，该点未计入原子账本。`2026-08-24T04:12:23Z` 复核四个 writer 均存活、两个固定 worktree clean，六类下游/终局文件继续缺席；`execution_complete=false`、`method_effectiveness_conclusion=null`、`changes_scientific_verdict=false`。

`2026-08-24T04:20:40Z`，watcher 首次正式接纳 VOC2012 新 cell `z0 / seed 7319` epoch 2 与 NYUv2 `random_feature_local / seed 7319` epoch 39。VOC 强审计 SHA-256 为 `61af7ea55c0acbb76df2d7d75ba6c9c0aea388e1a98517a382904d0de4ee2a15`，累计 `660` steps、`2636` exposures，validation mean-IoU/pixel accuracy 为 `0.031858671456575394 / 0.6690319180488586`；该极低中间 mIoU 原样保留，不作正向解释。NYUv2 强审计 SHA-256 为 `6ea0b015dcdfa0c18fa24bbe44e1be25b526afa3c8d4df500cb6d0c3bd645ca6`，累计 `6981` steps、`27885` exposures，validation abs-rel/delta1 为 `0.4036115974187851 / 0.3344071388244629`。两份均为 `returncode=0`、`status=passed`、`problems=[]` 且 37 项 checks 全为 true。

watcher state SHA-256 更新为 `e4978cb9444491e74a34ba96d3164b6a63a5654a67324175818b4d905e0f42d4`，累计 `183` 份中间审计、`18` 份 terminal 审计，`status=active`、`failures=[]`、`transient_audit_races=1`。四个任务重新都有已强审计 active cell，active subtotal 为 `89516` steps、`4779666` exposures；全局原子账本为 `1274136 / 51013200` steps、`141510976 / 725922600` exposures。`2026-08-24T04:21:38Z` 复核两个固定 revision worktree clean、六类下游/终局产物缺席，四任务唯一 writer 与 `seed_workers=1` 不变；goal 继续 active。

`2026-08-24T04:30:51Z`，watcher 接纳 NYUv2 `random_feature_local / seed 7319` epoch 41 与 VOC2012 `z0 / seed 7319` epoch 5；NYUv2 连续 history 包含未单独采样的 epoch 40。NYUv2 强审计 SHA-256 为 `3607f777ad8f2c1c08fa2a06a6c8edf6be1f6a602cc59f2e7755436f097da102`，累计 `7339` steps、`29315` exposures，validation abs-rel/delta1 为 `0.39901761114597323 / 0.27187175750732423`。VOC 强审计 SHA-256 为 `6a8d50522ee545b09bbff280a27d695085393118b7908b749e8d78cfc6a3be75`，累计 `1650` steps、`6590` exposures，validation mean-IoU/pixel accuracy 为 `0.031858671456575394 / 0.6690319180488586`。两份均为 `returncode=0`、`status=passed`、`problems=[]` 且 37 项 checks 全为 true；低指标原样保留。

watcher state SHA-256 更新为 `a9ae93653872e8322cff3527756dab114247f830aa886920bc23bbf541914fce`，累计 `185` 份中间审计、`18` 份 terminal 审计，active subtotal 为 `90864` steps、`4785050` exposures；全局原子账本为 `1275484 / 51013200` steps、`141516360 / 725922600` exposures。`2026-08-24T04:38:27Z` 复核固定 worktree clean、下游终局产物缺席，四个唯一 writer 继续运行；科学状态不变。

`2026-08-24T04:41:02Z`，watcher 接纳 NYUv2 `random_feature_local / seed 7319` epoch 42 与 VOC2012 `z0 / seed 7319` epoch 7。NYUv2 强审计 SHA-256 为 `4df6c752c8f8418219d40952d793bd6391220f9b55cad3f86181cce84a53305d`，累计 `7518` steps、`30030` exposures，validation abs-rel/delta1 为 `0.422716423869133 / 0.36467480659484863`。VOC 强审计 SHA-256 为 `360f0453bb38dd58291b0b540ca86973403f0719c598635b750ce35889d772d1`，累计 `2310` steps、`9226` exposures，validation mean-IoU/pixel accuracy 为 `0.031858671456575394 / 0.6690319180488586`。两份均为 `returncode=0`、`status=passed`、`problems=[]` 且 37 项 checks 全为 true；VOC 的持续极低 mIoU 原样保留。

watcher state SHA-256 更新为 `3ec16443cf5439d4d70cd65b5bbd9782c2715d2218d1d5c33c40f524720d0889`，累计 `187` 份中间审计、`18` 份 terminal 审计，active subtotal 为 `91703` steps、`4788401` exposures；全局原子账本为 `1276323 / 51013200` steps、`141519711 / 725922600` exposures。`2026-08-24T04:41:50Z` 复核固定 worktree clean、六类下游/终局产物缺席，四个唯一 writer 和门顺序保持合规。

`2026-08-24T04:51:13Z`，watcher 接纳 NYUv2 `random_feature_local / seed 7319` epoch 43 与 VOC2012 `z0 / seed 7319` epoch 10。NYUv2 强审计 SHA-256 为 `e6b7f85b70aee63c2f962afe1f59c608debb1b0fab28b912b44e0b02d1bc4902`，累计 `7697` steps、`30745` exposures，validation abs-rel/delta1 为 `0.4148321807384491 / 0.3705596923828125`。VOC 强审计 SHA-256 为 `8cfb41795f0fcb72b4e236c1e24d5a210fa38942c03c3eaea35826000286f91c`，累计 `3300` steps、`13180` exposures，validation mean-IoU/pixel accuracy 为 `0.03200000524520874 / 0.6689719557762146`。两份均为 `returncode=0`、`status=passed`、`problems=[]` 且 37 项 checks 全为 true；低指标原样保留。

watcher state SHA-256 更新为 `ba7d6c6edb0d06887d36c7021823a874594a95cf7c94078b83c97efa1aa51007`，累计 `189` 份中间审计、`18` 份 terminal 审计，active subtotal 为 `92872` steps、`4793070` exposures；全局原子账本为 `1277492 / 51013200` steps、`141524380 / 725922600` exposures。`2026-08-24T04:57:53Z` 复核固定 worktree clean、六类下游/终局产物缺席，四个唯一 writer 与门顺序继续合规。

`2026-08-24T05:01:23Z`，watcher 接纳 NYUv2 `random_feature_local / seed 7319` epoch 44 与 VOC2012 `z0 / seed 7319` epoch 13。NYUv2 强审计 SHA-256 为 `32d94b6e1c6646a471cc3bdb192b6ff600828c1fa551108855f76c72d4851f9f`，累计 `7876` steps、`31460` exposures，validation abs-rel/delta1 为 `0.4179470896720886 / 0.34183354377746583`。VOC 强审计 SHA-256 为 `265ee16f5e1a2193c79f4a6217259021d973a2940e3aaccbe74f05900cacccb6`，累计 `4290` steps、`17134` exposures，validation mean-IoU/pixel accuracy 为 `0.03974246233701706 / 0.6705969572067261`。两份均为 `returncode=0`、`status=passed`、`problems=[]` 且 37 项 checks 全为 true；单 cell 中间指标不形成总体结论。

watcher state SHA-256 更新为 `c002d438eab02c840b8e858d280af7f6198cbc96e0bd80b47c89930509fa7816`，累计 `191` 份中间审计、`18` 份 terminal 审计，active subtotal 为 `94041` steps、`4797739` exposures；全局原子账本为 `1278661 / 51013200` steps、`141529049 / 725922600` exposures。`2026-08-24T05:02:15Z` 复核固定 worktree clean、六类下游/终局产物缺席，四个唯一 writer 和门顺序保持合规。

`2026-08-24T05:11:36Z`，watcher 接纳 NYUv2 `random_feature_local / seed 7319` epoch 46 与 VOC2012 `z0 / seed 7319` epoch 15；NYUv2 连续 history 包含未单独采样的 epoch 45。NYUv2 强审计 SHA-256 为 `b4a2f672c810c74c423d0a6dbb6dc5deacb3f2d821ad04ed96f6a4c111ece9c9`，累计 `8234` steps、`32890` exposures，validation abs-rel/delta1 为 `0.4137462623417377 / 0.32526569366455077`。VOC 强审计 SHA-256 为 `cf40f5f3d17fdc8ef1a52b0225d24f4b7aed3bee875105671882260e846a3ac1`，累计 `4950` steps、`19770` exposures，validation mean-IoU/pixel accuracy 为 `0.039665382355451584 / 0.6688825488090515`。两份均为 `returncode=0`、`status=passed`、`problems=[]` 且 37 项 checks 全为 true；中间指标原样保留。

watcher state SHA-256 更新为 `bb5a8cc6d75e079ed3fe09fd6c86831f46aac5ce9a4e3acf85b7b644e458013c`，累计 `193` 份中间审计、`18` 份 terminal 审计，active subtotal 为 `95059` steps、`4801805` exposures；全局原子账本为 `1279679 / 51013200` steps、`141533115 / 725922600` exposures。`2026-08-24T05:17:20Z` 复核固定 worktree clean、六类下游/终局产物缺席，四个唯一 writer 与门顺序继续合规。

`2026-08-24T05:21:49Z`，watcher 接纳 NYUv2 `random_feature_local / seed 7319` epoch 47 与 VOC2012 `z0 / seed 7319` epoch 18。NYUv2 强审计 SHA-256 为 `d8b120f989b30752fbf4582d9641d311b52d49feb3e027615b5874801f27b593`，累计 `8413` steps、`33605` exposures，validation abs-rel/delta1 为 `0.4217566207051277 / 0.3539432525634766`。VOC 强审计 SHA-256 为 `b0cb1c905733b9119679dbcde662558716e39f9d94ec4cad7fc1753e97c6effc`，累计 `5940` steps、`23724` exposures，validation mean-IoU/pixel accuracy 为 `0.04210272431373596 / 0.6713262796401978`。两份均为 `returncode=0`、`status=passed`、`problems=[]` 且 37 项 checks 全为 true；中间指标原样保留。

watcher state SHA-256 更新为 `cd4e89a513c7d218b9fac9656e3b7fbedac834af16cad49f276456aa8b875e3e`，累计 `195` 份中间审计、`18` 份 terminal 审计，active subtotal 为 `96228` steps、`4806474` exposures；全局原子账本为 `1280848 / 51013200` steps、`141537784 / 725922600` exposures。`2026-08-24T05:22:54Z` 复核固定 worktree clean、六类下游/终局产物缺席，四个唯一 writer 和门顺序保持合规。

`2026-08-24T05:32:10Z`，watcher 接纳 NYUv2 `random_feature_local / seed 7319` epoch 48 与 VOC2012 `z0 / seed 7319` epoch 21。NYUv2 强审计 SHA-256 为 `c26953fe742d5f0781ff5ec194d134f90757aacb469de9df2080148cbba8e028`，累计 `8592` steps、`34320` exposures，validation abs-rel/delta1 为 `0.42104145511984825 / 0.35950455665588377`。VOC 强审计 SHA-256 为 `6e3710eddb071b37ddf0f857b0c0ece056a44d951c96fa1ce89e64529dc93a74`，累计 `6930` steps、`27678` exposures，validation mean-IoU/pixel accuracy 为 `0.04425127059221268 / 0.6695227026939392`。两份均为 `returncode=0`、`status=passed`、`problems=[]` 且 37 项 checks 全为 true；这些仍是单 cell 中间证据，低指标原样保留，不形成方法有效性裁决。

watcher state SHA-256 更新为 `b50746f11173dec322b236f16d06a81ced91b780afe5c5b7378d1dc3499fb6f8`，累计 `197` 份中间审计、`18` 份 terminal 审计，active subtotal 为 `97397` steps、`4811143` exposures；全局原子账本为 `1282017 / 51013200` steps、`141542453 / 725922600` exposures。`2026-08-24T05:35:19Z` 复核两棵固定 worktree clean、四个唯一 writer 与 watcher 存活、六类下游/终局产物缺席；`execution_complete=false`、`method_effectiveness_conclusion=null`、`changes_scientific_verdict=false`，goal 继续 active。

`2026-08-24T05:42:32Z`，watcher 同轮接纳 ImageNet-100 `velocity / seed 7319` epoch 41、NYUv2 `random_feature_local / seed 7319` epoch 49 与 VOC2012 `z0 / seed 7319` epoch 24。三份强审计均为 `returncode=0`、`status=passed`、`problems=[]` 且全部 37 项 checks 为 true。ImageNet 强审计 SHA-256 为 `f9d373491b9687b4c8393fa40869d2e2852952ba8792ec28d983ae8226ac061d`，累计 `37310` steps、`4774655` exposures，validation top-1/top-5 为 `0.04142194744976816 / 0.1750386398763524`。NYUv2 强审计 SHA-256 为 `a1ed8f71f7b73ffd06c46748eb30fc3432004a73eb7048bd1c31c66a49096360`，累计 `8771` steps、`35035` exposures，validation abs-rel/delta1 为 `0.40853045880794525 / 0.3127132415771484`。VOC 强审计 SHA-256 为 `417dce73089f44482a32dc7b43fed82973f462b5879fe7463ec8756b0888bc58`，累计 `7920` steps、`31632` exposures，validation mean-IoU/pixel accuracy 为 `0.04229811951518059 / 0.6701062321662903`。这些中间指标继续原样保留，不推导总体科学结论。

watcher state SHA-256 更新为 `6e925b3e7771509377d27d3c539817a18d5f2b3eee7b4af5f15626fff06e5c90`，累计 `200` 份中间审计、`18` 份 terminal 审计，active subtotal 为 `99476` steps、`4932267` exposures；全局原子账本为 `1284096 / 51013200` steps、`141663577 / 725922600` exposures。`2026-08-24T05:44:13Z` 复核两棵固定 worktree clean、四个唯一 writer 与 watcher 存活、六类下游/终局产物仍缺席；ADE20K 没有新增原子点，继续以 retry 2 epoch 5 计账并完整保留既有失败与极低 mIoU。goal 继续 active。

`2026-08-24T05:53:06Z` 至 `06:03:31Z`，watcher 接纳 NYUv2 `random_feature_local / seed 7319` epoch 51、epoch 52，以及 VOC2012 `z0 / seed 7319` epoch 27。三份强审计均为 `returncode=0`、`status=passed`、`problems=[]` 且 37 项 checks 全为 true。NYUv2 epoch 52 强审计 SHA-256 为 `d38f29a50228b2d1d6c7122d2344c608eff863193326bf3558f3caa70da829e2`，累计 `9308` steps、`37180` exposures，validation abs-rel/delta1 为 `0.41734748929738996 / 0.3316162109375`。VOC epoch 27 强审计 SHA-256 为 `8d439ca9d736b88ef26c75fe6e4dd484611e1dcf71ef706e65368d7849341305`，累计 `8910` steps、`35586` exposures，validation mean-IoU/pixel accuracy 为 `0.0651971697807312 / 0.6721295118331909`。epoch 51 的 intermediate artifact 同样保留；账本采用 watcher 当前唯一 active 点 epoch 52/27，不累计 successive epochs。

同一轮 watcher 发现 VOC epoch 29 审计期间 report 前进到 epoch 30，产生 `returncode=1` 的非权威 race artifact `artifacts/reports/voc2012_z0_seed7319_epoch29_strong_audit_non_authoritative_failed_race.json`，SHA-256 为 `6f74424f03f69af9a9323576f85d066a04799a2351ec909491936100832fa5bb`。分类为 `report_advanced_during_audit`，`failed_artifact_preserved=true`、`unresolved_scientific_failure=false`；该失败不计入科学账本，且不删除或正向解释。

watcher state SHA-256 更新为 `08fb06357b175183b9c14659145549d4350e54e8c0dbcf6d6b824c26db4b92ef`，累计 `203` 份中间审计、`18` 份 terminal 审计、`2` 份 transient audit race，active subtotal 为 `101003` steps、`4938366` exposures；全局原子账本为 `1285623 / 51013200` steps、`141669676 / 725922600` exposures。`2026-08-24T06:11:24Z` 复核固定 worktree clean、四个唯一 writer 与 watcher 存活、六类下游/终局产物缺席；`execution_complete=false`、`method_effectiveness_conclusion=null`、`changes_scientific_verdict=false`，goal 继续 active。

`2026-08-24T06:13:53Z`，watcher 接纳 NYUv2 `random_feature_local / seed 7319` epoch 53 与 VOC2012 `z0 / seed 7319` epoch 32。两份强审计均为 `returncode=0`、`status=passed`、`problems=[]` 且 37 项 checks 全为 true。NYUv2 强审计 SHA-256 为 `f731025ffeeaab96f83e03d7b6a1c37927b83e3afccf06b3b85631b949b29feb`，累计 `9487` steps、`37895` exposures，validation abs-rel/delta1 为 `0.4332916304469109 / 0.3578005790710449`。VOC 强审计 SHA-256 为 `a09308fb6e620171579d93c767d6d37d4addde50ffea2ab72cc1c04273a3d069`，累计 `10560` steps、`42176` exposures，validation mean-IoU/pixel accuracy 为 `0.059853389859199524 / 0.6712396144866943`。中间指标原样保留，不形成总体科学裁决。

watcher state SHA-256 更新为 `13964ee5906795f3df761cca0f63f6ba6a318db42279b2b786e33c34e6a577da`，累计 `205` 份中间审计、`18` 份 terminal 审计、`2` 份 transient audit race，active subtotal 为 `102832` steps、`4945671` exposures；全局原子账本为 `1287452 / 51013200` steps、`141676981 / 725922600` exposures。`2026-08-24T06:15:00Z` 复核两个固定 worktree clean、四个唯一 writer 与 watcher 存活、六类下游/终局产物缺席；goal 继续 active。

`2026-08-24T06:24:16Z`，watcher 接纳 NYUv2 `random_feature_local / seed 7319` epoch 55 与 VOC2012 `z0 / seed 7319` epoch 35。两份强审计均为 `returncode=0`、`status=passed`、`problems=[]` 且 37 项 checks 全为 true。NYUv2 强审计 SHA-256 为 `8a704717b514dd992d5f87939d76935cb1a5773a79e62a929507842dd64c4003`，累计 `9845` steps、`39325` exposures，validation abs-rel/delta1 为 `0.41795268207788466 / 0.3183135509490967`。VOC 强审计 SHA-256 为 `b5f44a2ac1ac2f33194057011d2df64f9d968b31d749a46b3bd1dcfe8077ad93`，累计 `11550` steps、`46130` exposures，validation mean-IoU/pixel accuracy 为 `0.07094760984182358 / 0.6735751628875732`。中间指标原样保留，不形成总体科学裁决。

watcher state SHA-256 更新为 `4a817c733ba762bfb48d518bb0df2dc16c2ff588cb18ebfb08deaa1b9d5a3c75`，累计 `207` 份中间审计、`18` 份 terminal 审计、`2` 份 transient audit race，active subtotal 为 `104180` steps、`4951055` exposures；全局原子账本为 `1288800 / 51013200` steps、`141682365 / 725922600` exposures。`2026-08-24T06:26:00Z` 复核两个固定 worktree clean、四个唯一 writer 与 watcher 存活、六类下游/终局产物缺席；goal 继续 active。

`2026-08-24T06:34:40Z`，watcher 同轮接纳 ADE20K retry 2 `random_feature_local / seed 4121` epoch 6、NYUv2 `random_feature_local / seed 7319` epoch 56 与 VOC2012 `z0 / seed 7319` epoch 38。三份强审计均为 `returncode=0`、`status=passed`、`problems=[]` 且 37 项 checks 全为 true。ADE 强审计 SHA-256 为 `91aae3325cefade072cad7c40041240cb9187eb80b9d75a261a089cc8afb10f9`，累计 `54570` steps、`109134` exposures，validation mean-IoU/pixel accuracy 为 `0.0036331734154373407 / 0.19416868686676025`；该极低 mIoU 原样保留，不作正向解释。NYUv2 强审计 SHA-256 为 `b0622bdbdca56a73cf7b6411b6e42b463a3d821cf43c17d219f0e33720113793`，累计 `10024` steps、`40040` exposures，validation abs-rel/delta1 为 `0.4154364041984081 / 0.32084050178527834`。VOC 强审计 SHA-256 为 `70545fe571b78b31bfd4d6a9d385df8956545b78d4e668bf3a30d640ffdbafe7`，累计 `12540` steps、`50084` exposures，validation mean-IoU/pixel accuracy 为 `0.0717378705739975 / 0.669947624206543`。中间指标原样保留，不形成总体科学裁决。

watcher state SHA-256 更新为 `af2cbf700802d426c6e8cc6d19ec5513c86cb8d0ea63f102108a3573775fc3ba`，累计 `210` 份中间审计、`18` 份 terminal 审计、`2` 份 transient audit race，active subtotal 为 `114444` steps、`4973913` exposures；全局原子账本为 `1299064 / 51013200` steps、`141705223 / 725922600` exposures。`2026-08-24T06:36:00Z` 复核两个固定 worktree clean、四个唯一 writer 与 watcher 存活、六类下游/终局产物缺席；goal 继续 active。

`2026-08-24T06:45:06Z`，watcher 接纳 NYUv2 `random_feature_local / seed 7319` epoch 57 与 VOC2012 `z0 / seed 7319` epoch 41。两份强审计均为 `returncode=0`、`status=passed`、`problems=[]` 且 37 项 checks 全为 true。NYUv2 强审计 SHA-256 为 `1dfe14f02628958017cff60e1e008c5d5d9c06a27601a17295fccccae59a1ae8`，累计 `10203` steps、`40755` exposures，validation abs-rel/delta1 为 `0.43107205256819725 / 0.3612088203430176`。VOC 强审计 SHA-256 为 `03211f4e78ac2fe2d971a6e189b70321e867e344e3b11bc75d1311007b19aa2f`，累计 `13530` steps、`54038` exposures，validation mean-IoU/pixel accuracy 为 `0.0805656909942627 / 0.6773018836975098`。中间指标原样保留，不形成总体科学裁决。

watcher state SHA-256 更新为 `c712eb4c190a685caa584b255dc81d18cc5e291378a38746d4fb8f6174a3bc42`，累计 `212` 份中间审计、`18` 份 terminal 审计、`2` 份 transient audit race，active subtotal 为 `115613` steps、`4978582` exposures；全局原子账本为 `1300233 / 51013200` steps、`141709892 / 725922600` exposures。`2026-08-24T06:47:00Z` 复核两个固定 worktree clean、四个唯一 writer 与 watcher 存活、六类下游/终局产物缺席；goal 继续 active。

`2026-08-24T06:55:26Z`，watcher 接纳 NYUv2 `random_feature_local / seed 7319` epoch 58 与 VOC2012 `z0 / seed 7319` epoch 43。两份强审计均为 `returncode=0`、`status=passed`、`problems=[]` 且 37 项 checks 全为 true。NYUv2 强审计 SHA-256 为 `c3afdd74037d7dfcc7223a2eae0347b4e6c35d55ba4e4c64700f81d58a4b5378`，累计 `10382` steps、`41470` exposures，validation abs-rel/delta1 为 `0.4420463897287846 / 0.36069860458374026`。VOC 强审计 SHA-256 为 `5cb0736c8ae941071a30eead4e1f0eddd21f656f7fad1def42e2cb68bbeb1f07`，累计 `14190` steps、`56674` exposures，validation mean-IoU/pixel accuracy 为 `0.09137611836194992 / 0.675451934337616`。中间指标原样保留，不形成总体科学裁决。

watcher state SHA-256 更新为 `181c4ea2dddcf5c5daa35388eb2f03acc9e0ec545e77f147d5b0314bd36bbf27`，累计 `214` 份中间审计、`18` 份 terminal 审计、`2` 份 transient audit race，active subtotal 为 `116452` steps、`4981933` exposures；全局原子账本为 `1301072 / 51013200` steps、`141713243 / 725922600` exposures。`2026-08-24T06:57:00Z` 复核两个固定 worktree clean、四个唯一 writer 与 watcher 存活、六类下游/终局产物缺席；goal 继续 active。

`2026-08-24T07:05:38Z`，watcher 接纳 NYUv2 `random_feature_local / seed 7319` epoch 60 与 VOC2012 `z0 / seed 7319` epoch 46。两份强审计均为 `returncode=0`、`status=passed`、`problems=[]` 且 37 项 checks 全为 true。NYUv2 强审计 SHA-256 为 `3a90d2362faeb7e431d81f2be6658a29c39c89f52000ba014d751d25f63be5ce`，累计 `10740` steps、`42900` exposures，validation abs-rel/delta1 为 `0.4361932411789894 / 0.35814990997314455`。VOC 强审计 SHA-256 为 `bc50c9843af98cc642c46522a542a9d1c5e09f92bb9a07fb5077d9ec39d86ee5`，累计 `15180` steps、`60628` exposures，validation mean-IoU/pixel accuracy 为 `0.08154608309268951 / 0.6733913421630859`。中间指标原样保留，不形成总体科学裁决。

watcher state SHA-256 更新为 `fae240cc9c386a703c6b60a3779a565678fe3dc51e1f2834c4cc6db336d61394`，累计 `216` 份中间审计、`18` 份 terminal 审计、`2` 份 transient audit race，active subtotal 为 `117800` steps、`4987317` exposures；全局原子账本为 `1302420 / 51013200` steps、`141718627 / 725922600` exposures。`2026-08-24T07:07:00Z` 复核两个固定 worktree clean、四个唯一 writer 与 watcher 存活、六类下游/终局产物缺席；goal 继续 active。

`2026-08-24T07:15:51Z`，watcher 接纳 NYUv2 `random_feature_local / seed 7319` epoch 61 与 VOC2012 `z0 / seed 7319` epoch 49。两份强审计均为 `returncode=0`、`status=passed`、`problems=[]` 且 37 项 checks 全为 true。NYUv2 强审计 SHA-256 为 `eddf1cec97e85e7a7b830b2bc8445864b402a4a28fd1db369bb39e98ac1c0cf9`，累计 `10919` steps、`43615` exposures，validation abs-rel/delta1 为 `0.4320360839366913 / 0.3517143249511719`。VOC 强审计 SHA-256 为 `e4d8f13f15dce890799e045c357a742a822ab751f668fd9445d5b07c9bb19dc4`，累计 `16170` steps、`64582` exposures，validation mean-IoU/pixel accuracy 为 `0.09471210092306137 / 0.6766591668128967`。中间指标原样保留，不形成总体科学裁决。

watcher state SHA-256 更新为 `d01664405e2d86f9fe696f9fca88e6b6ade8fd9b36eb3159b9e713aef38a1616`，累计 `218` 份中间审计、`18` 份 terminal 审计、`2` 份 transient audit race，active subtotal 为 `118969` steps、`4991986` exposures；全局原子账本为 `1303589 / 51013200` steps、`141723296 / 725922600` exposures。`2026-08-24T07:17:00Z` 复核两个固定 worktree clean、四个唯一 writer 与 watcher 存活、六类下游/终局产物缺席；goal 继续 active。

`2026-08-24T14:00:22Z`，经完整 Ruff、`4 passed in 3.81s` 定向测试、
`182 passed in 2854.28s` 全测试及两步 CPU toy smoke
`cache_reload_equal=true` 后，仅原子审计 watcher 由 PID `2393317` 滚动切换到内容寻址
版本 PID `1542686`，SHA-256 为
`00a04df6d6eae0ddcd13a4a57f74f8222842d1954b4a3400a8a9819fa44253c7`。
recovery、四个正式 worker、A800 monitors/supervisors 均未重启，两棵固定 revision
worktree 继续 clean。

此前 VOC2012 `z0 / seed 104729 / epoch 55` 的失败审计原件 SHA-256
`09bce9d05511805f18ae959db99014795cb31984553bef9709c035edc0f37d76`
保持不变。该点稳定 report 为 epoch 55，而稳定 checkpoint 已到 epoch 56；相同身份的
epoch 58 通过强审计 SHA-256
`7d7f2a06dde816ea438b92e8ea15411335b5f49c005e3d60ce788feca316c210`
覆盖该 checkpoint 边界，故严格分类为
`checkpoint_ahead_of_stable_report_confirmed_by_later_strong_audit`，移入已保留的
transient audit races，不作为 unresolved scientific failure，也不删除或正向解释失败。

切换后 watcher state SHA-256 为
`44b6d6336f78f9624e2bf215da0a5d3446a68997516cc21e3fe8b916354a35bc`，累计
299 份中间审计、20 份 terminal 审计、5 份完整保留的 audit races 和 0 个 unresolved
failures。不可变账本
`artifacts/reports/four_task_atomic_ledger_reconstruction_20260824T140028Z.json`
SHA-256 为 `abc1d3e53e4fc354850c810a773d9948707780667c2244570a616f5b17827484`，
活动边界为 ImageNet-100 epoch 45、ADE20K epoch 8、NYUv2 epoch 45、VOC2012
epoch 63；`status=passed`、`problems=[]`、`execution_complete=false`、
`changes_scientific_verdict=false`。六类下游/终局产物继续缺席，goal 保持 active。

`2026-08-24T14:10:28Z`，内容寻址 watcher 的下一轮原子写回继续
`failures=[]`，接受 NYUv2 `random_feature_local / seed 104729` epoch 48 与
VOC2012 `z0 / seed 104729` epoch 66；ImageNet-100 与 ADE20K 的接受边界分别保持
epoch 45/8。watcher state SHA-256 为
`5f611daaf0e7ba7941eb5ec454d03b9a17ecabffdc75527ddde496494661b084`，累计
301 份中间审计、20 份 terminal 审计、5 份完整保留的 audit races 和 0 个 unresolved
failures。

不可变账本
`artifacts/reports/four_task_atomic_ledger_reconstruction_20260824T141035Z.json`
SHA-256 为 `957d9fa690559f7f562572d2bc1bd36ca6da2366c084d1d57157854a25deb9f9`；
terminal 仍为 `20/240`，active subtotal 为 `144082` steps、`5507295` exposures，
全局原子账本为 `1369422 / 51013200` steps、
`142401245 / 725922600` exposures。跨主机 pretrigger 审计再次确认四个唯一 writer、
两个固定 worktree clean、六类下游/终局产物缺席，门决定仍为
`wait_for_all_240_main_matrix_cells`；`status=passed`、`execution_complete=false`、
`method_effectiveness_conclusion=null`、`changes_scientific_verdict=false`，goal 继续 active。

`2026-08-24T14:20:35Z`，watcher 原子接受 NYUv2
`random_feature_local / seed 104729` epoch 52 与 VOC2012 `z0 / seed 104729`
epoch 69，ImageNet-100/ADE20K 接受边界保持 epoch 45/8。state SHA-256 为
`adc44be632edab7291f01bbd0acdff3f8d2dbc6a882609b75a64a8e799f29086`，累计
303 份中间审计、20 份 terminal 审计、5 份完整保留的 races 与 0 个 unresolved
failures。

不可变账本 SHA-256
`665351f155e107f44fd48df945eb493cc38a2c961354732300271d67e9557278`，
active subtotal 为 `145788` steps、`5514109` exposures，全局为
`1371128 / 51013200` steps、`142408059 / 725922600` exposures。
`2026-08-24T14:23:01Z` 增量 pretrigger 复核两棵固定 worktree clean、所有注册 PID
存活、epoch-55 失败原件哈希不变、六类下游/终局产物缺席。terminal 仍为 `20/240`，
门决定保持 `wait_for_all_240_main_matrix_cells`；科学状态不变，goal 继续 active。

`2026-08-24T14:40:50Z`，watcher 接受 ADE20K `random_feature_local / seed 4121`
epoch 9、NYUv2 `random_feature_local / seed 104729`
epoch 59 与 VOC2012 `z0 / seed 104729` epoch 75，ImageNet-100 保持 epoch 45。
state SHA-256 为 `6222242903797aae8ff57b07b6ee83d32adba9aa21014abf645c1c9dd80a602f`，
累计 308 份中间审计、20 份 terminal 审计、5 份完整保留的 races、0 个 unresolved
failures。

不可变账本 SHA-256 为 `c26af0c58e02cf832cc41c2969dc18b48e1a63e3f94cc951c9dd6df5b90d91ef`，
active subtotal 为 `158116` steps、`5545211` exposures，全局为
`1383456 / 51013200` steps、`142439161 / 725922600` exposures。
pretrigger 继续确认六类下游/终局产物缺席，goal 保持 active。

`2026-08-24T14:30:42Z`，watcher 接受 NYUv2 `random_feature_local / seed 104729`
epoch 55 与 VOC2012 `z0 / seed 104729` epoch 72，ImageNet-100/ADE20K 保持
epoch 45/8。state SHA-256 为
`035a8a282fac32125edee4272b4e36f1e78b54649abbdbc247363e71ddc9c8a8`，
累计 305 份中间审计、20 份 terminal 审计、5 份保留 races、0 个 unresolved
failures。

账本 SHA-256 为 `a53ffa28ffade6a8729d6b00bc648728ff2eff2148f77b87a45d35213c03f61f`，
active subtotal 为 `147315` steps、`5520208` exposures，全局为
`1372655 / 51013200` steps、`142414158 / 725922600` exposures。
`14:32:44Z` 增量 pretrigger 复核继续通过，terminal 未增加，所有后续门保持关闭，
`execution_complete=false`、`changes_scientific_verdict=false`，goal 继续 active。

2026-08-24T14:51:00Z cross-host reconstruction remains 20/240 terminal
cells with 310 accepted intermediate audits, five classified transient races,
and zero unresolved failures. Accepted active boundaries are ImageNet-100 45,
ADE20K 9, NYUv2 63, and VOC2012 78. Ledger SHA-256:
eef69d202d730315fe3cc768dee116ccdf5cc60179833f9214b4a762198959b5.
The incremental pretrigger audit found both fixed worktrees clean, all
registered workers and supervisors alive, and all downstream/final artifacts
absent; status=passed, execution_complete=false, and
changes_scientific_verdict=false remain unchanged.

2026-08-24T17:13:11.804717Z cross-host update: the independent ledger accepted
338 intermediate audits and reached 23/240 terminal cells (ImageNet-100 13,
NYUv2 4, VOC2012 6, ADE20K 0). Active boundaries are ADE20K epoch 10,
ImageNet-100 epoch 47, and VOC2012 zt/seed-4121 epoch 69. Six transient audit
races and zero unresolved failures remain. Ledger SHA-256:
56e7531000f605c6322a89b8c1390239415054939ba22443891c0388dba1fd88.
The corrected cross-host pretrigger audit at 17:22:45Z passed direct H200 and
A800 identity checks and remains closed at
`wait_for_all_240_main_matrix_cells`; execution_complete=false and
changes_scientific_verdict=false remain unchanged. The earlier SSH-alias
false-negative recheck is preserved as operational evidence.

2026-08-24T16:52:50.031719Z cross-host update: 335 accepted intermediate
audits and 22/240 terminal cells. Active accepted boundaries are
ImageNet-100/ADE20K/NYUv2/VOC2012 = 47/10/63/50. The VOC2012 epoch-56 audit
failed its consistency checks (checkpoint/report epoch, step, exposure,
optimizer and scheduler mismatch) and is retained as the sixth classified
transient race, SHA-256
8f2d1063ba6c471b4938c4eb88004257e8c3d9378e728d6d73f04f3fbbe25a23; unresolved
failures remain zero. Ledger SHA-256:
23c2312848fe79d4a7508129d42877372bc97ee967f7632e1887e8e06d236a91.
The matching pretrigger audit passed; downstream remains closed and
changes_scientific_verdict=false.

2026-08-24T17:03:03.384715Z cross-host update: 337 accepted intermediate
audits and 22/240 terminal cells. Active boundaries are
ImageNet-100/ADE20K/NYUv2/VOC2012 = 47/10/73/63, with six classified transient
races and zero unresolved failures. The VOC epoch-56 failed consistency audit
remains preserved. Ledger SHA-256:
841a1def635a8f1df6ba33bdabb55869293f02436addac653d2cdc1074a154ed.
The matching pretrigger absence audit passed; downstream remains closed,
execution_complete=false, and changes_scientific_verdict=false remain unchanged.

2026-08-24T17:33:35.427708Z update: 341 accepted intermediate audits and
23/240 terminal cells. Active boundaries are ADE20K epoch 10, ImageNet-100
epoch 47, NYUv2 z0/seed-7319 epoch 19, and VOC2012 zt/seed-4121 epoch 75.
Six transient races remain classified and unresolved failures remain zero.
Ledger SHA-256:
6b3b3bc38c1004bff5c3e6b78d97aa4e79fc5d9f7aa4dfd7e3cc394db4c38a70.
The matching pretrigger audit passed; downstream remains closed and
execution_complete=false and changes_scientific_verdict=false remain unchanged.

2026-08-24T17:43:39.667722Z update: 343 accepted intermediate audits and
24/240 terminal cells. NYUv2 is at active epoch 27, VOC2012 has started
zt/seed-7319 at epoch 5, ADE20K remains epoch 10, and ImageNet-100 epoch 47.
Six transient races remain classified and unresolved failures remain zero.
Ledger SHA-256:
14678cf57ad60c9f629ac4f2491df34093d0a5866607d7f65e6377d5180cdc99.
The matching pretrigger audit passed; downstream remains closed and
execution_complete=false and changes_scientific_verdict=false remain unchanged.

2026-08-24T17:42:15Z read-only reconstruction: report/checkpoint activity was
observed on the A800 lanes, but the accepted ledger remained 341 audits and
23/240 terminal cells with six classified transient races and zero unresolved
failures. The duplicate ledger SHA-256 remains
6b3b3bc38c1004bff5c3e6b78d97aa4e79fc5d9f7aa4dfd7e3cc394db4c38a70; the
matching pretrigger gate passed and downstream remains closed.

2026-08-24T16:42:37.370706Z cross-host update: 333 accepted intermediate
audits and 22/240 terminal cells. Active boundaries are
ImageNet-100/ADE20K/NYUv2/VOC2012 = 47/9/54/50, with five classified transient
races and zero unresolved failures. Ledger SHA-256:
98822cc6a5d7f8b9ccffed0cb11fede3a9a0c21c701327a7f98c53a54f1050ca.
The matching pretrigger absence audit passed; downstream remains closed,
execution_complete=false, and changes_scientific_verdict=false remain unchanged.

2026-08-24T16:22:19.740711Z cross-host update: 328 accepted intermediate
audits and 22/240 terminal cells. Active boundaries are
ImageNet-100/ADE20K/NYUv2/VOC2012 = 46/9/34/37, with five classified transient
races and zero unresolved failures. Ledger SHA-256:
53526279519e3ab0dff0dbc920ad0e995125ba86da9a42d565455c5593a4dc72.
The matching pretrigger absence audit passed; downstream remains closed,
execution_complete=false, and changes_scientific_verdict=false remain unchanged.

2026-08-24T16:12:10.983707Z cross-host update: the watcher accepted 326
intermediate audits and the independent ledger remains at 22/240 terminal
cells. Active boundaries are ImageNet-100/ADE20K/NYUv2/VOC2012 = 46/9/25/31,
with five classified transient races and zero unresolved failures. Ledger
SHA-256: a04f8c048d308cb0cd8e918e24c96a6338155a6a2bcc77bf29f5bed3e88b2183.
The matching pretrigger absence audit passed; all downstream artifacts remain
absent, execution_complete=false, and changes_scientific_verdict=false.

2026-08-24T15:01:07Z watcher/reconstructor update: NYUv2 advanced to accepted
epoch 65. The ledger has 311 accepted intermediate audits, 20/240 terminal
cells, five classified transient races, and zero unresolved failures; active
boundaries are ImageNet-100/ADE20K/NYUv2/VOC2012 = 45/9/65/78. Ledger SHA-256:
44e8ed2cf2c705907ffc00b3f30f96eeb488b37590541ea8e35f35a8cf41ed25.
Pretrigger absence recheck passed with all downstream artifacts absent;
execution_complete=false and changes_scientific_verdict=false remain unchanged.

2026-08-24T15:11:11Z update: 314 accepted intermediate audits and 21/240
terminal cells. Active boundaries are ImageNet-100 46, ADE20K 9, NYUv2 68,
and VOC2012 zt/seed-4121 epoch 2; five classified races and zero unresolved
failures remain. Ledger SHA-256:
0fda8f7fa3dc15e2d2d79478f19a9786496ad04eb7b5999b82ae20a7c6623426.
The pretrigger absence audit passed; execution_complete=false and
changes_scientific_verdict=false remain unchanged.

2026-08-24T15:21:26Z update: 316 accepted intermediate audits and 21/240
terminal cells; active boundaries ImageNet-100/ADE20K/NYUv2/VOC2012 = 46/9/72/6.
Five transient races remain classified and unresolved failures remain zero.
Ledger SHA-256: 31fe7b51081fb79a7b8d27d2e91a778901d4911bb49c57272c144f3cdd24e4ee.
Pretrigger absence recheck passed; execution_complete=false and
changes_scientific_verdict=false remain unchanged.

2026-08-24T16:02:03.818706Z cross-host update: 324 accepted intermediate
audits and 22/240 terminal cells, with five classified transient races and
zero unresolved failures. NYUv2 reached its third terminal cell. Active
boundaries are ImageNet-100/ADE20K/NYUv2/VOC2012 = 46/9/15/24 (the latter two
include their representation and seed identifiers in the ledger). Ledger
SHA-256: 93fc7382da587996031644227481dc73cbbe3b705b2f8cc9280ed1cab23451be.
The matching cross-host pretrigger absence audit passed, downstream remains
closed, and execution_complete=false and changes_scientific_verdict=false
remain unchanged.

2026-08-24T15:31:34Z update: 318 accepted intermediate audits and 21/240
terminal cells; active boundaries ImageNet-100/ADE20K/NYUv2/VOC2012 = 46/9/75/9.
Five transient races remain classified and unresolved failures remain zero.
Ledger SHA-256: 8cdc397be10aa262dd40065aef9fa9f1ddc791fd7ee8bac9da2c2a2a473d69a4.
Pretrigger absence recheck passed; execution_complete=false and
changes_scientific_verdict=false remain unchanged.

2026-08-24T15:41:41Z update: 320 accepted intermediate audits and 21/240
terminal cells; active boundaries ImageNet-100/ADE20K/NYUv2/VOC2012 = 46/9/79/12.
Five transient races remain classified and unresolved failures remain zero.
Ledger SHA-256: 5fe709c9341d33cf7a8ca6a068db5f61677b6e2d0484bd52cf0e9adc982075e5.
Pretrigger absence recheck passed; execution_complete=false and
changes_scientific_verdict=false remain unchanged.

2026-08-24T16:32:29.181718Z cross-host update: 330 accepted intermediate
audits and 22/240 terminal cells. Active boundaries are
ImageNet-100/ADE20K/NYUv2/VOC2012 = 46/9/44/43, with five classified transient
races and zero unresolved failures. Ledger SHA-256:
9b350ce8454d35bc4e75999d40f82a08d5d46bf33ee293f55648ecee36d09ddd.
The matching pretrigger absence audit passed; downstream remains closed,
execution_complete=false, and changes_scientific_verdict=false remain unchanged.

2026-08-24T17:23:27.488721Z update: 340 accepted intermediate audits and
23/240 terminal cells. Active boundaries are ADE20K epoch 10, ImageNet-100
epoch 47, NYUv2 z0/seed-7319 epoch 11, and VOC2012 zt/seed-4121 epoch 75;
six transient races remain classified and unresolved failures remain zero.
Ledger SHA-256:
d28a2042e67bff10eac84468bfe98cd57f8ba6d2e623f6cf56340aab2848536b.
The matching cross-host pretrigger audit passed and downstream remains closed;
execution_complete=false and changes_scientific_verdict=false remain unchanged.

2026-08-24T17:50:23Z read-only reconstruction remained at 343 accepted audits
and 24/240 terminal cells. The duplicate ledger SHA-256 is
14678cf57ad60c9f629ac4f2491df34093d0a5866607d7f65e6377d5180cdc99; report
writes on the A800 lanes remain activity evidence only, and the matching gate
passed without opening downstream.

2026-08-24T17:53:54.424712Z update: 345 accepted intermediate audits and
24/240 terminal cells. Active boundaries are NYUv2 z0/seed-7319 epoch 36,
VOC2012 zt/seed-7319 epoch 11, ADE20K epoch 10, and ImageNet-100 epoch 47.
Six transient races remain classified and unresolved failures remain zero.
Ledger SHA-256:
26aca22cd338d82d253152b5b1bf85f46c8bd735d3afa026b28c290be52aa5cb.
The matching pretrigger audit passed; downstream remains closed and no
scientific verdict changed.

2026-08-24T18:04:03.118707Z update: 347 accepted intermediate audits and
24/240 terminal cells. Active boundaries are NYUv2 epoch 46, VOC2012 epoch 17,
ADE20K epoch 10, and ImageNet-100 epoch 47. Six transient races remain
classified and unresolved failures remain zero. Ledger SHA-256:
2e6e9019811bf30cc6e0643ffc2fddea58ead26a5bf61f0cafcf2bc84c7afba9.
The matching pretrigger audit passed; downstream remains closed and no
scientific verdict changed.

At 2026-08-24T18:18:27Z, an independent reconstruction after the cross-host
workers' latest atomic report writes remained unchanged at 350 accepted
intermediate audits and 24/240 terminal cells. Active boundaries are ADE20K
epoch 10, ImageNet-100 velocity/seed-7319 epoch 48, NYUv2 z0/seed-7319 epoch
55, and VOC2012 zt/seed-7319 epoch 23; six transient races remain classified
and unresolved failures remain zero. The retained ledger
`artifacts/reports/four_task_atomic_ledger_reconstruction_20260825T021827Z.json`
has SHA-256
`4cadfffa0e395bc8389019edbf3fd8a75d59a6e502619753cdc0907877b4aec1`.
No new downstream gate was needed because the ledger's scientific content did
not change; downstream remains closed, `execution_complete=false`, and
`changes_scientific_verdict=false`.

At 2026-08-24T19:45:34Z, the watcher accepted three further cross-host
intermediate audits, bringing the accepted total to 370 while terminal cells
remain 25/240. Active boundaries are ADE20K random_feature_local/seed-4121
epoch 11, ImageNet-100 velocity/seed-7319 epoch 49, NYUv2
z0/seed-104729 epoch 57, and VOC2012 zt/seed-7319 epoch 77. Seven
transient races remain classified and unresolved failures remain zero. Ledger
`artifacts/reports/four_task_atomic_ledger_reconstruction_20260824T201000Z.json`
has SHA-256
`0f130327da2ec01e9df86122df189f6d4dfe245b0bae68cf76c861131ca4e602`.
The matching pretrigger gate
`artifacts/reports/downstream_gate_cross_host_pretrigger_audit_20260824T201000Z.json`
has SHA-256
`e28b03746590366deccc1e14ec3b16697cbc8852525dc13596946be59b1b6e59`;
downstream remains closed, `execution_complete=false`, and
`changes_scientific_verdict=false`.

At 2026-08-24T19:35:27Z, the cross-host atomic watcher accepted two new
intermediate audits from the A800 lanes, bringing the accepted total to 367
while terminal cells remain 25/240. Active boundaries are ADE20K
random_feature_local/seed-4121 epoch 11, ImageNet-100 velocity/seed-7319
epoch 48, NYUv2 z0/seed-104729 epoch 48, and VOC2012 zt/seed-7319 epoch 71.
Seven transient races remain classified and unresolved failures remain zero.
Ledger
`artifacts/reports/four_task_atomic_ledger_reconstruction_20260824T193600Z.json`
has SHA-256
`577d83961a11ec48005cbeba37466ff54cc59b7fe0f2092638e824ed4c43d440`.
The matching pretrigger gate
`artifacts/reports/downstream_gate_cross_host_pretrigger_audit_20260824T193600Z.json`
has SHA-256
`77f939dab4ce13247125e1af7e84f14b492e13d9659193e06ddd6a05d4b8ac2c`;
downstream remains closed, `execution_complete=false`, and
`changes_scientific_verdict=false`.

At 2026-08-24T19:26:26Z, the reconstruction accepted 365 intermediate
audits while terminal cells remained 25/240. Active boundaries advanced to
ADE20K epoch 11, ImageNet-100 velocity/seed-7319 epoch 48, NYUv2
z0/seed-104729 epoch 39, and VOC2012 zt/seed-7319 epoch 65. Seven transient
races remain classified and unresolved failures remain zero. Ledger
`artifacts/reports/four_task_atomic_ledger_reconstruction_20260824T192600Z.json`
has SHA-256
`720105ac401d9456b4dec45ae624cf3f99b1a3584da095e11becd2abadc4cff5`.
The matching pretrigger gate
`artifacts/reports/downstream_gate_cross_host_pretrigger_audit_20260824T192626Z.json`
has SHA-256
`f20b1214cb56788972e2fc61eb1caebe0973bf1e12ef714353b5fc7490942424`;
downstream remains closed, `execution_complete=false`, and
`changes_scientific_verdict=false`.

At 2026-08-24T19:18:47Z, the reconstruction accepted 363 intermediate
audits while terminal cells remained 25/240. Active boundaries advanced to
ADE20K epoch 11, ImageNet-100 velocity/seed-7319 epoch 48, NYUv2
z0/seed-104729 epoch 30, and VOC2012 zt/seed-7319 epoch 59. Seven transient
races remain classified and unresolved failures remain zero. Ledger
`artifacts/reports/four_task_atomic_ledger_reconstruction_20260824T191800Z.json`
has SHA-256
`d1f101be89043a1949e873854206790531b3f472014fc486df9af3916ea6a36f`.
The matching pretrigger gate
`artifacts/reports/downstream_gate_cross_host_pretrigger_audit_20260824T191847Z.json`
has SHA-256
`384ef71bf69a50cf33753e4d1f3b64c1462ce6e9ea0063e59142d67fe9b33dd7`;
downstream remains closed, `execution_complete=false`, and
`changes_scientific_verdict=false`.

At 2026-08-24T19:07:06Z, the reconstruction accepted 361 intermediate
audits while terminal cells remained 25/240. Active boundaries advanced to
ADE20K epoch 11, ImageNet-100 velocity/seed-7319 epoch 48, NYUv2
z0/seed-104729 epoch 21, and VOC2012 zt/seed-7319 epoch 53. Seven transient
races remain classified and unresolved failures remain zero. Ledger
`artifacts/reports/four_task_atomic_ledger_reconstruction_20260824T190600Z.json`
has SHA-256
`e383a1e3fda95e3271f54c79c75a084078c7e4d05b37e4e81a420dbfcc1905de`.
The matching pretrigger gate
`artifacts/reports/downstream_gate_cross_host_pretrigger_audit_20260824T190706Z.json`
has SHA-256
`cbab6433d6defd50363bc956df810d9f1131c1ba31b581690373f60802049a1c`;
downstream remains closed, `execution_complete=false`, and
`changes_scientific_verdict=false`.

At 2026-08-24T18:56:09Z, the reconstruction accepted 359 intermediate
audits while terminal cells remained 25/240. Active boundaries advanced to
ADE20K epoch 11, ImageNet-100 velocity/seed-7319 epoch 48, NYUv2
z0/seed-104729 epoch 12, and VOC2012 zt/seed-7319 epoch 47. Seven transient
races remain classified and unresolved failures remain zero. Ledger
`artifacts/reports/four_task_atomic_ledger_reconstruction_20260824T185500Z.json`
has SHA-256
`5579c0420107f075ad7d9a9b02ab6000de3609f9c9776d4b8359cf8eda2418d5`.
The matching live-identity pretrigger gate
`artifacts/reports/downstream_gate_cross_host_pretrigger_audit_20260824T185609Z.json`
has SHA-256
`fbec18519751ec3ba7e2e71c0c5c53bbe11bbe5a69eb6a51317c180380e99ee4`;
downstream remains closed, `execution_complete=false`, and
`changes_scientific_verdict=false`.

At 2026-08-24T18:46:22Z, the reconstruction accepted 356 intermediate audits
and advanced the main matrix to 25/240 terminal cells. NYUv2 completed its
fifth terminal cell and moved to z0/seed-104729 epoch 2; other active
boundaries are ADE20K epoch 10, ImageNet-100 velocity/seed-7319 epoch 48,
and VOC2012 zt/seed-7319 epoch 41. Seven transient races remain classified
and unresolved failures remain zero. Ledger
`artifacts/reports/four_task_atomic_ledger_reconstruction_20260824T184500Z.json`
has SHA-256
`479d886f76c5c3641093f1d694662074e087087967c8d1c9f2f635300645d86d`.
The matching pretrigger gate
`artifacts/reports/downstream_gate_cross_host_pretrigger_audit_20260824T184622Z.json`
has SHA-256
`83ed10773f2dc0a723d31c09b85d3901e706f75476620b1bb884398bd8d2d8b7`;
downstream remains closed, `execution_complete=false`, and
`changes_scientific_verdict=false`.

At 2026-08-24T18:30:48Z, a subsequent reconstruction accepted 352
intermediate audits while terminal cells remained 24/240. Active boundaries
are ADE20K epoch 10, ImageNet-100 velocity/seed-7319 epoch 48, NYUv2
z0/seed-7319 epoch 65, and VOC2012 zt/seed-7319 epoch 29. Seven transient
races are classified and unresolved failures remain zero. Ledger
`artifacts/reports/four_task_atomic_ledger_reconstruction_20260825T0226Z.json`
has SHA-256
`d35db363e9bc42cce0c0f02815131fe57eacdf18a34049f93d538a70b7b83c2f`.
The matching live-identity pretrigger gate
`artifacts/reports/downstream_gate_cross_host_pretrigger_audit_20260824T183048Z.json`
has SHA-256
`8667441ee0d91db2d83f7ea17a2eac7edbf58bd39a26c1f404743cfd84b52e1d`;
downstream remains closed and `execution_complete=false` and
`changes_scientific_verdict=false` remain unchanged.

At 2026-08-24T18:36:51Z, the atomic reconstruction accepted 354
intermediate audits while terminal cells remained 24/240. Active boundaries
are ADE20K epoch 10, ImageNet-100 velocity/seed-7319 epoch 48, NYUv2
z0/seed-7319 epoch 74, and VOC2012 zt/seed-7319 epoch 35. Seven transient
races remain classified and unresolved failures remain zero. Ledger
`artifacts/reports/four_task_atomic_ledger_reconstruction_20260824T183700Z.json`
has SHA-256
`e938ca3bfbcbd2aec3ab76fb303ed0d036a7cad85ab176aa94aaa09b46658957`.
The matching live-identity pretrigger gate
`artifacts/reports/downstream_gate_cross_host_pretrigger_audit_20260824T183651Z.json`
has SHA-256
`6236d2347b7e01d4a671165d773f75896e1eda28e92fd95fc7936aff657d8548`;
downstream remains closed, `execution_complete=false`, and
`changes_scientific_verdict=false`.

At 2026-08-24T19:55:44Z, the watcher accepted a VOC2012 zt/seed-7319
terminal cell plus new NYUv2 and VOC2012 intermediate audits. The accepted
total is 372 and terminal cells are now 26/240. Active boundaries are ADE20K
random_feature_local/seed-4121 epoch 11, ImageNet-100 velocity/seed-7319
epoch 49, NYUv2 z0/seed-104729 epoch 64, and VOC2012 zt/seed-104729 epoch 1.
Seven transient races remain classified and unresolved failures remain zero.
Ledger
`artifacts/reports/four_task_atomic_ledger_reconstruction_20260824T195700Z.json`
has SHA-256
`25a7649618537a5703fcde97f9af06bc93aabd52acebc895bad45d46ef062713`.
The matching pretrigger gate
`artifacts/reports/downstream_gate_cross_host_pretrigger_audit_20260824T195700Z.json`
has SHA-256
`6b890c7d473fea3c4a631c92975bfc1e9946fe041a0f6ce830c945fdd1ab5a18`;
downstream remains closed, `execution_complete=false`, and
`changes_scientific_verdict=false`.

At 2026-08-24T20:05:56Z, the watcher accepted two further intermediate
audits. The accepted total is 374 and terminal cells remain 26/240. Active
boundaries are ADE20K random_feature_local/seed-4121 epoch 11, ImageNet-100
velocity/seed-7319 epoch 49, NYUv2 z0/seed-104729 epoch 73, and VOC2012
zt/seed-104729 epoch 7. Seven transient races remain classified and
unresolved failures remain zero. Ledger
`artifacts/reports/four_task_atomic_ledger_reconstruction_20260824T200700Z.json`
has SHA-256
`4700179ad4d07c4386a62208c661bd48cedefe8967885f26b263877d6200118b`.
The matching pretrigger gate
`artifacts/reports/downstream_gate_cross_host_pretrigger_audit_20260824T200700Z.json`
has SHA-256
`1043424a9e52cbf9dcc18e42ecda84be096857e44b2f3e70c7298d9d8c9a210c`;
downstream remains closed, `execution_complete=false`, and
`changes_scientific_verdict=false`.

At 2026-08-24T20:16:03Z, the watcher accepted a new NYUv2
z0/seed-104729 terminal cell and two further intermediate audits. The
accepted total is 376 and terminal cells are now 27/240. Active boundaries
are ADE20K random_feature_local/seed-4121 epoch 11, ImageNet-100
velocity/seed-7319 epoch 49, NYUv2 zt/seed-4121 epoch 2, and VOC2012
zt/seed-104729 epoch 13. Seven transient races remain classified and
unresolved failures remain zero. Ledger
`artifacts/reports/four_task_atomic_ledger_reconstruction_20260824T201700Z.json`
has SHA-256
`955546ddfc7a7c5afb4996f199f1c012206f769374a8e2a874e4888048f7b582`.
The matching pretrigger gate
`artifacts/reports/downstream_gate_cross_host_pretrigger_audit_20260824T201700Z.json`
has SHA-256
`4326c1138b71211d2c1c9e83be8025f3f235c6e2e5050a769526b6062f949668`;
downstream remains closed, `execution_complete=false`, and
`changes_scientific_verdict=false`.

At 2026-08-24T20:26:15Z, the watcher accepted two further intermediate
audits. The accepted total is 378 and terminal cells remain 27/240. Active
boundaries are ADE20K random_feature_local/seed-4121 epoch 11, ImageNet-100
velocity/seed-7319 epoch 49, NYUv2 zt/seed-4121 epoch 11, and VOC2012
zt/seed-104729 epoch 19. Seven transient races remain classified and
unresolved failures remain zero. Ledger
`artifacts/reports/four_task_atomic_ledger_reconstruction_20260824T202700Z.json`
has SHA-256
`51a3e221d70fa495a3110ac7f9144cc416920557263bdd1798a06b368e167a02`.
The matching pretrigger gate
`artifacts/reports/downstream_gate_cross_host_pretrigger_audit_20260824T202700Z.json`
has SHA-256
`04f80fa6bb97ea0abdd1b48c80eed8fccb062d0b33a13fa36412a2f65e2724e0`;
downstream remains closed, `execution_complete=false`, and
`changes_scientific_verdict=false`.

At 2026-08-24T20:36:22Z, the watcher accepted two further intermediate
audits. The accepted total is 380 and terminal cells remain 27/240. Active
boundaries are ADE20K random_feature_local/seed-4121 epoch 11, ImageNet-100
velocity/seed-7319 epoch 49, NYUv2 zt/seed-4121 epoch 21, and VOC2012
zt/seed-104729 epoch 25. Seven transient races remain classified and
unresolved failures remain zero. Ledger
`artifacts/reports/four_task_atomic_ledger_reconstruction_20260824T203700Z.json`
has SHA-256
`d57ddd7e7558e572c18a81f65e4862c2243bfa73b0acba6b0842d478caba82a4`.
The matching pretrigger gate
`artifacts/reports/downstream_gate_cross_host_pretrigger_audit_20260824T203700Z.json`
has SHA-256
`898fd0979d7dfc0d53fe61fcbcd86f2c81c8e142c67e7ed0c3af135c7ecfea9c`;
downstream remains closed, `execution_complete=false`, and
`changes_scientific_verdict=false`.

At 2026-08-24T20:46:28Z, the watcher accepted two further intermediate
audits. The accepted total is 382 and terminal cells remain 27/240. Active
boundaries are ADE20K random_feature_local/seed-4121 epoch 11, ImageNet-100
velocity/seed-7319 epoch 49, NYUv2 zt/seed-4121 epoch 30, and VOC2012
zt/seed-104729 epoch 31. Seven transient races remain classified and
unresolved failures remain zero. Ledger
`artifacts/reports/four_task_atomic_ledger_reconstruction_20260824T204700Z.json`
has SHA-256
`f9a680d031457665e578f33ea16f77ab7aba28b1c21e9cba932bb62b6975a7a4`.
The matching pretrigger gate
`artifacts/reports/downstream_gate_cross_host_pretrigger_audit_20260824T204700Z.json`
has SHA-256
`d40c3a944f306ecfbfe6e6351fcc892225c55df7690754a3d728a181b6274527`;
downstream remains closed, `execution_complete=false`, and
`changes_scientific_verdict=false`.

At 2026-08-24T20:56:36Z, the watcher accepted two further intermediate
audits. The accepted total is 384 and terminal cells remain 27/240. Active
boundaries are ADE20K random_feature_local/seed-4121 epoch 11, ImageNet-100
velocity/seed-7319 epoch 49, NYUv2 zt/seed-4121 epoch 40, and VOC2012
zt/seed-104729 epoch 37. Seven transient races remain classified and
unresolved failures remain zero. Ledger
`artifacts/reports/four_task_atomic_ledger_reconstruction_20260824T205700Z.json`
has SHA-256
`d23febd4041c785838e8bcdbf5b8394f36d1dbb2c28babdd6bb13c122ea2f32a`.
The matching pretrigger gate
`artifacts/reports/downstream_gate_cross_host_pretrigger_audit_20260824T205700Z.json`
has SHA-256
`555f27f09b4b074955335d9e202770f936835fe5297178e286046775fbf15c13`;
downstream remains closed, `execution_complete=false`, and
`changes_scientific_verdict=false`.

At 2026-08-24T21:06:43Z, the watcher preserved a new failed intermediate
audit for NYUv2 zt/seed-4121 epoch 49 with
`stable_checkpoint_during_hash_and_load`. The failure is not classified yet;
`unresolved_scientific_failure=false`, but the independent ledger therefore
reports `status=failed` and `unresolved_failures=1`. The failed artifact and
watcher state remain preserved. The failed reconstruction ledger
`artifacts/reports/four_task_atomic_ledger_reconstruction_20260824T210700Z.json`
has SHA-256
`568a2a3d900cda1dae3b4667e1567a27406bf415f928968a6b6d76537d632528`.
No pretrigger gate was generated from this failed ledger; all downstream
stages remain closed and `changes_scientific_verdict=false`.

At 2026-08-24T21:16:55Z, subsequent strong audits were present for NYUv2
zt/seed-4121 epoch 59, VOC2012 zt/seed-104729 epoch 50, and ImageNet-100
velocity/seed-7319 epoch 50, but the watcher had not yet migrated the prior
NYUv2 epoch-49 failure. The refreshed independent ledger therefore contains
389 accepted intermediate audits but still reports `status=failed` and one
unresolved watcher failure; no passed gate was generated. Ledger
`artifacts/reports/four_task_atomic_ledger_reconstruction_20260824T211800Z.json`
has SHA-256
`61e30f3ffc71783156fac5b81dc495f84f3527ff1ecef5260eb481f11dacac8e`.

At 2026-08-24T21:27:07Z, the watcher classified the prior NYUv2
zt/seed-4121 epoch-49 failure as
`unstable_observation_confirmed_by_later_strong_audit`, preserving its failed
artifact and increasing classified transient races to 8. A new unresolved
failure was then recorded for NYUv2 zt/seed-4121 epoch 68. The refreshed
ledger therefore remains `status=failed` with 390 accepted intermediate
audits and one unresolved failure; no passed gate was generated. Ledger
`artifacts/reports/four_task_atomic_ledger_reconstruction_20260824T212800Z.json`
has SHA-256
`a40bc90c70311cca7e0b88c88026b37174b3c670fdd3bff2e620ffddacb867d5`.

At 2026-08-24T21:37:16Z, the watcher still retained one unresolved
intermediate failure for NYUv2 zt/seed-4121 epoch 68
(`stable_checkpoint_during_hash_and_load`). A later NYUv2 epoch-77 strong audit
passed, but the watcher had not yet migrated the epoch-68 artifact. The
independent reconstruction therefore reports 392 accepted intermediate audits,
27/240 terminal cells, eight classified transient races, `status=failed`, and
`unresolved_failures=1`; terminal counts are ImageNet-100 13, NYUv2 6, VOC2012
8, and ADE20K 0. The failed artifact remains preserved, no pretrigger gate was
generated, and `execution_complete=false` and
`changes_scientific_verdict=false` remain unchanged. Ledger
`artifacts/reports/four_task_atomic_ledger_reconstruction_20260824T213900Z.json`
has SHA-256
`b0b87600ad1828bff95f65d21a231899bbfee118676849f3a8c4dd9aa6014d21`.

At 2026-08-24T21:47:25Z, the watcher migrated the NYUv2 zt/seed-4121
epoch-68 stability failure using later strong evidence. The watcher state now
has `failures=[]` and nine classified transient races. Independent ledger
reconstruction accepted 394 intermediate audits and 28/240 terminal cells:
ImageNet-100 13, NYUv2 7, VOC2012 8, ADE20K 0. Four cells remain active
(ADE20K random_feature_local/4121 epoch 12, ImageNet-100 velocity/7319 epoch
50, NYUv2 zt/7319 epoch 6, VOC2012 zt/104729 epoch 68). The ledger is
`status=passed` with no reconstruction problems, but
`execution_complete=false` and `changes_scientific_verdict=false`; downstream
remains fail-closed pending all 240 main cells. Ledger
`artifacts/reports/four_task_atomic_ledger_reconstruction_20260824T214800Z.json`
has SHA-256
`4a1857ae5d6d982ed5d8502ad5a3027c5b72677aa24ec42b87a6cf4f4927d233`.
The matching live-identity pretrigger gate
`artifacts/reports/downstream_gate_cross_host_pretrigger_audit_20260824T214800Z.json`
has SHA-256
`b259531caef74957b322c8fa317fa1a1c099102497647fa175f4670a4a4818fd` and
decides `wait_for_all_240_main_matrix_cells`; no causal, extension, replay, or
final artifact was triggered. An initial shell-quoting attempt also produced a
non-authoritative ledger filename ending in `reconstruction_.json`; it is
retained as operation provenance and is not the authoritative ledger.

At 2026-08-24T21:57:40Z, the watcher accepted four further strong audits
without failures (NYUv2 zt/seed-7319 epoch 15 and VOC2012 zt/seed-104729 epoch
74 among them). The refreshed ledger contains 396 accepted intermediate audits
and 28/240 terminal cells, with nine classified transient races and
`unresolved_failures=0`. Terminal counts remain ImageNet-100 13, NYUv2 7,
VOC2012 8, ADE20K 0; active cells are ADE20K random_feature_local/4121 epoch
12, ImageNet-100 velocity/7319 epoch 50, NYUv2 zt/7319 epoch 15, and VOC2012
zt/104729 epoch 74. Ledger status is `passed`, but
`execution_complete=false` and `changes_scientific_verdict=false`; the
pretrigger gate remains `wait_for_all_240_main_matrix_cells` and no downstream
artifact was triggered. Ledger
`artifacts/reports/four_task_atomic_ledger_reconstruction_20260824T215800Z.json`
has SHA-256
`77cc24c7408440b00891300908d30790181fcc833cab4a1a27b2fc939b9b533f`; matching
gate `artifacts/reports/downstream_gate_cross_host_pretrigger_audit_20260824T215800Z.json`
has SHA-256
`2820e624486f695f6cada7bfb91daeb57a2545ddf0de10b17b306cecfd4c15b3`.

At 2026-08-24T22:07:48Z, the watcher accepted NYUv2 zt/seed-7319 epoch 23;
no failure was observed. The refreshed ledger contains 397 accepted
intermediate audits and 28/240 terminal cells, with nine classified transient
races and `unresolved_failures=0`. Terminal counts remain ImageNet-100 13,
NYUv2 7, VOC2012 8, ADE20K 0. The four active cells are ADE20K
random_feature_local/4121 epoch 12, ImageNet-100 velocity/7319 epoch 50, NYUv2
zt/7319 epoch 23, and VOC2012 zt/104729 epoch 74. Ledger status is `passed`,
while `execution_complete=false` and `changes_scientific_verdict=false`; the
pretrigger decision remains `wait_for_all_240_main_matrix_cells`. Ledger
`artifacts/reports/four_task_atomic_ledger_reconstruction_20260824T220800Z.json`
has SHA-256
`2c8d6ac5ac0602f7742ff65a7b4a1d5eb720fa041c8457b75e67eb89c5765fbb`; matching
gate `artifacts/reports/downstream_gate_cross_host_pretrigger_audit_20260824T220800Z.json`
has SHA-256
`7c2db15ce0aec251855669e8835dc2c3fe062461b9a973bfe3f8f86b8cbbf04e`.
