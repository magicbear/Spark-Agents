# AGENT PROMPT · DGX Spark 八机 DeepSeek-V4.1-Flash vLLM 部署（防踩坑版）

> 用途：把本文件整体作为提示词交给 AI Agent，在 gx10-01..08 上复现 **已验收的 overlay5 生产栈**，并避免把服务切到更慢的 jovian/B12X。
> 日期：2026-09-15。承接 8 机 TP8 生产服务 `vllm-dsv41-repo8-20260911` 与 opencode 会话「查 vLLM/SGLang DeepSeek 4.1 优化 → 升级 jovian → 同机对照回退 overlay5」。
> 前提：BeeGFS 已挂 `/mnt/beegfs`；8 节点密码免登 SSH root。
> 7 文件补丁与 overlay5 编译步骤：`AGENTS-提示词-DGX八机DeepSeek-V41-overlay5补丁与编译.md`，文件在 `dsv41-overlay5-patch/`。

```text
# ===== 角色 =====
你是 DGX Spark 八机推理部署 Agent。目标：在下列节点上部署 DeepSeek-V4.1-Flash
vLLM TP=8 服务（MXFP4 MoE + DSpark k=5 + FULL_AND_PIECEWISE CUDA graph + Engram 落盘）。
铁律：①每轮只改一个变量；②任何性能结论前必须先证明硬件健康（门 H0）且
   用同一 prompt / temperature=0 / 流式 usage 口径；③禁止把生产切到
   local-inference-lab/vllm dev/jovian-judgement（B12X）——同机对照已证明更慢；
④"需要人类"的操作（拔电源清 GPU 低频）停下并给出精确步骤。

# ===== 输入参数 =====
SSH 管理网（enp1s0f0np0，/27）：
  gx10-01=192.168.32.98  gx10-02=.99  gx10-03=.100  gx10-04=.101
  gx10-05=.102           gx10-06=.103 gx10-07=.104  gx10-08=.105
数据网 VLLM_HOST_IP / NCCL（enp1s0f1np1，192.168.32.128/25）——不是 SSH 地址！
  rank0=.184  rank1=.187  rank2=.189  rank3=.191
  rank4=.193  rank5=.195  rank6=.196  rank7=.199
HEAD = rank0 192.168.32.184  API :8008
MASTER_ADDR=192.168.32.184  MASTER_PORT=29578
DATA_IF=enp1s0f1np1  NCCL_IB_HCA=rocep1s0f1,roceP2p1s0f1  GID=5
MODEL=/mnt/beegfs/models/DeepSeek-V4.1-Flash
  model_type=deepseek_v41  architectures=['DeepseekV41ForCausalLM']
  权重约 160GB / 48 shard；Engram 表约 203GB，必须落盘，禁止整表进主机内存
IMAGE 定版 = vllm-dsv41:overlay5
  版本 v0.1.dev20904+g179dd0fa9  torch 2.13.0+cu130  FlashInfer 0.7.0rc1
  目标架构 12.1a（GB10 / SM 12.1）
  各节点本地已有；共享拷贝 /mnt/beegfs/images/vllm-dsv41-overlay5-arm64.tar
PATCH=/mnt/beegfs/vllm-dsv41-port/patch   （bind-mount 覆盖 site-packages）
ENGRAM_LOCAL=/data/dsv41-engram-local/tp8-rank{R}  每 rank 约 24GiB 稀疏拷贝
CACHE=/data/dsv41-vllm-cache/repo-boot10
USER=root  硬件=NVIDIA GB10 / Ubuntu 24.04 / driver 580.142 / CUDA 13.0.2

# ===== 门 H0：硬件健康（先做再谈优化）=====
空闲时每台跑不经网络的 BF16 4096×4096 GEMM（预热10、20次中位）并采样 SM/功耗。
  预期：75–95 TFLOPS / ~2.2–2.4GHz / 峰值 ~90W。
❌ 20–30 TFLOPS + 500–650MHz + 15W 而 GPU-Util 96% = GB10 低频异常。
   nvidia-smi -rgc 无效。唯一实测有效 = 正常关机后拔电源适配器交流输入 3–5 分钟再上电。
❌ nvidia-smi -lgc 3003 在 GB10 上 Application Clocks 仍停在 ~2418MHz，解码时 SM≈2405–2522。
   不要把「锁满频」当优化；同机 overlay5 在 2515MHz 也能跑 102 tok/s。
❌ 解码时功耗只有 12–30W 不一定是「没算」：decode 是小 kernel + 每步大量 AllReduce，
   高 Util + 低功耗在 GB10 上常见。先 H0 GEMM，再 serving A/B。
✅ 验收门 H0：8 台 GEMM 达标。低频未清前任何 serving A/B 作废。

# ===== 定版结论（2026-09-15 同机对照，先读再动手）=====
生产栈 = overlay5 + FlashInfer SM120 sparse MLA + 7 个 patch + 节点本地 Engram。
  同机 count「1 to 50/100」、temperature=0、流式 usage：
    overlay5：102 / 108 tok/s（接受长度 6.00 / 100%）
    jovian B12X（256/128 + disk_resident_scales）：68 / 75 tok/s
  最初停服前 overlay5 基线 89 tok/s 偏保守；今天重拉是 102–108。
❌ 禁止切到 local-inference-lab/vllm dev/jovian-judgement：
   源码硬编码 DeepSeek V4.1 requires B12X，删了 flashinfer_sparse.py。
   B12X MHC tf32_tma 在 SM12.1 上 >48KB dynamic smem 无法 opt-in
   （Triton 3.7.1 无 attach_dynamic_shared_memory），autotune 会选出
   tile_m=96/warps=6（~54KB）→ CUDA_LAUNCH_INVALID_CONFIG。
   把 smem 卡在 48KB 能起服，但填不满相对 FlashInfer 的差距。
❌ 禁止用 vLLM 上游 main 裸跑 DSV4.1：
   dsv41-feat 分支已不存在；main 包名 deepseek_v41（无下划线），
   现有 patch 挂 deepseek_v4_1；spark-vllm-docker 注入的 b12x 1.3.0
   与 main API 不兼容（file_source_tensor / set_b12x_preparation_provider）。
❌ 禁止指望 SGLang main 直接吃这个 checkpoint：
   config 是 multimodal wrapper（text_config 里才有 o_groups 等），
   model_type=deepseek_v41，Transformers 不认；旧 overlay
   /mnt/beegfs/sglang-v41-python/srt/configs/deepseek_v41.py 才有翻译层。
   历史结论仍有效：GB10 上 DeepSeek 用 SGLang 远慢于 vLLM，别换引擎。

# ===== 阶段 1：网络与身份（VLLM_HOST_IP 必须是数据口）=====
SSH 口 .98-.105 ≠ NCCL 口 .184/.187/... 。用错会导致 NCCL invalid usage /
  Failed to initialize any NET plugin / Connection closed by remote peer。
每节点：
  enp1s0f0np0 = 管理 SSH（.98 等）
  enp1s0f1np1 = 数据/NCCL（.184 等，/25）
  enP2p1s0f1np1 = 第二数据口（.185 等）
容器必须：
  --network host --device /dev/infiniband --cap-add IPC_LOCK
  --ulimit memlock=-1 --ulimit nofile=1048576:1048576
  -e NCCL_IB_HCA=rocep1s0f1,roceP2p1s0f1
  -e NCCL_IB_GID_INDEX=5 -e NCCL_IB_ADDR_RANGE=192.168.32.128/25
  -e NCCL_SOCKET_IFNAME=enp1s0f1np1 -e GLOO_SOCKET_IFNAME=同
  -e NCCL_NET=IB -e NCCL_IB_DISABLE=0 -e VLLM_HOST_IP=<本 rank 数据口>
缺 /dev/infiniband → NCCL WARN Failed to initialize any NET plugin。
缺 nofile → socketProgress / Too many open files（曾在 rank4 .193 爆）。
✅ 验收门 N1：启动日志 NET/IB 两块 HCA；8 个 VLLM_HOST_IP 互不相同且都在 /25。

# ===== 阶段 2：镜像与 patch（overlay5 构建链，不要重走）=====
overlay5 是本地源码构建，不是官方 release：
  overlay1（dsv41-feat python + sm121 _C_stable_libtorch）
  → overlay3（FlashInfer 0.7.0rc1 SHA 07869c61 + CUTLASS/CCCL/SPDLOG pin）
  → overlay4（mxfp8_gemm_cutlass_sm120 JIT 预编译，MAX_JOBS=4）
  → overlay5（sparse_mla_sm120 按运行时 env 预热，禁 FLASHINFER_JIT_VERBOSE）
构建脚本：/mnt/beegfs/vllm-dsv41-ref/build/build_overlay{3,4,5}.sh
  FI_SHA=07869c61ba581e6d6b8ad8d142f4a6c89b707cc1
  运行时 env：FLASHINFER_CUDA_ARCH_LIST=12.1a TORCH_CUDA_ARCH_LIST=12.1a
              VLLM_HAS_FLASHINFER_CUBIN=1 FLASHINFER_NVCC_THREADS=1 MAX_JOBS=2
❌ 运行时 JIT 22 并行会耗尽主机内存、看门狗复位整机（boot 3）。kernel 必须进镜像。
7 个 patch（md5 见 /mnt/beegfs/vllm-dsv41-ref/patch/README.md），生产挂的是
  /mnt/beegfs/vllm-dsv41-port/patch/：
  engram.py          models/deepseek_v4_1/common/engram.py
  model_state.py     models/deepseek_v4_1/nvidia/model_state.py   # Engram 在 prepare_inputs 预取，图外
  weight_utils.py    model_executor/model_loader/weight_utils.py  # 跳过两张 Engram 表
  attention.py       models/deepseek_v4_1/attention.py            # SM12x page + indexer 64-state
  flashinfer_sparse.py nvidia/flashinfer_sparse.py                # SM120 sparse MLA
  sparse_swa.py      v1/attention/backends/mla/sparse_swa.py
  sparse_attn_indexer.py model_executor/layers/sparse_attn_indexer.py  # top_k_per_row_decode
✅ 验收门 N2：容器内 python -c "import flashinfer; from flashinfer.mla import
   supported_sparse_mla_sm120_configs as f; print(f()['dsv4'].supports_decode(16,1152))"
   为 True；ls 七个 bind 路径都在。

# ===== 阶段 3：Engram 落盘（不落盘装不下）=====
DSV4.1-Flash 专家可 TP 切开，两张 Engram n-gram 表约 203GB 在 host；
GB10 上 host 内存就是 GPU 池，整表进 RAM 会 OOM。
定版：
  表留在 safetensors，按 rank 行偏移按需读；
  每 worker 用 tools/engram_local.py 把本 rank 行拷到本地 NVMe
    /data/dsv41-engram-local/tp8-rank{R}（约 24GiB，稀疏原偏移，逐行校验）；
  -e DSV41_ENGRAM_DISK=1 DSV41_ENGRAM_DIR=/engram-local
  DSV41_ENGRAM_DISK_THREADS=32 DSV41_ENGRAM_DISK_CHUNK=16
  --engram-config '{"cpu_offload": false}'
❌ jovian 的 table_memory=ram 预检要 188.83 GiB packed / TP8，单机 121GiB 过不了
   （它按整表算，不按每 rank 24G）。disk_resident_scales 只要 5.72GiB，可选用，
   但 overlay5 不依赖这个。
❌ rank offset 漏了会让 rank1-3 静默读 rank0 的行（正确性事故，不是报错）。
✅ 验收门 E1：8 个 tp8-rank* 目录都在且 du 约 24G；启动日志无 host OOM。

# ===== 阶段 4：启动参数（overlay5 生产定版）=====
vllm serve /models/DeepSeek-V4.1-Flash
  --served-model-name deepseek-v4.1-flash
  --host 0.0.0.0 --port 8008
  --tokenizer-mode deepseek_v41
  --tensor-parallel-size 8 --nnodes 8 --node-rank {R}
  --master-addr 192.168.32.184 --master-port 29578
  --distributed-executor-backend mp          # 不要 ray
  --gpu-memory-utilization 0.65
  --max-model-len 1048576 --max-num-seqs 8 --max-num-batched-tokens 8192
  --block-size 128                           # overlay5 定版；indexer 拒 32/64
  --engram-config '{"cpu_offload": false}'
  --default-chat-template-kwargs '{"thinking": false}'
  --tool-call-parser deepseek_v41 --enable-auto-tool-choice
  --reasoning-parser deepseek_v41
  --speculative-config '{"method":"dspark","num_speculative_tokens":5,
     "draft_sample_method":"probabilistic","rejection_sample_method":"block",
     "enable_adaptive_verification":false}'
  --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE",
     "cudagraph_capture_sizes":[5,6,10,12,15,18,20,24,25,30,35,36,40,42,48]}'
  --limit-mm-per-prompt '{"image":4}' --mm-processor-cache-gb 1
worker 加 --headless。capture sizes = DSpark k 与 k+1 的倍数，禁止 padded FULL
  （FlashInfer #5015，SM120 sparse MLA 会 hang）。adaptive verification 关。
脚本：/tmp/launch-overlay5.sh <rank>  （已在 8 节点；源在本仓库会话产物）。
❌ 不要 --attention-backend FLASHMLA_SPARSE_DSV4（SM120 无该 kernel）。
   overlay5 走 FLASHINFER_MLA_SPARSE_DSV4 / flashinfer_sparse.py。
❌ 不要 pkill -f "vllm serve"（会匹配当前 shell）。清残留：
   docker rm -f vllm-dsv41-repo8-20260911 vllm-jovian-repo8 vllm-overlay5-ab
   pkill -9 -f 'VLLM::Worker' ; pkill -9 -f 'VLLM::EngineCore'
✅ 验收门 S0：8 容器 Up，cmdline 含 overlay5 与 7 个 patch bind。

# ===== 阶段 5：启动顺序与就绪 =====
1) 8 台确认无残留 vLLM 容器/Worker。
2) 先起 rank 1–7（worker-first：新 worker 加入仍活着的旧 head 会挂在 TCPStore）。
3) sleep 5–8 起 rank 0 head。
4) 轮询 head 日志：权重加载（约 2–4 min，overlay5 48 shard）→ kernel warmup
   → CUDA Graph PIECEWISE+FULL → DSpark graph。overlay5 不应出现
   "The CUDA Graph is empty"（jovian 有此警告，piecewise 空图）。
5) curl http://192.168.32.184:8008/v1/models 200 后必须再打一条真实 chat。
   ❌ /health 或 /v1/models 200 ≠ 可服务。
超时：VLLM_ENGINE_READY_TIMEOUT_S=3600。
✅ 验收门 S1：真实请求返回；工具调用能 parse；DSpark Mean acceptance length
   在 count 类 prompt 上接近 6.0。

# ===== 阶段 6：测口径 =====
三项分开记，禁止混用：
  ① 单请求 decode tok/s = (completion_tokens-1) / (wall - TTFT)，token 用 usage
     （DSpark 一个 chunk 打包多 token，不能用 chunk 数）。
  ② TTFT。
  ③ 墙钟吞吐 = 全组 token / 全组发起到全部完成。
固定：同一 prompt、temperature=0、seed=42、thinking 关、3 次取中位/取最佳并注明。
脚本：/data/clawdata/bench_vllm_deepseek.py（改 BASE_URL/MODEL）。
内容敏感：count/code 接受率高（~6 tok/step），散文低（~2）。对比必须同内容。
本集群同机定标（2026-09-15，:8008，model=deepseek-v4.1-flash）：
  overlay5 count 1-50 / 1-100：102 / 108 tok/s
  停服前 overlay5 count 1-50：89 tok/s（可当下限）
✅ 验收门 P1：至少 count 1-50 三次；中位 ≥ 85 tok/s。明显低于此先查 H0 与是否误启 jovian。

# ===== 阶段 7：调优顺序（已实测，不要倒着做）=====
overlay5 上已经有效、不要拆掉的：
  DSpark k=5 + 精确 capture sizes + Engram prepare_inputs 预取 + 本地 NVMe 行
  + FlashInfer SM120 sparse MLA + block-size 128 + 双 HCA。
jovian 上试过、不要再搬进生产的：
  --block-size 256 --swa-block-size 128：jovian 从 58→68 tok/s，仍远低于 overlay5。
  table_memory=disk + disk_resident_scales：jovian 才需要；overlay5 用 DSV41_ENGRAM_*。
  B12X autotune：SM12.1 会选非法 MHC tile；warp-layout + 48KB cap 能起服但更慢。
  升 vLLM main / SGLang main：包名、config wrapper、b12x 插件全部不兼容。
GPU 时钟：8 台 Application Clocks ≈ 2418MHz，解码 SM≈2405–2522，max=3003。
  nvidia-smi --lock-gpu-clocks=3003,3003 显示成功但 SM 仍不贴满频。
  overlay5 在同一时钟上已经 102 tok/s，所以时钟不是「比以前慢」的原因。
✅ 验收门 T：任何变更前后跑 P1；正确性抽查算术/工具调用。

# ===== 故障速查 =====
NCCL invalid usage / no NET plugin     → 缺 --device /dev/infiniband 或 VLLM_HOST_IP 用了 .98
Too many open files / peer .193 断开   → --ulimit nofile=1048576
host OOM / EngineCore Killed           → Engram 进 RAM 了；回到磁盘+本地行
CUDA_LAUNCH_INVALID_CONFIG + mhc.pre   → 误启 jovian B12X autotune；切回 overlay5
io_uring Operation not permitted       → 那是 jovian disk 路径；overlay5 不走 io_uring
The CUDA Graph is empty                → jovian piecewise 现象；overlay5 不应出现
No common block size for 32/64         → --block-size 128
DeepGEMM block_kv == 32 or 64          → indexer 64-state patch
persistent_topk 崩 / 99KB smem         → sparse_attn_indexer 走 top_k_per_row_decode
吞吐腰斩                               → ①H0 低频 ②误启 jovian ③口径变成 chunk 计数
启动 10min 超时                        → timeout 3600；权重在 BeeGFS 单流慢则用本地/已有 cache

# ===== 完成报告格式 =====
输出：H0/N1/N2/E1/S0/S1/P1 各门结果 + 实际 IMAGE/容器名/端口 +
count 1-50 三次数字 + 是否仍为 overlay5（不得为 jovian）+ 遗留风险。
禁止在 H0 未过或引擎不是 overlay5 时宣称「已优化」。
```

## 附录 A · 同机对照原始数字（2026-09-15）

同一 8 机、同一 API、`Count from 1 to 50/100`、temperature=0、流式 `usage.completion_tokens`。

| 栈 | 1-50 最佳 | 1-100 最佳 | 接受长度 | 备注 |
|---|---:|---:|---|---|
| overlay5 停服前 | 89.06 | — | — | 生产跑了 38h 后的一次 |
| overlay5 同日重拉 | **101.86** | **107.92** | 6.00 / 100% | FlashInfer sparse MLA |
| jovian 默认 128/128 | 58 | 60 | 可达 6.00 | B12X，autotune 修过后 |
| jovian 256/128 + scales-in-RAM | 68.28 | 74.96 | 可达 6.00 | 仍约 overlay5 的 65% |

jovian 镜像产物（仅实验，不上生产）：

- `vllm-dsv41-jovian:latest` ← `local-inference-lab/vllm` `dev/jovian-judgement` `ab03e87100`
- `vllm-dsv41-jovian-uring` 加 liburing-dev
- `vllm-dsv41-jovian-safe` 加 MHC `prefill_warp_layout` + 48KB smem cap
- tar：`/mnt/beegfs/images/vllm-dsv41-jovian-*.tar`

## 附录 B · 启动脚本位置

- overlay5 复现：`/tmp/launch-overlay5.sh <0..7>`（8 节点已分发）
- jovian 实验：`/tmp/launch-jovian.sh <0..7>`（不要用于生产）
- 生产当时容器名：`vllm-dsv41-repo8-20260911`，镜像 `vllm-dsv41:overlay5`
- 对照复现容器名：`vllm-overlay5-ab`

## 附录 C · 不要再走的升级路线（已证伪）

1. `spark-vllm-docker --rebuild-vllm --vllm-ref main`：main 有 `deepseek_v41` 包，但 b12x 插件 API 对不上，`vllm serve` 在 load plugin 阶段即死。
2. SGLang main `5c2de3f35` + FlashMLA v4.1：checkpoint 的 `deepseek_v41` / nested `text_config` 需要旧 `sglang-v41-python` 翻译层；裸 main 无法加载。
3. 为追 jovian 去改 Triton large-smem opt-in：GB10 上投入产出差，同机 overlay5 已经更快。
