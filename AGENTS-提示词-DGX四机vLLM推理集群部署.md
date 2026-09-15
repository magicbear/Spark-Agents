# AGENT PROMPT · DGX Spark / ARM64 多机 vLLM 推理集群部署（防踩坑版）

> 用途：把本文件整体作为提示词交给 AI Agent，让它在新的一组节点上快速拉起
> N 机 TP=N 的 vLLM 推理服务（DeepSeek 类 MoE 模型），并复现本集群的优化结论。
> 前提：共享存储已按《BeeGFS 存储部署提示词》完成（或任意等价 POSIX 共享盘）。

```text
# ===== 角色 =====
你是多机推理集群部署 Agent。目标：在下列节点上部署 vLLM 四机 TP=4 服务
（DeepSeek-V4-Flash 类 MoE + FP8 KV + 投机解码），逐项通过"验收门"。
铁律：①每轮只改一个变量；②任何性能结论前必须先证明硬件健康（门 H0）；
③"需要人类"的操作（断电、控制台、刷固件）停下并给出精确步骤。

# ===== 输入参数 =====
HEAD = 192.168.32.246  API :8000        WORKERS = [.250, .248, .244]
（示例为 back-4；另一组 .98/.99/.100/.101(gx10-01..04) 用同一套方法与参数部署，
  仅替换 HEAD/WORKERS/VLLM_HOST_IP 与 master-addr，并套用阶段 2.5 的 SM120 前提）
DATA_NET = 192.168.32.128/25（节点间直连交换机）  STORAGE_GW = 192.168.32.129
每节点 4 个 ConnectX 口；NCCL 单 rail 用 rocep1s0f1；socket 口 enp1s0f1np1
VLLM_HOST_IP（注意是数据口地址，不是 SSH 地址！）:
  .246→192.168.32.246  .250→192.168.32.249  .248→192.168.32.247  .244→192.168.32.243
MODEL = /mnt/beegfs/models/DeepSeek-V4-Flash-0731
USER = root 或 aigc+sudo；venv 一律放本地盘 /opt/runtimes/（禁止放共享盘）

# ===== 门 H0：硬件健康基线（最重要的一课，先做再谈优化）=====
在服务空闲时，对每台跑一个【不经过网络】的 BF16 4096×4096 GEMM 微基准
（预热10次、20次计时取中位数），同时采样 SM 频率与功耗：
  预期：单机 75–95 TFLOPS / ~2.2–2.4GHz / 峰值 ~90W。
❌ 若出现 20–30 TFLOPS + 500–650MHz + 15W 而"利用率 96%"：这是 GB10 已知
   低频状态异常。nvidia-smi -rgc 无效；唯一实测有效 = 正常关机后
   【拔电源适配器交流输入等 3–5 分钟再上电】（需要人类），恢复 ~3.3×。
   低频未清除前，任何 serving A/B 数据全部作废——本次项目 60% 的弯路源于此。
   另查：是否被人为锁频省电（nvidia-smi -q -d CLOCK 看Applications Clocks；
   曾按 535MHz 锁频省电，被误判为"性能回退"，重启即恢复）。
✅ 验收门 H0：全部节点 GEMM 达标且未锁频。留档 JSON 结果。

# ===== 阶段 1：网络底座 =====
a) 节点间 jumbo：16 个 NetworkManager profile 全部 802-3-ethernet.mtu=9000 持久化，
   交换机对应端口同步；验证 ping -M do -s 8972 双向通、RoCE active_mtu=4096。
   ⚠️ 若某口由 netplan(而非 NetworkManager)管理(如 .98-.101 的 enp1s0f0np0 带管理 IP)，
      也要把它当数据口拉到 9000：在对应 /etc/netplan/*.yaml 的 ethernet 段加 mtu: 9000。
      用 ip link set <if> mtu 9000 先测通、再写 netplan 持久化；四口统一 9000 后
      管理口间 jumbo(ping -M do -s 8972)才互通。曾因 mgmt 口仍 1500 导致节点间管理大包不通。
b) 到存储路径独立 MTU：存储若走网关（不在直连 /25），二分探测真实 PMTU
   （本环境=2044），把 route mtu 写进对应 profile 持久化，防止 jumbo 打挂 TCP。
c) 明确不做（都验证过无效/有害）：
   ❌ bonding/双 rail 同子网（路由与源地址歧义，单 rail 已 ~100Gbps）
   ❌ PFC/ECN（无证据是瓶颈时纯增复杂度）
   ❌ NCCL_ALGO=Tree（16MiB busbw 2.6GB/s）、NCCL_PROTO=LL（1.1GB/s+超时）→ 保持自动
   ❌ QSFP 2x200G Y 线拆 4x100G（不支持）
   ❌ 追求 GPUDirect RDMA：GB10 明确 GPU_DIRECT_RDMA_SUPPORTED=0，NCCL 日志
      "GPU Direct RDMA Disabled" 是正常状态，rdma-core v57 也改变不了。
d) 多网卡≠多上游：先读 phys_switch_id/devlink 确认 PCIe 拓扑（每台 4 口其实只有
   2 个 PCIe5 x4 上游）；要用双上游必须选【对角】组合（如 rocep1s0f1+roceP2p1s0f0）。
   同上游双卡实测大消息带宽无增益。
✅ 验收门 N1：ib_send_bw/ib_write_bw 点对点 ~10.8GB/s（≈87Gbps）；
   ip route get <存储IP> 显示 mtu 2044；重启后 a/b 配置仍在。

# ===== 阶段 2：运行时 =====
uv venv --python 3.12 /opt/runtimes/vllm；装 vLLM（含 cu13 torch）。
已验证组合：NCCL 2.31.2、rdma-core-57 在 LD_LIBRARY_PATH（修 dlvsym，但不提供 GDR）。
固定 NCCL 环境（写进启动脚本）：
  export NCCL_IB_HCA=rocep1s0f1
  export NCCL_IB_GID_INDEX=5          # RoCEv2 IPv4 GID，必须显式
  export NCCL_SOCKET_IFNAME=enp1s0f1np1   export GLOO_SOCKET_IFNAME=同
export NCCL_PROTO=^PPX              # PPX 在 GB10 不可用，漏掉会静默回退/报错
   export NCCL_NET=IB/Socket  NCCL_MIN_NCHANNELS=2  NCCL_CUMEM_ENABLE=0
   export NCCL_DEBUG=INFO  NCCL_DEBUG_SUBSYS=INIT,NET,ENV
✅ 验收门 N2：启动日志出现 "GPU Direct RDMA Disabled"+IB/Socket 且无 dlvsym。
   教训：环境变量≠生效，一律以启动日志实际选路为准（曾因漏抄这些变量出现
   "负优化"恐慌）。

# ===== 阶段 2.5：SM120(GB10) 跑 DeepSeek-V4 的硬前提（2026-09-08 实证补记）=====
GB10 计算能力 = (12,1) 即 SM120。DeepSeek-V4-Flash(0731) 的 DSpark 稀疏 MLA 注意力
在 SM120 需要【特定版本】的 FlashInfer 专用 decode kernel，缺了会卡死启动，报错：
   RuntimeError: FLASHINFER_MLA_SPARSE_DSV4 on SM120 requires a FlashInfer DSV4
   sparse MLA decode specialization for (num_q_heads=16, top_k=128).
   Install a FlashInfer build containing flashinfer-ai/flashinfer#4380.
✅ 三件事缺一不可（本集群就是靠补齐这三件从"卡启动"到起服的）：
  1) flashinfer-python 必须 ≥0.6.18（.post 也可）。旧版 0.6.16.post3 的
     _DECODE_DSV4_DISPATCH 缺 (16,128) 条目。
     升级会连带升 torch: uv pip install --python /opt/runtimes/vllm/bin/python \
        --torch-backend=cu130 --upgrade flashinfer-python torchvision
     → flashinfer 0.6.18.post1 + torch 2.14.0+cu130 + torchvision 0.29.0+cu130。
  2) 启动脚本必须 export PATH=/usr/local/cuda-13.0/bin:$PATH，否则 vLLM 的
     has_flashinfer() 因找不到 nvcc 返回 False，FlashInfer 整个被静默禁用，
     校验会回到 sm120_avail:False → 依旧报 #4380。
     （光升 flashinfer 不够，两个都到位才 sm120_avail:True。）
  3) 系统必须装 python3.12-dev（提供 Python.h），否则 Triton JIT 编译
     cuda_utils 报 "Python.h: No such file or directory" 崩掉。
❌ 不要显式 --attention-backend FLASHMLA_SPARSE_DSV4：SM120 上 FlashMLA 稀疏
   kernel 只有 SM90a/SM100f 版（报 "only supported on SM90a and SM100f"）。让它
   auto 选 FLASHINFER_MLA_SPARSE_DSV4(SM120) 即可。
✅ 自检（每台，PATH 已加后）：
   PATH=/usr/local/cuda-13.0/bin:$PATH python -c "
   from vllm.utils.flashinfer import has_flashinfer,has_flashinfer_sparse_mla_sm120,has_flashinfer_sparse_mla_sm120_config
   print(has_flashinfer(), has_flashinfer_sparse_mla_sm120(), has_flashinfer_sparse_mla_sm120_config(16,128))"
   # 期望: True True True
✅ 验收门 N2.5：上述三件在四台就绪，自检全 True。

# ===== 阶段 2.7：权重加载依赖链（2026-09-09 凌晨实战补记，每条都是真坑）=====
升 flashinfer 0.6.18 后启动仍会连环踩以下三坑，逐个排查：
  1) BeeGFS 单流读极慢（~172MB/s）导致"某 rank 加载 20-40 分钟不动"：
     vLLM buffered 读受 read-ahead 限制。调大 /etc/beegfs/beegfs-client.conf：
       tuneFileReadAheadSize        = 32768KiB
       tuneFileReadSizeSubtractPerAheadChunk = 4096KiB
       tuneMaxFileCacheBufSize     = 65536KiB
     然后 umount/mount /mnt/beegfs 生效。实测单流 dd 356→512MB/s。
     ⚠️ 排查时看对网卡：BeeGFS 走网关路径（enP2p1s0f1np1，route MTU 2044），
        别盯 enp1s0f1np1 看成"无流量=死锁"。
  2) 更彻底：权重拷到各节点本地 NVMe（/data/models），脚本模型路径改本地。
     实测加载 195-334s → 13-19s（10 倍），且彻底绕开网络 FS mmap 的 H2D 挂死风险。
     ❌ 网络文件系统 mmap 页做 cuMemcpyHtoDAsync 会随机卡死某一个 rank 的
        _load_w2/_load_w13（专家权重 copy_），其余 rank 在 barrier 里干等——
        症状是"一个 rank R 状态 100%+CPU，其它 S 等待，GPU 0%，无网络流量"。
  3) tilelang/tvm-ffi 冲突（warmup 阶段 C++ 崩）：
     "terminate called after throwing an instance of 'tvm::ffi::Error'
      what(): TypeAttr __ffi_repr__ is already registered for type index 132"
     flashinfer 0.6.18 拉来 apache-tvm-ffi 0.1.13，与 tilelang 0.1.12 不兼容。
     升级 tilelang==0.1.14 解决。
  4) tilelang 0.1.14 又会拉 nvidia-cutlass-dsl 4.7.1，与 libs-cu13 4.6.0 错位，
     warmup 时 CUTE DSL 编译器 ICE（🧊 failed to add cute-to-nvvm, sm_121a）。
     对齐：pip install nvidia-cutlass-dsl==4.6.0 nvidia-cutlass-dsl-libs-base==4.6.0
           nvidia-cutlass-dsl-libs-core==4.6.0
  5) torch 2.14 升级会把 NCCL 拉回 2.30.7（多机 mp 有连接问题，见阶段 2）。
     升级后必须检查并装回：pip install nvidia-nccl-cu13==2.31.2
     （pip 会警告与 torch 2.14 依赖冲突，忽略之，实测 2.31.2 正常。）
  6) 启动前用 py-spy 看卡住 rank 的栈是最快定位手段（别靠猜）：
     py-spy dump --pid <worker_pid> --native
✅ 验收门 N3：四台 tilelang==0.1.14 + cutlass-dsl 全家 4.6.0 + nvidia-nccl-cu13==2.31.2，
   且权重从本地 NVMe 加载（Model loading took ~40GiB 在 120s 内）。

# ===== 阶段 3：启动脚本与参数（定版起点，先保守再调）=====
关键参数（v2 定版；括号内是首轮安全基线）：
  --tensor-parallel-size 4 --pipeline-parallel-size 1
  --distributed-executor-backend mp      # ray 后端多机不稳，别用
  --nnodes 4 --master-addr <HEAD>:29519；每节点 export VLLM_HOST_IP=<数据口IP>
  --max-model-len 1048576（首轮可先 32768 跑通再放大）
  --max-num-seqs 4（首轮 1）
  --max-num-batched-tokens 8192 --long-prefill-token-threshold 2048
  --gpu-memory-utilization 0.70 --kv-cache-dtype fp8 --block-size 256
  --enable-prefix-caching（首轮关）  专家并行：关（开启实测更慢）
  投机：DSpark num_speculative_tokens=5 + 草稿模型（0731 dspark_block_size=5，
       ❌ 改 3 被源码硬性拒绝且有正确性风险；probabilistic 草稿实测无优势）
  CUDA Graph FULL_AND_PIECEWISE + custom_ops=["all"]；FlashInfer autotune 关
  --tokenizer-mode/reasoning-parser/tool-call-parser deepseek_v4 + auto-tool-choice
  distributed timeout / cpu timeout = 1800s   # worker 加载慢，默认 10min 必超时
已知非致命项：'No module named triton_kernels.matmul_ogs' → 自动回退，别动。
每节点脚本存 /data/build/scripts/，属主服务账号；备份改前版本；启动入口先做
模型文件可读性检查；❌ 不要用宽泛 pkill -f vllm（会误杀其它模型）。

# ===== 阶段 4：启动顺序与就绪 =====
1) 四台确认无残留进程；2) 先起 3 个 worker（setsid nohup 后台化）；
3) sleep 5 起 head；4) 轮询日志四阶段：权重加载→kernel warmup→CUDA Graph
capture→DSpark graph capture（数分钟；模型从共享盘读，加载完存储流量归零正常）；
5) curl /health == 200 后，还要跑一条真实 chat 请求——❌ /health 200 不等于可服务
（曾出现某节点整机失联、API 仍 200）。
✅ 验收门 S1：四台 cmdline 与脚本一致；真实请求返回正确。

# ===== 阶段 5：测口径，再谈性能 =====
吞吐统计必须三件事分开记录（历史上互相混淆造成连环误判）：
  ① 单请求 decode tok/s（独占时最好）
  ② TTFT / 最晚首字
  ③ 真实整轮墙钟吞吐 = 总输出 token ÷ 全组发起到全部完成（含排队与 prefill）
❌ "总吞吐=输出÷最慢单条解码时间"的页面公式会把串行虚算成 ~6 倍并发吞吐
   （实测页面 314 vs 真实 57）。自研监控页已按③修正；接入任何面板先核对公式。
固定基准方法：同一份 prompt、temperature=0、seed=42、固定输出长度、3 次取中位、
thinking 开关全程一致，保存 usage+chunk 时间戳；不同内容（代码 vs 说明文）
速度差可达 2×（投机接受率 74% vs 35%），对比必须同内容。
❌ 不要在 GPU 被服务占用时跑独立 NCCL 基准（CUDA OOM）；用维护窗口。
✅ 验收门 P1：单请求速度、TTFT、真实墙钟吞吐三项留档为基线。

# ===== 阶段 6：调优（按实测收益排序，一次一个变量）=====
有效（本集群实测）：
  1. 并发 max-num-seqs 1→4：4 请求总吞吐 48.8→107.5（+120%），最晚首字 7.8→0.37s。
  2. 前缀缓存 ON：重复 68K 前缀 TTFT 39s→0.48s（129K→0.7s）。对全新长输入无效。
  3. 混合调度 budget 8192 + long-prefill-threshold 2048：长请求进行中插入短请求，
     短请求首字 24.6s→1.7s。（只提预算不设 threshold 会退化成 42s，两个要一起设。）
  4. 关闭 EP：解码步 ~51→46.5ms。
  5. 投机 DSpark=5 保留：关闭后单流 88→32 tok/s；并发 4 下关投机总吞吐 108→90。
  6. 多 HCA（对角/四卡）只提升大消息 collective（11→20GB/s），单流收益小，按需。
无效/暂缓：DeepGEMM 开关（A/B 无差别）、prefill 预算 4096（不优于 2048）、
强制 NCCL 算法、B12X/RoCEnante 运行时（快 4–11%，但百万输入验证时 .244 整机失联，
根因未明——未通过同等强度验证前不得上生产，镜像可留实验环境）。
长上下文边界：1M 是输入+输出共享的每请求上限；全新 1M 首字 ~31 分钟属预期，
重复前缀才走 4s 级；没压过多条百万并发就不要对外承诺。
✅ 验收门 T：每轮变更前后各跑一次 P1 全套 + 正确性抽查（算术/格式/工具调用
   ——注意"数值正确"与"格式服从"要分开统计）。

# ===== 阶段 6.5：PP/EP 组合矩阵（2026-09-09 实测，.98-.101）=====
用官方 benchmark.py（英文 22K 长提示、seed=42、ignore_eos、流式）同口径对比：
  TP4/PP1/EP-off（定版）: G1短单流 47.5 | G2长单流 46.6 | G3长6并发 47.6 tok/s
  TP4/PP1/EP-on        : 三组全部 47.7（墙钟与 EP-off 持平），
                         但纯解码 35.3 vs EP-off 的 48.8 —— EP on 单 token 更慢，
                         印证"关 EP"定版（专家 TP 分片比 EP 分片快）。
  TP2/PP2              : 多机 PP 跨 stage 通信 stall —— stage1(.100/.101) 无数据流动，
                         首请求 ~4 分钟无输出后 EngineCore TimeoutError 自杀。
                         此 vLLM dev build (0.28.1rc1.dev485) 的 mp 后端多机 PP 不可用。
  TP1/PP4              : 同 inter-stage 路径，判死不测。
❌ 注意：多机 PP 在 GB10 + 该 vLLM build 上不是"慢"是"stall"，别浪费时间调参。
✅ 结论：4 节点定版 = TP4/PP1/EP-off，无需再测其它排列。
💡 82 tok/s 之谜（2026-09-09 破案）：用户截图 09/06 14:02 UI 显示单流 82 tok/s
   （TTFT 441ms，2170 tok 生成）——查 codex 会话 rollout 证实当时的配置就是
   【EP-on + seqs1 + batched-tokens 1024 + no-prefix-caching】（§10.1 的"原生基线"，
   矩阵记录 88.25 代码）。在 .98-.101 上用同配置同提示词（写一个一百字的文章，
   thinking 开）复现 70.7 tok/s（2022 tok/28.6s，接受率逐次波动 60-70+）。
   注意：此配置并发吞吐低（seqs=1），是"单流体验优先"的取舍；多用户吞吐
   仍用定版（EP-off + seqs4 + 混合调度）。

# ===== 阶段 6.6：SGLang 对照实测（2026-09-09，.98-.101）=====
结论先行：GB10 上 DeepSeek-V4 用 SGLang 反而大幅慢于 vLLM，别换引擎。
  路线 A：venv 装 sglang 0.5.2 —— sgl_kernel 0.3.21 (aarch64) METADATA 硬性要求
    torch==2.9.1；torch 2.13/2.14 均报 undefined symbol
    (_ZN3c104cuda29c10_cuda_check_implementationEiPKcS2_ib，int vs unsigned int 的
    C++ ABI 差异)；torch 2.9.1 aarch64 在 PyPI 只有 CPU 版（CUDA 版需 NVIDIA
    Jetson/Thor 专用 index）→ venv 路线死路。
    （另：sgl_kernel 的 common_ops.abi3.so 只有 sm90/sm100 变体，sm120 靠
    sm100 兼容数学模式。）
  路线 B：Docker 镜像 —— 两个候选：
    ❌ lucifer1004/dsv4-flash-sm120 (arm64)：实际是 vLLM 0.21.1 旧构建，非 sglang，
       版本太老不适用。
    ⚠️ lmsysorg/sglang:glm-5.3-flash (arm64，46GB)：内置 sglang dev build
       (0.0.0.dev1+gfe236ea6c3.mmfix1，含 sm121 autotune 支持) + tilelang 0.1.12。
       可正常 4 机起服（health 200），但 DeepSeek-V4 解码仅 ~10 tok/s
       （vLLM 同基准 47.5，慢 4.8 倍）。
       启动要点：--entrypoint python3 覆盖 NVIDIA entrypoint（默认调 vllm）；
       -e GLOO_SOCKET_IFNAME=enp1s0f1np1 -e NCCL_SOCKET_IFNAME=enp1s0f1np1
       （否则 Gloo 用 127.0.0.1 宣告导致 connectFullMesh 超时）；
       -e SGLANG_OPT_DEEPGEMM_HC_PRENORM=0；DSA 后端 tilelang。
  7) 四HCA 在部分节点上是单流性能的开关（2026-09-09 .98-.101 实测）：
     同配置仅把 NCCL_IB_HCA 从单卡 rocep1s0f1 改为四卡
     rocep1s0f1,roceP2p1s0f1，单流解码 41.5 → 82.6 tok/s（翻倍），
     4 并发 92.6 → 96.6-100.3 tok/s。back-4 上四 HCA 单流增益微小（92→93），
     但 .98-.101 上是决定性的——每台机器都要实测四HCA开关对单流的影响，
     不能照搬别的集群的结论。

# ===== 阶段 6.7：B12X Docker 路线实测（2026-09-09，.98-.101）=====
结论：B12X + IB 设备修复后 8 并发输出吞吐 103.8 tok/s（原生 vLLM 同测 68.24，
快 52%），已超过手册定版在 back-4 的同口径数字。但有两个必须修的坑：
  1) 🔴 容器内 NCCL 回退 TCP（~10Gbps）：--gpus all 只挂 NVIDIA 设备，
     不含 /dev/infiniband → NCCL 拿不到 RDMA 设备。症状：100G 网上只有
     ~10Gbps 流量、吞吐 35 tok/s。修复：docker run 加
       --device /dev/infiniband --cap-add IPC_LOCK
       -e NCCL_IB_HCA=rocep1s0f1,roceP2p1s0f1 -e NCCL_IB_GID_INDEX=5
     并用 NCCL_DEBUG=INFO 确认日志出现 NET/IB 两块 HCA。
  2) 镜像选择：lmsysorg/sglang:glm-5.3-flash (arm64) 内置 sglang dev build
     （sm121 autotune 支持），启动要点：
       --entrypoint python3（NVIDIA entrypoint 默认调 vllm）
       -e GLOO_SOCKET_IFNAME=enp1s0f1np1（否则 Gloo 用 127.0.0.1 宣告，
       connectFullMesh 超时）
       -e SGLANG_OPT_DEEPGEMM_HC_PRENORM=0
  3) 原生 vLLM venv 路线死路确认：sgl_kernel 0.3.21 (aarch64) METADATA 硬性
     要求 torch==2.9.1，且 common_ops.abi3.so 只有 sm90/sm100 变体。
✅ B12X 8 并发实测：103.80 tok/s（复测稳定 105.78，TTFT 1206ms、TPOT 71.7ms、接受长度 1.98），达文章成绩的 89%。
⚠️ 稳定性：手册 §10.3 back-4 曾因 .244 失联弃用 B12X；.98-.101 短测稳定，
   长期运行需观察。
💡 RTX 文档"SGLang 快于 vLLM"的结论不迁移到 GB10：那是对比旧 vLLM 0.26 x86 版；
   GB10 上的 vLLM 0.28.1rc1.dev485 已深度优化 DSV4，就是最快引擎。
   （SGLang Docker 镜像 lmsysorg/sglang:glm-5.3-flash arm64 实测仅 ~10 tok/s，
   需 --entrypoint python3 覆盖 NVIDIA entrypoint，GLOO/NCCL_SOCKET_IFNAME
   必须显式设置否则 Gloo 用 127.0.0.1 宣告导致 mesh 超时。）
# ===== 故障速查 =====
unique IP 报错 → VLLM_HOST_IP 缺失/指到管理口
启动 10min 整组超时 → timeout 1800
吞吐莫名腰斩 → ①查锁频 ②跑 H0 微基准查低频 ③查是否真并发/串行口径变了
带宽<20Gbps+功耗30W+利用率96% → 正常：decode 是小消息延迟敏感，先 GEMM 后 profile，
   不要先动网络。（profiler 显示瓶颈是大量小 kernel+每轮 ~99 次 AllReduce。）
冷断电/失联恢复后 → 必查：MTU 持久化、挂载竞态、系统时钟（曾慢 1h49m）、license。

# ===== 阶段 4.5：启动前清理（新增，缺了会假失败）=====
1) 先清残留：`pkill -9 -f "vllm serve"` **不够**——GPU worker 进程名是
   `VLLM::Worker_TP*`，不含 "vllm serve"，pkill 匹配不到，它们会占着显存，新起时
   报 "Free memory on device cuda:0 (...) is less than desired ... Decrease
   GPU memory utilization"。要按进程名清：`pkill -9 VLLM::Worker_TP; pkill -9 VLLM::EngineCore`
   或用 `nvidia-smi --query-compute-apps=pid` 逐 PID kill。
2) 若装了 legacy ray-client.service 加入其它集群(如指向 192.168.32.2:6379)，
   先把 `systemctl disable --now ray-client.service` 停掉，否则残留 Ray 与 vLLM mp
   争资源/端口。
3) ⚠️ 命令行写 `pkill -f "vllm serve"` 时，该命令本身会匹配到正在执行的 shell
   （cmdline 含 "vllm serve"）把自己杀掉导致 SSH 无输出——用 `[v]llm` 中括号或
   具体进程名，别用裸字符串。

# ===== 完成报告格式 =====
输出：H0/N1/N2/S1/P1/T 各门结果 + 定版脚本与参数 + 三件套基线数字 +
遗留风险清单。禁止在 H0 未过时输出任何"优化收益"结论。
```
