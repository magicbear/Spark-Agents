# AGENTS 提示词 · DGX Spark 四机 Qwen（SGLang）部署与优化操作手册（整合版）

> **版本标识**：Qwen3.8-Flash-Next-FP8 / Qwen4Exp，2026-09-09 终版。  
> **适用硬件**：4 台 DGX Spark（GB10 / ARM64: `192.168.32.193 / 195 / 197 / 199`）。  
> **服务端口**：集群对外统一监听端口 `30000`。  
> **权重存储**：统一使用 BeeGFS 挂载目录 `/mnt/beegfs/models/Qwen3.8-Flash-Next-FP8`，无需拷贝至本地。

---

## 一、角色、目标与硬规则

### 1. 角色与目标
你是一名集群部署与性能调优 Agent。目标是在 4 台 DGX Spark 节点上以最少误差复现 SGLang 部署 `Qwen3.8-Flash-Next-FP8`，采用验证通过的 **Attention TP2 / PP1 / EP4 / MTP3** 拓扑，并在同口径下完成功能与吞吐量验收。不得复用或干扰 DeepSeek/vLLM 的服务和配置。

### 2. 五大硬规则（必须严格遵守）
1. **探活与监控红线（至关重要）**：
   - **严禁使用 `/health` 做压测前后或高频心跳探活**！SGLang 的 `/health` 接口会发起真实的 dummy generation，会导致 GPU 被占用、KV Cache 被扰动，严重污染吞吐测试数据；
   - **基线探活与健康检查统一使用 `/model_info` 或 `/get_model_info`**。
2. **权重路径规范**：
   - 权重统一使用 BeeGFS 共享挂载点 `/mnt/beegfs/models/Qwen3.8-Flash-Next-FP8`，不要将权重复制到本地，保持各节点存储视图一致。
3. **隔离与安全停止**：
   - 严禁使用 `pkill -f vllm` 或粗暴 kill；各服务均使用独立的容器名（`sglang-qwen-bench`）与 systemd unit（`sglang-qwen-bench.service`）进行隔离生命周期管理。
4. **端口与节点职责分配**：
   - 对外服务端口固定为 `30000`（仅在 Node Rank 0 / 193 暴露对外 API）；
   - Worker 节点（195/197/199，Rank 1~3）必须添加 `--headless`，避免在非协调节点开放多余的 HTTP 监听。
5. **性能评测标准口径**：
   - 统一使用真实壁钟（Wall-clock）吞吐：全组总生成 Token ÷（全组首字发出到全部完成的共同壁钟时间），杜绝排除排队延迟导致的虚高吞吐；3 轮测试取中位数。

---

## 二、固定输入与集群参数

```ini
# 集群拓扑与通信
HEAD = 192.168.32.193:30000
RANK0 = 192.168.32.193
RANK1 = 192.168.32.195
RANK2 = 192.168.32.197
RANK3 = 192.168.32.199
MASTER_PORT = 29571

# 网络与 RDMA
DATA_INTERFACE = enp1s0f1np1
RDMA_HCA = rocep1s0f1,roceP2p1s0f1
ROCE_GID_INDEX = 5

# 模型与镜像
MODEL = /mnt/beegfs/models/Qwen3.8-Flash-Next-FP8
WHEEL_CACHE = /mnt/beegfs/qwen38-wheels
SGLANG_IMAGE_TAR = /mnt/beegfs/images/sglang-dev-qwen38-next-local-arm64.tar
IMAGE = lmsysorg/sglang:dev-qwen38-next-local
IMAGE_ID = sha256:9d2a843c706c74bc259c0d9abf360551eb2734e1e7d255ab012a6965f10480b6
IMAGE_COMMIT = 4ccff141db

# 实验目录
EXPERIMENT_ROOT = /data/clawdata/experiments/spark-qwen-topology-bench-20260908
```

---

## 三、拓扑结论与参数映射

### 1. 推荐生产拓扑：Attention TP2 / PP1 / EP4 / MTP3
SGLang 开启 DP-Attention 时，CLI 的 `--tp-size` 代表整个模型并行组，注意力有效 TP 为 `tp_size / dp_size`。

| 配置项 | 推荐参数值 | 作用与机制说明 |
| :--- | :--- | :--- |
| **并行拓扑** | `--tp-size 4 --dp-size 2 --enable-dp-attention --pp-size 1 --ep-size 4` | 4 节点跨机，有效 Attention TP=2，MoE EP=4 |
| **投机加速** | `--speculative-algorithm NEXTN --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4` | 原生 MTP3 投机采样，每步生成 4 个草稿 Token |
| **跨网同步优化** | `--speculative-skip-dp-mlp-sync` | **关键优化**：消除 DP-Attention 与投机解码间的冗余跨机 MLP 同步屏障，降低 step 延迟 |
| **Mamba 缓存** | `--max-mamba-cache-size 20` | **切勿低于 20**！DP split 会将缓存均分，20 刚好能满足 4 并发每请求 5 个状态槽的要求 |
| **上下文与图** | `--context-length 1048576 --cuda-graph-max-bs-decode 4` | 1M 上下文上限，CUDA Graph 解码批大小 4 |
| **显存与分配** | `--mem-fraction-static 0.72` | 适度预留动态内存，配合 PyTorch 内存参数根治驱动报警 |

### 2. 性能实测基准（2026-09-09 热态复测）
- **C1（单并发）**：单流生成速度 **47.89 tok/s**，TTFT **0.26s**
- **C2（双并发）**：单流 **49.5 tok/s**，聚合吞吐 **98.17 tok/s**
- **C4（四并发）**：单流 **41.15 tok/s**，聚合吞吐 **148.56 ~ 153.24 tok/s**，TTFT **0.228s**
- 对比同节点 vLLM TP4 基线（141.07 tok/s）：SGLang 聚合吞吐高 **5.3% ~ 8.6%**，单流解码高 **8.5% ~ 10.4%**。

### 3. 已知禁用拓扑与失败归因
- ❌ **传统 CLI TP4 / EP4（无 DP-Attention）**：被 GB10 / SM121 的 QSA Head Layout 限制击退（仅接受 TP1 24Q/2KV 或 TP2 12Q/1KV）。
- ❌ **TP1 / PP4 / EP2 / EP1**：在权重加载阶段报 `ZeroDivisionError` 或 `unsupported PLE weight layout`。
- ❌ **TP2 / PP2 / EP2**：PP 后续 stage 无法接收并解析 PLE 的 `shard_N` 嵌入表布局。

---

## 四、快速启动与控制（控制脚本方式）

集群控制已封装在 `/data/clawdata/experiments/spark-qwen-topology-bench-20260908`：

```bash
cd /data/clawdata/experiments/spark-qwen-topology-bench-20260908

# 1. 生成并同步 4 机脚本
python3 sglang_matrix_control.py sglang-attntp2-ep4-mtp3c4 prepare

# 2. 启动 4 机集群（启动顺序：195 -> 197 -> 199 -> 193）
python3 sglang_matrix_control.py sglang-attntp2-ep4-mtp3c4 start

# 3. 检查 4 机运行状态
python3 sglang_matrix_control.py sglang-attntp2-ep4-mtp3c4 status
```

---

## 五、单机分步部署命令（底层排障与原生执行）

若需手动排查或单步启动，在各节点执行以下标准化环境与命令：

### 1. 通用环境变量（4 机均需导入）
```bash
export NCCL_IB_HCA=rocep1s0f1,roceP2p1s0f1
export NCCL_SOCKET_IFNAME=enp1s0f1np1
export GLOO_SOCKET_IFNAME=enp1s0f1np1
export NCCL_IB_GID_INDEX=5
export SGLANG_HOST_IP=$(ip -4 addr show enp1s0f1np1 | awk '/inet /{print $2}' | cut -d/ -f1)
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
# 根治驱动 NV_ERR_NO_MEMORY 的关键内存分配配置：
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512
```

### 2. 启动命令（各节点命令差异）

#### 协调节点 Head Node（192.168.32.193, Rank 0，最后启动）：
```bash
docker run --rm --name sglang-qwen-bench \
  --gpus all --network host --ipc host --privileged --ulimit memlock=-1 \
  -v /mnt/beegfs:/mnt/beegfs:ro \
  -e NCCL_IB_HCA -e NCCL_SOCKET_IFNAME -e GLOO_SOCKET_IFNAME -e NCCL_IB_GID_INDEX \
  -e SGLANG_HOST_IP -e SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN -e PYTORCH_CUDA_ALLOC_CONF \
  lmsysorg/sglang:dev-qwen38-next-local python3 -m sglang.launch_server \
  --model-path /mnt/beegfs/models/Qwen3.8-Flash-Next-FP8 \
  --served-model-name /mnt/beegfs/models/Qwen3.8-Flash-Next-FP8 \
  --host 0.0.0.0 --port 30000 \
  --dist-init-addr 192.168.32.193:29571 --nnodes 4 --node-rank 0 \
  --tp-size 4 --dp-size 2 --enable-dp-attention --pp-size 1 --ep-size 4 \
  --moe-a2a-backend none --mem-fraction-static 0.72 \
  --max-running-requests 4 --max-mamba-cache-size 20 \
  --speculative-algorithm NEXTN --speculative-num-steps 3 \
  --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 \
  --speculative-skip-dp-mlp-sync \
  --cuda-graph-max-bs-decode 4 \
  --no-ple-offload-embedding --context-length 1048576 \
  --chunked-prefill-size 4096 --mamba-ssm-dtype bfloat16 \
  --reasoning-parser auto --tool-call-parser qwen3_coder
```

Agent/函数调用场景必须保留 `--tool-call-parser qwen3_coder`。该 checkpoint 的
chat template 使用 `<tool_call><function=...><parameter=...>` XML 格式，正好对应
SGLang 的 `qwen3_coder` detector。当前镜像不提供 `--enable-auto-tool-choice`
启动参数；这是 vLLM 参数，不能原样加到 SGLang。SGLang 配置 parser 后即可接受
OpenAI 请求中的 `tools` 及 `tool_choice=auto/required`。

端到端验收：

```bash
cd /data/clawdata/experiments/spark-qwen-topology-bench-20260908
python3 tool_call_smoke.py
```

脚本测试非流式 `tool_choice=required`、非流式 `tool_choice=auto` 和流式
`tool_choice=required`。三项都必须满足：`finish_reason=tool_calls`、函数名和参数
JSON 正确、普通 `content` 中没有残留 `<tool_call>` XML；只看到模型生成 XML 文本
不能判定 Agent 功能可用。

启用 parser 后的 C4 回归结果为 143.20 / 151.84 / 153.50 tok/s，三轮中位数
**151.84 tok/s**，请求 decode 中位数 **41.66 tok/s**，TTFT 中位数
**0.228s**；工具解析没有造成可见吞吐回退。

#### 工作节点 Worker Nodes（195 / 197 / 199, Rank 1/2/3，先启动）：
在上述命令中替换 `--node-rank 1`（或 `2`、`3`），并追加 `--headless` 参数。

---

## 六、就绪探活、预热与验收

### 1. 就绪探测（纯净探活）
四机日志中出现：
- `Load weight begin.` -> `Multi-thread loading shards` 完成
- `Engine startup timings (...)`
- `The server is fired up and ready to roll!`

执行探测命令（**切勿调用 /health**）：
```bash
curl -sS http://192.168.32.193:30000/model_info
curl -sS http://192.168.32.193:30000/v1/models
```

### 2. 自动 Warm-up（消除首轮 15.7s TTFT 惩罚）
探活通过后，发送 1 轮单请求将 Triton 算子与 Graph 编译完毕：
```bash
curl -sS http://192.168.32.193:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "/mnt/beegfs/models/Qwen3.8-Flash-Next-FP8", "messages": [{"role": "user", "content": "Hello"}], "max_tokens": 16}' > /dev/null
```

### 3. 验收基准测试
```bash
cd /data/clawdata/experiments/spark-qwen-topology-bench-20260908

# 1. 基础功能 Canary（验证 37*19 准确输出 703）
python3 -c '
import urllib.request, json
req = urllib.request.Request("http://192.168.32.193:30000/v1/chat/completions",
    data=json.dumps({"model":"/mnt/beegfs/models/Qwen3.8-Flash-Next-FP8","messages":[{"role":"user","content":"37 * 19 = ?"}],"temperature":0}).encode(),
    headers={"Content-Type":"application/json"})
resp = json.loads(urllib.request.urlopen(req).read())
print("Result:", resp["choices"][0]["message"]["content"])
'

# 2. 并发扫描与聚合吞吐验收（C1/C2/C4）
python3 concurrency_sweep.py sglang-attntp2-ep4-mtp3-concurrency-sweep-$(date +%Y%m%d).jsonl
```

---

## 七、集群停止与清理收敛

```bash
cd /data/clawdata/experiments/spark-qwen-topology-bench-20260908

# 优雅停止集群
python3 sglang_matrix_control.py sglang-attntp2-ep4-mtp3c4 stop

# 收集四机运行与错误日志
python3 sglang_matrix_control.py sglang-attntp2-ep4-mtp3c4 collect

# 若遇节点失联，执行底层强行停止：
ansible -i "192.168.32.193,192.168.32.195,192.168.32.197,192.168.32.199," all -m shell -a "systemctl stop --no-block sglang-qwen-bench.service || docker rm -f sglang-qwen-bench"
```
