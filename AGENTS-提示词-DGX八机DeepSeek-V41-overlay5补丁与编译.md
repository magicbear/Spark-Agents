# AGENT PROMPT · DeepSeek-V4.1-Flash overlay5 GB10 补丁应用与镜像编译

> 用途：把本文件整体作为提示词交给 AI Agent。目标只有两件：①把 7 文件补丁打到 `vllm-dsv41:overlay5`；②必要时从零编译该镜像。
> 补丁目录：`docs/Spark-Agents/dsv41-overlay5-patch/`（统一 diff + 整文件替换 + `apply.sh`）。
> 部署与起服见《AGENTS-提示词-DGX八机DeepSeek-V41-Flash-vLLM.md》。本文件不负责切 jovian。
> 底本：vLLM `0.1.dev20904+g179dd0fa9`（已删除的 `dsv41-feat` @ `179dd0fa9`），镜像 `vllm-dsv41:overlay5`。
> 生产补丁来自 `/mnt/beegfs/vllm-dsv41-port/patch/`（`engram.py` 比 recipe 仓库多 DP head-shard）。

```text
# ===== 角色 =====
你是 vLLM overlay5 补丁/编译 Agent。铁律：①补丁针对 overlay5 原文件，不要打到
vLLM main / jovian；②生产用 bind-mount，不要把 7 文件 bake 进镜像（方便回滚）；
③运行时 JIT 禁止 MAX_JOBS>4，否则 GB10 主存打满看门狗复位；④一次只改一个变量。

# ===== 输入 =====
PATCH_DIR = docs/Spark-Agents/dsv41-overlay5-patch
FILES     = $PATCH_DIR/files          # 7 个 .py + mounts.txt
UNIFIED   = $PATCH_DIR/overlay5-gb10.patch
APPLY     = $PATCH_DIR/apply.sh
SITE      = /usr/local/lib/python3.12/dist-packages/vllm
IMAGE     = vllm-dsv41:overlay5
NIGHTLY   = vllm/vllm-openai:nightly-8a728663c1c3eeace834a95f5654fa653cc1998c
FI_SHA    = 07869c61ba581e6d6b8ad8d142f4a6c89b707cc1   # FlashInfer 0.7.0rc1
CUTLASS   = b46b16d003484063bca4ed365e44095c4c6ed633
CCCL      = 16bd510c9b712e82b0ab6cbb630d8e29ba1f7116
SPDLOG    = c3aed4b68373955e1cc94307683d44dca1515d2b
ARCH      = 12.1a   # GB10 SM 12.1

# ===== 补丁清单（7 文件，相对 vllm/）=====
engram.py              models/deepseek_v4_1/common/engram.py
  Engram 表不进 host RAM：safetensors preadv 按 rank 行偏移读 + 可选 DSV41_ENGRAM_DIR
  本地 NVMe 行；rank-offset 修正；共享读池；EngramDiskStager；DP head-shard /
  dummy_hashes / gather_engram_hashes（8 机生产版）。
model_state.py         models/deepseek_v4_1/nvidia/model_state.py
  Engram 行在 prepare_inputs 预取，forward 无 host round-trip，才能进 CUDA graph。
weight_utils.py        model_executor/model_loader/weight_utils.py
  loader 跳过两张 Engram 表（203GB）。
attention.py           models/deepseek_v4_1/attention.py
  SM12x page；indexer cache 每页 64 state（DeepGEMM paged MQA 只收 32/64）。
flashinfer_sparse.py   models/deepseek_v4_1/nvidia/flashinfer_sparse.py
  64-state compressed page；64-token SWA backend（SM120 sparse MLA 唯一 page size）。
sparse_swa.py          v1/attention/backends/mla/sparse_swa.py
  get_swa_block_size() hook。
sparse_attn_indexer.py model_executor/layers/sparse_attn_indexer.py
  SM12x decode top-k 走 top_k_per_row_decode（persistent_topk 在 GB10 会打满 48 SM）。

sha256 见 $PATCH_DIR/SHA256SUMS。生产 md5：
  engram.py 5fcc1ac8  model_state.py 0a14bee6  weight_utils.py 7e1027f1
  attention.py da9ef196  flashinfer_sparse.py af0f8447
  sparse_swa.py cc419353  sparse_attn_indexer.py a9b73756

# ===== 阶段 A：应用补丁（已有 overlay5 镜像时走这条，生产定版）=====
三种等价方法，推荐 A1。

A1 bind-mount（生产，不改镜像）
  docker run ... $(bash $APPLY bind $FILES $SITE) ...
  或手写 7 条 -v，与 mounts.txt 一致。例：
    -v $FILES/engram.py:$SITE/models/deepseek_v4_1/common/engram.py:ro
  ✅ 验收门 A1：容器内 md5 与上表一致；镜像层内原文件仍是未打补丁的 md5
     （engram 原 9a3b8cd9）。回滚 = 去掉 -v。

A2 patch -p1（改源码树或容器内 site-packages）
  # 根目录必须含 vllm/ 包：
  bash $APPLY patch $UNIFIED /usr/local/lib/python3.12/dist-packages
  # 或 vLLM 源码根：
  bash $APPLY patch $UNIFIED /path/to/vllm
  统一 diff 已对 overlay5 原文件 dry-run 通过。
  ❌ 不要打到 vLLM main / jovian（包名 deepseek_v41、无 flashinfer_sparse.py）。

A3 整文件覆盖
  bash $APPLY copy $FILES $SITE

✅ 验收门 A：
  python3 -c "import hashlib,pathlib
  p=pathlib.Path('$SITE')
  print(hashlib.md5((p/'models/deepseek_v4_1/common/engram.py').read_bytes()).hexdigest()[:8])"
  期望 5fcc1ac8。缺文件或 md5 不对禁止起服。

# ===== 阶段 B：从零编译 overlay5（仅镜像丢失时）=====
overlay5 = overlay1（dsv41-feat python + sm121 扩展）
         + overlay3（FlashInfer 0.7.0rc1 源码安装）
         + overlay4（mxfp8_gemm_cutlass_sm120 预编译）
         + overlay5（sparse_mla_sm120 按运行时 env 再预热，禁 debug JIT）
每台节点本地 build；不要把中间层放到共享盘。MAX_JOBS=2~4。

B0 前提
  aarch64、Docker、CUDA 13.0 nvcc、主机内存 ≥80GiB 空闲。
  dsv41-feat 分支已从 GitHub 消失。python 树来源二选一：
    ① 从现存 overlay5 镜像导出：
       docker create --name x vllm-dsv41:overlay5
       docker cp x:/usr/local/lib/python3.12/dist-packages/vllm ./vllm
       docker rm x
    ② 历史 nightly + 当时 dsv41-feat checkout @ 179dd0fa9（现已不可拉）。
  ❌ 用 vLLM main 替换 python 树会与 b12x/包名全部不兼容。

B1 overlay1
  FROM $NIGHTLY
  COPY vllm/ $SITE/
  清 __pycache__，python3 -c "import vllm; print(vllm.__version__)"
  期望 0.1.dev20904+g179dd0fa9。
  然后在带源码的容器里只编 _C_stable_libtorch（sm_121a）：
    TORCH_CUDA_ARCH_LIST=12.1a
    cmake -S /src -B /src/build -G Ninja -DCMAKE_BUILD_TYPE=Release
      -DVLLM_TARGET_DEVICE=cuda -DNVCC_THREADS=2
    cmake --build /src/build --target _C_stable_libtorch -j20
  参考脚本：/mnt/beegfs/vllm-dsv41-ref/build/build_stable_ext.sh
  docker tag 为 vllm-dsv41:overlay1。
  ✅ 门 B1：import vllm 成功；_C_stable_libtorch*.so 存在。

B2 overlay3 = overlay1 + FlashInfer 0.7.0rc1
  pip uninstall -y flashinfer-jit-cache flashinfer-cubin flashinfer-python
  拉 FI/CUTLASS/CCCL/SPDLOG 四个 SHA（见输入），
  BUILD_NVEP=0 FLASHINFER_BUILD_NO_PIP=1 pip install --no-deps --no-build-isolation .
  ENV VLLM_HAS_FLASHINFER_CUBIN=1
  断言：
    from flashinfer.mla import supported_sparse_mla_sm120_configs as f
    assert f()['dsv4'].supports_decode(num_heads=16, topk=1152)
  完整 Dockerfile：/mnt/beegfs/vllm-dsv41-ref/build/build_overlay3.sh
  ✅ 门 B2：flashinfer 0.7.0rc1；supports_decode(16,1152)=True。
  ⚠️ overlay3 里用 FLASHINFER_JIT_VERBOSE 预热的 sparse_mla 是 debug 构建，
     运行时 cache key 对不上，必须被 overlay5 覆盖。不要把 overlay3 当生产镜像。

B3 overlay4 = overlay3 + mxfp8 GEMM 预编译
  FLASHINFER_CUDA_ARCH_LIST=12.1a MAX_JOBS=4 FLASHINFER_NVCC_THREADS=1
  python3 -c "from flashinfer.jit.gemm import gen_gemm_sm120_module_cutlass_mxfp8 as gen;
    spec=gen(); (spec.build(verbose=True) if hasattr(spec,'build') else spec.build_and_load())"
  产物：/root/.cache/flashinfer/0.7.0rc1/121a/cached_ops/mxfp8_gemm_cutlass_sm120/
  ❌ MAX_JOBS=22 会在 8 机同时编译时打满主机内存（boot 3 看门狗复位）。
  脚本：build_overlay4.sh。✅ 门 B3：该目录存在且镜像 tag 为 overlay4。

B4 overlay5 = overlay4 + sparse_mla 按运行时 env 预热
  必须与起服 env 一致，否则 cache miss 又会运行时编译：
    FLASHINFER_CUDA_ARCH_LIST=12.1a
    TORCH_CUDA_ARCH_LIST=12.1a
    FLASHINFER_DISABLE_VERSION_CHECK=1
    VLLM_HAS_FLASHINFER_CUBIN=1
    FLASHINFER_NVCC_THREADS=1
    MAX_JOBS=4
    且不要设置 FLASHINFER_JIT_VERBOSE / JIT_DEBUG / JIT_LINEINFO
  COPY $PATCH_DIR/prewarm5.py 后 RUN python3 /tmp/prewarm5.py
  校验（有 GPU）：
    docker run --rm --gpus all -e FLASHINFER_CUDA_ARCH_LIST=12.1a \
      -e TORCH_CUDA_ARCH_LIST=12.1a -e FLASHINFER_DISABLE_VERSION_CHECK=1 \
      -e VLLM_HAS_FLASHINFER_CUBIN=1 -e MAX_JOBS=2 -e FLASHINFER_NVCC_THREADS=1 \
      -v $PATCH_DIR/verify5.py:/v.py:ro --entrypoint python3 \
      vllm-dsv41:overlay5 /v.py
    期望两行 VERIFY … HIT，且无 Building JIT。
  脚本：build_overlay5.sh + prewarm5.py + verify5.py（已拷进补丁目录）。
  ✅ 门 B4：镜像 vllm-dsv41:overlay5；verify HIT；仍未 bake 7 文件补丁。

# ===== 阶段 C：补丁 + 镜像一起验证 =====
1) 镜像是 overlay5，补丁是 bind-mount（A1）。
2) import：
   from flashinfer.mla import supported_sparse_mla_sm120_configs as f
   print(f()['dsv4'].supports_decode(16,1152))   # True
3) 7 文件 md5 对上。
4) 起服后日志无 runtime JIT、无 "No common block size for 64"、
   无 persistent_topk 崩、无 host OOM。
5) --block-size 128 仍然要带（补丁改的是 indexer/SWA page，不是 CLI 默认值）。
✅ 门 C：真实 chat 返回；count 类 prompt DSpark 接受长度接近 6.0。

# ===== 不要做 =====
❌ 把补丁打进 jovian/B12X 或 vLLM main。
❌ 运行时 FLASHINFER_JIT_VERBOSE=1（cache 对不上，下一次又现场编译）。
❌ 追求把 Engram 整表装进 RAM（188GiB，GB10 只有 ~121GiB）。
❌ pkill -f "vllm serve"（会杀掉当前 shell）。
❌ 编译时 MAX_JOBS 开满。

# ===== 完成报告 =====
A 门 md5、用的方法（bind/copy/patch）、B 门是否重建镜像、C 门一条真实请求。
未过 A 禁止起服。
```

## 目录

```
docs/Spark-Agents/dsv41-overlay5-patch/
  overlay5-gb10.patch   # 对 overlay5 原文件的 unified diff（patch -p1）
  apply.sh              # bind | copy | patch
  files/                # 生产用整文件 + mounts.txt
  SHA256SUMS
  prewarm5.py verify5.py
```

生产 8 机起服仍用 bind-mount，启动脚本见 `/tmp/launch-overlay5.sh`。
