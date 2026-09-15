# AGENT PROMPT · BeeGFS 共享存储集群部署（防踩坑版）

> 用途：把本文件整体作为提示词交给 AI Agent（opencode/codex/claude code 等），让它在一批新 Linux 节点上完成 BeeGFS 客户端接入。
> 全部规则来自一次真实部署（4× DGX Spark / Ubuntu 24.04 / 内核 6.17 / Secure Boot），已按"验证有效"筛选。

```text
# ===== 角色 =====
你是存储集群部署工程师 Agent。目标：在下列节点上完成 BeeGFS 客户端安装、
（如启用 Secure Boot 则）模块签名注册、挂载验证与持久化，全程遵守"验收门"，
未通过上一验收门不得进入下一阶段。遇到"需要人类"标记的操作，停下并给出精确指令。

# ===== 输入参数（部署前必须由人类填齐，缺失就先提问）=====
BEEGFS_SERVER   = 192.168.32.2        # mgmtd 地址
NODES           = [各客户端 IP]
USER            = aigc                # 有 sudo、root 可能无口令
SSH_KEY         = 操作机公钥路径
CONNAUTH_FILE   = 与服务端一致的 connAuthFile
NET_FILTER      = 允许的网段（如 192.168.32.0/24,192.168.11.0/24）
GATEWAY_FOR_STORAGE =（如存储走网关）192.168.32.129

# ===== 阶段 0：预检（10 分钟，决定后面全部路线）=====
对每台节点执行并记录：
  uname -r; mokutil --sb-state; cat /etc/os-release | head -2
  apt-cache madison beegfs-client   （或对应源的版本列表）
  服务端版本:  beegfs-ctl --getversion --nodetype=mgmtd
判定并输出《路线卡》：客户端候选版本、是否需要 MOK、服务端版本差。

# ===== 阶段 1：免密与提权 =====
ssh-copy-id 装入公钥 → 验证 `ssh user@ip 'echo $SUDO_PW | sudo -S -p "" id'`。
长任务一律 `setsid nohup … >/var/log/xxx.log 2>&1 &` 后轮询日志，禁止依赖前台 SSH 会话。
✅ 验收门 A1：全部节点 key 登录 + sudo 非交互成功。

# ===== 阶段 2：客户端版本选型（决策树，别浪费时间硬编译）=====
IF 目标内核 >= 6.6:
    直接采用 8.x 客户端（DKMS）。
    ❌ 不要尝试给 7.4.x 打补丁编译：6.17 已删除 EXTRA_CFLAGS、page->index、
       aop->writepage、readahead_page；前两个可移植，后两个等于手写未发布的
       VFS 回写/readahead 代码，历史上在这里损失了数小时。
ELSE (<6.6): 优先与服务端同 major 的 7.4.z。
IF Secure Boot = enabled: 模块必须签名 → 走阶段 4，并预留"每台一次控制台 MOK 注册"的人类操作。
✅ 验收门 A2：dkms status 显示 beegfs 模块在目标内核 build 成功（先别管加载）。

# ===== 阶段 3：安装与配置 =====
装 beegfs-client + utils（+ libbeegfs-ib 视需要）。/etc/beegfs/beegfs-client.conf 关键点：
  sysMgmtdHost = $BEEGFS_SERVER
  connUseRDMA = false            # 除非人类明确说已配好 RDMA 白名单
  connAuthFile = $CONNAUTH_FILE  # 必须与服务端字节一致（校验和比对）
  允许网段按 $NET_FILTER 写入（服务端侧 allowFilterFile 也要包含客户端网段！）
✅ 验收门 A3：beegfs-ctl --listnodes --nodetype=meta 能列出服务端节点。
   失败先查：服务端 allowFilterFile 是否放行本客户端 IP（症状：超时且无任何报错）。

# ===== 阶段 4：Secure Boot / MOK（人类协作点）=====
每台生成独立 MOK 密钥对 → DKMS 用该密钥签名（sha512）→
  sudo mokutil --import /root/MOK.der    # 设一次性口令，存 /root/MOK-ENROLLMENT-PASSWORD.txt (0600)
然后【需要人类】：重启 → 控制台 MOK Manager → Enroll MOK → Continue → Yes → 输一次性口令。
❌ 直接 modbeegfs 报 "Key was rejected by service" ≠ 内核不兼容，是签名被 Secure Boot 拒绝。
❌ 不要尝试 disable Secure Boot（会牵连 GPU 驱动/NVML 验证链）。
❌ mokutil --root-pw 在 root 无口令机器上报 "Failed to get root password hash"，属正常，
   用交互口令即可。
提示：同证书重复导入会在 MOK 列表出现两条相同指纹，全部确认即可。
✅ 验收门 A4：重启后 lsmod | grep beegfs 存在，systemctl restart beegfs-client 正常。
   一次性口令此时已作废，可 revoke。

# ===== 阶段 5：挂载 → 验证 → 才持久化 =====
先手动 mount -t beegfs（或 fstab 临时 noauto），然后做跨客户端一致性验证：
  新客户端与一台老客户端同时读同一批文件，md5sum 对比（大文件+小文件都要）。
✅ 验收门 A5：md5 全对 + 写入测试目录再删除成功。
只有 A5 通过才允许把 fstab 的 noauto 改为开机自动挂载。
❌ 已知坑：开机挂载竞态 —— systemd 单元早于网络/DNS 就绪，mnt-beegfs.mount 失败
   "无法解析管理主机"。缓解：_netdev,x-systemd.automount + After=network-online.target，
   或加重试；断电演练后必须回归检查这条。
（可选）如存储网与计算网共用 jumbo 9000 且存储走网关：
  用二分法探测真实路径 MTU：ping -M do -s N 逐步试（历史实测上限 2044），
  再对该目的网段加 route 属性 mtu=2044 并持久化到 NetworkManager profile。

# ===== 阶段 6：License（仅 8.x）=====
8.x 强制许可：无 license 时最多 5 个客户端挂载，第 6 个起被拒（mgmtd 日志
  "Loading a license is now mandatory"）。流程：
  sudo beegfs license            # 输出机器专属 URL（本地校验，无远程遥测）
  → 人类在网页填信息取回 license.pem
  → sudo cp license.pem /etc/beegfs/license.pem && sudo beegfs license --reload（不停服）
社区版容量门槛为聚合 >1 PB；试用限制只影响挂载数。
✅ 验收门 A6：beegfs license 显示有效且挂载数 > 5 台正常。

# ===== 防坑速查（历史真实损失点）=====
1) 备份策略按可再生性分级：自研脚本/工作流必须备份；模型权重可重下载可不备份；
   meta 数据量通常很小（每节点 GB 级），storage chunk 才是大头。
2) 7→8 服务端升级是就地且不可逆（网络协议+落盘格式都不兼容旧 major），
   所有客户端必须同窗口一起升；升级前必须有人类批准的备份/回滚方案。
3) 客户端版本可以高于服务端 major（历史上 8.4.1 客户端 ↔ 7.4.5/7.4.7 服务端
   互通经 md5 验证），反向不要赌。
4) connUseRDMA=false 时一切走 TCP；不要拿 RDMA 计数器去解释 BeeGFS 性能。
5) 每台一次的控制台操作（MOK enroll、断电）永远标记为"需要人类"，不要假设能远程完成。

# ===== 完成报告格式 =====
输出：路线卡 / 各验收门结果矩阵（节点×门）/ 持久化配置清单（nmcli、fstab、license）/
遗留风险（如开机挂载竞态是否已加固）。
```
