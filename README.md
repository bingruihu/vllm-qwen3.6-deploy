# Qwen3.6 系列模型 vLLM 部署文档

基于 Docker 部署 **Qwen3.6-27B (Dense)** 和 **Qwen3.6-35B-A3B-FP8 (MoE)** 两个模型,使用 vLLM 提供 OpenAI 兼容 API 及 Anthropic Messages API 服务,支持远程 Windows 客户端访问。

---

## 一、模型概览

| 特性 | Qwen3.6-27B (Dense) | Qwen3.6-35B-A3B-FP8 (MoE) |
|------|---------------------|---------------------------|
| model_type | `qwen3_5` | `qwen3_5_moe` |
| 架构 | Dense (64 层) | MoE (40 层, 256 专家, 8 激活 + 1 共享) |
| 量化 | BF16 原始精度 | FP8 量化 |
| 上下文长度 | 262,144 (256K) | 262,144 (256K) |
| vision 编码器 | 有 (multimodal) | 有 (multimodal) |
| 权重文件 | 15 个 safetensors 分片 | 40 个 layers + mtp + outside |
| 权重显存 | ~54GB | ~30GB |

> ℹ️ **架构说明**: Qwen3.6 系列使用 `qwen3_5` 架构 (Gated DeltaNet + Gated Attention 混合注意力),与原版 Qwen3 (`qwen3`) 不同。vLLM v0.23.0 原生支持,不需要 `--trust-remote-code`。

### 何时用哪个模型

| 场景 | 推荐模型 | 原因 |
|------|----------|------|
| 通用文本任务、快速推理 | **27B Dense** | 推理更快、显存更充裕 |
| 复杂推理、工具调用 | **35B-A3B-FP8** | MoE 稀疏能力、FP8 显存效率高 |
| 高并发场景 | **27B Dense** | 单卡可承载更多请求 |
| 低显存环境 (32GB) | **35B-A3B-FP8** | FP8 量化大幅降低显存需求 |

### 目标环境

| 项目 | 值 |
|------|-----|
| 服务器 IP | `YOUR_SERVER_IP` |
| 操作系统 | Linux 6.8.0 (Ubuntu) |
| Docker 版本 | 28.2.2 |
| vLLM 镜像 | `vllm/vllm-openai:v0.23.0-ubuntu2404` (本地已存在) |
| 目标 GPU | GPU 5 (NVIDIA RTX PRO 6000 Blackwell, 97GB) |
| 对外端口 | `18000` (35B-A3B-FP8), `28000` (27B) |

---

## 二、目录结构

```
/home/$USER/WorkStation/Vllm/
├── README.md                        # 本文档
├── load-local.sh                    # 切换模型为本地 vLLM 的脚本
├── load-glm.sh                      # 切换模型为 GLM 的脚本
└── models/
    ├── Qwen36-27B/                  # Qwen3.6-27B Dense (BF16, 41GB, 15 分片)
    │   ├── config.json
    │   ├── model-00001-of-00015.safetensors
    │   └── ... (15 个 safetensors 分片 + tokenizer + 配置文件)
    └── Qwen36-35B-A3B-FP8/          # Qwen3.6-35B-A3B-FP8 MoE (FP8, ~30GB)
        ├── config.json
        ├── layers-*.safetensors (40 个)
        ├── mtp.safetensors
        ├── outside.safetensors
        ├── tokenizer.json
        ├── chat_template.jinja
        └── ...
```

---

## 三、部署步骤

### 1. 准备镜像 (本地已有可跳过)

```bash
docker pull vllm/vllm-openai:v0.23.0-ubuntu2404
```

### 2. 部署 Qwen3.6-35B-A3B-FP8 (MoE 模型)

> 该命令使用 `docker run` 创建一个**持久化容器** (不带 `--rm`),之后用 `docker start/stop` 控制启停。

```bash
docker run -d \
  --name qwen-vllm \
  --gpus '"device=5"' \
  --ipc=host \
  --restart=no \
  -p 18000:8000 \
  -v /home/$USER/WorkStation/Vllm/models:/models:ro \
  -e VLLM_MOE_FORCE_MARLIN=1 \
  vllm/vllm-openai:v0.23.0-ubuntu2404 \
  --model /models/Qwen36-35B-A3B-FP8 \
  --served-model-name qwen3.6-35b \
  --host 0.0.0.0 \
  --port 8000 \
  --max-model-len 262144 \
  --gpu-memory-utilization 0.95 \
  --quantization fp8 \
  --dtype bfloat16 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3 \
  --enable-prefix-caching \
  --enable-chunked-prefill \
  --max-num-seqs 8 \
  --block-size 32 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":4}'
```

**参数说明:**

| 参数 | 说明 | 适用 |
|------|------|------|
| `-d` | 后台运行 | 通用 |
| `--name qwen-vllm` | 容器名,后续用该名称启停 | 通用 |
| `--gpus '"device=5"'` | 指定 GPU 5; 换卡改此参数 (如 `'"device=7"'`) | 通用 |
| `--ipc=host` | 共享主机共享内存,PyTorch 多进程通信所需 | 通用 |
| `--restart=no` | 不自动重启,完全由手动 start/stop 控制 | 通用 |
| `-p 18000:8000` | 端口映射: 宿主机 18000 → 容器 8000 | 仅 35B-A3B-FP8 |
| `-v .../models:/models:ro` | 只读挂载模型目录 | 通用 |
| `-e VLLM_MOE_FORCE_MARLIN=1` | **强制 MoE 使用 Marlin 量化内核** (FP8 MoE 推理必需) | **仅 35B-A3B-FP8** |
| `--model` | 模型路径 | 各自不同 |
| `--served-model-name` | API 中使用的模型名 | 各自不同 |
| `--max-model-len 262144` | 最大上下文长度 (256K tokens) | 通用 |
| `--gpu-memory-utilization 0.95` | GPU 显存占用比例 | 通用 |
| `--quantization fp8` | 模型量化格式 (FP8) | **仅 35B-A3B-FP8** |
| `--dtype bfloat16` | 计算数据类型 | 通用 |
| `--enable-auto-tool-choice` | 开启自动工具选择 (Claude Code / Agent 必需) | 通用 |
| `--tool-call-parser qwen3_coder` | Qwen3.6 系列工具调用解析器 | 通用 |
| `--reasoning-parser qwen3` | Qwen3 思维链解析器 | 通用 |
| `--enable-prefix-caching` | 启用前缀缓存,加速重复 prompt 推理 | 通用 |
| `--enable-chunked-prefill` | 启用分块 prefill,提高首 token 延迟 | 通用 |
| `--max-num-seqs 8` | 最大并发序列数 | 通用 |
| `--block-size 32` | KV Cache 块大小 (tokens) | 通用 |

> 📌 关于 `--gpus` 语法: 必须使用 `'"device=5"'` (外层单引号 + 内层双引号), 这是 NVIDIA Container Toolkit 的要求。

> 💡 **若要让 Claude Code 接入**: `--enable-auto-tool-choice` 和 `--tool-call-parser qwen3_coder` 两个参数**必不可少**, 否则 Claude Code 无法调用工具。详见 [八、接入 Claude Code](#八接入-claude-code)。

### 3. 部署 Qwen3.6-27B (Dense 模型)

```bash
docker run -d \
  --name qwen27b-vllm \
  --gpus '"device=4"' \
  --ipc=host \
  --restart=no \
  -p 28000:8000 \
  -v /home/$USER/WorkStation/Vllm/models:/models:ro \
  vllm/vllm-openai:v0.23.0-ubuntu2404 \
  --model /models/Qwen36-27B-FP8 \
  --served-model-name qwen3.6-27b \
  --host 0.0.0.0 \
  --port 8000 \
  --max-model-len 262144 \
  --max-num-batched-tokens 8192 \
  --gpu-memory-utilization 0.9 \
  --dtype bfloat16 \
  --quantization fp8 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3 \
  --enable-prefix-caching \
  --enable-chunked-prefill \
  --max-num-seqs 32 \
  --block-size 16 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":4}'
```

> ⚠️ **与 35B-A3B-FP8 的差异**:
> - 容器名: `qwen27b-vllm` (而非 `qwen-vllm`)
> - 端口: `-p 28000:8000` (而非 `-p 18000:8000`)
> - 模型名: `qwen3.6-27b` (而非 `qwen3.6-35b`)
> - **无** `--quantization fp8` (BF16 原始精度)
> - **无** `VLLM_MOE_FORCE_MARLIN` 环境变量 (非 MoE 模型)
> - **无** `--trust-remote-code` (vLLM v0.23.0 原生支持)
>
> **可选**: 如果只需要文本推理 (不需要 vision), 可加 `--language-model-only` 跳过视觉编码器, 节省约 3GB 显存。

### 4. 两模型显存对比

| 模型 | 权重显存 | KV Cache (262K 上下文) | 推荐 GPU | 显存余量 |
|------|----------|----------------------|----------|----------|
| 27B BF16 | ~54GB | 约 8-12GB | GPU 5 (97GB) | ~30GB |
| 35B-A3B FP8 | ~30GB | 约 6-10GB | GPU 5 (97GB) | ~57GB |

两个模型均可用 GPU 5 (97GB Blackwell) 稳定运行。27B 推理更快但权重更大, 35B-A3B-FP8 权重更小且有 MoE 稀疏能力。

---

## 四、服务启停

### 启动服务

```bash
# 35B-A3B-FP8
docker start qwen-vllm

# 27B Dense
docker start qwen27b-vllm
```

### 停止服务

```bash
# 35B-A3B-FP8
docker stop qwen-vllm

# 27B Dense
docker stop qwen27b-vllm
```

### 重启服务

```bash
# 重启单个服务
docker restart qwen-vllm
docker restart qwen27b-vllm

# 同时重启
docker restart qwen-vllm && docker restart qwen27b-vllm
```

### 查看运行状态

```bash
docker ps -a --filter name=qwen-vllm
docker ps -a --filter name=qwen27b-vllm
```

### 查看实时日志 (首次加载模型需 1~5 分钟)

```bash
docker logs -f qwen-vllm       # 35B-A3B-FP8
docker logs -f qwen27b-vllm    # 27B
```

看到以下日志表示服务就绪:
```
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
INFO:     Application startup complete.
```

---

## 五、访问服务

### 5.1 本机验证

```bash
# === 35B-A3B-FP8 (端口 18000) ===

# 查看可用模型
curl http://localhost:18000/v1/models

# 文本对话测试
curl http://localhost:18000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.6-35b",
    "messages": [{"role":"user","content":"你好,介绍一下你自己"}],
    "max_tokens": 256
  }'

# === 27B Dense (端口 28000) ===

# 查看可用模型
curl http://localhost:28000/v1/models

# 文本对话测试
curl http://localhost:28000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.6-27b",
    "messages": [{"role":"user","content":"你好,介绍一下你自己"}],
    "max_tokens": 256
  }'
```

### 5.2 远程 Windows 访问

**前提**: 服务器防火墙需放行 18000 和/或 28000 端口。在服务器执行:
```bash
# 查看/开放防火墙 (若使用 ufw)
sudo ufw allow 18000/tcp
sudo ufw allow 28000/tcp
```

**在 Windows 上访问**, 把 `localhost` 换成服务器 IP:

| 模型 | 模型列表 | 对话接口 | 健康检查 |
|------|----------|----------|----------|
| 35B-A3B-FP8 | `http://YOUR_SERVER_IP:18000/v1/models` | `http://YOUR_SERVER_IP:18000/v1/chat/completions` | `http://YOUR_SERVER_IP:18000/health` |
| 27B | `http://YOUR_SERVER_IP:28000/v1/models` | `http://YOUR_SERVER_IP:28000/v1/chat/completions` | `http://YOUR_SERVER_IP:28000/health` |

### 5.3 通过 SSH 端口隧道远程访问 (推荐)

如果服务器防火墙未放行端口, 或你希望**加密传输**, 可以通过 SSH 隧道将远程服务映射到本地。

#### 5.3.1 使用 MobaXterm 建立隧道

1. 打开 MobaXterm, 新建 SSH 会话, 连接 `YOUR_SERVER_IP`
2. 连接成功后, 左侧侧边栏会自动显示 **Session tunneling** → **Port Forwarding**
3. 点击 **Port forwarding settings**, 添加转发规则:

   | 模型 | Forwarded port | Destination host | Destination port |
   |------|---------------|------------------|------------------|
   | 35B-A3B-FP8 | `18000` | `127.0.0.1` | `18000` |
   | 27B | `28000` | `127.0.0.1` | `28000` |

4. 点击 **OK** 保存

**验证隧道是否生效:**
```bash
# 35B-A3B-FP8
curl http://localhost:18000/v1/models

# 27B
curl http://localhost:28000/v1/models
```

#### 5.3.2 使用命令行 SSH 隧道

如果你有 Git Bash、WSL 或其他支持 ssh 的终端:

```bash
# 35B-A3B-FP8
ssh -N -L 18000:127.0.0.1:18000 YOUR_USERNAME@YOUR_SERVER_IP

# 27B
ssh -N -L 28000:127.0.0.1:28000 YOUR_USERNAME@YOUR_SERVER_IP
```

### 5.3.3 通过隧道配置 Claude Code

隧道建立后, 在 `~/.claude/settings-local.json` 写入:

```json
{
  "env": {
    "ANTHROPIC_BASE_URL": "http://localhost:18000",
    "ANTHROPIC_API_KEY": "dummy",
    "ANTHROPIC_AUTH_TOKEN": "dummy",
    "ANTHROPIC_DEFAULT_HAIKU_MODEL": "qwen3.6-35b",
    "ANTHROPIC_DEFAULT_SONNET_MODEL": "qwen3.6-35b",
    "ANTHROPIC_DEFAULT_OPUS_MODEL": "qwen3.6-35b",
    "CLAUDE_CODE_ATTRIBUTION_HEADER": "0",
    "API_TIMEOUT_MS": "3000000"
  }
}
```

> 📌 如果要使用 27B 模型, 将 `ANTHROPIC_BASE_URL` 改为 `http://localhost:28000`, 将 `ANTHROPIC_DEFAULT_*_MODEL` 改为 `qwen3.6-27b`。`settings-local.json` 一次只能指向一个服务, 需切换时用 `load-glm.sh` / `load-local.sh` 脚本。

### 5.4 反向隧道 / 第三方客户端

> 与原来相同, 详见 5.4 反向隧道和 5.5 第三方客户端章节 (略, 内容不变)。

---

## 六、常见问题

### 6.1 SSH 隧道连接断开后怎么办

隧道只是临时连接, 断开后重新建立即可。服务本身不受影响, 客户端会因连接 `localhost` 失败而报错, 隧道恢复后自动恢复。

### 6.2 更换 GPU

默认使用 GPU 5 (97GB Blackwell)。两个模型的 GPU 选择建议:

**35B-A3B-FP8 各 GPU 显存适配:**

| GPU | 型号 | 显存 | 是否能装下 ~30GB 模型 |
|-----|------|------|----------------------|
| 0–3 | RTX 4090 | 24GB | ❌ 装不下 |
| 4 | RTX PRO 6000 Blackwell | 97GB | ✅ 可用 |
| 5 | RTX PRO 6000 Blackwell | 97GB | ✅ **推荐** |
| 6–7 | RTX 4090 | 24GB | ❌ 装不下 |

**27B BF16 各 GPU 显存适配:**

| GPU | 型号 | 显存 | 是否能装下 ~54GB 模型 |
|-----|------|------|----------------------|
| 0–3, 6–7 | RTX 4090 | 24GB | ❌ 装不下 |
| 4, 5 | RTX PRO 6000 Blackwell | 97GB | ✅ **可用 (需 97GB)** |

> 💡 若启动日志出现 `CUDA out of memory`, 说明所选 GPU 显存不足, 请换用 97GB 的 Blackwell 卡 (GPU 4 或 5)。

### 6.3 端口冲突

```bash
# 查看占用
ss -tlnp | grep -E '(18000|28000)'

# 35B-A3B-FP8 默认 18000, 27B 默认 28000, 互不冲突
# 如需更换, 重新创建容器时修改 -p 参数即可
```

### 6.4 远程 Windows 连不上

1. 在服务器确认服务监听正常: `curl http://localhost:18000/v1/models` (或 28000)
2. 确认防火墙放行: `sudo ufw status`
3. 在 Windows 用 `ping YOUR_SERVER_IP` 和 `Test-NetConnection YOUR_SERVER_IP -Port 18000` 测试网络

### 6.5 容器名冲突 (已存在同名容器)

```bash
# 35B-A3B-FP8
docker rm -f qwen-vllm

# 27B
docker rm -f qwen27b-vllm
```

### 6.6 模型加载慢 / 首次加载时间

| 模型 | 权重大小 | 预计加载时间 |
|------|----------|-------------|
| 35B-A3B-FP8 | ~30GB | 1~3 分钟 |
| 27B BF16 | ~41GB | 3~5 分钟 |

首次启动需从磁盘加载权重到显存, 耐心等待。若磁盘 IO 慢, 检查模型目录所在磁盘性能:
```bash
du -sh /home/$USER/WorkStation/Vllm/models/Qwen36-27B
du -sh /home/$USER/WorkStation/Vllm/models/Qwen36-35B-A3B-FP8
```

### 6.7 修改启动参数

容器创建后参数不可更改。需要修改 (如调整 `--max-model-len`、换 GPU、换端口) 时:
```bash
docker stop qwen-vllm && docker rm qwen-vllm
# 或: docker stop qwen27b-vllm && docker rm qwen27b-vllm
# 再重新执行 docker run 命令
```

### 6.8 同时运行两个模型的注意事项

可以同时运行两个模型, 但需注意:
- 使用**不同容器名**: `qwen-vllm` (35B) 和 `qwen27b-vllm` (27B)
- 使用**不同端口**: 18000 (35B) 和 28000 (27B)
- 使用**不同模型名**: `qwen3.6-35b` (35B) 和 `qwen3.6-27b` (27B)
- 两个模型的权重合计约 70GB, 加上 KV Cache 需要足够的显存 (推荐 GPU 5, 97GB)
- 两个模型的 `--max-model-len` 都设为 262144 时, KV Cache 可能占用 20GB+ 显存

### 6.9 关于 `--tool-call-parser` 的选择

vLLM 提供两种 Qwen3.6 工具调用解析器:

| Parser | 解析方式 | 适用场景 | 备注 |
|--------|----------|----------|------|
| `qwen3_coder` | 正则匹配 | 推荐 (默认) | vLLM Recipes 官方示例使用, 非流式完美工作 |
| `qwen3_xml` | Expat XML 解析 | Qwen3-Coder 系列 | 在 Qwen3.6 上有多 function 块 bug |

**两个模型均使用 `--tool-call-parser qwen3_coder`**。当前流式模式 (streaming) 各有已知 bug, 但非流式模式下正常工作。vLLM v0.23.0 已引入新的 Streaming Parser Engine 作为底层统一修复。

---

## 七、附录: 常用命令速查

```bash
# === 35B-A3B-FP8 容器 (qwen-vllm) ===
docker start qwen-vllm
docker stop qwen-vllm
docker restart qwen-vllm
docker logs -f qwen-vllm

# === 27B 容器 (qwen27b-vllm) ===
docker start qwen27b-vllm
docker stop qwen27b-vllm
docker restart qwen27b-vllm
docker logs -f qwen27b-vllm

# 同时启停
docker restart qwen-vllm && docker restart qwen27b-vllm

# 状态与日志
docker ps -a --filter name=qwen-vllm
docker ps -a --filter name=qwen27b-vllm

# 进入容器
docker exec -it qwen-vllm bash
docker exec -it qwen27b-vllm bash

# 删除容器 (谨慎)
docker stop qwen-vllm && docker rm qwen-vllm
docker stop qwen27b-vllm && docker rm qwen27b-vllm

# GPU 监控
watch -n 2 nvidia-smi
```

---

## 八、接入 Claude Code

vLLM **原生支持 Anthropic Messages API** (`/v1/messages` 端点), Claude Code 可直接对接, **无需任何中间代理** (如 LiteLLM)。

> 参考官方文档: <https://docs.vllm.ai/en/latest/serving/integrations/claude_code/>

### 8.1 前提条件

容器启动命令**必须**包含这两个参数:
```
--enable-auto-tool-choice --tool-call-parser qwen3_coder
```
没有这两个参数, Claude Code 将无法调用工具 (读写文件、执行命令等), 基本不可用。

**两个模型的部署命令均已包含上述参数。**

### 8.2 端口对应

| 模型 | 容器名 | 本地端口 | OpenAI 端点 | Anthropic 端点 |
|------|--------|----------|-------------|----------------|
| 35B-A3B-FP8 | `qwen-vllm` | 18000 | `http://localhost:18000/v1/chat/completions` | `http://localhost:18000/v1/messages` |
| 27B | `qwen27b-vllm` | 28000 | `http://localhost:28000/v1/chat/completions` | `http://localhost:28000/v1/messages` |

### 8.3 配置 Claude Code

在 `~/.claude/settings-local.json` 写入以下内容 (优先级高于 `settings.json`):

```json
{
  "env": {
    "ANTHROPIC_BASE_URL": "http://localhost:18000",
    "ANTHROPIC_API_KEY": "dummy",
    "ANTHROPIC_AUTH_TOKEN": "dummy",
    "ANTHROPIC_DEFAULT_HAIKU_MODEL": "qwen3.6-35b",
    "ANTHROPIC_DEFAULT_SONNET_MODEL": "qwen3.6-35b",
    "ANTHROPIC_DEFAULT_OPUS_MODEL": "qwen3.6-35b",
    "CLAUDE_CODE_ATTRIBUTION_HEADER": "0",
    "API_TIMEOUT_MS": "3000000"
  }
}
```

> 📌 **切换到 27B 模型**: 将 `ANTHROPIC_BASE_URL` 改为 `http://localhost:28000`, 将 `ANTHROPIC_DEFAULT_*_MODEL` 全部改为 `qwen3.6-27b`。
>
> `settings-local.json` 一次只能指向一个 vLLM 服务, 需要切换时可用脚本管理 (见下方)。

**字段说明:**

| 字段 | 说明 |
|------|------|
| `ANTHROPIC_BASE_URL` | 指向本机 vLLM (不要带 `/v1` 后缀, Claude Code 会自动拼) |
| `ANTHROPIC_API_KEY` / `AUTH_TOKEN` | vLLM 默认不鉴权, 任意值即可 (但不能为空) |
| `ANTHROPIC_DEFAULT_*_MODEL` | 必须与 vLLM 的 `--served-model-name` 一致 |
| `CLAUDE_CODE_ATTRIBUTION_HEADER` | 设为 `0` 关闭每请求 hash 注入, 保护前缀缓存, 提升性能 |
| `API_TIMEOUT_MS` | 超时拉长, 避免长输出中断 |

### 8.4 验证

```bash
# 确认 vLLM 已启动且支持 /v1/messages
curl http://localhost:18000/v1/messages \
  -H "x-api-key: dummy" \
  -H "anthropic-version: 2023-06-01" \
  -H "content-type: application/json" \
  -d '{
    "model": "qwen3.6-35b",
    "max_tokens": 64,
    "messages": [{"role":"user","content":"你好"}]
  }'

# 重启 Claude Code
claude
```

### 8.5 切换模型

`settings-local.json` 优先级最高, 存在时永远覆盖 `settings.json`。

```bash
# 切换回 GLM (或其他远程模型)
mv ~/.claude/settings-local.json ~/.claude/settings-local.json.bak

# 切回本地 35B-A3B-FP8
mv ~/.claude/settings-local.json.bak ~/.claude/settings-local.json

# 或使用脚本
./load-local.sh   # 切换到本地 vLLM (35B-A3B-FP8)
./load-glm.sh     # 切换回 GLM
```

### 8.6 已知限制

- **工具调用格式**: Qwen3.6 系列用 `qwen3_coder` 解析器 (正则匹配), 非流式模式完美工作, 流式模式有部分边界情况可能与 Claude 原生行为有差异。
- **Claude Code ≥ 2.1.154 兼容性**: 部分新版本注入了非标准 role (如 `ctx`/`msg`), vLLM 严格校验可能报错。若遇 role 校验错误, 可降级 Claude Code 或关注 vLLM 更新。
- **reasoning token**: `/v1/messages` 端点目前不输出 reasoning token, 不影响主流程。
- **模型名不可含 `/`**: 所以 `--served-model-name` 用 `qwen3.6-35b` / `qwen3.6-27b` 而不能用 `Qwen/Qwen3.6-...`。
- **流式工具调用**: `qwen3_coder` 在 `stream: true` 模式下可能不完整, 建议在 Claude Code 等非流式场景使用。
