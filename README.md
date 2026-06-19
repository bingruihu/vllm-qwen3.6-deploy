# Qwen3.6-35B-A3B-FP8 vLLM 部署文档

基于 Docker 部署 **Qwen3.6-35B-A3B-FP8** 文本大模型,使用 vLLM 提供 OpenAI 兼容 API 服务,支持远程 Windows 客户端访问。

---

## 一、环境信息

| 项目 | 值 |
|------|-----|
| 服务器 IP | `YOUR_SERVER_IP` |
| 操作系统 | Linux 6.8.0 (Ubuntu) |
| Docker 版本 | 28.2.2 |
| vLLM 镜像 | `vllm/vllm-openai:v0.23.0-ubuntu2404` (本地已存在) |
| 模型路径(宿主机) | `/home/$USER/WorkStation/Vllm/models/Qwen36-35B-A3B-FP8` |
| 模型架构 | `qwen3_moe`(MoE,文本对话/工具调用/思维链) |
| 目标 GPU | GPU 5(NVIDIA RTX PRO 6000 Blackwell,97GB) |
| 对外端口 | `18000`(容器内 8000) |

> ℹ️ **GPU 选择说明**:模型 FP8 权重合计约 **30GB**,需选择显存充足的 GPU。
> 本文档默认使用 **GPU 5(RTX PRO 6000 Blackwell,97GB,当前空闲)**,显存充足可稳定运行。
> 如需更换 GPU,只需修改 `--gpus` 参数,详见 [六、常见问题](#六常见问题)。

---

## 二、目录结构

```
/home/$USER/WorkStation/Vllm/
├── README.md                        # 本文档
└── models/
    └── Qwen36-35B-A3B-FP8/          # 模型权重(40 个 layers 分片 + mtp + outside)
        ├── config.json
        ├── model.safetensors.index.json
        ├── layers-*.safetensors
        ├── outside.safetensors
        ├── mtp.safetensors
        ├── tokenizer.json
        ├── chat_template.jinja
        └── ...
```

---

## 三、部署步骤

### 1. 准备镜像(本地已有可跳过)

```bash
docker pull vllm/vllm-openai:v0.23.0-ubuntu2404
```

### 2. 创建容器(仅执行一次)

> 该命令使用 `docker run` 创建一个**持久化容器**(不带 `--rm`),之后用 `docker start/stop` 控制启停。

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
  --tool-call-parser qwen3_xml \
  --reasoning-parser qwen3 \
  --trust-remote-code \
  --enable-prefix-caching \
  --enable-chunked-prefill \
  --max-num-seqs 8 \
  --block-size 32
```

> 💡 **若要让 Claude Code 接入**:`--enable-auto-tool-choice` 和 `--tool-call-parser qwen3_xml` 两个参数**必不可少**,否则 Claude Code 无法调用工具(读写文件、执行命令)。详见 [八、接入 Claude Code](#八接入-claude-code)。

**参数说明:**

| 参数 | 说明 |
|------|------|
| `-d` | 后台运行 |
| `--name qwen-vllm` | 容器名,后续用该名称启停 |
| `--gpus '"device=5"'` | **指定 GPU 5**;换卡改此参数(如 `'"device=7"'`) |
| `--ipc=host` | 共享主机共享内存,PyTorch 多进程通信所需 |
| `--restart=no` | 不自动重启,完全由手动 `start/stop` 控制 |
| `-p 18000:8000` | 端口映射:宿主机 18000 → 容器 8000 |
| `-v .../models:/models:ro` | 只读挂载模型目录 |
| `-e VLLM_MOE_FORCE_MARLIN=1` | **强制 MoE 使用 Marlin 量化内核**(FP8 MoE 推理必需) |
| `--served-model-name` | API 中使用的模型名 |
| `--max-model-len 262144` | 最大上下文长度(256K tokens,可按显存调整) |
| `--gpu-memory-utilization 0.95` | GPU 显存占用比例 |
| `--quantization fp8` | 模型量化格式(FP8) |
| `--dtype bfloat16` | 计算数据类型 |
| `--enable-auto-tool-choice` | 开启自动工具选择(Claude Code / Agent 必需) |
| `--tool-call-parser qwen3_xml` | Qwen3 系列 XML 格式工具调用解析器 |
| `--reasoning-parser qwen3` | Qwen3 思维链解析器 |
| `--trust-remote-code` | 信任模型远程代码(自定义架构需要) |
| `--enable-prefix-caching` | 启用前缀缓存,加速重复 prompt 推理 |
| `--enable-chunked-prefill` | 启用分块 prefill,提高首 token 延迟 |
| `--max-num-seqs 8` | 最大并发序列数 |
| `--block-size 32` | KV Cache 块大小(tokens) |

> 📌 关于 `--gpus` 语法:必须使用 `'"device=5"'`(外层单引号 + 内层双引号),这是 NVIDIA Container Toolkit 的要求。

---

## 四、服务启停

### 启动服务
```bash
docker start qwen-vllm
```

### 停止服务
```bash
docker stop qwen-vllm
```

### 重启服务
```bash
docker restart qwen-vllm
```

### 查看运行状态
```bash
docker ps -a --filter name=qwen-vllm
```

### 查看实时日志(首次加载模型需 1~3 分钟)
```bash
docker logs -f qwen-vllm
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
```

### 5.2 远程 Windows 访问

**前提**:服务器防火墙需放行 18000 端口。在服务器执行:
```bash
# 查看/开放防火墙(若使用 ufw)
sudo ufw allow 18000/tcp

# 或使用 iptables
sudo iptables -A INPUT -p tcp --dport 18000 -j ACCEPT
```

**在 Windows 上访问**,把 `localhost` 换成服务器 IP:

| 用途 | 地址 |
|------|------|
| 模型列表 | `http://YOUR_SERVER_IP:18000/v1/models` |
| 对话接口 | `http://YOUR_SERVER_IP:18000/v1/chat/completions` |
| 健康检查 | `http://YOUR_SERVER_IP:18000/health` |

**Windows PowerShell 测试示例:**
```powershell
# 测试连通性
curl http://YOUR_SERVER_IP:18000/v1/models

# 对话请求
$body = @{
    model = "qwen3.6-35b"
    messages = @(@{role="user"; content="用 Python 写一个快速排序"})
    max_tokens = 512
} | ConvertTo-Json
Invoke-RestMethod -Uri "http://YOUR_SERVER_IP:18000/v1/chat/completions" -Method Post -ContentType "application/json" -Body $body
```

### 5.3 通过 SSH 端口隧道远程访问(推荐)

如果服务器防火墙未放行 18000 端口,或你希望**加密传输**(对话内容不裸奔),可以通过 SSH 隧道将远程服务映射到本地。

> ✅ **为什么推荐隧道**:不需要改服务器防火墙、不需要管理员权限、全程加密,且 Claude Code 和所有客户端都只需配置 `localhost`。

#### 5.3.1 使用 MobaXterm 建立隧道

1. 打开 MobaXterm,新建 SSH 会话,连接 `YOUR_SERVER_IP`(密钥+密码认证)
2. 连接成功后,左侧侧边栏会自动显示 **Session tunneling** → **Port Forwarding**
3. 点击 **Port forwarding settings**(或使用菜单栏 Session → SSH settings → Tunnels 标签页)
4. 添加一条转发规则:

   | 字段 | 值 |
   |------|-----|
   | **Forwarded port** | `18000` |
   | **Destination host** | `127.0.0.1` |
   | **Destination port** | `18000` |

5. 点击 **OK** 保存,MobaXterm 会自动将本地 `localhost:18000` 转发到远程服务器

**验证隧道是否生效:**
```bash
# 在 Windows 的 PowerShell 或 CMD 中执行
curl http://localhost:18000/v1/models
```

如果返回模型列表,说明隧道成功。

#### 5.3.2 使用命令行 SSH 隧道

如果你有 Git Bash、WSL 或其他支持 ssh 的终端:

```bash
ssh -N -L 18000:127.0.0.1:18000 YOUR_USERNAME@YOUR_SERVER_IP
```

| 参数 | 说明 |
|------|------|
| `-N` | 不打开远程 shell,仅转发端口 |
| `-L 18000:127.0.0.1:18000` | 本地 18000 → 远程服务器 127.0.0.1:18000 |

隧道保持运行时,所有指向 `localhost:18000` 的请求都会通过 SSH 加密通道转发到远程服务。

#### 5.3.3 通过隧道配置 Claude Code

隧道建立后,Claude Code 的配置与本地完全一致,在 `~/.claude/settings-local.json` 写入:

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

> 📌 隧道不关闭,Claude Code 每次启动都能直接使用,无需重新配置。

#### 5.3.4 通过隧道配置第三方客户端

所有 OpenAI 兼容客户端同理,将 API Base 设为 `http://localhost:18000/v1` 即可:

- **Cursor / Continue / Cline**: API Base → `http://localhost:18000/v1`
- **OpenAI Python SDK**: `base_url="http://localhost:18000/v1"`
- **Apifox / Postman**: URL → `http://localhost:18000/v1/chat/completions`

---

### 5.4 通过 SSH 反向隧道将本地 vLLM 共享给远程服务器(openclaw)

如果你希望将本机(或本地网络内)运行的 vLLM 服务通过 SSH 反向端口转发暴露给另一台远程服务器(如 `REMOTE_SERVER_IP`),让该服务器的 root 用户也能访问本地 vLLM。

#### 5.4.1 建立反向隧道

在**本机**执行:

```bash
ssh -N -f -R 18000:localhost:18000 root@REMOTE_SERVER_IP
```

| 参数 | 说明 |
|------|------|
| `-N` | 不打开远程 shell,仅转发端口 |
| `-f` | 后台运行 |
| `-R 18000:localhost:18000` | 远程服务器 18000 → 本机 localhost 18000 |

#### 5.4.2 验证连接

在 **REMOTE_SERVER_IP** 上执行:

```bash
# v1 模型列表接口
curl http://localhost:18000/v1/models

# 对话测试
curl http://localhost:18000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.6-35b",
    "messages": [{"role":"user","content":"你好"}],
    "max_tokens": 256
  }'
```

> ⚠️ **注意**: 不要使用 `/models` 路径(vLLM 不是这个端点),正确路径为 `/v1/models`。

#### 5.4.3 断开隧道

```bash
# 方法 1: 通过 SSH 控制连接关闭
ssh -O stop -R 18000:localhost:18000 root@REMOTE_SERVER_IP

# 方法 2: kill 隧道进程
pkill -f "18000:localhost:18000"
```

> 💡 隧道只是临时连接,断开后重新建立即可。服务本身不受影响,客户端会因连接失败而报错,隧道恢复后自动恢复。

---

### 5.5 在第三方客户端中使用(直连模式)

如果已在服务器上架放防火墙(`sudo ufw allow 18000/tcp`),可以直接访问:

- **Cursor / Continue / Cline**:OpenAI API Base 设为 `http://YOUR_SERVER_IP:18000/v1`,API Key 任意填
- **OpenAI Python SDK**:
  ```python
  from openai import OpenAI
  client = OpenAI(
      base_url="http://YOUR_SERVER_IP:18000/v1",
      api_key="EMPTY"
  )
  resp = client.chat.completions.create(
      model="qwen3.6-35b",
      messages=[{"role":"user","content":"hello"}]
  )
  print(resp.choices[0].message.content)
  ```
- **Apifox / Postman**:直接调用 `http://YOUR_SERVER_IP:18000/v1/chat/completions`

---

## 六、常见问题

### 6.1 SSH 隧道连接断开后怎么办

隧道只是临时连接,断开后重新建立即可:
- **MobaXterm**:关闭隧道后重新连接 SSH 会话即可
- **命令行**:关闭终端或 Ctrl+C 后,重新执行 `ssh -N -L ...`

服务本身不受影响,Claude Code 和所有客户端会因连接 `localhost` 失败而报错,隧道恢复后自动恢复。

### 6.2 更换 GPU

默认使用 GPU 5。若需切换到其它 GPU(例如与其它任务共享、或 GPU 5 被占用),只需修改 `--gpus` 参数:

```bash
# 1. 停止并删除旧容器
docker stop qwen-vllm
docker rm qwen-vllm

# 2. 用新 GPU 重新创建容器(只改 --gpus,例如换成 GPU 4)
docker run -d \
  --name qwen-vllm \
  --gpus '"device=4"' \
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
  --tool-call-parser qwen3_xml \
  --reasoning-parser qwen3 \
  --trust-remote-code \
  --enable-prefix-caching \
  --enable-chunked-prefill \
  --max-num-seqs 8 \
  --block-size 32

docker logs -f qwen-vllm
```

**各 GPU 显存适配建议:**

| GPU | 型号 | 显存 | 是否能装下 30GB 模型 |
|-----|------|------|----------------------|
| 0–3 | RTX 4090 | 24GB | ❌ 装不下 |
| 4 | RTX PRO 6000 Blackwell | 97GB | ✅ 可用(当前被占用) |
| 5 | RTX PRO 6000 Blackwell | 97GB | ✅ **推荐(当前空闲)** |
| 6 | RTX 4090 | 24GB | ❌ 装不下 |
| 7 | RTX 4090 | 24GB | ❌ 装不下 |

查看 GPU 实时占用:
```bash
nvidia-smi
# 或只看某几张卡
nvidia-smi -i 4,5 --query-gpu=index,name,memory.used,memory.total --format=csv
```

> 💡 若启动日志出现 `CUDA out of memory`,说明所选 GPU 显存不足,请按上表换用 97GB 的 Blackwell 卡(GPU 4 或 5)。

### 6.3 端口 18000 被占用

```bash
# 查看占用
ss -tlnp | grep 18000

# 换一个端口,例如 28000,重新创建容器时把 -p 改成 -p 28000:8000
```

### 6.4 远程 Windows 连不上

1. 在服务器确认服务监听正常:`curl http://localhost:18000/v1/models`
2. 确认防火墙放行:`sudo ufw status`
3. 在 Windows 用 `ping YOUR_SERVER_IP` 和 `Test-NetConnection YOUR_SERVER_IP -Port 18000` 测试网络
4. 确认服务器与 Windows 在同一网段或路由可达

### 6.5 容器名冲突(已存在同名容器)

```bash
docker rm -f qwen-vllm   # 强制删除旧容器后重新 run
```

### 6.6 模型加载慢 / 下载 tokenizer 卡住

首次启动需从磁盘加载 ~30GB 权重到显存,耐心等待。若磁盘 IO 慢,检查模型目录所在磁盘性能:
```bash
du -sh /home/$USER/WorkStation/Vllm/models/Qwen36-35B-A3B-FP8
```

### 6.7 修改启动参数

容器创建后参数不可更改。需要修改(如调整 `--max-model-len`、换 GPU、换端口)时:
```bash
docker stop qwen-vllm && docker rm qwen-vllm
# 再重新执行 docker run 命令
```

---

## 七、附录:常用命令速查

```bash
# 启停
docker start qwen-vllm
docker stop qwen-vllm
docker restart qwen-vllm

# 状态与日志
docker ps -a --filter name=qwen-vllm
docker logs -f qwen-vllm
docker stats qwen-vllm

# 进入容器
docker exec -it qwen-vllm bash

# 删除容器(谨慎)
docker stop qwen-vllm && docker rm qwen-vllm

# GPU 监控
watch -n 2 nvidia-smi
```

---

## 八、接入 Claude Code

vLLM **原生支持 Anthropic Messages API**(`/v1/messages` 端点),Claude Code 可直接对接,**无需任何中间代理**(如 LiteLLM)。

> 参考官方文档:<https://docs.vllm.ai/en/latest/serving/integrations/claude_code/>

### 8.1 前提条件

容器启动命令**必须**包含这两个参数(已在第三章的 `docker run` 中加入):
```
--enable-auto-tool-choice --tool-call-parser qwen3_xml
```
没有这两个参数,Claude Code 将无法调用工具(读写文件、执行命令等),基本不可用。

### 8.2 端口对应

| 客户端 | 端点 |
|--------|------|
| Claude Code(本机) | `http://localhost:18000` |
| OpenAI 协议客户端 | `http://localhost:18000/v1/chat/completions` |
| Anthropic 协议客户端 | `http://localhost:18000/v1/messages` |

### 8.3 配置 Claude Code

在 `~/.claude/settings-local.json` 写入以下内容(优先级高于 `settings.json`,会覆盖现有模型配置):

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

**字段说明:**

| 字段 | 说明 |
|------|------|
| `ANTHROPIC_BASE_URL` | 指向本机 vLLM(不要带 `/v1` 后缀,Claude Code 会自动拼) |
| `ANTHROPIC_API_KEY` / `AUTH_TOKEN` | vLLM 默认不鉴权,任意值即可(但不能为空) |
| `ANTHROPIC_DEFAULT_*_MODEL` | 必须与 vLLM 的 `--served-model-name` 一致(此处为 `qwen3.6-35b`) |
| `CLAUDE_CODE_ATTRIBUTION_HEADER` | 设为 `0` 关闭每请求 hash 注入,保护前缀缓存,提升性能 |
| `API_TIMEOUT_MS` | 超时拉长,避免长输出中断 |

### 8.4 验证

```bash
# 1. 确认 vLLM 已启动且支持 /v1/messages
curl http://localhost:18000/v1/messages \
  -H "x-api-key: dummy" \
  -H "anthropic-version: 2023-06-01" \
  -H "content-type: application/json" \
  -d '{
    "model": "qwen3.6-35b",
    "max_tokens": 64,
    "messages": [{"role":"user","content":"你好"}]
  }'

# 2. 重启 Claude Code
claude
```

### 8.5 切换回其他模型(如 GLM)

`settings-local.json` 优先级最高,存在时永远覆盖 `settings.json`。临时切回其它服务:

```bash
# 备份本地配置 → 回退到 settings.json 的 GLM
mv ~/.claude/settings-local.json ~/.claude/settings-local.json.bak

# 需要时再切回本地 vLLM
mv ~/.claude/settings-local.json.bak ~/.claude/settings-local.json
```

### 8.6 已知限制

- **工具调用格式**:Qwen3 系列用 `qwen3_xml` 解析器(基于 XML 标签),大部分场景可用,但复杂 tool_use 边界情况可能与 Claude 原生行为有差异。
- **Claude Code ≥ 2.1.154 兼容性**:部分新版本注入了非标准 role(如 `ctx`/`msg`),vLLM 严格校验可能报错。若遇 role 校验错误,可降级 Claude Code 或关注 vLLM 更新。
- **reasoning token**:`/v1/messages` 端点目前不输出 reasoning token,不影响主流程。
- **模型名不可含 `/`**:所以 `--served-model-name` 用 `qwen3.6-35b` 而不能用 `Qwen/Qwen3.6-...`。
