#!/usr/bin/env bash
# switch-claude.sh — 一键切换 Claude Code 配置
#
# 用法:
#   bash switch-claude.sh <modelname>
#   例如: bash switch-claude.sh local-qwen36-27b
#
# 行为:
#   将 ~/.claude/settings-<modelname>.json 复制覆盖 ~/.claude/settings.json
#   覆盖前自动备份当前 settings.json 到 settings.json.bak
#   源文件 settings-<modelname>.json 保持不变，可反复切换
#
# 切换后需重启 Claude Code 才会生效。

set -euo pipefail

CLAUDE_DIR="${HOME}/.claude"
TARGET="${CLAUDE_DIR}/settings.json"
BAK="${CLAUDE_DIR}/settings.json.bak"

print_available() {
  # 列出所有 settings-*.json，去掉路径/settings- 前缀/.json 后缀
  ls -1 "${CLAUDE_DIR}"/settings-*.json 2>/dev/null \
    | sed 's#.*/settings-##; s#\.json$##' \
    | sed 's/^/  - /'
}

if [[ $# -lt 1 ]]; then
  echo "用法: bash $0 <modelname>" >&2
  echo "例如: bash $0 local-qwen36-27b" >&2
  echo "" >&2
  echo "可用配置 (modelname):" >&2
  print_available >&2
  exit 1
fi

MODELNAME="$1"
SOURCE="${CLAUDE_DIR}/settings-${MODELNAME}.json"

if [[ ! -f "$SOURCE" ]]; then
  echo "错误: 配置文件不存在: ${SOURCE}" >&2
  echo "" >&2
  echo "可用配置 (modelname):" >&2
  print_available >&2
  exit 1
fi

# 覆盖前备份当前 settings.json
if [[ -f "$TARGET" ]]; then
  cp "$TARGET" "$BAK"
fi

cp "$SOURCE" "$TARGET"

echo "✅ 已切换配置: ${MODELNAME}"
echo "   ${SOURCE}"
echo "   -> ${TARGET}"
[[ -f "$BAK" ]] && echo "   (原配置已备份到 ${BAK})"
echo ""
echo "当前 settings.json 内容:"
cat "$TARGET"
echo ""
echo "请重启 Claude Code 使配置生效。"
