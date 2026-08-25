#!/usr/bin/env bash
set -euo pipefail

#setup warmup model for warmup script
echo "$ACTIVE_MODEL"
echo "SERVED_NAME=$SERVED_NAME" > "$ACTIVE_MODEL"

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
if [[ "$(basename "$SCRIPT_DIR")" == "launcher" && -d "$SCRIPT_DIR/../profiles" ]]; then
  MANAGER_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)
else
  MANAGER_ROOT="$SCRIPT_DIR"
fi
PROJECT_ROOT="$MANAGER_ROOT"
PROJECT_RELEASE_FILE=${PROJECT_RELEASE_FILE:-"$MANAGER_ROOT/PROJECT_RELEASE.env"}
# shellcheck source=/dev/null
source "$PROJECT_RELEASE_FILE"
RUNTIME_ROOT=${RUNTIME_ROOT:-"$PROJECT_ROOT"}
PROFILE_DIR=${PROFILE_DIR:-"$RUNTIME_ROOT/profiles"}
TEMPLATE_DIR=${TEMPLATE_DIR:-"$PROFILE_DIR/templates"}
LOG_DIR=${LOG_DIR:-"$RUNTIME_ROOT/run-logs"}
STATE_FILE=${STATE_FILE:-"$LOG_DIR/start-manager.state"}
STAMP=$(date +%Y%m%d-%H%M%S)
VERSION=${VERSION:-$FORK_RELEASE}

# ---- Helper functions ----

normalize_bool() {
  case "${1,,}" in
    1|yes|y|true|on) echo 1 ;;
    *) echo 0 ;;
  esac
}

guess_model_family() {
  local dir=${1,,}
  if [[ "$dir" == *gemma* ]]; then
    echo gemma4
  else
    echo qwen
  fi
}

guess_quantization() {
  local dir=${1,,}
  if [[ "$dir" == *fp8* ]]; then
    echo fp8
  elif [[ "$dir" == *gptq* ]]; then
    echo gptq_marlin
  elif [[ "$dir" == *awq* ]]; then
    echo awq_marlin
  elif [[ "$dir" == *quark* ]]; then
    echo quark
  else
    echo ""
  fi
}

default_context_tokens() {
  local quantization=${QUANTIZATION:-$(guess_quantization "${MODEL_DIR:-}")}
  if [[ "$quantization" == "fp8" ]]; then
    echo 102400
  elif [[ "$quantization" == "quark" ]]; then
    echo 8192
  else
    echo 131072
  fi
}

default_gpu_util() {
  local quantization=${QUANTIZATION:-$(guess_quantization "${MODEL_DIR:-}")}
  if [[ "$quantization" == "fp8" ]]; then
    echo 0.92
  else
    echo 0.90
  fi
}

reasoning_parser_is_disabled() {
  case "${REASONING_PARSER:-}" in
    off|none|disabled|disable)
      return 0
      ;;
  esac
  return 1
}

resolve_template_file() {
  local template=${1:-}
  [[ -n "$template" ]] || return 1
  if [[ -f "$template" ]]; then
    printf '%s\n' "$template"
    return 0
  fi
  if [[ -f "$TEMPLATE_DIR/$template" ]]; then
    printf '%s\n' "$TEMPLATE_DIR/$template"
    return 0
  fi
  return 1
}

gpu_device_count() {
  local devices=${1:-}
  local count=0 part
  devices=${devices// /}
  [[ -n "$devices" ]] || {
    echo 0
    return 0
  }
  IFS=',' read -r -a parts <<< "$devices"
  for part in "${parts[@]}"; do
    [[ -n "$part" ]] && count=$((count + 1))
  done
  echo "$count"
}

detect_default_gpu_devices() {
  local detected
  detected=$(
    nvidia-smi --query-gpu=index,name --format=csv,noheader 2>/dev/null |
      awk -F'\t' '
        BEGIN { sep = "" }
        tolower($2) ~ /2080[[:space:]]*ti/ {
          out = out sep $1
          sep = ","
          count++
          if (count == 2) {
            print out
            exit
          }
        }
      '
  ) || true
  if [[ -n "$detected" ]]; then
    printf '%s\n' "$detected"
  elif [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    printf '%s\n' "$CUDA_VISIBLE_DEVICES"
  else
    printf '0,1\n'
  fi
}

# ---- Environment setup (from set_sm75_runtime_env) ----

export STABLE_ROOT="$RUNTIME_ROOT"
export HOME=${RUN_HOME:-"$HOME"}

if [[ -z "${CUDA_HOME:-}" ]]; then
  for cuda_candidate in "/usr/local/cuda-${PRIMARY_CUDA_VERSION:-12.8}" \
    /usr/local/cuda-13.0 /usr/local/cuda-12.8 /usr/local/cuda-13 \
    /usr/local/cuda-12 /usr/local/cuda; do
    if [[ -x "$cuda_candidate/bin/nvcc" ]]; then
      CUDA_HOME="$cuda_candidate"
      break
    fi
  done
fi
export CUDA_HOME
if [[ -z "${CUDA_HOME:-}" || ! -x "$CUDA_HOME/bin/nvcc" ]]; then
  echo "ERROR: CUDA toolkit not found; set CUDA_HOME explicitly." >&2
  exit 1
fi

export CUDA_PATH="$CUDA_HOME"
export CUDACXX="$CUDA_HOME/bin/nvcc"

if [[ -z "${CC:-}" ]]; then
  for gcc_candidate in "/usr/bin/gcc-${PRIMARY_GCC_MAJOR:-12}" \
    /usr/bin/gcc-15 /usr/bin/gcc-14 /usr/bin/gcc-13 /usr/bin/gcc-12; do
    if [[ -x "$gcc_candidate" ]]; then
      CC="$gcc_candidate"
      break
    fi
  done
fi
export CC
if [[ -z "${CXX:-}" && -n "${CC:-}" ]]; then
  CXX=${CC/gcc/g++}
fi
export CXX
if [[ -z "${CXX:-}" && -x /usr/bin/g++-12 ]]; then
  export CXX=/usr/bin/g++-12
fi
if [[ -z "${CUDAHOSTCXX:-}" && -n "${CXX:-}" ]]; then
  export CUDAHOSTCXX="$CXX"
fi

export TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-7.5}
export CUDA_DEVICE_ORDER=${CUDA_DEVICE_ORDER:-PCI_BUS_ID}

# GPU devices: prefer GPU_DEVICES env, then CUDA_VISIBLE_DEVICES, then detect
if [[ -z "${GPU_DEVICES:-}" ]]; then
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    GPU_DEVICES="$CUDA_VISIBLE_DEVICES"
  else
    GPU_DEVICES="$(detect_default_gpu_devices)"
  fi
fi
export CUDA_VISIBLE_DEVICES="$GPU_DEVICES"

# FLASHQLA_ROOT detection
if [[ -z "${FLASHQLA_ROOT:-}" ]]; then
  runtime_parent=$(cd -- "$RUNTIME_ROOT/.." && pwd)
  for flashqla_candidate in \
    "$RUNTIME_ROOT/.deps/FlashQLA-SM70-SM75" \
    "$MANAGER_ROOT/.deps/FlashQLA-SM70-SM75" \
    "$RUNTIME_ROOT/FlashQLA-SM70-SM75" \
    "$MANAGER_ROOT/FlashQLA-SM70-SM75" \
    "$runtime_parent/FlashQLA-SM70-SM75" \
    /opt/FlashQLA-SM70-SM75; do
    if [[ -d "$flashqla_candidate/flash_qla" ]]; then
      FLASHQLA_ROOT="$flashqla_candidate"
      break
    fi
  done
fi
export PYTHONPATH="$RUNTIME_ROOT${FLASHQLA_ROOT:+:$FLASHQLA_ROOT}${PYTHONPATH:+:$PYTHONPATH}"
export PATH="$RUNTIME_ROOT/.venv/bin:${CUDA_HOME}/bin:$PATH"

if [[ -n "${FLASHQLA_ROOT:-}" ]]; then
  export TORCH_EXTENSIONS_DIR=${TORCH_EXTENSIONS_DIR:-"$FLASHQLA_ROOT/.torch_extensions_vllm_flashqla_legacy"}
fi

export FLASHINFER_ENABLE_AOT=${FLASHINFER_ENABLE_AOT:-1}

# TQ/int8KV env vars (from set_sm75_runtime_env)
if [[ "${KV_CACHE_DTYPE:-}" == "int8_per_token_head" ]]; then
  export VLLM_INT8KV_FA_PREFILL=${VLLM_INT8KV_FA_PREFILL:-1}
  if [[ "${MODE:-normal}" == "safe" ]]; then
    export VLLM_INT8KV_FA_FIRST_CHUNK_DEQUANT=${VLLM_INT8KV_FA_FIRST_CHUNK_DEQUANT:-1}
  else
    export VLLM_INT8KV_FA_FIRST_CHUNK_DEQUANT=${VLLM_INT8KV_FA_FIRST_CHUNK_DEQUANT:-0}
  fi
  export VLLM_INT8KV_FA_CONTINUATION_DEQUANT=${VLLM_INT8KV_FA_CONTINUATION_DEQUANT:-1}
  export VLLM_INT8KV_FA_CASCADE_DEQUANT=${VLLM_INT8KV_FA_CASCADE_DEQUANT:-1}
  export VLLM_INT8KV_FA_CASCADE_TILE_TOKENS=${VLLM_INT8KV_FA_CASCADE_TILE_TOKENS:-65536}
fi

if [[ "${KV_CACHE_DTYPE:-}" == turboquant_* ]]; then
  tq_continuation_reserve_default=65536

  if [[ "${ENABLE_PREFIX_CACHING:-1}" == "1" ]]; then
    tq_max_model_len=${MAX_MODEL_LEN:-0}
    tq_max_num_seqs=${MAX_NUM_SEQS:-1}

    if [[ "$tq_max_model_len" =~ ^[0-9]+$ && "$tq_max_num_seqs" =~ ^[0-9]+$ ]] \
      && (( tq_max_model_len >= 240000 )) \
      && (( tq_max_num_seqs <= 1 )); then
      tq_continuation_reserve_default=262144
    fi
  fi

  export VLLM_TURBOQUANT_USE_FLASHINFER_PREFILL=${VLLM_TURBOQUANT_USE_FLASHINFER_PREFILL:-1}
  if [[ "${MODE:-normal}" == "fast" || "${MODE:-normal}" == "aggressive" ]]; then
    export VLLM_TURBOQUANT_REQUIRE_FLASHINFER_PREFILL=1
  else
    export VLLM_TURBOQUANT_REQUIRE_FLASHINFER_PREFILL=${VLLM_TURBOQUANT_REQUIRE_FLASHINFER_PREFILL:-0}
  fi
  export VLLM_TURBOQUANT_FLASHINFER_BACKEND=${VLLM_TURBOQUANT_FLASHINFER_BACKEND:-fa2}
  export VLLM_TURBOQUANT_CONTINUATION_PREFIX_COMBINE=${VLLM_TURBOQUANT_CONTINUATION_PREFIX_COMBINE:-auto}
  export VLLM_TURBOQUANT_CONTINUATION_PREFIX_COMBINE_MIN_TOKENS=${VLLM_TURBOQUANT_CONTINUATION_PREFIX_COMBINE_MIN_TOKENS:-20480}
  export VLLM_TURBOQUANT_CONTINUATION_WORKSPACE_RESERVE_TOKENS=${VLLM_TURBOQUANT_CONTINUATION_WORKSPACE_RESERVE_TOKENS:-$tq_continuation_reserve_default}
  export VLLM_TURBOQUANT_CUDAGRAPH_SPEC_DECODE_SAFE=${VLLM_TURBOQUANT_CUDAGRAPH_SPEC_DECODE_SAFE:-1}
  export VLLM_TURBOQUANT_CUDAGRAPH_SPEC_PREFIX_ROWS=${VLLM_TURBOQUANT_CUDAGRAPH_SPEC_PREFIX_ROWS:-1}
  export VLLM_TURBOQUANT_CONTINUATION_SDPA_Q_CHUNK=${VLLM_TURBOQUANT_CONTINUATION_SDPA_Q_CHUNK:-512}
  export VLLM_TURBOQUANT_CONTINUATION_SDPA_MAX_QK_CELLS=${VLLM_TURBOQUANT_CONTINUATION_SDPA_MAX_QK_CELLS:-16777216}
  export VLLM_TURBOQUANT_SPEC_CONTINUATION_DECODE_FASTPATH=${VLLM_TURBOQUANT_SPEC_CONTINUATION_DECODE_FASTPATH:-1}

  if [[ -n "${VLLM_TURBOQUANT_MAX_KV_SPLITS:-}" ]]; then
    export VLLM_TURBOQUANT_MAX_KV_SPLITS
  fi

  export VLLM_TURBOQUANT_DECODE_BLOCK_KV=${VLLM_TURBOQUANT_DECODE_BLOCK_KV:-2}
fi

# Torch cache dirs
export TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-"$MANAGER_ROOT/torchinductor-cache"}
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-"$MANAGER_ROOT/triton-cache"}
export PYTHONUNBUFFERED=1

if [[ "${ENABLE_AUTO_TOOL_CHOICE:-0}" == "1" ]]; then
  export VLLM_ENFORCE_STRICT_TOOL_CALLING=${VLLM_ENFORCE_STRICT_TOOL_CALLING:-1}
fi

if [[ -n "${REASONING_BUDGET:-}" ]]; then
  export VLLM_DEFAULT_THINKING_TOKEN_BUDGET="$REASONING_BUDGET"
else
  unset VLLM_DEFAULT_THINKING_TOKEN_BUDGET
fi

# ---- Mode defaults (from apply_mode for normal mode) ----

MODE=${MODE:-normal}
case "$MODE" in
  normal)
    ENFORCE_EAGER=0
    DISABLE_LOG_STATS=1
    VLLM_SM75_SPEC_SYNC_MODE=safe
    VLLM_ALLOW_MAMBA_SPEC_FULL_CUDAGRAPH=0
    ;;
  fast)
    ENFORCE_EAGER=0
    DISABLE_LOG_STATS=1
    VLLM_SM75_SPEC_SYNC_MODE=safe
    VLLM_ALLOW_MAMBA_SPEC_FULL_CUDAGRAPH=1
    ;;
  aggressive)
    ENFORCE_EAGER=0
    DISABLE_LOG_STATS=1
    VLLM_SM75_SPEC_SYNC_MODE=nosync
    VLLM_ALLOW_MAMBA_SPEC_FULL_CUDAGRAPH=1
    ;;
  safe)
    ENFORCE_EAGER=1
    DISABLE_LOG_STATS=0
    VLLM_SM75_SPEC_SYNC_MODE=safe
    VLLM_ALLOW_MAMBA_SPEC_FULL_CUDAGRAPH=0
    ;;
  *)
    echo "ERROR: MODE must be safe, normal, fast, or aggressive." >&2
    exit 1
    ;;
esac

# ---- Profile sourcing ----

if [[ -n "${PROFILE:-}" ]]; then
  profile_file=""
  if [[ -f "$PROFILE_DIR/$PROFILE" ]]; then
    profile_file="$PROFILE_DIR/$PROFILE"
  elif [[ -f "$PROFILE_DIR/${PROFILE%.env}.env" ]]; then
    profile_file="$PROFILE_DIR/${PROFILE%.env}.env"
  fi
  if [[ -n "$profile_file" && -f "$profile_file" ]]; then
    while IFS= read -r key; do
      [[ -n "$key" ]] || continue
      value=$(awk -F= -v key="$key" '$1 == key { value = substr($0, index($0, "=") + 1); gsub(/^[ \t]+|[ \t]+$/, "", value); gsub(/^'\''|'\''$/, "", value); gsub(/^"|"$/, "", value); print value; exit }' "$profile_file")
      export "$key"="$value"
    done < <(sed -nE 's/^([A-Za-z_][A-Za-z0-9_]*)=.*/\1/p' "$profile_file" | sort -u)
  else
    echo "ERROR: Profile not found: $PROFILE (searched $PROFILE_DIR/$PROFILE and $PROFILE_DIR/${PROFILE%.env}.env)" >&2
    exit 1
  fi
fi

# ---- Runtime defaults (from prepare_runtime_defaults) ----

if [[ -z "${MODEL_DIR:-}" ]]; then
  echo "ERROR: MODEL_DIR is required. Set it as an environment variable." >&2
  exit 1
fi
if [[ ! -d "$MODEL_DIR" ]]; then
  echo "ERROR: Model directory does not exist: $MODEL_DIR" >&2
  exit 1
fi

MODEL_FAMILY=${MODEL_FAMILY:-$(guess_model_family "$MODEL_DIR")}
SERVED_NAME=${SERVED_NAME:-$(basename "$MODEL_DIR")}
TEMPLATE_DIR=${TEMPLATE_DIR:-"$PROFILE_DIR/templates"}
GPU_DEVICES=${GPU_DEVICES:-$(detect_default_gpu_devices)}
PP_SIZE=${PP_SIZE:-1}
if [[ -z "${TP_SIZE:-}" ]]; then
  gpu_count=$(gpu_device_count "$GPU_DEVICES")
  if [[ "$PP_SIZE" =~ ^[1-9][0-9]*$ ]] && (( gpu_count % PP_SIZE == 0 )); then
    TP_SIZE=$((gpu_count / PP_SIZE))
  else
    TP_SIZE=$gpu_count
  fi
fi
QUANTIZATION=${QUANTIZATION:-$(guess_quantization "$MODEL_DIR")}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-$(default_context_tokens)}
GPU_UTIL=${GPU_UTIL:-$(default_gpu_util)}
MAX_BATCHED_TOKENS=${MAX_BATCHED_TOKENS:-2048}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-1}
MTP_K=${MTP_K:-0}
PORT=${PORT:-8000}
SERVICE_SCOPE=${SERVICE_SCOPE:-local}

# Prefix cache defaults
ENABLE_PREFIX_CACHING=$(normalize_bool "${ENABLE_PREFIX_CACHING:-1}")
ENABLE_PROMPT_TOKENS_DETAILS=$(normalize_bool "${ENABLE_PROMPT_TOKENS_DETAILS:-1}")
DISABLE_PREFIX_CACHING=$(normalize_bool "${DISABLE_PREFIX_CACHING:-0}")

if [[ "$DISABLE_PREFIX_CACHING" == "1" ]]; then
  ENABLE_PREFIX_CACHING=0
fi

# Mamba cache mode (Qwen prefix-cache default)
if [[ "$ENABLE_PREFIX_CACHING" == "1" && "$MODEL_FAMILY" == qwen* ]]; then
  MAMBA_CACHE_MODE=${MAMBA_CACHE_MODE:-align}
fi

# Message type defaults
normalize_message_type_defaults() {
  if [[ "${MESSAGE_TYPE:-}" == "text+image" || -n "${MM_LIMIT_JSON:-}" || "${LANGUAGE_MODEL_ONLY:-1}" == "0" ]]; then
    MESSAGE_TYPE=text+image
  else
    MESSAGE_TYPE=text-only
  fi

  if [[ "$MESSAGE_TYPE" == "text+image" ]]; then
    MM_LIMIT_JSON=${MM_LIMIT_JSON:-'{"image":1,"video":0,"audio":0}'}
    LANGUAGE_MODEL_ONLY=0
    SKIP_MM_PROFILING=$(normalize_bool "${SKIP_MM_PROFILING:-0}")
  else
    MM_LIMIT_JSON=""
    LANGUAGE_MODEL_ONLY=1
    SKIP_MM_PROFILING=1
  fi
}

normalize_message_type_defaults

# Tool calling defaults
ENABLE_AUTO_TOOL_CHOICE=$(normalize_bool "${ENABLE_AUTO_TOOL_CHOICE:-0}")
if [[ "$ENABLE_AUTO_TOOL_CHOICE" == "1" ]]; then
  TOOL_CALL_PARSER=${TOOL_CALL_PARSER:-qwen3_xml}
  VLLM_ENFORCE_STRICT_TOOL_CALLING=${VLLM_ENFORCE_STRICT_TOOL_CALLING:-1}
fi

# Reasoning defaults
apply_family_reasoning_defaults() {
  if [[ "${MODEL_FAMILY:-}" != qwen* ]]; then
    return 0
  fi
  case "${REASONING_PARSER:-}" in
    off|none|disabled|disable)
      return 0
      ;;
  esac
  local model_dir_l served_l profile_l group_l
  model_dir_l=${MODEL_DIR,,}
  served_l=${SERVED_NAME,,}
  profile_l=${PROFILE:-}
  profile_l=${profile_l,,}
  group_l=${PROFILE_GROUP:-}
  group_l=${group_l,,}

  case "$group_l" in
    qwen3*|qwen36*)
      REASONING_PARSER=${REASONING_PARSER:-qwen3}
      return 0
      ;;
  esac
  case "$profile_l" in
    qwen27b/*)
      REASONING_PARSER=${REASONING_PARSER:-qwen3}
      return 0
      ;;
  esac
  case "$model_dir_l $served_l" in
    *qwen3*|*qwen-3*|*qwen_3*|*qwopus3*|*qwen36*)
      REASONING_PARSER=${REASONING_PARSER:-qwen3}
      return 0
      ;;
  esac
  if [[ "${REASONING_PARSER:-}" == "qwen3" ]]; then
    REASONING_PARSER=""
  fi
}

apply_family_reasoning_defaults

# Parallelism constraints
enforce_parallelism_constraints() {
  local pp=${PP_SIZE:-1}
  local mtp=${MTP_K:-0}

  if [[ "$pp" =~ ^[2-9][0-9]*$ ]] && [[ "$mtp" =~ ^[1-9][0-9]*$ ]]; then
    if [[ "${NO_ASYNC_SCHEDULING:-0}" != "1" ]]; then
      NO_ASYNC_SCHEDULING=1
      PP_MTP_ASYNC_AUTO_DISABLED=1
    fi
  fi
}

enforce_parallelism_constraints

# ---- Build vLLM args (from build_args) ----

build_args() {
  local host_arg=$1

  VLLM_ARGS=(
    --host "$host_arg"
    --port "$PORT"
    --model "$MODEL_DIR"
    --served-model-name "$SERVED_NAME"
    --dtype half
    --tensor-parallel-size "${TP_SIZE:-2}"
    --pipeline-parallel-size "${PP_SIZE:-1}"
    --generation-config vllm
    --max-model-len "$MAX_MODEL_LEN"
    --enable-chunked-prefill
    --max-num-seqs "$MAX_NUM_SEQS"
    --max-num-batched-tokens "$MAX_BATCHED_TOKENS"
    --enable-log-requests

  )

  if [[ -n "${QUANTIZATION:-}" && "$QUANTIZATION" != "auto" ]]; then
    VLLM_ARGS+=(--quantization "$QUANTIZATION")
  fi
  [[ -n "${KV_CACHE_DTYPE:-}" ]] && VLLM_ARGS+=(--kv-cache-dtype "$KV_CACHE_DTYPE")
  if [[ -n "${KV_CACHE_MEMORY_BYTES:-}" ]]; then
    VLLM_ARGS+=(--kv-cache-memory-bytes "$KV_CACHE_MEMORY_BYTES")
  else
    VLLM_ARGS+=(--gpu-memory-utilization "$GPU_UTIL")
  fi
  [[ -n "${MAMBA_CACHE_MODE:-}" ]] && VLLM_ARGS+=(--mamba-cache-mode "$MAMBA_CACHE_MODE")
  [[ "${ENFORCE_EAGER:-0}" == "1" ]] && VLLM_ARGS+=(--enforce-eager)
  [[ "${NO_ASYNC_SCHEDULING:-0}" == "1" ]] && VLLM_ARGS+=(--no-async-scheduling)
  [[ "${DISABLE_HYBRID_KV_CACHE_MANAGER:-0}" == "1" ]] && VLLM_ARGS+=(--disable-hybrid-kv-cache-manager)
  if [[ "${DISABLE_PREFIX_CACHING:-0}" == "1" ]]; then
    VLLM_ARGS+=(--no-enable-prefix-caching)
  elif [[ "${ENABLE_PREFIX_CACHING:-1}" == "1" ]]; then
    VLLM_ARGS+=(--enable-prefix-caching)
  fi
  [[ "${ENABLE_PROMPT_TOKENS_DETAILS:-1}" == "1" ]] && VLLM_ARGS+=(--enable-prompt-tokens-details)
  [[ "${LANGUAGE_MODEL_ONLY:-0}" == "1" ]] && VLLM_ARGS+=(--language-model-only)
  [[ "${SKIP_MM_PROFILING:-0}" == "1" ]] && VLLM_ARGS+=(--skip-mm-profiling)
  [[ "${DISABLE_CUSTOM_ALL_REDUCE:-0}" == "1" ]] && VLLM_ARGS+=(--disable-custom-all-reduce)
  [[ -n "${ATTENTION_BACKEND:-}" ]] && VLLM_ARGS+=(--attention-backend "$ATTENTION_BACKEND")
  if [[ -n "${REASONING_PARSER:-}" ]] && ! reasoning_parser_is_disabled; then
    VLLM_ARGS+=(--reasoning-parser "$REASONING_PARSER")
  fi
  [[ -n "${DEFAULT_CHAT_TEMPLATE_KWARGS:-}" ]] && VLLM_ARGS+=(--default-chat-template-kwargs "$DEFAULT_CHAT_TEMPLATE_KWARGS")
  [[ -n "${TOOL_PARSER_PLUGIN:-}" ]] && VLLM_ARGS+=(--tool-parser-plugin "$TOOL_PARSER_PLUGIN")
  [[ -n "${TOOL_CALL_PARSER:-}" ]] && VLLM_ARGS+=(--tool-call-parser "$TOOL_CALL_PARSER")
  [[ "${ENABLE_AUTO_TOOL_CHOICE:-0}" == "1" ]] && VLLM_ARGS+=(--enable-auto-tool-choice)
  [[ -n "${ADDITIONAL_CONFIG_JSON:-}" ]] && VLLM_ARGS+=(--additional-config "$ADDITIONAL_CONFIG_JSON")
  [[ -n "${HF_OVERRIDES_JSON:-}" ]] && VLLM_ARGS+=(--hf-overrides "$HF_OVERRIDES_JSON")

  if [[ -n "${MM_LIMIT_JSON:-}" ]]; then
    VLLM_ARGS+=(--limit-mm-per-prompt "$MM_LIMIT_JSON")
  elif [[ "$MODEL_FAMILY" == qwen* && -z "${ADDITIONAL_CONFIG_JSON:-}" ]]; then
    VLLM_ARGS+=(--additional-config '{"gdn_prefill_backend":"flashqla_legacy"}')
  elif [[ "$MODEL_FAMILY" == gemma* ]]; then
    VLLM_ARGS+=(--limit-mm-per-prompt '{"image":0,"video":0,"audio":0}')
  fi

  if [[ "$MODEL_FAMILY" == qwen* && -n "${MM_LIMIT_JSON:-}" && -z "${ADDITIONAL_CONFIG_JSON:-}" ]]; then
    VLLM_ARGS+=(--additional-config '{"gdn_prefill_backend":"flashqla_legacy"}')
  fi

  if [[ -n "${CHAT_TEMPLATE_PRESET:-}" ]]; then
    local resolved_template
    if resolved_template=$(resolve_template_file "$CHAT_TEMPLATE_PRESET"); then
      CHAT_TEMPLATE_FILE="$resolved_template"
    fi
  fi
  [[ -n "${CHAT_TEMPLATE_FILE:-}" ]] && VLLM_ARGS+=(--chat-template "$CHAT_TEMPLATE_FILE")

  local capture=$((MTP_K + 1))
  local prefill_capture=${MAX_BATCHED_TOKENS:-2048}
  if (( prefill_capture < capture )); then
    prefill_capture=$capture
  fi
  if [[ -n "${SPECULATIVE_CONFIG:-}" ]]; then
    VLLM_ARGS+=(--speculative-config "$SPECULATIVE_CONFIG")
  elif (( MTP_K > 0 )); then
    VLLM_ARGS+=(--speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":${MTP_K}}")
  fi

  local cudagraph_mode
  case "$MODE" in
    safe)
      cudagraph_mode=PIECEWISE
      ;;
    normal)
      if (( MTP_K > 0 )) || [[ -n "${SPECULATIVE_CONFIG:-}" ]]; then
        cudagraph_mode=PIECEWISE
      else
        cudagraph_mode=FULL_AND_PIECEWISE
      fi
      ;;
    fast|aggressive)
      cudagraph_mode=FULL_AND_PIECEWISE
      ;;
    *)
      cudagraph_mode=PIECEWISE
      ;;
  esac

  if [[ -n "${COMPILATION_CONFIG_JSON:-}" ]]; then
    VLLM_ARGS+=(--compilation-config "$COMPILATION_CONFIG_JSON")
  elif [[ -n "${SPECULATIVE_CONFIG:-}" || "$MTP_K" -gt 0 ]]; then
    VLLM_ARGS+=(--compilation-config "{\"cudagraph_mode\":\"${cudagraph_mode}\",\"cudagraph_capture_sizes\":[${capture},${prefill_capture}],\"max_cudagraph_capture_size\":${prefill_capture}}")
  else
    VLLM_ARGS+=(--compilation-config "{\"cudagraph_mode\":\"${cudagraph_mode}\",\"cudagraph_capture_sizes\":[1,${prefill_capture}],\"max_cudagraph_capture_size\":${prefill_capture}}")
  fi
}

# ---- Host binding ----

if [[ "$SERVICE_SCOPE" == "lan" ]]; then
  host_arg="0.0.0.0"
else
  host_arg="127.0.0.1"
fi

build_args "$host_arg"

# ---- Launch summary ----

guess_precision_scheme() {
  local dir=${1:-${MODEL_DIR:-}}
  local quantization=${2:-${QUANTIZATION:-$(guess_quantization "$dir")}}
  dir=${dir,,}

  if [[ "$dir" == *w8a8* || "$quantization" == "quark" ]]; then
    echo W8A8
  elif [[ "$dir" == *nvfp4* || "$dir" == *mxfp4* ]]; then
    echo W4A16
  elif [[ "$dir" == *fp8* || "$quantization" == "fp8" ]]; then
    echo W8A16
  elif [[ "$dir" == *w8a16* || "$dir" == *int8* ]]; then
    echo W8A16
  elif [[ "$dir" == *w4a16* || "$dir" == *int4* || "$dir" == *4bit* || "$dir" == *awq* ]]; then
    echo W4A16
  elif [[ "$quantization" == "awq_marlin" || "$quantization" == "gptq_marlin" ]]; then
    echo W4A16
  else
    echo auto
  fi
}

current_prefix_cache_label() {
  if [[ "${DISABLE_PREFIX_CACHING:-0}" == "1" ]]; then
    printf 'disabled'
  elif [[ "${ENABLE_PREFIX_CACHING:-1}" == "1" ]]; then
    printf 'enabled'
  else
    printf 'auto'
  fi
}

current_tool_calling_label() {
  local label plugin_label
  if [[ "${ENABLE_AUTO_TOOL_CHOICE:-0}" == "1" ]]; then
    label="auto"
    if [[ -n "${TOOL_CALL_PARSER:-}" ]]; then
      label+=" / parser=$TOOL_CALL_PARSER"
    else
      label+=" / parser=<unset>"
    fi
  else
    label="off"
    if [[ -n "${TOOL_CALL_PARSER:-}" ]]; then
      label+=" / parser=$TOOL_CALL_PARSER"
    fi
  fi
  if [[ -n "${TOOL_PARSER_PLUGIN:-}" ]]; then
    plugin_label=${TOOL_PARSER_PLUGIN##*/}
    label+=" / plugin=$plugin_label"
  fi
  printf '%s\n' "$label"
}

current_reasoning_label() {
  local label="template default"
  if [[ -n "${DEFAULT_CHAT_TEMPLATE_KWARGS:-}" ]]; then
    case "${DEFAULT_CHAT_TEMPLATE_KWARGS//[[:space:]]/}" in
      *'"enable_thinking":false'*|*"\"enable_thinking\":false"*)
        label="thinking off"
        ;;
      *'"enable_thinking":true'*|*"\"enable_thinking\":true"*)
        label="thinking on"
        ;;
      *)
        label="custom kwargs"
        ;;
    esac
  fi
  if [[ -n "${REASONING_PARSER:-}" ]]; then
    label+=" / parser=$REASONING_PARSER"
  fi
  if [[ -n "${REASONING_BUDGET:-}" ]]; then
    label+=" / default budget=$REASONING_BUDGET"
  fi
  printf '%s\n' "$label"
}

current_tq_diagnostics_label() {
  if [[ "${KV_CACHE_DTYPE:-}" != turboquant_* ]]; then
    printf 'n/a'
    return 0
  fi
  printf 'FORCE_DECODE_SDPA=%s, FORCE_CONTINUATION_SDPA=%s, PREFIX_COMBINE=%s@%s, MAX_KV_SPLITS=%s, DECODE_BLOCK_KV=%s, K8V4_FP8_FORMAT=%s' \
    "${VLLM_TURBOQUANT_FORCE_DECODE_SDPA:-0}" \
    "${VLLM_TURBOQUANT_FORCE_CONTINUATION_SDPA:-0}" \
    "${VLLM_TURBOQUANT_CONTINUATION_PREFIX_COMBINE:-auto}" \
    "${VLLM_TURBOQUANT_CONTINUATION_PREFIX_COMBINE_MIN_TOKENS:-20480}" \
    "${VLLM_TURBOQUANT_MAX_KV_SPLITS:-auto}" \
    "${VLLM_TURBOQUANT_DECODE_BLOCK_KV:-4}" \
    "${VLLM_TURBOQUANT_K8V4_FP8_FORMAT:-auto}"
}

printf '============================================================\n'
printf ' %s v%s\n' "$PROJECT_NAME" "$VERSION"
printf ' Runtime identity: %s\n' "$RUNTIME_IDENTITY"
printf ' Base vLLM: %s\n' "$BASE_VLLM_VERSION"
printf ' Launch time: %s\n' "$(date '+%F %T %Z')"
printf ' Served name: %s\n' "$SERVED_NAME"
printf ' Model: %s\n' "$MODEL_DIR"
printf ' Profile: %s\n' "${PROFILE:-manual}"
printf ' Mode: %s\n' "$MODE"
printf ' GPU devices: %s\n' "${GPU_DEVICES:-}"
printf ' TP / PP: %s / %s\n' "${TP_SIZE:-2}" "${PP_SIZE:-1}"
printf ' Port: %s\n' "$PORT"
printf ' Scope: %s\n' "$SERVICE_SCOPE"
printf ' Quantization: %s\n' "${QUANTIZATION:-auto}"
printf ' W/A type: %s\n' "$(guess_precision_scheme "$MODEL_DIR" "${QUANTIZATION:-}")"
printf ' KV precision: %s\n' "${KV_CACHE_DTYPE:-fp16}"
printf ' TQ diagnostics: %s\n' "$(current_tq_diagnostics_label)"
printf ' Prefix cache: %s\n' "$(current_prefix_cache_label)"
printf ' Mamba cache mode: %s\n' "${MAMBA_CACHE_MODE:-auto}"
printf ' Context tokens: %s\n' "$MAX_MODEL_LEN"
printf ' GPU util: %s\n' "$GPU_UTIL"
printf ' Max batched tokens: %s\n' "$MAX_BATCHED_TOKENS"
printf ' Max sequences: %s\n' "$MAX_NUM_SEQS"
printf ' MTP tokens: %s\n' "$MTP_K"
printf ' Message type: %s\n' "${MESSAGE_TYPE:-text-only}"
printf ' Chat template: %s\n' "${CHAT_TEMPLATE_PRESET:-model default}"
printf ' Reasoning default: %s\n' "$(current_reasoning_label)"
printf ' Tool calling: %s\n' "$(current_tool_calling_label)"
printf ' MTP graph policy: VLLM_SM75_SPEC_SYNC_MODE=%s, VLLM_ALLOW_MAMBA_SPEC_FULL_CUDAGRAPH=%s\n' "${VLLM_SM75_SPEC_SYNC_MODE:-auto}" "${VLLM_ALLOW_MAMBA_SPEC_FULL_CUDAGRAPH:-0}"
printf ' Async scheduling: %s\n' "$(if [[ "${NO_ASYNC_SCHEDULING:-0}" == "1" ]]; then printf 'disabled'; else printf 'enabled'; fi)"
printf ' Strict tool calling: VLLM_ENFORCE_STRICT_TOOL_CALLING=%s\n' "${VLLM_ENFORCE_STRICT_TOOL_CALLING:-0}"
printf '============================================================\n'
printf '\n'

# ---- Execute vLLM ----

args_text=$(printf '%q ' "${VLLM_ARGS[@]}")
printf 'Command: %s/.venv/bin/python -m vllm.entrypoints.openai.api_server %s\n' "$RUNTIME_ROOT" "$args_text"
printf '\n'

exec "$RUNTIME_ROOT/.venv/bin/python" -m vllm.entrypoints.openai.api_server "${VLLM_ARGS[@]}"
