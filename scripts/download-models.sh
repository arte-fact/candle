#!/usr/bin/env bash
# ============================================================================
# download-models.sh — Fetch the benchmark/test GGUF collection from Hugging Face.
#
# All checkpoints live in Unsloth's GGUF repos. Files are byte-verified against
# the expected Content-Length so a partial or redirected download is caught.
#
# Usage:
#   scripts/download-models.sh                 # fetch every missing model
#   scripts/download-models.sh --list          # show the catalog and status
#   scripts/download-models.sh --check         # verify sizes, download nothing
#   scripts/download-models.sh NAME [NAME...]  # fetch specific filename(s)
#
# Environment:
#   MODELS_DIR   override destination (default: <repo>/models)
#   HF_TOKEN     optional Hugging Face access token for gated repos
# ============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODELS_DIR="${MODELS_DIR:-$REPO_ROOT/models}"
HF_BASE="https://huggingface.co"

# filename <TAB> hf_repo <TAB> expected_bytes
CATALOG=$(cat <<'EOF'
Devstral-Small-2-24B-Instruct-2512-UD-Q4_K_XL.gguf	unsloth/Devstral-Small-2-24B-Instruct-2512-GGUF	14506151072
Qwen3-Coder-30B-A3B-Instruct-1M-Q4_0.gguf	unsloth/Qwen3-Coder-30B-A3B-Instruct-1M-GGUF	17379990880
Qwen3-Coder-Next-Q4_0.gguf	unsloth/Qwen3-Coder-Next-GGUF	45330098208
Qwen3.5-9B-BF16.gguf	unsloth/Qwen3.5-9B-GGUF	17920697312
Qwen3.5-27B-Q4_1.gguf	unsloth/Qwen3.5-27B-GGUF	17182934944
Qwen3.5-35B-A3B-MXFP4_MOE.gguf	unsloth/Qwen3.5-35B-A3B-GGUF	21587638912
gemma-4-E4B-it-Q8_0.gguf	unsloth/gemma-4-E4B-it-GGUF	8192950976
gemma-4-26B-A4B-it-MXFP4_MOE.gguf	unsloth/gemma-4-26B-A4B-it-GGUF	16630345024
gemma-4-31B-it-Q4_K_M.gguf	unsloth/gemma-4-31B-it-GGUF	18323730976
EOF
)

human_size() {
    awk -v b="$1" 'BEGIN{
        split("B KiB MiB GiB TiB", u, " ");
        for (i=1; b>=1024 && i<5; i++) b/=1024;
        printf "%.2f %s", b, u[i];
    }'
}

local_size() { stat -c '%s' "$1" 2>/dev/null || echo 0; }

cmd_list() {
    printf '%-55s %-55s %12s  %s\n' "FILE" "REPO" "SIZE" "STATUS"
    while IFS=$'\t' read -r file repo bytes; do
        [[ -z "$file" ]] && continue
        path="$MODELS_DIR/$file"
        got=$(local_size "$path")
        if [[ "$got" -eq "$bytes" ]]; then status="ok"
        elif [[ "$got" -eq 0 ]]; then status="missing"
        else status="partial ($got/$bytes)"
        fi
        printf '%-55s %-55s %12s  %s\n' "$file" "$repo" "$(human_size "$bytes")" "$status"
    done <<< "$CATALOG"
}

fetch_one() {
    local file="$1" repo="$2" want="$3"
    local dest="$MODELS_DIR/$file"
    local url="$HF_BASE/$repo/resolve/main/$file"

    local got; got=$(local_size "$dest")
    if [[ "$got" -eq "$want" ]]; then
        echo "[ok]    $file ($(human_size "$want")) — already complete"
        return 0
    fi

    echo "[fetch] $file  <-  $repo  ($(human_size "$want"))"
    local auth=()
    [[ -n "${HF_TOKEN:-}" ]] && auth=(-H "Authorization: Bearer $HF_TOKEN")

    curl -L --fail --retry 5 --retry-delay 5 --retry-connrefused \
         -C - --progress-bar \
         "${auth[@]}" \
         -o "$dest" "$url"

    got=$(local_size "$dest")
    if [[ "$got" -ne "$want" ]]; then
        echo "[err]   $file size mismatch: got $got, expected $want" >&2
        return 1
    fi
    echo "[done]  $file verified ($(human_size "$want"))"
}

cmd_check() {
    local bad=0
    while IFS=$'\t' read -r file repo bytes; do
        [[ -z "$file" ]] && continue
        got=$(local_size "$MODELS_DIR/$file")
        if [[ "$got" -ne "$bytes" ]]; then
            echo "[bad]   $file  got=$got  want=$bytes"
            bad=$((bad + 1))
        fi
    done <<< "$CATALOG"
    if [[ "$bad" -eq 0 ]]; then
        echo "All models present and size-verified."
    else
        echo "$bad model(s) missing or incomplete." >&2
        exit 1
    fi
}

cmd_fetch() {
    local -a filter=("$@")
    mkdir -p "$MODELS_DIR"
    local failed=0
    while IFS=$'\t' read -r file repo bytes; do
        [[ -z "$file" ]] && continue
        if [[ ${#filter[@]} -gt 0 ]]; then
            local match=0
            for f in "${filter[@]}"; do [[ "$f" == "$file" ]] && match=1; done
            [[ "$match" -eq 0 ]] && continue
        fi
        fetch_one "$file" "$repo" "$bytes" || failed=$((failed + 1))
    done <<< "$CATALOG"
    [[ "$failed" -eq 0 ]] || { echo "$failed download(s) failed." >&2; exit 1; }
}

case "${1:-}" in
    -h|--help) sed -n '3,17p' "$0"; exit 0 ;;
    --list)    cmd_list ;;
    --check)   cmd_check ;;
    "")        cmd_fetch ;;
    *)         cmd_fetch "$@" ;;
esac
