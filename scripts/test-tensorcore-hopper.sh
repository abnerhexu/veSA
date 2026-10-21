#!/bin/sh
# Run one task to completion; no background jobs and no GPU dependency.
set -eu
TENSORCORE_HOPPER_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$TENSORCORE_HOPPER_DIR/.."
export JAVA_TOOL_OPTIONS="${JAVA_TOOL_OPTIONS:+$JAVA_TOOL_OPTIONS }-Xmx1536m -XX:ActiveProcessorCount=1"
export TENSORCORE_SIM=verilator
export MAKEFLAGS=-j1
case "${1:-}" in
  compile) exec mill --no-server -j 1 integration.compile ;;
  config) exec mill --no-server -j 1 tensorcore.test.testOnly gemmini.tensorcore.TensorProfilesSpec ;;
  cpu) exec "${TENSORCORE_PYTHON:-../.venv/bin/python}" modeling/tensorcore/validate.py --architecture hopper --cases 150000 ;;
  rtl-e4m3|rtl-fp16)
    exec mill --no-server -j 1 integration.runMain gemmini.integration.GenerateGemminiIntegration "hopper-${1#rtl-}" "out/integration-rtl-hopper-${1#rtl-}"
    ;;
  pe-*|mesh-*|integration-*)
    kind=${1%%-*}
    profile=${1#*-}
    case "$profile" in
      e4m3-f32|e4m3-f16|e5m2-f32|e5m2-f16|fp16-f32|fp16-f16|bf16-f32) ;;
      *) echo 'Hopper supports seven unscaled FP8/FP16/BF16 profiles only' >&2; exit 2 ;;
    esac
    case "$kind" in
      pe) exec mill --no-server -j 1 tensorcore.test.testOnly gemmini.tensorcore.HopperPESpec -- -z "$profile" ;;
      mesh) exec mill --no-server -j 1 tensorcore.test.testOnly gemmini.tensorcore.HopperMeshSpec -- -z "$profile" ;;
      integration) exec mill --no-server -j 1 integration.test.testOnly gemmini.integration.HopperRoCCSpec -- -z "$profile" ;;
    esac
    ;;
  *) echo 'Usage: scripts/test-tensorcore-hopper.sh <compile|config|cpu|rtl-e4m3|rtl-fp16|pe-PROFILE|mesh-PROFILE|integration-PROFILE>' >&2; exit 2 ;;
esac
