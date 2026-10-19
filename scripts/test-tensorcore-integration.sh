#!/bin/sh
# Compile/simulate one integration target at a time, including native compiles.
set -eu
TENSORCORE_INTEGRATION_SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$TENSORCORE_INTEGRATION_SCRIPT_DIR/.."
export JAVA_TOOL_OPTIONS="${JAVA_TOOL_OPTIONS:+$JAVA_TOOL_OPTIONS }-Xmx1536m -XX:ActiveProcessorCount=1"
export TENSORCORE_SIM=verilator
case "${1:-}" in
  compile) exec mill --no-server -j 1 integration.compile ;;
  rtl-e4m3|rtl-fp16|rtl-nvfp4|rtl-disabled)
    exec mill --no-server -j 1 integration.runMain gemmini.integration.GenerateGemminiIntegration "${1#rtl-}" "out/integration-rtl-${1#rtl-}"
    ;;
  e4m3-f32|e4m3-f16|e5m2-f32|e5m2-f16|fp16-f32|fp16-f16|bf16-f32|mxfp4|nvfp4)
    exec mill --no-server -j 1 integration.test.testOnly gemmini.integration.GemminiRoCCSpec -- -z "$1"
    ;;
  *) echo 'Usage: scripts/test-tensorcore-integration.sh <compile|rtl-e4m3|rtl-fp16|rtl-nvfp4|rtl-disabled|PROFILE>' >&2; exit 2 ;;
esac
