#!/bin/sh
# One configuration and one native build at a time; never launch tasks in parallel.
set -eu
TENSORCORE_SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$TENSORCORE_SCRIPT_DIR/.."
export TENSORCORE_SIM=verilator
case "${1:-}" in
  fence)
    exec mill --no-server -j 1 tensorcore.test.testOnly gemmini.tensorcore.TensorCommandFrontendSpec
    ;;
  isolation)
    exec mill --no-server -j 1 tensorcore.test.testOnly gemmini.tensorcore.TensorCommandIsolationSpec
    ;;
  cmd-e4m3-f32|cmd-e4m3-f16|cmd-e5m2-f32|cmd-e5m2-f16|cmd-fp16-f32|cmd-fp16-f16|cmd-bf16-f32|cmd-mxfp4|cmd-nvfp4)
    exec mill --no-server -j 1 tensorcore.test.testOnly gemmini.tensorcore.TensorCommandExecuteSpec -- -z "${1#cmd-}"
    ;;
  cmd-single-e4m3-f32|cmd-single-mxfp4|cmd-single-nvfp4)
    exec mill --no-server -j 1 tensorcore.test.testOnly gemmini.tensorcore.TensorCommandSinglePortSpec -- -z "${1#cmd-single-}"
    ;;
  control)
    exec mill --no-server -j 1 tensorcore.test.testOnly gemmini.tensorcore.TensorExecuteControlSpec
    ;;
  contracts)
    exec mill --no-server -j 1 tensorcore.test.testOnly gemmini.tensorcore.AccumulatorContractsSpec
    ;;
  router)
    exec mill --no-server -j 1 tensorcore.test.testOnly gemmini.tensorcore.AccumulatorRouterSpec
    ;;
  acc-single-e4m3-f32|acc-single-mxfp4|acc-single-nvfp4)
    exec mill --no-server -j 1 tensorcore.test.testOnly gemmini.tensorcore.TensorAccumulatorSinglePortSpec -- -z "${1#acc-single-}"
    ;;
  acc-e4m3-f32|acc-e4m3-f16|acc-e5m2-f32|acc-e5m2-f16|acc-fp16-f32|acc-fp16-f16|acc-bf16-f32|acc-mxfp4|acc-nvfp4)
    exec mill --no-server -j 1 tensorcore.test.testOnly gemmini.tensorcore.TensorAccumulatorExecuteSpec -- -z "${1#acc-}"
    ;;
  e4m3-f32|e4m3-f16|e5m2-f32|e5m2-f16|fp16-f32|fp16-f16|bf16-f32|mxfp4|nvfp4)
    exec mill --no-server -j 1 tensorcore.test.testOnly gemmini.tensorcore.TensorExecuteSpec -- -z "$1"
    ;;
  *)
    echo 'Usage: scripts/test-tensorcore-memory.sh <control|contracts|router|fence|isolation|PROFILE|acc-PROFILE|acc-single-PROFILE|cmd-PROFILE|cmd-single-PROFILE>' >&2
    echo 'PROFILE: e4m3-f32 e4m3-f16 e5m2-f32 e5m2-f16 fp16-f32 fp16-f16 bf16-f32 mxfp4 nvfp4; acc-single: e4m3-f32 mxfp4 nvfp4' >&2
    exit 2
    ;;
esac
