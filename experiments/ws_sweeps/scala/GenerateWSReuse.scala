package gemmini.wsexperiments

import chisel3.stage.ChiselStage
import gemmini.tensorcore.{BlackwellConfig, TensorFormat}

object GenerateWSReuse extends App {
  require(args.length == 5,
    "GenerateWSReuse PHYSICAL_DEPTH ROWS COLUMNS MAX_K OUTPUT_DIR")
  val physicalDepth = args(0).toInt
  val rows = args(1).toInt
  val columns = args(2).toInt
  val maxK = args(3).toInt
  val output = args(4)
  (new ChiselStage).emitSystemVerilog(
    new WSReuseExperimentMesh(BlackwellConfig(TensorFormat.E4M3), rows,
      columns, physicalDepth, maxK),
    Array("--target-dir", output, "--output-file", "WSReuseExperimentMesh.sv"))
}
