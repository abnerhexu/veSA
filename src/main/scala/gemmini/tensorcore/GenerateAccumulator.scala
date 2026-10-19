package gemmini.tensorcore

import chisel3._
import gemmini.{AccumulatorBankRouter, AccumulatorMemCore}

/** Export the real raw C/D storage path; 32-bit ports are storage, not A/B MMA. */
object GenerateAccumulator extends App {
  require(args.length == 2, "Usage: GenerateAccumulator <bridge|router|dual|single> <output-directory>")
  require(Set("bridge", "router", "dual", "single").contains(args(0)))
  def design(): Module = {
    val mem = TensorMemoryConfig()
    val row = Vec(mem.dim, Vec(1, UInt(32.W)))
    args(0) match {
      case "bridge" => new TensorAccumulatorBridge(mem, row, UInt(32.W))
      case "router" => new AccumulatorBankRouter(mem.accDepth, row, UInt(32.W))
      case _ => new AccumulatorMemCore(mem.accDepth, row, (x: UInt, _: UInt) => x,
        UInt(32.W), args(0) == "single", if (args(0) == "single") 2 else -1,
        false, false, 2, UInt(32.W), false)
    }
  }
  (new chisel3.stage.ChiselStage).emitSystemVerilog(design(), Array("--target-dir", args(1)))
}
