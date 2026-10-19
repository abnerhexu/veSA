package gemmini.tensorcore

/** Standalone RTL generation; e.g. tensorcore.runMain ...Generate e4m3 f32 out/rtl */
object Generate extends App {
  require(args.length == 3, "Usage: Generate <e4m3|e5m2|fp16|bf16> <f32|f16> <output-directory>")
  val formats = Seq(TensorFormat.E4M3, TensorFormat.E5M2, TensorFormat.FP16, TensorFormat.BF16)
    .map(f => f.name -> f).toMap
  require(formats.contains(args(0)), "Only FP16/BF16 and lower input formats are supported")
  require(Set("f32", "f16").contains(args(1)))
  val config = BlackwellConfig(formats(args(0)), args(1) == "f16")
  (new chisel3.stage.ChiselStage).emitSystemVerilog(new TensorMesh(config),
    Array("--target-dir", args(2)))
}
