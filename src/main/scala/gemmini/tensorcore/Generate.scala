package gemmini.tensorcore

private[tensorcore] object ElaborationProfile {
  def parse(args: Array[String]): TensorConfig = {
    require(args.length == 3, "Usage: <e4m3|e5m2|fp16|bf16|mxfp4|nvfp4> <f32|f16> <output-directory>")
    val formats = Seq(TensorFormat.E4M3, TensorFormat.E5M2, TensorFormat.FP16, TensorFormat.BF16)
      .map(f => f.name -> f).toMap
    val hopper = args(0).startsWith("hopper-")
    val name = args(0).stripPrefix("hopper-")
    require(formats.contains(name) || Set("mxfp4", "nvfp4").contains(name),
      "Only FP16/BF16 and lower input formats are supported")
    require(Set("f32", "f16").contains(args(1)))
    require(!hopper || formats.contains(name), "Hopper has no native FP4 profile")
    name match {
      case "mxfp4" => require(args(1) == "f32"); BlackwellConfig.mxfp4
      case "nvfp4" => require(args(1) == "f32"); BlackwellConfig.nvfp4
      case _ => if (hopper) HopperConfig(formats(name), args(1) == "f16")
        else BlackwellConfig(formats(name), args(1) == "f16")
    }
  }
}

/** Standalone RTL generation; e.g. tensorcore.runMain ...Generate e4m3 f32 out/rtl */
object Generate extends App {
  val config = ElaborationProfile.parse(args)
  (new chisel3.stage.ChiselStage).emitSystemVerilog(new TensorMesh(config),
    Array("--target-dir", args(2)))
}

/** Memory execution adapter, without Rocket/Chipyard or a GPU. */
object GenerateExecute extends App {
  val config = ElaborationProfile.parse(args)
  (new chisel3.stage.ChiselStage).emitSystemVerilog(new TensorExecute(config),
    Array("--target-dir", args(2)))
}

/** Command/fence RTL only; not a substitute for full Gemmini elaboration. */
object GenerateCommandFrontend extends App {
  val config = ElaborationProfile.parse(args)
  (new chisel3.stage.ChiselStage).emitSystemVerilog(new TensorCommandFrontend(config),
    Array("--target-dir", args(2)))
}
