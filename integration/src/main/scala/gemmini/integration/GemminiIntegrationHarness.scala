package gemmini.integration

import chisel3._
import chisel3.util._
import freechips.rocketchip.diplomacy._
import freechips.rocketchip.rocket.RocketCoreParams
import freechips.rocketchip.system.BaseConfig
import freechips.rocketchip.tile.{RocketTileParams, TileKey, TileVisibilityNodeKey}
import freechips.rocketchip.tilelink._
import gemmini._
import gemmini.Arithmetic.SIntArithmetic
import gemmini.tensorcore._
import org.chipsalliance.cde.config.{Config, Parameters}

object IntegrationParameters {
  def apply(): Parameters = (new Config((site, here, up) => {
    case TileKey => RocketTileParams(core = RocketCoreParams(useVM = false, nPMPs = 0))
  }) ++ new BaseConfig)
  def accelerator(config: GemminiArrayConfig[SInt, Float, Float])
                 (implicit p: Parameters): Gemmini[SInt, Float, Float] = new Gemmini(config)
}

/** Real Gemmini RoCC accelerator, TLB, ROB, controllers, DMA and internal SRAM.
  * A second TileLink client lets the test initialize/read system memory. No
  * CPU boot, VM page walk or software toolchain is needed for machine-mode
  * command validation; no accelerator component is replaced with a stub.
  */
class GemminiIntegrationHarness(profile: BlackwellConfig = BlackwellConfig(),
                                singlePorted: Boolean = false,
                                tensorEnabled: Boolean = true)
                               (implicit p: Parameters) extends LazyModule {
  java.nio.file.Files.createDirectories(java.nio.file.Paths.get("out"))
  val config = TensorCoreConfigs.blackwell(profile, singlePortedAccumulator = singlePorted).copy(
    dma_buswidth = 64, headerFileName = "out/tensorcore-integration-params.h",
    tensorcore = if (tensorEnabled) Some(profile) else None)
  val visibility = TLEphemeralNode()
  val accelerator = LazyModule(IntegrationParameters.accelerator(config)(p.alterPartial {
    case TileVisibilityNodeKey => visibility
  }))
  val host = TLClientNode(Seq(TLMasterPortParameters.v1(Seq(TLMasterParameters.v1(
    name = "integration_host", sourceId = IdRange(0, 1))))))
  val bus = TLXbar()
  val ram = LazyModule(new TLRAM(AddressSet(0, 0xffff), beatBytes = 8))
  ram.node := TLFragmenter(8, 64) := bus
  bus := visibility := accelerator.tlNode
  bus := host

  lazy val module = new Impl
  class Impl extends LazyModuleImp(this) {
    val io = IO(new Bundle {
      val cmd = Flipped(Decoupled(new TensorCommand()))
      val response = Decoupled(new TensorCommandResponse())
      val busy = Output(Bool())
      val interrupt = Output(Bool())
      val memory = Flipped(new TLBundle(host.out.head._2.bundle))
    })
    host.out.head._1 <> io.memory
    val rocc = accelerator.module.io
    rocc.cmd.valid := io.cmd.valid
    io.cmd.ready := rocc.cmd.ready
    rocc.cmd.bits := 0.U.asTypeOf(rocc.cmd.bits)
    rocc.cmd.bits.inst.opcode := "h7b".U // custom3
    rocc.cmd.bits.inst.funct := io.cmd.bits.funct
    rocc.cmd.bits.inst.rd := io.cmd.bits.rd
    rocc.cmd.bits.inst.xd := io.cmd.bits.xd
    rocc.cmd.bits.inst.xs1 := true.B
    rocc.cmd.bits.inst.xs2 := true.B
    rocc.cmd.bits.rs1 := io.cmd.bits.rs1
    rocc.cmd.bits.rs2 := io.cmd.bits.rs2
    rocc.cmd.bits.status.prv := 3.U
    rocc.cmd.bits.status.dprv := 3.U
    io.response.valid := rocc.resp.valid
    io.response.bits.rd := rocc.resp.bits.rd
    io.response.bits.data := rocc.resp.bits.data
    rocc.resp.ready := io.response.ready
    io.busy := rocc.busy
    io.interrupt := rocc.interrupt
    rocc.exception := false.B
    rocc.mem := DontCare // Gemmini DMA uses the diplomatic TileLink node.
    rocc.fpu_req.ready := false.B
    rocc.fpu_resp.valid := false.B
    rocc.fpu_resp.bits := 0.U.asTypeOf(rocc.fpu_resp.bits)
    for (port <- rocc.ptw) {
      port.req.ready := true.B
      port.resp.valid := false.B
      port.resp.bits := 0.U.asTypeOf(port.resp.bits)
      port.ptbr := 0.U.asTypeOf(port.ptbr)
      port.hgatp := 0.U.asTypeOf(port.hgatp)
      port.vsatp := 0.U.asTypeOf(port.vsatp)
      port.status := 0.U.asTypeOf(port.status)
      port.hstatus := 0.U.asTypeOf(port.hstatus)
      port.gstatus := 0.U.asTypeOf(port.gstatus)
      port.pmp := 0.U.asTypeOf(port.pmp)
      port.customCSRs := DontCare
      assert(!port.req.valid, "Machine-mode integration unexpectedly requested a page walk")
    }
  }
}

object GenerateGemminiIntegration extends App {
  require(args.length == 2, "Usage: GenerateGemminiIntegration <e4m3|fp16|nvfp4|disabled> <output-directory>")
  val profile = args(0) match {
    case "e4m3" | "disabled" => BlackwellConfig()
    case "fp16" => BlackwellConfig(TensorFormat.FP16)
    case "nvfp4" => BlackwellConfig.nvfp4
    case _ => throw new IllegalArgumentException("Unsupported integration generation profile")
  }
  implicit val p: Parameters = IntegrationParameters()
  (new chisel3.stage.ChiselStage).emitSystemVerilog(
    LazyModule(new GemminiIntegrationHarness(profile, tensorEnabled = args(0) != "disabled")).module,
    Array("--target-dir", args(1)))
}
