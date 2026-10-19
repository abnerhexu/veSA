package gemmini.tensorcore

import chisel3._
import chisel3.util._

/** GST-FDPA step 1: exact 16-element dots, then scale significands.
  * No FP normalization occurs between the group dot and the fused sum.
  */
class GroupedDotFrontend(config: BlackwellConfig) extends Module {
  require(config.grouped)
  val io = IO(new Bundle {
    val start = Input(Bool())
    val in = Flipped(Decoupled(new TensorOperands(config)))
    val out = Decoupled(new TensorTerm)
  })
  private val active = RegInit(false.B)
  private val element = Reg(UInt(4.W))
  private val group = Reg(UInt(2.W))
  // Products are exact quarter units; 16 products are bounded by +/-2304.
  private val partial = Reg(SInt(13.W))
  private val anyProduct = Reg(Bool())
  private val alpha = Reg(UInt(8.W))
  private val beta = Reg(UInt(8.W))
  private val halfUnits = VecInit(Seq(0, 1, 2, 3, 4, 6, 8, 12).map(_.U(4.W)))
  private val magnitude = halfUnits(io.in.bits.a(2, 0)) * halfUnits(io.in.bits.b(2, 0))
  private val product = Mux(io.in.bits.a(3) ^ io.in.bits.b(3), -magnitude.zext, magnitude.zext)
  private val total = partial +& product
  private val totalMagnitude = Mux(total < 0.S, (-total).asUInt, total.asUInt)
  private val first = element === 0.U
  private val last = element === 15.U
  private val nvfp4 = config.blockScale.contains(BlockScale.UE4M3)
  private val inputAlpha = if (nvfp4) io.in.bits.alpha.get & 127.U else io.in.bits.alpha.get
  private val inputBeta = if (nvfp4) io.in.bits.beta.get & 127.U else io.in.bits.beta.get
  private val currentAlpha = Mux(first, inputAlpha, alpha)
  private val currentBeta = Mux(first, inputBeta, beta)

  private def scale(raw: UInt): (UInt, SInt, Bool) = {
    if (nvfp4) {
      val decoded = TensorArithmetic.decode(Cat(0.U(1.W), raw(6, 0)), 4, 3, finiteOnly = true)
      ((decoded.magnitude >> 20)(3, 0), decoded.exponent, decoded.nan)
    } else {
      (1.U(4.W), raw.zext - 127.S(12.W), raw === 255.U)
    }
  }
  private val (sa, ea, nanA) = scale(currentAlpha)
  private val (sb, eb, nanB) = scale(currentBeta)
  private val allZeroProducts = !anyProduct && magnitude === 0.U

  io.in.ready := active && (!last || io.out.ready)
  io.out.valid := active && last && io.in.valid
  io.out.bits.sign := total < 0.S
  io.out.bits.magnitude := (totalMagnitude * sa * sb) << (if (nvfp4) 15 else 21)
  io.out.bits.exponent := Mux(allZeroProducts || sa === 0.U || sb === 0.U,
    config.zeroExponent.S, ea + eb)
  io.out.bits.nan := nanA || nanB
  io.out.bits.infinity := false.B

  when(io.start) {
    assert(!active)
    active := true.B
    element := 0.U
    group := 0.U
    partial := 0.S
    anyProduct := false.B
  }
  when(io.in.fire) {
    when(first) {
      if (config.scaleBlockSize == 32) {
        when(group(0)) {
          assert(inputAlpha === alpha && inputBeta === beta,
            "MXFP4 scales must be shared by two 16-element compute groups")
        }
      }
      alpha := inputAlpha
      beta := inputBeta
    }.otherwise {
      assert(inputAlpha === alpha && inputBeta === beta, "Scale changed inside a block")
    }
    when(last) {
      partial := 0.S
      anyProduct := false.B
      element := 0.U
      group := group + 1.U
      when(group === 3.U) { active := false.B }
    }.otherwise {
      partial := total
      anyProduct := anyProduct || (magnitude =/= 0.U)
      element := element + 1.U
    }
  }
}
