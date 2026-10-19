package gemmini.tensorcore

import chisel3._
import chisel3.util._

class TensorRequest(config: BlackwellConfig) extends Bundle {
  val c = UInt(config.outputWidth.W)
  val length = UInt(log2Ceil(config.fusionLength + 1).W)
}

class TensorOperands(config: BlackwellConfig) extends Bundle {
  val a = UInt(config.format.width.W)
  val b = UInt(config.format.width.W)
}

/** A two-pass fused-dot PE. One block in flight; all state advances on fire.
  * C is included in max-exponent selection and truncation, not added afterward.
  */
class TensorPE(val config: BlackwellConfig = BlackwellConfig()) extends Module {
  val io = IO(new Bundle {
    val req = Flipped(Decoupled(new TensorRequest(config)))
    val in = Flipped(Decoupled(new TensorOperands(config)))
    val out = Decoupled(UInt(config.outputWidth.W))
  })
  private val idle :: collect :: replay :: finish :: respond :: Nil = Enum(5)
  private val state = RegInit(idle)
  private val countWidth = log2Ceil(config.fusionLength + 1)
  private val length = Reg(UInt(countWidth.W))
  private val index = Reg(UInt(countWidth.W))
  private val terms = Reg(Vec(config.fusionLength + 1, new TensorTerm))
  private val maximum = Reg(SInt(12.W))
  private val sum = Reg(SInt(config.sumWidth.W))
  private val nan = Reg(Bool())
  private val positiveInfinity = Reg(Bool())
  private val negativeInfinity = Reg(Bool())
  private val result = Reg(UInt(config.outputWidth.W))

  io.req.ready := state === idle
  io.in.ready := state === collect
  io.out.valid := state === respond
  io.out.bits := result

  private val c = TensorArithmetic.decode(io.req.bits.c,
    if (config.halfOutput) 5 else 8, if (config.halfOutput) 10 else 23)
  private val product = TensorArithmetic.product(io.in.bits.a, io.in.bits.b,
    config.format, config.zeroExponent)

  when(io.req.fire) {
    assert(io.req.bits.length > 0.U && io.req.bits.length <= config.fusionLength.U)
    length := io.req.bits.length
    index := 0.U
    terms(config.fusionLength) := c
    terms(config.fusionLength).exponent := Mux(c.magnitude === 0.U,
      config.zeroExponent.S, c.exponent)
    maximum := Mux(c.magnitude === 0.U, config.zeroExponent.S, c.exponent)
    sum := 0.S
    nan := c.nan
    positiveInfinity := c.infinity && !c.sign
    negativeInfinity := c.infinity && c.sign
    state := collect
  }
  when(io.in.fire) {
    terms(index) := product
    maximum := Mux(product.exponent > maximum, product.exponent, maximum)
    nan := nan || product.nan
    positiveInfinity := positiveInfinity || (product.infinity && !product.sign)
    negativeInfinity := negativeInfinity || (product.infinity && product.sign)
    when(index === length - 1.U) {
      index := 0.U
      state := replay
    }.otherwise { index := index + 1.U }
  }
  private val term = terms(Mux(index === length, config.fusionLength.U, index))
  private val delta = (maximum - term.exponent).asUInt
  private val aligned = (term.magnitude << (config.fractionalBits - 23)) >> delta
  private val signedTerm = Wire(SInt(config.sumWidth.W))
  signedTerm := Mux(term.sign, -aligned.zext, aligned.zext)
  when(state === replay) {
    sum := sum + signedTerm
    when(index === length) { state := finish }
      .otherwise { index := index + 1.U }
  }
  when(state === finish) {
    val canonicalNaN = ((BigInt(1) << (config.outputWidth - 1)) - 1).U
    val infinity = (if (config.halfOutput) BigInt(0x7c00) else BigInt(0x7f800000)).U(config.outputWidth.W)
    val sign = (BigInt(1) << (config.outputWidth - 1)).U(config.outputWidth.W)
    result := Mux(nan || (positiveInfinity && negativeInfinity), canonicalNaN,
      Mux(positiveInfinity || negativeInfinity, infinity | Mux(negativeInfinity, sign, 0.U),
        TensorArithmetic.pack(sum, maximum, config.fractionalBits, config.halfOutput)))
    state := respond
  }
  when(io.out.fire) { state := idle }
}
