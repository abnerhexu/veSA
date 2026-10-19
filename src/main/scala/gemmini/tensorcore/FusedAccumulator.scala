package gemmini.tensorcore

import chisel3._
import chisel3.util._

class FusedRequest(outputWidth: Int, maxTerms: Int) extends Bundle {
  val c = UInt(outputWidth.W)
  val length = UInt(log2Ceil(maxTerms + 1).W)
}

/** Paper776 truncated fused summation, shared by T-FDPA and GST-FDPA.
  * Do not replace a cancelling group's prescribed exponent with zeroExponent.
  */
class FusedAccumulator(config: BlackwellConfig, maxTerms: Int) extends Module {
  val io = IO(new Bundle {
    val req = Flipped(Decoupled(new FusedRequest(config.outputWidth, maxTerms)))
    val in = Flipped(Decoupled(new TensorTerm))
    val out = Decoupled(UInt(config.outputWidth.W))
  })
  private val idle :: collect :: replay :: finish :: respond :: Nil = Enum(5)
  private val state = RegInit(idle)
  private val countWidth = log2Ceil(maxTerms + 1)
  private val length = Reg(UInt(countWidth.W))
  private val index = Reg(UInt(countWidth.W))
  private val terms = Reg(Vec(maxTerms + 1, new TensorTerm))
  private val maximum = Reg(SInt(12.W))
  private val sum = Reg(SInt(config.sumWidth.W))
  private val nan = Reg(Bool())
  private val positiveInfinity = Reg(Bool())
  private val negativeInfinity = Reg(Bool())
  private val result = Reg(UInt(config.outputWidth.W))
  private val c = TensorArithmetic.decode(io.req.bits.c,
    if (config.halfOutput) 5 else 8, if (config.halfOutput) 10 else 23)

  io.req.ready := state === idle
  io.in.ready := state === collect
  io.out.valid := state === respond
  io.out.bits := result

  when(io.req.fire) {
    assert(io.req.bits.length > 0.U && io.req.bits.length <= maxTerms.U)
    length := io.req.bits.length
    index := 0.U
    terms(maxTerms) := c
    terms(maxTerms).exponent := Mux(c.magnitude === 0.U, config.zeroExponent.S, c.exponent)
    maximum := Mux(c.magnitude === 0.U, config.zeroExponent.S, c.exponent)
    sum := 0.S
    nan := c.nan
    positiveInfinity := c.infinity && !c.sign
    negativeInfinity := c.infinity && c.sign
    state := collect
  }
  when(io.in.fire) {
    val term = io.in.bits
    terms(index) := term
    maximum := Mux(term.exponent > maximum, term.exponent, maximum)
    nan := nan || term.nan
    positiveInfinity := positiveInfinity || (term.infinity && !term.sign)
    negativeInfinity := negativeInfinity || (term.infinity && term.sign)
    when(index === length - 1.U) {
      index := 0.U
      state := replay
    }.otherwise { index := index + 1.U }
  }
  private val term = terms(Mux(index === length, maxTerms.U, index))
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
