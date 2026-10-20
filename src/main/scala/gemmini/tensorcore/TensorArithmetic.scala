package gemmini.tensorcore

import chisel3._
import chisel3.util._

/** Unnormalized value = (-1)^sign * magnitude * 2^(exponent-23). */
class TensorTerm extends Bundle {
  val sign = Bool()
  // A scaled GST group is at most 2304 * 2^23, requiring 35 magnitude bits.
  val magnitude = UInt(36.W)
  val exponent = SInt(12.W)
  val nan = Bool()
  val infinity = Bool()
}

object TensorArithmetic {
  def decode(bits: UInt, exponentBits: Int, fractionBits: Int,
             finiteOnly: Boolean = false): TensorTerm = {
    val result = Wire(new TensorTerm)
    val exponent = bits(fractionBits + exponentBits - 1, fractionBits)
    val fraction = bits(fractionBits - 1, 0)
    val allExponent = exponent.andR
    val bias = (1 << (exponentBits - 1)) - 1
    result.sign := bits(exponentBits + fractionBits)
    result.magnitude := Cat(exponent =/= 0.U, fraction) << (23 - fractionBits)
    result.exponent := Mux(exponent === 0.U, 1.U, exponent).zext - bias.S(12.W)
    result.nan := (if (finiteOnly) allExponent && fraction.andR else allExponent && fraction.orR)
    result.infinity := (if (finiteOnly) false.B else allExponent && !fraction.orR)
    result
  }

  def product(a: UInt, b: UInt, format: TensorFormat, zeroExponent: Int): TensorTerm = {
    val left = decode(a, format.exponentBits, format.fractionBits, format.finiteOnly)
    val right = decode(b, format.exponentBits, format.fractionBits, format.finiteOnly)
    val result = Wire(new TensorTerm)
    // Decode used 23 fractional bits. Remove exactly the known padding before
    // multiplying, so this is a small exact significand multiplier.
    val lm = left.magnitude >> (23 - format.fractionBits)
    val rm = right.magnitude >> (23 - format.fractionBits)
    result.magnitude := (lm * rm) << (23 - 2 * format.fractionBits)
    result.sign := left.sign ^ right.sign
    result.exponent := Mux(result.magnitude === 0.U, zeroExponent.S,
      left.exponent + right.exponent)
    result.nan := left.nan || right.nan ||
      (left.infinity && right.magnitude === 0.U && !right.infinity) ||
      (right.infinity && left.magnitude === 0.U && !left.infinity)
    result.infinity := left.infinity || right.infinity
    result
  }

  /** Exact fixed-point sum -> RZ-FP32 or RNE-FP16; no intermediate FP rounding. */
  def pack(sum: SInt, exponent: SInt, fractionalBits: Int, halfOutput: Boolean,
           outputFractionBits: Int = 23): UInt = {
    val width = sum.getWidth
    val fraction = if (halfOutput) 10 else outputFractionBits
    require(halfOutput || Set(13, 23).contains(fraction))
    val storedFraction = if (halfOutput) 10 else 23
    val bias = if (halfOutput) 15 else 127
    val exponentBits = if (halfOutput) 5 else 8
    val sign = sum < 0.S
    val magnitude = Mux(sign, (-sum).asUInt, sum.asUInt)
    val leading = (width - 1).U - PriorityEncoder(Reverse(magnitude))
    val valueExponent = exponent + leading.zext - fractionalBits.S
    val normalExponent = Mux(valueExponent < (1 - bias).S, (1 - bias).S, valueExponent)
    val shift = normalExponent - fraction.S - exponent + fractionalBits.S
    val shiftWidth = log2Ceil(width + 1)
    val safeShift = Wire(UInt(shiftWidth.W))
    val safeLeft = Wire(UInt(shiftWidth.W))
    safeShift := Mux(shift > width.S, width.U, shift.asUInt)
    safeLeft := Mux(shift < (-width).S, width.U, (-shift).asUInt)
    val base = Wire(UInt((width + 1).W))
    base := Mux(shift >= 0.S, magnitude >> safeShift,
      (magnitude << safeLeft)(width, 0))
    val round = WireDefault(false.B)
    if (halfOutput) {
      val mask = ((1.U((width + 1).W) << safeShift) - 1.U)(width - 1, 0)
      val remainder = magnitude & mask
      val halfway = (1.U((width + 1).W) << (safeShift - 1.U))(width - 1, 0)
      round := shift > 0.S && shift <= width.S &&
        (remainder > halfway || (remainder === halfway && base(0)))
    }
    val rounded = base + round.asUInt
    val carry = rounded(fraction + 1)
    val significand = Mux(carry, rounded >> 1, rounded)
    val finalExponent = normalExponent + carry.asUInt.zext
    val encodedExponent = Wire(UInt(exponentBits.W))
    encodedExponent := Mux(significand(fraction), (finalExponent + bias.S).asUInt, 0.U)
    val finite = Cat(sign, encodedExponent, significand(fraction - 1, 0),
      0.U((storedFraction - fraction).W))
    val infinity = Cat(sign, Fill(exponentBits, 1.U(1.W)), 0.U(storedFraction.W))
    Mux(magnitude === 0.U, 0.U, Mux(finalExponent > bias.S, infinity, finite))
  }
}
