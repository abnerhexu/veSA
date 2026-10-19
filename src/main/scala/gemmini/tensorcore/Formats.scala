package gemmini.tensorcore

/** Operand encodings are raw bits. FP32 is ONLY an accumulator/output format. */
case class TensorFormat(name: String, exponentBits: Int, fractionBits: Int,
                        finiteOnly: Boolean = false) {
  val width: Int = 1 + exponentBits + fractionBits
  val bias: Int = (1 << (exponentBits - 1)) - 1
  require(width <= 16)
}

object TensorFormat {
  val FP16 = TensorFormat("fp16", 5, 10)
  val BF16 = TensorFormat("bf16", 8, 7)
  val E4M3 = TensorFormat("e4m3", 4, 3, finiteOnly = true)
  val E5M2 = TensorFormat("e5m2", 5, 2)
}

/** Only Blackwell is implemented until its release gate passes. */
case class BlackwellConfig(format: TensorFormat = TensorFormat.E4M3,
                           halfOutput: Boolean = false) {
  require(Set(TensorFormat.FP16, TensorFormat.BF16,
    TensorFormat.E4M3, TensorFormat.E5M2).contains(format))
  require(!halfOutput || format != TensorFormat.BF16, "BF16 has no native FP16-output instruction")
  val fractionalBits = 25
  val fusionLength: Int = if (format.width == 8) 32 else 16
  val outputWidth: Int = if (halfOutput) 16 else 32
  val zeroExponent: Int = if (halfOutput) -22 else -133
  // Each unnormalized product has magnitude <4, C has magnitude <2.
  val integerBits: Int = 1 + (BigInt(4 * fusionLength + 2) - 1).bitLength
  val sumWidth: Int = fractionalBits + integerBits
}
