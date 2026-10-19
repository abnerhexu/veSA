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
  val E2M1 = TensorFormat("e2m1", 2, 1, finiteOnly = true)
}

sealed trait BlockScale
object BlockScale {
  case object E8M0 extends BlockScale
  case object UE4M3 extends BlockScale
}

/** Only Blackwell is implemented until its release gate passes. */
case class BlackwellConfig(format: TensorFormat = TensorFormat.E4M3,
                           halfOutput: Boolean = false,
                           blockScale: Option[BlockScale] = None) {
  require(Set(TensorFormat.FP16, TensorFormat.BF16,
    TensorFormat.E4M3, TensorFormat.E5M2, TensorFormat.E2M1).contains(format))
  val grouped: Boolean = blockScale.nonEmpty
  require(grouped == (format == TensorFormat.E2M1), "FP4 requires an explicit block-scale profile")
  require(!grouped || !halfOutput, "Dedicated block-scaled FP4 has FP32 C/D")
  require(!halfOutput || format != TensorFormat.BF16, "BF16 has no native FP16-output instruction")
  val fractionalBits: Int = if (grouped) 35 else 25
  val fusionLength: Int = if (grouped) 64 else if (format.width == 8) 32 else 16
  val scaleBlockSize: Int = if (blockScale.contains(BlockScale.UE4M3)) 16 else 32
  val outputWidth: Int = if (halfOutput) 16 else 32
  val zeroExponent: Int = if (grouped) -139 else if (halfOutput) -22 else -133
  // Each unnormalized product has magnitude <4, C has magnitude <2.
  // GST: group <=16*6*6, scale significands <2 each. Four groups plus C
  // are bounded by 4*576*4+2=9218, requiring 15 signed integer bits.
  val integerBits: Int = if (grouped) 15 else 1 + (BigInt(4 * fusionLength + 2) - 1).bitLength
  val sumWidth: Int = fractionalBits + integerBits
}

object BlackwellConfig {
  def mxfp4: BlackwellConfig = BlackwellConfig(TensorFormat.E2M1, blockScale = Some(BlockScale.E8M0))
  def nvfp4: BlackwellConfig = BlackwellConfig(TensorFormat.E2M1, blockScale = Some(BlockScale.UE4M3))
}
