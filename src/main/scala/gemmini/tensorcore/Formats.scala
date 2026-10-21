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

/** Static numerical profiles; no run-time precision reinterpretation. */
sealed trait TensorConfig {
  def format: TensorFormat
  def halfOutput: Boolean
  def blockScale: Option[BlockScale]
  def architecture: String
  def fractionalBits: Int
  def zeroExponent: Int
  def outputFractionBits: Int = if (halfOutput) 10 else 23
  final def grouped: Boolean = blockScale.nonEmpty
  final def fusionLength: Int = if (grouped) 64 else if (format.width == 8) 32 else 16
  final def scaleBlockSize: Int = if (blockScale.contains(BlockScale.UE4M3)) 16 else 32
  final def outputWidth: Int = if (halfOutput) 16 else 32
  // Unnormalized T products <4. Four scaled GST groups plus C <9218.
  final def integerBits: Int = if (grouped) 15 else 1 + (BigInt(4 * fusionLength + 2) - 1).bitLength
  final def sumWidth: Int = fractionalBits + integerBits
}

case class BlackwellConfig(format: TensorFormat = TensorFormat.E4M3,
                           halfOutput: Boolean = false,
                           blockScale: Option[BlockScale] = None) extends TensorConfig {
  require(Set(TensorFormat.FP16, TensorFormat.BF16,
    TensorFormat.E4M3, TensorFormat.E5M2, TensorFormat.E2M1).contains(format))
  require(grouped == (format == TensorFormat.E2M1), "FP4 requires an explicit block-scale profile")
  require(!grouped || !halfOutput, "Dedicated block-scaled FP4 has FP32 C/D")
  require(!halfOutput || format != TensorFormat.BF16, "BF16 has no native FP16-output instruction")
  val fractionalBits: Int = if (grouped) 35 else 25
  val zeroExponent: Int = if (grouped) -139 else if (halfOutput) -22 else -133
  val architecture = "blackwell"
}

/** Paper776 Table 6, pinned MMA-Sim WGMMA/HGMMA family (including e_zero). */
case class HopperConfig(format: TensorFormat = TensorFormat.E4M3,
                        halfOutput: Boolean = false) extends TensorConfig {
  require(Set(TensorFormat.FP16, TensorFormat.BF16, TensorFormat.E4M3, TensorFormat.E5M2).contains(format),
    "Hopper has no native FP4/block-scaled profile")
  require(!halfOutput || format != TensorFormat.BF16, "BF16 has no native FP16-output instruction")
  val blockScale: Option[BlockScale] = None
  val architecture = "hopper"
  val fractionalBits: Int = if (format.width == 8) 13 else 25
  val zeroExponent: Int = -133
  override def outputFractionBits: Int = if (!halfOutput && format.width == 8) 13 else super.outputFractionBits
}

object BlackwellConfig {
  def mxfp4: BlackwellConfig = BlackwellConfig(TensorFormat.E2M1, blockScale = Some(BlockScale.E8M0))
  def nvfp4: BlackwellConfig = BlackwellConfig(TensorFormat.E2M1, blockScale = Some(BlockScale.UE4M3))
}
