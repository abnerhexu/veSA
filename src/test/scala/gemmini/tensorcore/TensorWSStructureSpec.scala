package gemmini.tensorcore

import chisel3.stage.ChiselStage
import org.scalatest.flatspec.AnyFlatSpec

class TensorWSStructureSpec extends AnyFlatSpec {
  private def chirrtl(gen: => chisel3.RawModule): String =
    (new ChiselStage).emitChirrtl(gen)

  private def multipliers(text: String): Int = "\\bmul\\(".r.findAllIn(text).length

  "A Tensor WS PE" should "keep one input-product lane and a nine-bit exponent path" in {
    val e4m3 = chirrtl(new TensorWSPE(BlackwellConfig(), outputRows = 2, groupEnd = false))
    assert(multipliers(e4m3) == 1, "E4M3 PE must contain one significand multiplier")
    assert(e4m3.contains("maximum : SInt<9>"), "Exponent wave must stay nine bits wide")

    val nvRegular = chirrtl(new TensorWSPE(BlackwellConfig.nvfp4,
      outputRows = 2, groupEnd = false))
    assert(multipliers(nvRegular) == 1,
      "A regular NVFP4 PE must contain one quarter-product multiplier")

    val nvBoundary = chirrtl(new TensorWSPE(BlackwellConfig.nvfp4,
      outputRows = 2, groupEnd = true))
    assert(multipliers(nvBoundary) == 3,
      "An NVFP4 group boundary has one input-product and two auxiliary scale multipliers")
  }

  "The Tensor WS mesh" should "instantiate one fusion-length PE per output column" in {
    def instances(config: BlackwellConfig): Int = chirrtl(new TensorMesh(config))
      .linesIterator.count(_.trim.startsWith("inst pes_"))
    assert(instances(BlackwellConfig()) == 64)
    assert(instances(BlackwellConfig.nvfp4) == 128)
  }
}
