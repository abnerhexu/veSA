package gemmini.tensorcore

import org.scalatest.flatspec.AnyFlatSpec

class TensorProfilesSpec extends AnyFlatSpec {
  "Architecture profiles" should "separate internal and block-output precision" in {
    for (f <- Seq(TensorFormat.E4M3, TensorFormat.E5M2)) {
      val hopper = HopperConfig(f)
      assert(hopper.fractionalBits == 13 && hopper.outputFractionBits == 13)
      assert(hopper.outputWidth == 32 && hopper.fusionLength == 32)
      assert(HopperConfig(f, true).outputFractionBits == 10)
      assert(HopperConfig(f, true).fractionalBits == 13)
      assert(BlackwellConfig(f).fractionalBits == 25 && BlackwellConfig(f).outputFractionBits == 23)
    }
    assert(HopperConfig(TensorFormat.FP16).fractionalBits == 25)
    assert(HopperConfig(TensorFormat.BF16).fractionalBits == 25)
    assert(HopperConfig(TensorFormat.FP16, true).zeroExponent == -133)
    assert(BlackwellConfig(TensorFormat.FP16, true).zeroExponent == -22)
    assert(BlackwellConfig.mxfp4.fractionalBits == 35)
    assert(BlackwellConfig.nvfp4.fusionLength == 64)
  }
  it should "reject native Hopper FP4 and unsupported half-output profiles" in {
    intercept[IllegalArgumentException] { HopperConfig(TensorFormat.E2M1) }
    intercept[IllegalArgumentException] { HopperConfig(TensorFormat.BF16, true) }
    intercept[IllegalArgumentException] { TensorFormat("fp32", 8, 23) }
    intercept[IllegalArgumentException] { ElaborationProfile.parse(Array("hopper-mxfp4", "f32", "unused")) }
    intercept[IllegalArgumentException] { ElaborationProfile.parse(Array("hopper-nvfp4", "f32", "unused")) }
  }
}
