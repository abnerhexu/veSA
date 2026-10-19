package gemmini

import chisel3._
import gemmini.Arithmetic.SIntArithmetic
import gemmini.tensorcore.BlackwellConfig

/** Small first-release configurations. SInt is only a raw storage carrier for
  * the Tensor path; legacy integer MMA remains a separate command family.
  * Do not enable legacy DMA scales when loading encoded operands or raw C/D.
  */
object TensorCoreConfigs {
  def blackwell(profile: BlackwellConfig = BlackwellConfig(), dim: Int = 2,
                singlePortedAccumulator: Boolean = false): GemminiArrayConfig[SInt, Float, Float] = {
    require(Set(2, 4).contains(dim), "Start with a small Tensor array")
    val slot = profile.format.width max 8
    GemminiConfigs.defaultConfig.copy(
      inputType = SInt(slot.W), weightType = SInt(slot.W), accType = SInt(32.W),
      spatialArrayInputType = SInt(slot.W), spatialArrayWeightType = SInt(slot.W),
      spatialArrayOutputType = SInt(32.W),
      meshRows = dim, meshColumns = dim, tileRows = 1, tileColumns = 1,
      sp_capacity = CapacityInMatrices(4 * 256 / dim),
      acc_capacity = CapacityInMatrices(2 * 16 / dim),
      sp_banks = 4, acc_banks = 2, sp_singleported = true,
      acc_singleported = singlePortedAccumulator,
      acc_sub_banks = if (singlePortedAccumulator) 2 else -1,
      mvin_scale_args = None, mvin_scale_acc_args = None, acc_scale_args = None,
      has_loop_conv = false, has_training_convs = false,
      has_first_layer_optimizations = false, has_normalizations = false,
      has_nonlinear_activations = false, clock_gate = false,
      use_shared_ext_mem = false, use_tl_ext_mem = false,
      tensorcore = Some(profile))
  }
}

/** Compose in a compatible Chipyard tree; full system elaboration is still a
  * release gate, distinct from the independently tested command/bank path.
  */
class BlackwellTensorCoreConfig(profile: BlackwellConfig = BlackwellConfig(), dim: Int = 2,
                               singlePortedAccumulator: Boolean = false)
  extends DefaultGemminiConfig(TensorCoreConfigs.blackwell(profile, dim, singlePortedAccumulator))
