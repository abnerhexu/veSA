package gemmini.tensorcore

import chisel3._
import chisel3.util._

class TensorWSActivation(config: BlackwellConfig) extends Bundle {
  val a = UInt(config.format.width.W)
  val alpha = if (config.grouped) Some(UInt(8.W)) else None
  // False rows carry the vertical state through a short final segment without
  // interpreting stale weight-bank contents as zero-times-infinity products.
  val active = Bool()
}

class TensorWSWeightLoad(config: BlackwellConfig) extends Bundle {
  val bank = Bool()
  val b = UInt(config.format.width.W)
  val beta = if (config.grouped) Some(UInt(8.W)) else None
}

class TensorWSExponentState(config: BlackwellConfig, outputRows: Int) extends Bundle {
  // Nine signed bits cover every implemented product/group exponent:
  // Blackwell GST [-139, 254], unscaled products, and Hopper zero -133.
  val maximum = SInt(9.W)
  val groupAny = if (config.grouped) Some(Bool()) else None
  val tag = UInt(log2Ceil(outputRows max 2).W)
  val bank = Bool()
}

/** Narrow column-bottom feedback. Classification is recomputed during replay,
  * so only e_max plus routing control traverses the return FIFO.
  */
class TensorWSFeedback(outputRows: Int) extends Bundle {
  val maximum = SInt(9.W)
  val tag = UInt(log2Ceil(outputRows max 2).W)
  val bank = Bool()
}

class TensorWSSumState(config: BlackwellConfig, outputRows: Int) extends Bundle {
  val maximum = SInt(9.W)
  val sum = SInt(config.sumWidth.W)
  val nan = Bool()
  val positiveInfinity = Bool()
  val negativeInfinity = Bool()
  val groupSum = if (config.grouped) Some(SInt(13.W)) else None
  val groupAny = if (config.grouped) Some(Bool()) else None
  val tag = UInt(log2Ceil(outputRows max 2).W)
  val bank = Bool()
}

/** One physical PE in the weight-stationary Tensor array.
  *
  * Two weight registers let the narrow exponent lane prepass the next fusion
  * segment while the significand multiplier/add lane replays the current one.
  * Activations move horizontally; maximum-exponent or fixed-sum state moves
  * vertically. No product is retained in a PE.
  */
class TensorWSPE(config: BlackwellConfig, outputRows: Int, groupEnd: Boolean) extends Module {
  val io = IO(new Bundle {
    val load = Input(Valid(new TensorWSWeightLoad(config)))

    val exponentActivation = Input(Valid(new TensorWSActivation(config)))
    val exponentState = Input(Valid(new TensorWSExponentState(config, outputRows)))
    val exponentActivationOut = Output(Valid(new TensorWSActivation(config)))
    val exponentStateOut = Output(Valid(new TensorWSExponentState(config, outputRows)))

    val sumActivation = Input(Valid(new TensorWSActivation(config)))
    val sumState = Input(Valid(new TensorWSSumState(config, outputRows)))
    val sumActivationOut = Output(Valid(new TensorWSActivation(config)))
    val sumStateOut = Output(Valid(new TensorWSSumState(config, outputRows)))
  })

  private val weights = Reg(Vec(2, UInt(config.format.width.W)))
  private val betas = if (config.grouped) Some(Reg(Vec(2, UInt(8.W)))) else None
  when(io.load.valid) {
    weights(io.load.bits.bank) := io.load.bits.b
    if (config.grouped) { betas.get(io.load.bits.bank) := io.load.bits.beta.get }
  }

  when(io.exponentActivation.valid || io.exponentState.valid) {
    assert(io.exponentActivation.valid === io.exponentState.valid,
      "WS exponent activation and vertical wave lost alignment")
  }
  when(io.sumActivation.valid || io.sumState.valid) {
    assert(io.sumActivation.valid === io.sumState.valid,
      "WS significand activation and vertical wave lost alignment")
  }

  private val exponentFire = io.exponentActivation.valid && io.exponentState.valid
  private val exponentActivationReg = Reg(new TensorWSActivation(config))
  private val exponentStateReg = Reg(new TensorWSExponentState(config, outputRows))
  io.exponentActivationOut.valid := RegNext(exponentFire, false.B)
  io.exponentStateOut.valid := RegNext(exponentFire, false.B)
  io.exponentActivationOut.bits := exponentActivationReg
  io.exponentStateOut.bits := exponentStateReg

  when(exponentFire) {
    exponentActivationReg := io.exponentActivation.bits
    exponentStateReg := io.exponentState.bits
    if (config.grouped) {
      val nonzero = io.exponentActivation.bits.a(2, 0).orR &&
        weights(io.exponentState.bits.bank)(2, 0).orR
      val groupAny = io.exponentState.bits.groupAny.get || nonzero
      if (groupEnd) {
        val term = TensorArithmetic.scaleGroupMetadata(groupAny,
          io.exponentActivation.bits.alpha.get, betas.get(io.exponentState.bits.bank), config)
        exponentStateReg.maximum := Mux(term.exponent > io.exponentState.bits.maximum,
          term.exponent, io.exponentState.bits.maximum)
        exponentStateReg.groupAny.get := false.B
      } else {
        exponentStateReg.groupAny.get := groupAny
      }
    } else {
      val term = TensorArithmetic.productMetadata(io.exponentActivation.bits.a,
        weights(io.exponentState.bits.bank), config.format, config.zeroExponent)
      exponentStateReg.maximum := Mux(term.exponent > io.exponentState.bits.maximum,
        term.exponent, io.exponentState.bits.maximum)
    }
    when(!io.exponentActivation.bits.active) {
      exponentStateReg := io.exponentState.bits
    }
  }

  private val sumFire = io.sumActivation.valid && io.sumState.valid
  private val sumActivationReg = Reg(new TensorWSActivation(config))
  private val sumStateReg = Reg(new TensorWSSumState(config, outputRows))
  io.sumActivationOut.valid := RegNext(sumFire, false.B)
  io.sumStateOut.valid := RegNext(sumFire, false.B)
  io.sumActivationOut.bits := sumActivationReg
  io.sumStateOut.bits := sumStateReg

  when(sumFire) {
    sumActivationReg := io.sumActivation.bits
    sumStateReg := io.sumState.bits
    if (config.grouped) {
      val (quarter, nonzero) = TensorArithmetic.fp4QuarterProduct(
        io.sumActivation.bits.a, weights(io.sumState.bits.bank))
      val groupTotal = io.sumState.bits.groupSum.get + quarter
      val groupAny = io.sumState.bits.groupAny.get || nonzero
      if (groupEnd) {
        val term = TensorArithmetic.scaleGroup(groupTotal, groupAny,
          io.sumActivation.bits.alpha.get, betas.get(io.sumState.bits.bank), config)
        sumStateReg.sum := io.sumState.bits.sum + TensorArithmetic.align(
          term, io.sumState.bits.maximum, config)
        sumStateReg.nan := io.sumState.bits.nan || term.nan
        sumStateReg.positiveInfinity := io.sumState.bits.positiveInfinity ||
          (term.infinity && !term.sign)
        sumStateReg.negativeInfinity := io.sumState.bits.negativeInfinity ||
          (term.infinity && term.sign)
        sumStateReg.groupSum.get := 0.S
        sumStateReg.groupAny.get := false.B
      } else {
        sumStateReg.groupSum.get := groupTotal
        sumStateReg.groupAny.get := groupAny
      }
    } else {
      val term = TensorArithmetic.product(io.sumActivation.bits.a,
        weights(io.sumState.bits.bank), config.format, config.zeroExponent)
      sumStateReg.sum := io.sumState.bits.sum + TensorArithmetic.align(
        term, io.sumState.bits.maximum, config)
      sumStateReg.nan := io.sumState.bits.nan || term.nan
      sumStateReg.positiveInfinity := io.sumState.bits.positiveInfinity ||
        (term.infinity && !term.sign)
      sumStateReg.negativeInfinity := io.sumState.bits.negativeInfinity ||
        (term.infinity && term.sign)
    }
    when(!io.sumActivation.bits.active) {
      sumStateReg := io.sumState.bits
    }
  }
}
