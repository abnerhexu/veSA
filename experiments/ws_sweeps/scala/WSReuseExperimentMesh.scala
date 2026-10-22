package gemmini.wsexperiments

import chisel3._
import chisel3.util._
import gemmini.tensorcore._

class WSReuseRequest(config: BlackwellConfig, rows: Int, columns: Int) extends Bundle {
  val c = Vec(rows, Vec(columns, UInt(config.outputWidth.W)))
  val k = UInt(16.W)
  val tag = UInt(16.W)
}

class WSReuseOperands(config: BlackwellConfig, rows: Int, columns: Int) extends Bundle {
  val a = Vec(rows, UInt(config.format.width.W))
  val b = Vec(columns, UInt(config.format.width.W))
}

class WSReuseResult(config: BlackwellConfig, rows: Int, columns: Int) extends Bundle {
  val d = Vec(rows, Vec(columns, UInt(config.outputWidth.W)))
  val tag = UInt(16.W)
}

class WSReuseWeightLoad(config: BlackwellConfig) extends Bundle {
  val segmentBank = Bool()
  val chunk = Bool()
  val b = UInt(config.format.width.W)
}

class WSReuseActivation(config: BlackwellConfig) extends Bundle {
  val a = UInt(config.format.width.W)
}

class WSReuseExponentState(rows: Int) extends Bundle {
  val maximum = SInt(9.W)
  val tag = UInt(log2Ceil(rows max 2).W)
  val segmentBank = Bool()
  val chunk = Bool()
}

class WSReuseFeedback(rows: Int) extends Bundle {
  val maximum = SInt(9.W)
  val tag = UInt(log2Ceil(rows max 2).W)
  val segmentBank = Bool()
}

class WSReuseSumState(config: BlackwellConfig, rows: Int) extends Bundle {
  val maximum = SInt(9.W)
  val sum = SInt(config.sumWidth.W)
  val nan = Bool()
  val positiveInfinity = Bool()
  val negativeInfinity = Bool()
  val tag = UInt(log2Ceil(rows max 2).W)
  val segmentBank = Bool()
  val chunk = Bool()
}

/** One physical product lane with two segment buffers and two chunks/segment. */
class WSReuseExperimentPE(config: BlackwellConfig, rows: Int) extends Module {
  val io = IO(new Bundle {
    val load = Input(Valid(new WSReuseWeightLoad(config)))
    val exponentActivation = Input(Valid(new WSReuseActivation(config)))
    val exponentState = Input(Valid(new WSReuseExponentState(rows)))
    val exponentActivationOut = Output(Valid(new WSReuseActivation(config)))
    val exponentStateOut = Output(Valid(new WSReuseExponentState(rows)))
    val sumActivation = Input(Valid(new WSReuseActivation(config)))
    val sumState = Input(Valid(new WSReuseSumState(config, rows)))
    val sumActivationOut = Output(Valid(new WSReuseActivation(config)))
    val sumStateOut = Output(Valid(new WSReuseSumState(config, rows)))
  })

  // Two segment buffers x two 16-term chunks gives four stationary weights.
  private val weights = Reg(Vec(4, UInt(config.format.width.W)))
  private def slot(bank: Bool, chunk: Bool): UInt = Cat(bank, chunk)
  when(io.load.valid) {
    weights(slot(io.load.bits.segmentBank, io.load.bits.chunk)) := io.load.bits.b
  }

  private val exponentFire = io.exponentActivation.valid && io.exponentState.valid
  when(io.exponentActivation.valid || io.exponentState.valid) {
    assert(io.exponentActivation.valid === io.exponentState.valid)
  }
  private val exponentActivationReg = Reg(new WSReuseActivation(config))
  private val exponentStateReg = Reg(new WSReuseExponentState(rows))
  io.exponentActivationOut.valid := RegNext(exponentFire, false.B)
  io.exponentStateOut.valid := RegNext(exponentFire, false.B)
  io.exponentActivationOut.bits := exponentActivationReg
  io.exponentStateOut.bits := exponentStateReg
  when(exponentFire) {
    exponentActivationReg := io.exponentActivation.bits
    exponentStateReg := io.exponentState.bits
    val term = TensorArithmetic.productMetadata(io.exponentActivation.bits.a,
      weights(slot(io.exponentState.bits.segmentBank, io.exponentState.bits.chunk)),
      config.format, config.zeroExponent)
    exponentStateReg.maximum := Mux(term.exponent > io.exponentState.bits.maximum,
      term.exponent, io.exponentState.bits.maximum)
  }

  private val sumFire = io.sumActivation.valid && io.sumState.valid
  when(io.sumActivation.valid || io.sumState.valid) {
    assert(io.sumActivation.valid === io.sumState.valid)
  }
  private val sumActivationReg = Reg(new WSReuseActivation(config))
  private val sumStateReg = Reg(new WSReuseSumState(config, rows))
  io.sumActivationOut.valid := RegNext(sumFire, false.B)
  io.sumStateOut.valid := RegNext(sumFire, false.B)
  io.sumActivationOut.bits := sumActivationReg
  io.sumStateOut.bits := sumStateReg
  when(sumFire) {
    sumActivationReg := io.sumActivation.bits
    sumStateReg := io.sumState.bits
    val term = TensorArithmetic.product(io.sumActivation.bits.a,
      weights(slot(io.sumState.bits.segmentBank, io.sumState.bits.chunk)),
      config.format, config.zeroExponent)
    sumStateReg.sum := io.sumState.bits.sum +
      TensorArithmetic.align(term, io.sumState.bits.maximum, config)
    sumStateReg.nan := io.sumState.bits.nan || term.nan
    sumStateReg.positiveInfinity := io.sumState.bits.positiveInfinity ||
      (term.infinity && !term.sign)
    sumStateReg.negativeInfinity := io.sumState.bits.negativeInfinity ||
      (term.infinity && term.sign)
  }
}

/** A double-buffered 16xN WS array for a 32-term E4M3 FDA.
  *
  * The exponent lane computes product-only maxima for the next segment while
  * the multiplier lane replays the current segment. Only when replay starts is
  * the prior FP32 C exponent combined with that maximum. No products or
  * complete-K operand vectors are stored.
  */
class WSReuseExperimentMesh(
    val config: BlackwellConfig = BlackwellConfig(TensorFormat.E4M3),
    val rows: Int = 2,
    val columns: Int = 2,
    val physicalDepth: Int = 16,
    val maxK: Int = 256) extends Module {
  require(!config.grouped && !config.halfOutput)
  require(config.fusionLength == 2 * physicalDepth,
    "The experiment maps one 32-term FDA onto two 16-term chunks")
  require(rows > 0 && columns > 0 && physicalDepth > 0)
  require(maxK >= config.fusionLength && maxK % config.fusionLength == 0)

  val io = IO(new Bundle {
    val req = Flipped(Decoupled(new WSReuseRequest(config, rows, columns)))
    val in = Flipped(Decoupled(new WSReuseOperands(config, rows, columns)))
    val out = Decoupled(new WSReuseResult(config, rows, columns))
    val exponentPass = Output(Bool())
    val expectedIndex = Output(UInt(16.W))
    val busy = Output(Bool())
    val overlap = Output(Bool())
  })

  private val fusionLength = config.fusionLength
  private val laneWidth = log2Ceil(fusionLength + 1)
  private val tagWidth = log2Ceil(rows max 2)
  private val cycleWidth = log2Ceil(2 * physicalDepth + rows + columns + 2)
  private val resultWidth = log2Ceil(rows * columns + 1)

  private val idle :: active :: respond :: Nil = Enum(3)
  private val state = RegInit(idle)
  private val bankEmpty = 0.U(3.W)
  private val bankFull = 1.U(3.W)
  private val bankPrepassing = 2.U(3.W)
  private val bankPrepassed = 3.U(3.W)
  private val bankReplaying = 4.U(3.W)
  private val bankState = RegInit(VecInit(Seq.fill(2)(0.U(3.W))))
  private val bankSequence = Reg(Vec(2, UInt(16.W)))

  private val requestK = Reg(UInt(16.W))
  private val totalSegments = Reg(UInt(16.W))
  private val accepted = Reg(UInt(16.W))
  private val fillOffset = Reg(UInt(laneWidth.W))
  private val fillBank = RegInit(false.B)
  private val nextFillSequence = Reg(UInt(16.W))
  private val nextPrepassSequence = Reg(UInt(16.W))
  private val nextReplaySequence = Reg(UInt(16.W))
  private val tag = Reg(UInt(16.W))
  private val accumulators = Reg(Vec(rows, Vec(columns, UInt(config.outputWidth.W))))
  private val activations = Reg(Vec(2,
    Vec(rows, Vec(fusionLength, UInt(config.format.width.W)))))

  private val pes = Seq.tabulate(physicalDepth, columns) { (_, _) =>
    Module(new WSReuseExperimentPE(config, rows))
  }

  io.req.ready := state === idle
  io.in.ready := state === active && accepted < requestK && bankState(fillBank) === bankEmpty
  io.out.valid := state === respond
  io.out.bits.d := accumulators
  io.out.bits.tag := tag
  io.expectedIndex := accepted
  io.busy := state =/= idle

  private val fillChunk = fillOffset >= physicalDepth.U
  private val fillDepth = Mux(fillChunk, fillOffset - physicalDepth.U, fillOffset)
  for (depth <- 0 until physicalDepth; column <- 0 until columns) {
    val pe = pes(depth)(column)
    pe.io.load.valid := io.in.fire && fillDepth === depth.U
    pe.io.load.bits.segmentBank := fillBank
    pe.io.load.bits.chunk := fillChunk
    pe.io.load.bits.b := io.in.bits.b(column)
  }
  when(io.in.fire) {
    for (row <- 0 until rows) {
      activations(fillBank)(row)(fillOffset) := io.in.bits.a(row)
    }
    accepted := accepted + 1.U
    when(fillOffset === (fusionLength - 1).U) {
      bankSequence(fillBank) := nextFillSequence
      bankState(fillBank) := bankFull
      nextFillSequence := nextFillSequence + 1.U
      fillOffset := 0.U
      fillBank := !fillBank
    }.otherwise {
      fillOffset := fillOffset + 1.U
    }
  }

  private val prepassMaximum = Reg(Vec(2, Vec(rows, Vec(columns, SInt(9.W)))))
  private val prepassCandidateValid = WireDefault(false.B)
  private val prepassCandidate = WireDefault(false.B)
  when(bankState(0) === bankFull && bankSequence(0) === nextPrepassSequence) {
    prepassCandidateValid := true.B; prepassCandidate := false.B
  }
  when(bankState(1) === bankFull && bankSequence(1) === nextPrepassSequence) {
    prepassCandidateValid := true.B; prepassCandidate := true.B
  }

  private val exponentActive = RegInit(false.B)
  private val exponentBank = Reg(Bool())
  private val exponentChunk = RegInit(false.B)
  private val exponentCycle = Reg(UInt(cycleWidth.W))
  private val exponentFeedbackCount = Reg(UInt(resultWidth.W))
  when(state === active && !exponentActive && prepassCandidateValid) {
    exponentActive := true.B
    exponentBank := prepassCandidate
    exponentChunk := false.B
    exponentCycle := 0.U
    exponentFeedbackCount := 0.U
    bankState(prepassCandidate) := bankPrepassing
    for (row <- 0 until rows; column <- 0 until columns) {
      prepassMaximum(prepassCandidate)(row)(column) := (-256).S
    }
  }.elsewhen(exponentActive) {
    exponentCycle := exponentCycle + 1.U
  }

  for (depth <- 0 until physicalDepth; column <- 0 until columns) {
    val pe = pes(depth)(column)
    if (column == 0) {
      val valid = exponentActive && exponentCycle >= depth.U &&
        exponentCycle < (depth + rows).U
      val outputRow = (exponentCycle - depth.U)(tagWidth - 1, 0)
      pe.io.exponentActivation.valid := valid
      pe.io.exponentActivation.bits.a := activations(exponentBank)(outputRow)(
        Mux(exponentChunk, (depth + physicalDepth).U, depth.U))
    } else {
      pe.io.exponentActivation := pes(depth)(column - 1).io.exponentActivationOut
    }
    if (depth == 0) {
      val valid = exponentActive && exponentCycle >= column.U &&
        exponentCycle < (column + rows).U
      val outputRow = (exponentCycle - column.U)(tagWidth - 1, 0)
      pe.io.exponentState.valid := valid
      pe.io.exponentState.bits.maximum := prepassMaximum(exponentBank)(outputRow)(column)
      pe.io.exponentState.bits.tag := outputRow
      pe.io.exponentState.bits.segmentBank := exponentBank
      pe.io.exponentState.bits.chunk := exponentChunk
    } else {
      pe.io.exponentState := pes(depth - 1)(column).io.exponentStateOut
    }
  }

  private val feedbackValid = Wire(Vec(columns, Bool()))
  private val feedbackBits = Wire(Vec(columns, new WSReuseFeedback(rows)))
  for (column <- 0 until columns) {
    var valid = pes(physicalDepth - 1)(column).io.exponentStateOut.valid
    val bottom = pes(physicalDepth - 1)(column).io.exponentStateOut.bits
    var bits = Wire(new WSReuseFeedback(rows))
    bits.maximum := bottom.maximum
    bits.tag := bottom.tag
    bits.segmentBank := bottom.segmentBank
    for (_ <- 0 until physicalDepth) {
      val nextValid = RegNext(valid, false.B)
      val nextBits = RegEnable(bits, valid)
      valid = nextValid; bits = nextBits
    }
    feedbackValid(column) := valid
    feedbackBits(column) := bits
    when(valid) {
      prepassMaximum(bits.segmentBank)(bits.tag)(column) := bits.maximum
    }
  }
  private val feedbackThisCycle = PopCount(feedbackValid)
  when(exponentActive && feedbackThisCycle =/= 0.U) {
    exponentFeedbackCount := exponentFeedbackCount + feedbackThisCycle
    when(exponentFeedbackCount + feedbackThisCycle === (rows * columns).U) {
      when(!exponentChunk) {
        exponentChunk := true.B
        exponentCycle := 0.U
        exponentFeedbackCount := 0.U
      }.otherwise {
        exponentActive := false.B
        bankState(exponentBank) := bankPrepassed
        nextPrepassSequence := nextPrepassSequence + 1.U
      }
    }
  }

  private val replayCandidateValid = WireDefault(false.B)
  private val replayCandidate = WireDefault(false.B)
  when(bankState(0) === bankPrepassed && bankSequence(0) === nextReplaySequence) {
    replayCandidateValid := true.B; replayCandidate := false.B
  }
  when(bankState(1) === bankPrepassed && bankSequence(1) === nextReplaySequence) {
    replayCandidateValid := true.B; replayCandidate := true.B
  }

  private val replayActive = RegInit(false.B)
  private val replayBank = Reg(Bool())
  private val replayChunk = RegInit(false.B)
  private val replayCycle = Reg(UInt(cycleWidth.W))
  private val replayResultCount = Reg(UInt(resultWidth.W))
  private val replayMaximum = Reg(Vec(rows, Vec(columns, SInt(9.W))))
  private val fixedSums = Reg(Vec(rows, Vec(columns, SInt(config.sumWidth.W))))
  private val nan = Reg(Vec(rows, Vec(columns, Bool())))
  private val positiveInfinity = Reg(Vec(rows, Vec(columns, Bool())))
  private val negativeInfinity = Reg(Vec(rows, Vec(columns, Bool())))

  when(state === active && !replayActive && replayCandidateValid) {
    replayActive := true.B
    replayBank := replayCandidate
    replayChunk := false.B
    replayCycle := 0.U
    replayResultCount := 0.U
    bankState(replayCandidate) := bankReplaying
    for (row <- 0 until rows; column <- 0 until columns) {
      val c = TensorArithmetic.decode(accumulators(row)(column), 8, 23)
      val cExponent = Mux(c.magnitude === 0.U, config.zeroExponent.S, c.exponent)
      val productMaximum = prepassMaximum(replayCandidate)(row)(column)
      val maximum = Mux(cExponent > productMaximum, cExponent, productMaximum)
      replayMaximum(row)(column) := maximum
      fixedSums(row)(column) := TensorArithmetic.align(c, maximum, config)
      nan(row)(column) := c.nan
      positiveInfinity(row)(column) := c.infinity && !c.sign
      negativeInfinity(row)(column) := c.infinity && c.sign
    }
  }.elsewhen(replayActive) {
    replayCycle := replayCycle + 1.U
  }

  for (depth <- 0 until physicalDepth; column <- 0 until columns) {
    val pe = pes(depth)(column)
    if (column == 0) {
      val valid = replayActive && replayCycle >= depth.U &&
        replayCycle < (depth + rows).U
      val outputRow = (replayCycle - depth.U)(tagWidth - 1, 0)
      pe.io.sumActivation.valid := valid
      pe.io.sumActivation.bits.a := activations(replayBank)(outputRow)(
        Mux(replayChunk, (depth + physicalDepth).U, depth.U))
    } else {
      pe.io.sumActivation := pes(depth)(column - 1).io.sumActivationOut
    }
    if (depth == 0) {
      val valid = replayActive && replayCycle >= column.U &&
        replayCycle < (column + rows).U
      val outputRow = (replayCycle - column.U)(tagWidth - 1, 0)
      pe.io.sumState.valid := valid
      pe.io.sumState.bits.maximum := replayMaximum(outputRow)(column)
      pe.io.sumState.bits.sum := fixedSums(outputRow)(column)
      pe.io.sumState.bits.nan := nan(outputRow)(column)
      pe.io.sumState.bits.positiveInfinity := positiveInfinity(outputRow)(column)
      pe.io.sumState.bits.negativeInfinity := negativeInfinity(outputRow)(column)
      pe.io.sumState.bits.tag := outputRow
      pe.io.sumState.bits.segmentBank := replayBank
      pe.io.sumState.bits.chunk := replayChunk
    } else {
      pe.io.sumState := pes(depth - 1)(column).io.sumStateOut
    }
  }

  private def packed(result: WSReuseSumState): UInt = {
    val canonicalNaN = ((BigInt(1) << 31) - 1).U(32.W)
    val infinity = BigInt(0x7f800000L).U(32.W)
    val sign = BigInt(0x80000000L).U(32.W)
    Mux(result.nan || (result.positiveInfinity && result.negativeInfinity), canonicalNaN,
      Mux(result.positiveInfinity || result.negativeInfinity,
        infinity | Mux(result.negativeInfinity, sign, 0.U),
        TensorArithmetic.pack(result.sum, result.maximum,
          config.fractionalBits, halfOutput = false)))
  }

  private val replayBottomValid = Wire(Vec(columns, Bool()))
  for (column <- 0 until columns) {
    val bottom = pes(physicalDepth - 1)(column).io.sumStateOut
    replayBottomValid(column) := bottom.valid
    when(bottom.valid) {
      when(replayChunk) {
        accumulators(bottom.bits.tag)(column) := packed(bottom.bits)
      }.otherwise {
        fixedSums(bottom.bits.tag)(column) := bottom.bits.sum
        nan(bottom.bits.tag)(column) := bottom.bits.nan
        positiveInfinity(bottom.bits.tag)(column) := bottom.bits.positiveInfinity
        negativeInfinity(bottom.bits.tag)(column) := bottom.bits.negativeInfinity
      }
    }
  }
  private val replayThisCycle = PopCount(replayBottomValid)
  when(replayActive && replayThisCycle =/= 0.U) {
    replayResultCount := replayResultCount + replayThisCycle
    when(replayResultCount + replayThisCycle === (rows * columns).U) {
      when(!replayChunk) {
        replayChunk := true.B
        replayCycle := 0.U
        replayResultCount := 0.U
      }.otherwise {
        replayActive := false.B
        bankState(replayBank) := bankEmpty
        nextReplaySequence := nextReplaySequence + 1.U
        when(nextReplaySequence === totalSegments - 1.U) { state := respond }
      }
    }
  }

  io.exponentPass := exponentActive
  io.overlap := exponentActive && replayActive

  when(io.req.fire) {
    assert(io.req.bits.k > 0.U && io.req.bits.k <= maxK.U)
    assert(io.req.bits.k % fusionLength.U === 0.U)
    requestK := io.req.bits.k
    totalSegments := io.req.bits.k / fusionLength.U
    accepted := 0.U
    fillOffset := 0.U
    fillBank := false.B
    nextFillSequence := 0.U
    nextPrepassSequence := 0.U
    nextReplaySequence := 0.U
    bankState(0) := bankEmpty
    bankState(1) := bankEmpty
    exponentActive := false.B
    replayActive := false.B
    accumulators := io.req.bits.c
    tag := io.req.bits.tag
    state := active
  }
  when(io.out.fire) { state := idle }
}
