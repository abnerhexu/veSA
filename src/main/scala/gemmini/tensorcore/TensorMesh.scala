package gemmini.tensorcore

import chisel3._
import chisel3.util._

class TensorMatrixRequest(config: TensorConfig, rows: Int, columns: Int) extends Bundle {
  val c = Vec(rows, Vec(columns, UInt(config.outputWidth.W)))
  val k = UInt(16.W)
  val tag = UInt(16.W)
}

class TensorMatrixOperands(config: TensorConfig, rows: Int, columns: Int) extends Bundle {
  // One K slice: A[:, k] and B[k, :]. Skewing is internal to the mesh.
  val a = Vec(rows, UInt(config.format.width.W))
  val b = Vec(columns, UInt(config.format.width.W))
  val alpha = if (config.grouped) Some(Vec(rows, UInt(8.W))) else None
  val beta = if (config.grouped) Some(Vec(columns, UInt(8.W))) else None
}

class TensorMatrixResult(config: TensorConfig, rows: Int, columns: Int) extends Bundle {
  val d = Vec(rows, Vec(columns, UInt(config.outputWidth.W)))
  val tag = UInt(16.W)
}

/** Weight-stationary fused-dot systolic array.
  *
  * A physical column contains one PE per term in the architectural fusion
  * length. B is stationary in a double-buffered PE register. The first wave is
  * a multiplier-free (max,+) exponent pass; its narrow result
  * returns through a fusionLength-deep feedback FIFO. The second wave repeats
  * A from the left-edge buffer, aligns each exact product to that fixed exponent
  * and passes a fixed-point sum down the column. C is aligned at the top.
  *
  * Segments are normalized at the column bottom. That rounded value becomes C
  * for the next segment, preserving CoFDA boundaries for K > fusionLength.
  * While one bank is replayed, the other bank may be loaded and prepassed, so
  * the exponent and significand lanes operate concurrently in steady state.
  */
class TensorMesh(val config: TensorConfig = BlackwellConfig(),
                 val rows: Int = 2, val columns: Int = 2) extends Module {
  require(rows > 0 && columns > 0)
  val io = IO(new Bundle {
    val req = Flipped(Decoupled(new TensorMatrixRequest(config, rows, columns)))
    val in = Flipped(Decoupled(new TensorMatrixOperands(config, rows, columns)))
    val out = Decoupled(new TensorMatrixResult(config, rows, columns))
  })
  private val fusionLength = config.fusionLength
  private val fusionShift = log2Ceil(fusionLength)
  require((1 << fusionShift) == fusionLength, "WS segment length must be a power of two")
  private val laneWidth = log2Ceil(fusionLength + 1)
  private val tagWidth = log2Ceil(rows max 2)
  private val cycleWidth = log2Ceil(2 * fusionLength + rows + columns + 2)

  private val idle :: active :: respond :: Nil = Enum(3)
  private val state = RegInit(idle)
  private val bankEmpty = 0.U(3.W)
  private val bankFull = 1.U(3.W)
  private val bankPrepassing = 2.U(3.W)
  private val bankPrepassed = 3.U(3.W)
  private val bankReplaying = 4.U(3.W)
  private val bankState = RegInit(VecInit(Seq.fill(2)(0.U(3.W))))
  private val bankLength = Reg(Vec(2, UInt(laneWidth.W)))
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

  // Only A is retained at the array edge. B lives in the two stationary banks
  // inside every PE and is never replayed from a product buffer.
  private val activations = Reg(Vec(2,
    Vec(rows, Vec(fusionLength, UInt(config.format.width.W)))))
  private val alphas = if (config.grouped) Some(Reg(Vec(2,
    Vec(rows, Vec(fusionLength, UInt(8.W)))))) else None

  private val pes = Seq.tabulate(fusionLength, columns) { (r, _) =>
    Module(new TensorWSPE(config, rows, config.grouped && (r + 1) % 16 == 0))
  }

  io.req.ready := state === idle
  io.in.ready := state === active && accepted < requestK && bankState(fillBank) === bankEmpty
  io.out.valid := state === respond
  io.out.bits.d := accumulators
  io.out.bits.tag := tag

  // Load one K position into the selected weight bank and transpose the A rows
  // into the left-edge replay store.
  for (r <- 0 until fusionLength; c <- 0 until columns) {
    val pe = pes(r)(c)
    pe.io.load.valid := io.in.fire && fillOffset === r.U
    pe.io.load.bits.bank := fillBank
    pe.io.load.bits.b := io.in.bits.b(c)
    if (config.grouped) { pe.io.load.bits.beta.get := io.in.bits.beta.get(c) }
  }
  when(io.in.fire) {
    for (r <- 0 until rows) {
      activations(fillBank)(r)(fillOffset) := io.in.bits.a(r)
      if (config.grouped) { alphas.get(fillBank)(r)(fillOffset) := io.in.bits.alpha.get(r) }
    }
    accepted := accepted + 1.U
    when(fillOffset === (fusionLength - 1).U || accepted === requestK - 1.U) {
      bankLength(fillBank) := fillOffset + 1.U
      bankSequence(fillBank) := nextFillSequence
      bankState(fillBank) := bankFull
      nextFillSequence := nextFillSequence + 1.U
      fillOffset := 0.U
      fillBank := !fillBank
    }.otherwise {
      fillOffset := fillOffset + 1.U
    }
  }

  private val prepassCandidateValid = WireDefault(false.B)
  private val prepassCandidate = WireDefault(false.B)
  when(bankState(0) === bankFull && bankSequence(0) === nextPrepassSequence) {
    prepassCandidateValid := true.B
    prepassCandidate := false.B
  }
  when(bankState(1) === bankFull && bankSequence(1) === nextPrepassSequence) {
    prepassCandidateValid := true.B
    prepassCandidate := true.B
  }

  private val exponentActive = RegInit(false.B)
  private val exponentBank = Reg(Bool())
  private val exponentCycle = Reg(UInt(cycleWidth.W))
  private val exponentFeedbackCount = Reg(UInt(log2Ceil(rows * columns + 1).W))
  private val prepassResults = Reg(Vec(2,
    Vec(rows, Vec(columns, new TensorWSFeedback(rows)))))

  when(state === active && !exponentActive && prepassCandidateValid) {
    exponentActive := true.B
    exponentBank := prepassCandidate
    exponentCycle := 0.U
    exponentFeedbackCount := 0.U
    bankState(prepassCandidate) := bankPrepassing
  }.elsewhen(exponentActive) {
    exponentCycle := exponentCycle + 1.U
  }

  // Exponent activations enter each physical K row diagonally from the left;
  // product maxima enter each output column diagonally from the top.
  for (r <- 0 until fusionLength; c <- 0 until columns) {
    val pe = pes(r)(c)
    if (c == 0) {
      val valid = exponentActive && exponentCycle >= r.U &&
        exponentCycle < (r + rows).U
      val outputRow = (exponentCycle - r.U)(tagWidth - 1, 0)
      pe.io.exponentActivation.valid := valid
      pe.io.exponentActivation.bits.a := activations(exponentBank)(outputRow)(r)
      pe.io.exponentActivation.bits.active := r.U < bankLength(exponentBank)
      if (config.grouped) {
        pe.io.exponentActivation.bits.alpha.get := alphas.get(exponentBank)(outputRow)(r)
      }
    } else {
      pe.io.exponentActivation := pes(r)(c - 1).io.exponentActivationOut
    }
    if (r == 0) {
      val valid = exponentActive && exponentCycle >= c.U &&
        exponentCycle < (c + rows).U
      pe.io.exponentState.valid := valid
      pe.io.exponentState.bits := 0.U.asTypeOf(new TensorWSExponentState(config, rows))
      pe.io.exponentState.bits.maximum := (-256).S
      pe.io.exponentState.bits.tag := (exponentCycle - c.U)(tagWidth - 1, 0)
      pe.io.exponentState.bits.bank := exponentBank
    } else {
      pe.io.exponentState := pes(r - 1)(c).io.exponentStateOut
    }
  }

  // The column-bottom result returns to the top through a narrow FIFO. Its
  // depth models the physical feedback route and keeps the second wave honest.
  private val feedbackValid = Wire(Vec(columns, Bool()))
  private val feedbackBits = Wire(Vec(columns, new TensorWSFeedback(rows)))
  for (c <- 0 until columns) {
    var valid = pes(fusionLength - 1)(c).io.exponentStateOut.valid
    val bottom = pes(fusionLength - 1)(c).io.exponentStateOut.bits
    var bits = Wire(new TensorWSFeedback(rows))
    bits.maximum := bottom.maximum
    bits.tag := bottom.tag
    bits.bank := bottom.bank
    for (_ <- 0 until fusionLength) {
      val nextValid = RegNext(valid, false.B)
      val nextBits = RegEnable(bits, valid)
      valid = nextValid
      bits = nextBits
    }
    feedbackValid(c) := valid
    feedbackBits(c) := bits
    when(valid) {
      prepassResults(bits.bank)(bits.tag)(c) := bits
    }
  }
  private val feedbackThisCycle = PopCount(feedbackValid)
  when(exponentActive && feedbackThisCycle =/= 0.U) {
    exponentFeedbackCount := exponentFeedbackCount + feedbackThisCycle
    when(exponentFeedbackCount + feedbackThisCycle === (rows * columns).U) {
      exponentActive := false.B
      bankState(exponentBank) := bankPrepassed
      nextPrepassSequence := nextPrepassSequence + 1.U
    }
  }

  private val replayCandidateValid = WireDefault(false.B)
  private val replayCandidate = WireDefault(false.B)
  when(bankState(0) === bankPrepassed && bankSequence(0) === nextReplaySequence) {
    replayCandidateValid := true.B
    replayCandidate := false.B
  }
  when(bankState(1) === bankPrepassed && bankSequence(1) === nextReplaySequence) {
    replayCandidateValid := true.B
    replayCandidate := true.B
  }

  private val replayActive = RegInit(false.B)
  private val replayBank = Reg(Bool())
  private val replayCycle = Reg(UInt(cycleWidth.W))
  private val replayResultCount = Reg(UInt(log2Ceil(rows * columns + 1).W))
  when(state === active && !replayActive && replayCandidateValid) {
    replayActive := true.B
    replayBank := replayCandidate
    replayCycle := 0.U
    replayResultCount := 0.U
    bankState(replayCandidate) := bankReplaying
  }.elsewhen(replayActive) {
    replayCycle := replayCycle + 1.U
  }

  for (r <- 0 until fusionLength; c <- 0 until columns) {
    val pe = pes(r)(c)
    if (c == 0) {
      val valid = replayActive && replayCycle >= r.U && replayCycle < (r + rows).U
      val outputRow = (replayCycle - r.U)(tagWidth - 1, 0)
      pe.io.sumActivation.valid := valid
      pe.io.sumActivation.bits.a := activations(replayBank)(outputRow)(r)
      pe.io.sumActivation.bits.active := r.U < bankLength(replayBank)
      if (config.grouped) {
        pe.io.sumActivation.bits.alpha.get := alphas.get(replayBank)(outputRow)(r)
      }
    } else {
      pe.io.sumActivation := pes(r)(c - 1).io.sumActivationOut
    }
    if (r == 0) {
      val valid = replayActive && replayCycle >= c.U && replayCycle < (c + rows).U
      val outputRow = (replayCycle - c.U)(tagWidth - 1, 0)
      val cTerm = TensorArithmetic.decode(accumulators(outputRow)(c),
        if (config.halfOutput) 5 else 8, if (config.halfOutput) 10 else 23)
      val cExponent = Mux(cTerm.magnitude === 0.U, config.zeroExponent.S, cTerm.exponent)
      val product = prepassResults(replayBank)(outputRow)(c)
      val maximum = Wire(SInt(9.W))
      maximum := Mux(cExponent > product.maximum, cExponent, product.maximum)
      pe.io.sumState.valid := valid
      pe.io.sumState.bits := 0.U.asTypeOf(new TensorWSSumState(config, rows))
      pe.io.sumState.bits.maximum := maximum
      pe.io.sumState.bits.sum := TensorArithmetic.align(cTerm, maximum, config)
      pe.io.sumState.bits.nan := cTerm.nan
      pe.io.sumState.bits.positiveInfinity := cTerm.infinity && !cTerm.sign
      pe.io.sumState.bits.negativeInfinity := cTerm.infinity && cTerm.sign
      pe.io.sumState.bits.tag := outputRow
      pe.io.sumState.bits.bank := replayBank
    } else {
      pe.io.sumState := pes(r - 1)(c).io.sumStateOut
    }
  }

  private def packed(result: TensorWSSumState): UInt = {
    val canonicalNaN = ((BigInt(1) << (config.outputWidth - 1)) - 1).U
    val infinity = (if (config.halfOutput) BigInt(0x7c00) else BigInt(0x7f800000))
      .U(config.outputWidth.W)
    val sign = (BigInt(1) << (config.outputWidth - 1)).U(config.outputWidth.W)
    Mux(result.nan || (result.positiveInfinity && result.negativeInfinity), canonicalNaN,
      Mux(result.positiveInfinity || result.negativeInfinity,
        infinity | Mux(result.negativeInfinity, sign, 0.U),
        TensorArithmetic.pack(result.sum, result.maximum,
          config.fractionalBits, config.halfOutput, config.outputFractionBits)))
  }

  private val replayBottomValid = Wire(Vec(columns, Bool()))
  for (c <- 0 until columns) {
    val bottom = pes(fusionLength - 1)(c).io.sumStateOut
    replayBottomValid(c) := bottom.valid
    when(bottom.valid) {
      accumulators(bottom.bits.tag)(c) := packed(bottom.bits)
    }
  }
  private val replayThisCycle = PopCount(replayBottomValid)
  when(replayActive && replayThisCycle =/= 0.U) {
    replayResultCount := replayResultCount + replayThisCycle
    when(replayResultCount + replayThisCycle === (rows * columns).U) {
      replayActive := false.B
      bankState(replayBank) := bankEmpty
      nextReplaySequence := nextReplaySequence + 1.U
      when(nextReplaySequence === totalSegments - 1.U) { state := respond }
    }
  }

  when(io.req.fire) {
    assert(io.req.bits.k > 0.U)
    if (config.grouped) {
      assert(io.req.bits.k(5, 0) === 0.U, "FP4 K must be a multiple of 64")
    }
    requestK := io.req.bits.k
    totalSegments := (io.req.bits.k + (fusionLength - 1).U) >> fusionShift
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
