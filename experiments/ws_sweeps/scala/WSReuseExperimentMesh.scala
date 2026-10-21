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

/** A physicalDepth x columns WS array whose numerical fusion length is larger
  * than its physical K depth.
  *
  * The producer follows expectedIndex and sends each fusion block twice.  The
  * first pass sends exponents through the narrow max lane.  The second pass
  * reloads the same raw operands and sends significands through the multiplier
  * and fixed-sum lane.  Only one physical K chunk of A and B is resident; no
  * product or full-K operand buffer is instantiated.
  */
class WSReuseExperimentMesh(
    val config: BlackwellConfig = BlackwellConfig(TensorFormat.E4M3),
    val rows: Int = 2,
    val columns: Int = 2,
    val physicalDepth: Int = 16,
    val maxK: Int = 256) extends Module {
  require(!config.grouped, "The area/K sweep uses the unscaled FP8 datapath")
  require(!config.halfOutput, "The area/K sweep retains FP32 C/D")
  require(rows > 0 && columns > 0 && physicalDepth > 0)
  require(config.fusionLength % physicalDepth == 0,
    "Physical depth must divide the numerical fusion length")
  require(maxK >= config.fusionLength && maxK % config.fusionLength == 0)

  val io = IO(new Bundle {
    val req = Flipped(Decoupled(new WSReuseRequest(config, rows, columns)))
    val in = Flipped(Decoupled(new WSReuseOperands(config, rows, columns)))
    val out = Decoupled(new WSReuseResult(config, rows, columns))
    val exponentPass = Output(Bool())
    val expectedIndex = Output(UInt(16.W))
    val busy = Output(Bool())
  })

  private val fusionLength = config.fusionLength
  private val fillWidth = log2Ceil(physicalDepth max 2)
  private val tagWidth = log2Ceil(rows max 2)
  private val waveWidth = log2Ceil(2 * physicalDepth + rows + columns + 4)
  private val resultCountWidth = log2Ceil(rows * columns + 1)

  private val Seq(idle, initExponent, loadChunk, runWave, initSum,
    finishBlock, respond) = Enum(7)
  private val state = RegInit(idle)
  private val exponentPass = RegInit(true.B)
  private val requestK = Reg(UInt(16.W))
  private val blockBase = Reg(UInt(16.W))
  private val chunkOffset = Reg(UInt(log2Ceil(fusionLength max 2).W))
  private val fillOffset = Reg(UInt(fillWidth.W))
  private val activeBank = RegInit(false.B)
  private val waveCycle = Reg(UInt(waveWidth.W))
  private val resultCount = Reg(UInt(resultCountWidth.W))
  private val tag = Reg(UInt(16.W))

  private val accumulators = Reg(Vec(rows, Vec(columns, UInt(config.outputWidth.W))))
  private val maxima = Reg(Vec(rows, Vec(columns, SInt(9.W))))
  private val fixedSums = Reg(Vec(rows, Vec(columns, SInt(config.sumWidth.W))))
  private val nan = Reg(Vec(rows, Vec(columns, Bool())))
  private val positiveInfinity = Reg(Vec(rows, Vec(columns, Bool())))
  private val negativeInfinity = Reg(Vec(rows, Vec(columns, Bool())))
  private val activations = Reg(Vec(rows,
    Vec(physicalDepth, UInt(config.format.width.W))))

  private val pes = Seq.tabulate(physicalDepth, columns) { (_, _) =>
    Module(new TensorWSPE(config, rows, groupEnd = false))
  }

  io.req.ready := state === idle
  io.in.ready := state === loadChunk
  io.out.valid := state === respond
  io.out.bits.d := accumulators
  io.out.bits.tag := tag
  io.exponentPass := exponentPass
  io.expectedIndex := blockBase + chunkOffset + fillOffset
  io.busy := state =/= idle

  for (depth <- 0 until physicalDepth; column <- 0 until columns) {
    val pe = pes(depth)(column)
    pe.io.load.valid := io.in.fire && fillOffset === depth.U
    pe.io.load.bits.bank := activeBank
    pe.io.load.bits.b := io.in.bits.b(column)
  }

  when(io.in.fire) {
    for (row <- 0 until rows) {
      activations(row)(fillOffset) := io.in.bits.a(row)
    }
    when(fillOffset === (physicalDepth - 1).U) {
      fillOffset := 0.U
      waveCycle := 0.U
      resultCount := 0.U
      state := runWave
    }.otherwise {
      fillOffset := fillOffset + 1.U
    }
  }

  for (depth <- 0 until physicalDepth; column <- 0 until columns) {
    val pe = pes(depth)(column)
    if (column == 0) {
      val row = (waveCycle - depth.U)(tagWidth - 1, 0)
      val valid = state === runWave && waveCycle >= depth.U &&
        waveCycle < (depth + rows).U
      pe.io.exponentActivation.valid := valid && exponentPass
      pe.io.exponentActivation.bits := 0.U.asTypeOf(new TensorWSActivation(config))
      pe.io.exponentActivation.bits.a := activations(row)(depth)
      pe.io.exponentActivation.bits.active := true.B
      pe.io.sumActivation.valid := valid && !exponentPass
      pe.io.sumActivation.bits := 0.U.asTypeOf(new TensorWSActivation(config))
      pe.io.sumActivation.bits.a := activations(row)(depth)
      pe.io.sumActivation.bits.active := true.B
    } else {
      pe.io.exponentActivation := pes(depth)(column - 1).io.exponentActivationOut
      pe.io.sumActivation := pes(depth)(column - 1).io.sumActivationOut
    }

    if (depth == 0) {
      val row = (waveCycle - column.U)(tagWidth - 1, 0)
      val valid = state === runWave && waveCycle >= column.U &&
        waveCycle < (column + rows).U
      pe.io.exponentState.valid := valid && exponentPass
      pe.io.exponentState.bits := 0.U.asTypeOf(new TensorWSExponentState(config, rows))
      pe.io.exponentState.bits.maximum := maxima(row)(column)
      pe.io.exponentState.bits.tag := row
      pe.io.exponentState.bits.bank := activeBank

      pe.io.sumState.valid := valid && !exponentPass
      pe.io.sumState.bits := 0.U.asTypeOf(new TensorWSSumState(config, rows))
      pe.io.sumState.bits.maximum := maxima(row)(column)
      pe.io.sumState.bits.sum := fixedSums(row)(column)
      pe.io.sumState.bits.nan := nan(row)(column)
      pe.io.sumState.bits.positiveInfinity := positiveInfinity(row)(column)
      pe.io.sumState.bits.negativeInfinity := negativeInfinity(row)(column)
      pe.io.sumState.bits.tag := row
      pe.io.sumState.bits.bank := activeBank
    } else {
      pe.io.exponentState := pes(depth - 1)(column).io.exponentStateOut
      pe.io.sumState := pes(depth - 1)(column).io.sumStateOut
    }
  }

  // Match the production architecture's narrow physical return route.
  private val feedbackValid = Wire(Vec(columns, Bool()))
  private val feedbackBits = Wire(Vec(columns, new TensorWSFeedback(rows)))
  for (column <- 0 until columns) {
    var valid = pes(physicalDepth - 1)(column).io.exponentStateOut.valid
    val bottom = pes(physicalDepth - 1)(column).io.exponentStateOut.bits
    var bits = Wire(new TensorWSFeedback(rows))
    bits.maximum := bottom.maximum
    bits.tag := bottom.tag
    bits.bank := bottom.bank
    for (_ <- 0 until physicalDepth) {
      val nextValid = RegNext(valid, false.B)
      val nextBits = RegEnable(bits, valid)
      valid = nextValid
      bits = nextBits
    }
    feedbackValid(column) := valid
    feedbackBits(column) := bits
    when(valid) {
      maxima(bits.tag)(column) := bits.maximum
    }
  }

  private val sumBottomValid = Wire(Vec(columns, Bool()))
  for (column <- 0 until columns) {
    val bottom = pes(physicalDepth - 1)(column).io.sumStateOut
    sumBottomValid(column) := bottom.valid
    when(bottom.valid) {
      fixedSums(bottom.bits.tag)(column) := bottom.bits.sum
      nan(bottom.bits.tag)(column) := bottom.bits.nan
      positiveInfinity(bottom.bits.tag)(column) := bottom.bits.positiveInfinity
      negativeInfinity(bottom.bits.tag)(column) := bottom.bits.negativeInfinity
    }
  }

  private val completedThisCycle = Mux(exponentPass,
    PopCount(feedbackValid), PopCount(sumBottomValid))

  when(state === runWave) {
    waveCycle := waveCycle + 1.U
    when(completedThisCycle =/= 0.U) {
      resultCount := resultCount + completedThisCycle
    }
    // Keep the direct horizontal companion schedule active until every PE
    // register and the physical exponent-return route have drained.  Using a
    // fixed geometry-derived window avoids a stale feedback pulse ending the
    // following chunk early.
    when(waveCycle === (2 * physicalDepth + rows + columns + 1).U) {
      assert(resultCount === (rows * columns).U,
        "WS wave did not return every output")
      when((chunkOffset +& physicalDepth.U) < fusionLength.U) {
        chunkOffset := chunkOffset + physicalDepth.U
        activeBank := !activeBank
        fillOffset := 0.U
        state := loadChunk
      }.elsewhen(exponentPass) {
        state := initSum
      }.otherwise {
        state := finishBlock
      }
    }
  }

  when(state === initExponent) {
    for (row <- 0 until rows; column <- 0 until columns) {
      val c = TensorArithmetic.decode(accumulators(row)(column), 8, 23)
      maxima(row)(column) := Mux(c.magnitude === 0.U,
        config.zeroExponent.S, c.exponent)
    }
    exponentPass := true.B
    chunkOffset := 0.U
    fillOffset := 0.U
    activeBank := false.B
    state := loadChunk
  }

  when(state === initSum) {
    for (row <- 0 until rows; column <- 0 until columns) {
      val c = TensorArithmetic.decode(accumulators(row)(column), 8, 23)
      fixedSums(row)(column) := TensorArithmetic.align(c, maxima(row)(column), config)
      nan(row)(column) := c.nan
      positiveInfinity(row)(column) := c.infinity && !c.sign
      negativeInfinity(row)(column) := c.infinity && c.sign
    }
    exponentPass := false.B
    chunkOffset := 0.U
    fillOffset := 0.U
    activeBank := false.B
    state := loadChunk
  }

  private def packed(row: Int, column: Int): UInt = {
    val canonicalNaN = ((BigInt(1) << 31) - 1).U(32.W)
    val infinity = BigInt(0x7f800000L).U(32.W)
    val sign = BigInt(0x80000000L).U(32.W)
    Mux(nan(row)(column) ||
      (positiveInfinity(row)(column) && negativeInfinity(row)(column)), canonicalNaN,
      Mux(positiveInfinity(row)(column) || negativeInfinity(row)(column),
        infinity | Mux(negativeInfinity(row)(column), sign, 0.U),
        TensorArithmetic.pack(fixedSums(row)(column), maxima(row)(column),
          config.fractionalBits, halfOutput = false)))
  }

  when(state === finishBlock) {
    for (row <- 0 until rows; column <- 0 until columns) {
      accumulators(row)(column) := packed(row, column)
    }
    when(blockBase + fusionLength.U < requestK) {
      blockBase := blockBase + fusionLength.U
      state := initExponent
    }.otherwise {
      state := respond
    }
  }

  when(io.req.fire) {
    assert(io.req.bits.k > 0.U && io.req.bits.k <= maxK.U)
    assert(io.req.bits.k % fusionLength.U === 0.U,
      "K must contain complete numerical fusion blocks")
    requestK := io.req.bits.k
    accumulators := io.req.bits.c
    tag := io.req.bits.tag
    blockBase := 0.U
    state := initExponent
  }

  when(io.out.fire) {
    state := idle
  }
}
