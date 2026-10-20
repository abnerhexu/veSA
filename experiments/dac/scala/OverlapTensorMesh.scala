package gemmini.dac

import chisel3._
import chisel3.util._
import gemmini.tensorcore._

class OverlapRequest(config: TensorConfig) extends Bundle {
  val c = UInt(config.outputWidth.W)
  val k = UInt(16.W)
}

/** Two raw-operand banks decouple collection from fused-block replay.
  * C is decoded only after the previous block's output conversion, so the
  * speculative product maximum never substitutes for the final common exponent.
  */
class OverlapPE(config: TensorConfig, reuseSlots: Boolean = false) extends Module {
  require(!config.grouped, "This overlap experiment implements T-FDPA")
  private val size = config.fusionLength
  private val indexWidth = log2Ceil(size + 1)
  val io = IO(new Bundle {
    val req = Flipped(Decoupled(new OverlapRequest(config)))
    val in = Flipped(Decoupled(new TensorOperands(config)))
    val out = Decoupled(UInt(config.outputWidth.W))
  })
  private val busy = RegInit(false.B)
  private val rawA = Reg(Vec(if (reuseSlots) 1 else 2, Vec(size, UInt(config.format.width.W))))
  private val rawB = Reg(Vec(if (reuseSlots) 1 else 2, Vec(size, UInt(config.format.width.W))))
  private val full = RegInit(VecInit(Seq.fill(2)(false.B)))
  private val lengths = Reg(Vec(2, UInt(indexWidth.W)))
  private val maxima = Reg(Vec(2, SInt(12.W)))
  private val hasNaN = Reg(Vec(2, Bool()))
  private val hasPositiveInfinity = Reg(Vec(2, Bool()))
  private val hasNegativeInfinity = Reg(Vec(2, Bool()))
  private val writeBank = Reg(Bool())
  private val readBank = Reg(Bool())
  private val writeIndex = Reg(UInt(indexWidth.W))
  private val inputLeft = Reg(UInt(16.W))
  private val outputLeft = Reg(UInt(16.W))
  private val cBits = Reg(UInt(config.outputWidth.W))

  private val waiting :: replay :: finish :: respond :: Nil = Enum(4)
  private val state = RegInit(waiting)
  private val readIndex = Reg(UInt(indexWidth.W))
  private val length = Reg(UInt(indexWidth.W))
  private val maximum = Reg(SInt(12.W))
  private val sum = Reg(SInt(config.sumWidth.W))
  private val nan = Reg(Bool())
  private val positiveInfinity = Reg(Bool())
  private val negativeInfinity = Reg(Bool())

  io.req.ready := !busy
  // In reuse mode, the next block may overwrite only operands already read.
  // Reg reads use the old value on a simultaneous same-slot read/write edge.
  private val canWrite = if (reuseSlots)
    writeBank === readBank || state === finish || (state === replay && writeIndex <= readIndex)
  else true.B
  private val storeBank = if (reuseSlots) 0.U else writeBank.asUInt
  private val loadBank = if (reuseSlots) 0.U else readBank.asUInt
  // Completed lanes must not block draining later lanes in the mesh wavefront.
  io.in.ready := busy && (inputLeft === 0.U || (!full(writeBank) && canWrite))
  io.out.valid := busy && state === respond
  io.out.bits := cBits

  when(io.req.fire) {
    busy := true.B
    full.foreach(_ := false.B)
    writeBank := false.B
    readBank := false.B
    writeIndex := 0.U
    inputLeft := io.req.bits.k
    outputLeft := io.req.bits.k
    cBits := io.req.bits.c
    state := waiting
  }

  private val incoming = TensorArithmetic.product(io.in.bits.a, io.in.bits.b,
    config.format, config.zeroExponent)
  when(io.in.fire) {
    assert(inputLeft =/= 0.U)
    rawA(storeBank)(writeIndex) := io.in.bits.a
    rawB(storeBank)(writeIndex) := io.in.bits.b
    maxima(writeBank) := Mux(writeIndex === 0.U || incoming.exponent > maxima(writeBank),
      incoming.exponent, maxima(writeBank))
    hasNaN(writeBank) := incoming.nan || (writeIndex =/= 0.U && hasNaN(writeBank))
    hasPositiveInfinity(writeBank) := (incoming.infinity && !incoming.sign) ||
      (writeIndex =/= 0.U && hasPositiveInfinity(writeBank))
    hasNegativeInfinity(writeBank) := (incoming.infinity && incoming.sign) ||
      (writeIndex =/= 0.U && hasNegativeInfinity(writeBank))
    inputLeft := inputLeft - 1.U
    when(writeIndex === (size - 1).U || inputLeft === 1.U) {
      lengths(writeBank) := writeIndex + 1.U
      full(writeBank) := true.B
      writeBank := !writeBank
      writeIndex := 0.U
    }.otherwise { writeIndex := writeIndex + 1.U }
  }

  private val c = TensorArithmetic.decode(cBits, if (config.halfOutput) 5 else 8,
    if (config.halfOutput) 10 else 23)
  private val cExponent = Mux(c.magnitude === 0.U, config.zeroExponent.S, c.exponent)
  when(busy && state === waiting && full(readBank)) {
    maximum := Mux(cExponent > maxima(readBank), cExponent, maxima(readBank))
    nan := hasNaN(readBank) || c.nan
    positiveInfinity := hasPositiveInfinity(readBank) || (c.infinity && !c.sign)
    negativeInfinity := hasNegativeInfinity(readBank) || (c.infinity && c.sign)
    length := lengths(readBank)
    sum := 0.S
    readIndex := 0.U
    state := replay
  }
  private val product = TensorArithmetic.product(
    rawA(loadBank)(readIndex(log2Ceil(size)-1, 0)),
    rawB(loadBank)(readIndex(log2Ceil(size)-1, 0)), config.format, config.zeroExponent)
  private val cTerm = Wire(new TensorTerm)
  cTerm := c
  cTerm.exponent := cExponent
  private val term = Mux(readIndex === length, cTerm, product)
  private val delta = (maximum - term.exponent).asUInt
  private val aligned = if (config.fractionalBits >= 23)
    (term.magnitude << (config.fractionalBits - 23)) >> delta
  else term.magnitude >> (delta +& (23 - config.fractionalBits).U)
  private val signedTerm = Wire(SInt(config.sumWidth.W))
  signedTerm := Mux(term.sign, -aligned.zext, aligned.zext)
  when(busy && state === replay) {
    sum := sum + signedTerm
    when(readIndex === length) { state := finish }
      .otherwise { readIndex := readIndex + 1.U }
  }
  when(busy && state === finish) {
    val canonicalNaN = ((BigInt(1) << (config.outputWidth - 1)) - 1).U
    val infinity = (if (config.halfOutput) BigInt(0x7c00) else BigInt(0x7f800000)).U(config.outputWidth.W)
    val sign = (BigInt(1) << (config.outputWidth - 1)).U(config.outputWidth.W)
    cBits := Mux(nan || (positiveInfinity && negativeInfinity), canonicalNaN,
      Mux(positiveInfinity || negativeInfinity, infinity | Mux(negativeInfinity, sign, 0.U),
        TensorArithmetic.pack(sum, maximum, config.fractionalBits, config.halfOutput,
          config.outputFractionBits)))
    full(readBank) := false.B
    readBank := !readBank
    outputLeft := outputLeft - length
    state := Mux(outputLeft === length, respond, waiting)
  }
  when(io.out.fire) { busy := false.B }
}

/** Continuous wavefront; globally freeze forwarding when a PE's two banks fill.
  * Collection and replay are independent, including during input bubbles.
  */
class OverlapTensorMesh(config: TensorConfig, rows: Int = 2, columns: Int = 2,
                        reuseSlots: Boolean = false) extends Module {
  val io = IO(new Bundle {
    val req = Flipped(Decoupled(new TensorMatrixRequest(config, rows, columns)))
    val in = Flipped(Decoupled(new TensorMatrixOperands(config, rows, columns)))
    val out = Decoupled(new TensorMatrixResult(config, rows, columns))
  })
  private val idle :: active :: respond :: Nil = Enum(3)
  private val state = RegInit(idle)
  private val k = Reg(UInt(16.W))
  private val received = Reg(UInt(16.W))
  private val tag = Reg(UInt(16.W))
  private val result = Reg(Vec(rows, Vec(columns, UInt(config.outputWidth.W))))
  private val drainCycles = rows + columns - 2
  private val drainCount = Reg(UInt(log2Ceil((drainCycles + 1) max 2).W))
  private val pes = Seq.fill(rows, columns)(Module(new OverlapPE(config, reuseSlots)))
  private val allReady = pes.flatten.map(_.io.in.ready).reduce(_ && _)
  private val allValid = pes.flatten.map(_.io.out.valid).reduce(_ && _)
  private val draining = state === active && received === k && drainCount < drainCycles.U
  private val advance = io.in.fire || (draining && allReady)

  io.req.ready := state === idle
  io.in.ready := state === active && received < k && allReady
  io.out.valid := state === respond
  io.out.bits.d := result
  io.out.bits.tag := tag

  private def delayed(bits: UInt, valid: Bool, cycles: Int): (UInt, Bool) = {
    var data = bits
    var present = valid
    for (_ <- 0 until cycles) {
      val nextData = RegEnable(data, advance)
      val nextValid = RegInit(false.B)
      when(io.req.fire) { nextValid := false.B }
        .elsewhen(advance) { nextValid := present }
      data = nextData
      present = nextValid
    }
    (data, present)
  }
  private val a = Array.ofDim[(UInt, Bool)](rows, columns)
  private val b = Array.ofDim[(UInt, Bool)](rows, columns)
  for (r <- 0 until rows; col <- 0 until columns) {
    a(r)(col) = if (col == 0) delayed(io.in.bits.a(r), io.in.fire, r)
      else delayed(a(r)(col-1)._1, a(r)(col-1)._2, 1)
    b(r)(col) = if (r == 0) delayed(io.in.bits.b(col), io.in.fire, col)
      else delayed(b(r-1)(col)._1, b(r-1)(col)._2, 1)
    val pe = pes(r)(col)
    pe.io.req.valid := io.req.fire
    pe.io.req.bits.c := io.req.bits.c(r)(col)
    pe.io.req.bits.k := io.req.bits.k
    pe.io.in.valid := advance && a(r)(col)._2 && b(r)(col)._2
    pe.io.in.bits.a := a(r)(col)._1
    pe.io.in.bits.b := b(r)(col)._1
    pe.io.out.ready := state === active && allValid
  }
  when(io.req.fire) {
    k := io.req.bits.k
    received := 0.U
    drainCount := 0.U
    tag := io.req.bits.tag
    state := active
  }
  when(io.in.fire) { received := received + 1.U }
  when(draining && allReady) { drainCount := drainCount + 1.U }
  when(state === active && allValid) {
    for (r <- 0 until rows; col <- 0 until columns) result(r)(col) := pes(r)(col).io.out.bits
    state := respond
  }
  when(io.out.fire) { state := idle }
}
