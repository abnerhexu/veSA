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

/** Output-stationary systolic array with adjacent A/B forwarding.
  * Physical dimensions do not set the fusion boundary. The controller streams
  * up to fusionLength slices, drains the wavefront, rounds each PE exactly once,
  * and uses those rounded outputs as C for the next block. One request in flight.
  */
class TensorMesh(val config: TensorConfig = BlackwellConfig(),
                 val rows: Int = 2, val columns: Int = 2) extends Module {
  require(rows > 0 && columns > 0)
  val io = IO(new Bundle {
    val req = Flipped(Decoupled(new TensorMatrixRequest(config, rows, columns)))
    val in = Flipped(Decoupled(new TensorMatrixOperands(config, rows, columns)))
    val out = Decoupled(new TensorMatrixResult(config, rows, columns))
  })
  private val idle :: launch :: feed :: drain :: waitResults :: respond :: Nil = Enum(6)
  private val state = RegInit(idle)
  private val remaining = Reg(UInt(16.W))
  private val blockLength = Mux(remaining > config.fusionLength.U,
    config.fusionLength.U, remaining)
  private val sent = Reg(UInt(log2Ceil(config.fusionLength + 1).W))
  private val drainCycles = rows + columns - 2
  private val drainCount = Reg(UInt(log2Ceil((drainCycles + 1) max 2).W))
  private val accumulators = Reg(Vec(rows, Vec(columns, UInt(config.outputWidth.W))))
  private val tag = Reg(UInt(16.W))
  private val pes = Seq.fill(rows, columns)(Module(new TensorPE(config)))
  private val allReady = pes.flatten.map(_.io.req.ready).reduce(_ && _)
  private val allValid = pes.flatten.map(_.io.out.valid).reduce(_ && _)
  private val advance = io.in.fire || state === drain

  io.req.ready := state === idle
  io.in.ready := state === feed
  io.out.valid := state === respond
  io.out.bits.d := accumulators
  io.out.bits.tag := tag

  // Clear validity between fused blocks, and freeze forwarding on input bubbles.
  // PE arithmetic replay advances independently; consuming pairs is gated by advance.
  private def delayed(bits: UInt, valid: Bool, cycles: Int): (UInt, Bool) = {
    var data = bits
    var present = valid
    for (_ <- 0 until cycles) {
      val nextData = RegEnable(data, advance)
      val nextValid = RegInit(false.B)
      when(state === launch) { nextValid := false.B }
        .elsewhen(advance) { nextValid := present }
      data = nextData
      present = nextValid
    }
    (data, present)
  }

  private val a = Array.ofDim[(UInt, Bool)](rows, columns)
  private val b = Array.ofDim[(UInt, Bool)](rows, columns)
  for (r <- 0 until rows; c <- 0 until columns) {
    val inputA = if (config.grouped) Cat(io.in.bits.alpha.get(r), io.in.bits.a(r)) else io.in.bits.a(r)
    val inputB = if (config.grouped) Cat(io.in.bits.beta.get(c), io.in.bits.b(c)) else io.in.bits.b(c)
    a(r)(c) = if (c == 0) delayed(inputA, io.in.fire, r)
      else delayed(a(r)(c - 1)._1, a(r)(c - 1)._2, 1)
    b(r)(c) = if (r == 0) delayed(inputB, io.in.fire, c)
      else delayed(b(r - 1)(c)._1, b(r - 1)(c)._2, 1)
    val pe = pes(r)(c)
    pe.io.req.valid := state === launch && allReady
    pe.io.req.bits.c := accumulators(r)(c)
    pe.io.req.bits.length := blockLength
    pe.io.in.valid := advance && a(r)(c)._2 && b(r)(c)._2
    pe.io.in.bits.a := a(r)(c)._1(config.format.width - 1, 0)
    pe.io.in.bits.b := b(r)(c)._1(config.format.width - 1, 0)
    if (config.grouped) {
      pe.io.in.bits.alpha.get := a(r)(c)._1 >> config.format.width
      pe.io.in.bits.beta.get := b(r)(c)._1 >> config.format.width
    }
    pe.io.out.ready := state === waitResults && allValid
    when(pe.io.in.valid) { assert(pe.io.in.ready, "Systolic wave exceeded PE block length") }
  }

  when(io.req.fire) {
    assert(io.req.bits.k > 0.U)
    if (config.grouped) { assert(io.req.bits.k(5, 0) === 0.U, "FP4 K must be a multiple of 64") }
    remaining := io.req.bits.k
    accumulators := io.req.bits.c
    tag := io.req.bits.tag
    state := launch
  }
  when(state === launch && allReady) {
    sent := 0.U
    state := feed
  }
  when(io.in.fire) {
    sent := sent + 1.U
    when(sent === blockLength - 1.U) {
      drainCount := 0.U
      state := (if (drainCycles == 0) waitResults else drain)
    }
  }
  when(state === drain) {
    drainCount := drainCount + 1.U
    when(drainCount === (drainCycles - 1 max 0).U) { state := waitResults }
  }
  when(state === waitResults && allValid) {
    for (r <- 0 until rows; c <- 0 until columns) {
      accumulators(r)(c) := pes(r)(c).io.out.bits
    }
    remaining := remaining - blockLength
    state := Mux(remaining === blockLength, respond, launch)
  }
  when(io.out.fire) { state := idle }
}
