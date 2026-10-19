package gemmini.tensorcore

import chisel3._
import chisel3.util._
import gemmini.ScratchpadReadIO

/** Storage geometry, independent of the arithmetic fusion length.
  * FP4 occupies one byte per element. Accumulator slots always contain raw
  * 32-bit words; FP16 C/D use their low 16 bits, NOT numeric FP32 conversion.
  */
case class TensorMemoryConfig(dim: Int = 2, spBanks: Int = 4, spDepth: Int = 256,
                              accBanks: Int = 2, accDepth: Int = 16,
                              writeLatency: Int = 2) {
  require(dim >= 2)
  require(Seq(spBanks, accBanks).forall(x => x >= 1 && isPow2(x)))
  require(Seq(spDepth, accDepth).forall(x => x >= 2 && isPow2(x)))
  require(writeLatency >= 1)
  val spRows = spBanks * spDepth
  val accRows = accBanks * accDepth
  val accWidth = dim * 32
  def slotWidth(config: BlackwellConfig): Int = config.format.width max 8
  def spWidth(config: BlackwellConfig): Int = dim * slotWidth(config)
}

/** All addresses are flat ROW indices, not byte addresses or LocalAddr metadata.
  * A is explicitly K-major (one A column per row); B is row-major. Scale rows
  * contain one byte per lane, and advance once per scaleBlockSize K slices.
  * C/D are dim consecutive accumulator rows. One full physical output tile.
  */
class TensorExecuteRequest(config: BlackwellConfig) extends Bundle {
  val a = UInt(32.W)
  val b = UInt(32.W)
  val c = UInt(32.W)
  val d = UInt(32.W)
  val alpha = if (config.grouped) Some(UInt(32.W)) else None
  val beta = if (config.grouped) Some(UInt(32.W)) else None
  val zeroC = Bool()
  val k = UInt(16.W)
  val tag = UInt(16.W)
}

class TensorAccumulatorWrite(mem: TensorMemoryConfig) extends Bundle {
  val addr = UInt(log2Ceil(mem.accDepth).W)
  val data = UInt(mem.accWidth.W)
  val mask = Vec(mem.accWidth / 8, Bool())
  // Must remain false: C has ALREADY participated in the fused operation.
  val accumulate = Bool()
}

/** Conservative memory-to-mesh execution adapter. One outstanding memory read,
  * one matrix in flight. No numeric conversions, activation, or store-side add.
  * Accumulator read responses must be the original full-width raw words.
  * Completion is held until consumed and follows the final write commit delay.
  * The owner must fence other local-memory writers throughout busy.
  */
class TensorExecute(val config: BlackwellConfig = BlackwellConfig(),
                    val mem: TensorMemoryConfig = TensorMemoryConfig()) extends Module {
  val io = IO(new Bundle {
    val cmd = Flipped(Decoupled(new TensorExecuteRequest(config)))
    val spad = Vec(mem.spBanks, new ScratchpadReadIO(mem.spDepth, mem.spWidth(config)))
    val accRead = Vec(mem.accBanks, new ScratchpadReadIO(mem.accDepth, mem.accWidth))
    val accWrite = Vec(mem.accBanks, Decoupled(new TensorAccumulatorWrite(mem)))
    val completed = Decoupled(UInt(16.W))
    val busy = Output(Bool())
  })
  private val (idle :: cReq :: cResp :: launch :: scaleAReq :: scaleAResp ::
    scaleBReq :: scaleBResp :: aReq :: aResp :: bReq :: bResp :: feed ::
    waitResult :: write :: commit :: respond :: Nil) = Enum(17)
  private val state = RegInit(idle)
  private val request = Reg(new TensorExecuteRequest(config))
  private val row = Reg(UInt(log2Ceil(mem.dim).W))
  private val slice = Reg(UInt(16.W))
  private val commitCount = Reg(UInt(log2Ceil(mem.writeLatency + 1).W))
  private val c = Reg(Vec(mem.dim, Vec(mem.dim, UInt(config.outputWidth.W))))
  private val d = Reg(Vec(mem.dim, Vec(mem.dim, UInt(config.outputWidth.W))))
  private val a = Reg(UInt(mem.spWidth(config).W))
  private val b = Reg(UInt(mem.spWidth(config).W))
  private val alpha = Reg(UInt(mem.spWidth(config).W))
  private val beta = Reg(UInt(mem.spWidth(config).W))
  private val mesh = Module(new TensorMesh(config, mem.dim, mem.dim))

  io.cmd.ready := state === idle
  io.busy := state =/= idle
  io.completed.valid := state === respond
  io.completed.bits := request.tag

  private val spAddr = WireDefault(request.a + slice)
  private val spRequest = state === aReq || state === bReq || state === scaleAReq || state === scaleBReq
  private val spResponse = state === aResp || state === bResp || state === scaleAResp || state === scaleBResp
  when(state === bReq || state === bResp) { spAddr := request.b + slice }
  if (config.grouped) {
    when(state === scaleAReq || state === scaleAResp) {
      spAddr := request.alpha.get + (slice / config.scaleBlockSize.U)
    }
    when(state === scaleBReq || state === scaleBResp) {
      spAddr := request.beta.get + (slice / config.scaleBlockSize.U)
    }
  }
  private val spBank = if (mem.spBanks == 1) 0.U else
    (spAddr >> log2Ceil(mem.spDepth))(log2Ceil(mem.spBanks)-1, 0)
  for (i <- 0 until mem.spBanks) {
    io.spad(i).req.valid := spRequest && spBank === i.U
    io.spad(i).req.bits.addr := spAddr
    io.spad(i).req.bits.fromDMA := false.B
    io.spad(i).resp.ready := spResponse && spBank === i.U
  }
  private val spReqFire = io.spad(spBank).req.fire
  private val spRespFire = io.spad(spBank).resp.fire
  private val spData = io.spad(spBank).resp.bits.data
  when(spRespFire) { assert(!io.spad(spBank).resp.bits.fromDMA) }

  private val accAddr = Mux(state === write, request.d, request.c) + row
  private val accBank = if (mem.accBanks == 1) 0.U else
    (accAddr >> log2Ceil(mem.accDepth))(log2Ceil(mem.accBanks)-1, 0)
  for (i <- 0 until mem.accBanks) {
    io.accRead(i).req.valid := state === cReq && accBank === i.U
    io.accRead(i).req.bits.addr := accAddr
    io.accRead(i).req.bits.fromDMA := false.B
    io.accRead(i).resp.ready := state === cResp && accBank === i.U
    io.accWrite(i).valid := state === write && accBank === i.U
    io.accWrite(i).bits.addr := accAddr
    io.accWrite(i).bits.data := VecInit(d(row).map(_.pad(32))).asUInt
    io.accWrite(i).bits.mask.foreach(_ := true.B)
    io.accWrite(i).bits.accumulate := false.B
  }

  mesh.io.req.valid := state === launch
  mesh.io.req.bits.c := c
  mesh.io.req.bits.k := request.k
  mesh.io.req.bits.tag := request.tag
  mesh.io.in.valid := state === feed
  for (lane <- 0 until mem.dim) {
    val lo = lane * mem.slotWidth(config)
    mesh.io.in.bits.a(lane) := a(lo + config.format.width - 1, lo)
    mesh.io.in.bits.b(lane) := b(lo + config.format.width - 1, lo)
    if (config.grouped) {
      mesh.io.in.bits.alpha.get(lane) := alpha(lo + 7, lo)
      mesh.io.in.bits.beta.get(lane) := beta(lo + 7, lo)
    }
  }
  mesh.io.out.ready := state === waitResult

  when(io.cmd.fire) {
    def inRange(base: UInt, count: UInt, capacity: Int): Unit = {
      assert((base +& count) <= capacity.U, "Tensor local-memory range overflow")
    }
    assert(io.cmd.bits.k > 0.U)
    inRange(io.cmd.bits.a, io.cmd.bits.k, mem.spRows)
    inRange(io.cmd.bits.b, io.cmd.bits.k, mem.spRows)
    inRange(io.cmd.bits.d, mem.dim.U, mem.accRows)
    when(!io.cmd.bits.zeroC) { inRange(io.cmd.bits.c, mem.dim.U, mem.accRows) }
    if (config.grouped) {
      assert(io.cmd.bits.k(5, 0) === 0.U, "FP4 K must be a positive multiple of 64")
      val scaleRows = io.cmd.bits.k / config.scaleBlockSize.U
      inRange(io.cmd.bits.alpha.get, scaleRows, mem.spRows)
      inRange(io.cmd.bits.beta.get, scaleRows, mem.spRows)
    }
    request := io.cmd.bits
    row := 0.U
    slice := 0.U
    c := 0.U.asTypeOf(c)
    state := Mux(io.cmd.bits.zeroC, launch, cReq)
  }
  when(state === cReq && io.accRead(accBank).req.fire) { state := cResp }
  when(state === cResp && io.accRead(accBank).resp.fire) {
    assert(!io.accRead(accBank).resp.bits.fromDMA)
    val raw = io.accRead(accBank).resp.bits.data.asTypeOf(Vec(mem.dim, UInt(32.W)))
    for (lane <- 0 until mem.dim) { c(row)(lane) := raw(lane)(config.outputWidth-1, 0) }
    when(row === (mem.dim-1).U) { state := launch }
      .otherwise { row := row + 1.U; state := cReq }
  }
  when(mesh.io.req.fire) { state := (if (config.grouped) scaleAReq else aReq) }
  when(state === scaleAReq && spReqFire) { state := scaleAResp }
  when(state === scaleAResp && spRespFire) { alpha := spData; state := scaleBReq }
  when(state === scaleBReq && spReqFire) { state := scaleBResp }
  when(state === scaleBResp && spRespFire) { beta := spData; state := aReq }
  when(state === aReq && spReqFire) { state := aResp }
  when(state === aResp && spRespFire) { a := spData; state := bReq }
  when(state === bReq && spReqFire) { state := bResp }
  when(state === bResp && spRespFire) { b := spData; state := feed }
  when(mesh.io.in.fire) {
    slice := slice + 1.U
    when(slice === request.k - 1.U) { state := waitResult }
      .otherwise {
        if (config.grouped) {
          state := Mux((slice + 1.U) % config.scaleBlockSize.U === 0.U, scaleAReq, aReq)
        } else { state := aReq }
      }
  }
  when(mesh.io.out.fire) {
    assert(mesh.io.out.bits.tag === request.tag)
    d := mesh.io.out.bits.d
    row := 0.U
    state := write
  }
  when(state === write && io.accWrite(accBank).fire) {
    when(row === (mem.dim-1).U) { commitCount := 0.U; state := commit }
      .otherwise { row := row + 1.U }
  }
  when(state === commit) {
    commitCount := commitCount + 1.U
    when(commitCount === (mem.writeLatency-1).U) { state := respond }
  }
  when(io.completed.fire) { state := idle }
}
