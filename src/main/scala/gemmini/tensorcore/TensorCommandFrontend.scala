package gemmini.tensorcore

import chisel3._
import chisel3.util._

/** Private Gemmini commands, not NVIDIA instructions or a PTX ABI. */
object TensorCommandISA {
  val AddrAB = 32
  val AddrCD = 33
  val Scales = 34
  val Run = 35
  val Success = 0
  val MissingDescriptor = 1
  val BadEncoding = 2
  val BadRange = 3
  def matches(funct: UInt): Bool = funct >= AddrAB.U && funct <= Run.U
}

class TensorCommand(xLen: Int = 64) extends Bundle {
  val funct = UInt(7.W)
  val rs1 = UInt(xLen.W)
  val rs2 = UInt(xLen.W)
  val rd = UInt(5.W)
  val xd = Bool()
}

class TensorCommandResponse(xLen: Int = 64) extends Bundle {
  val rd = UInt(5.W)
  // [31:16] tag, [15:8] reserved zero, [7:0] status; higher bits zero.
  val data = UInt(xLen.W)
}

/** Preserve the complete caller payload (including RoCC status/ROB metadata).
  * No younger command, Tensor or legacy, may pass an active fence. The caller
  * drains already-dispatched work independently of this blocked input head.
  */
class TensorCommandGate[T <: Data](commandType: T, funct: T => UInt) extends Module {
  val io = IO(new Bundle {
    val in = Flipped(Decoupled(commandType.cloneType))
    val tensor = Decoupled(commandType.cloneType)
    val legacy = Decoupled(commandType.cloneType)
    val hold = Input(Bool())
  })
  private val isTensor = TensorCommandISA.matches(funct(io.in.bits))
  private val enabled = !io.hold && !reset.asBool
  io.tensor.bits := io.in.bits
  io.legacy.bits := io.in.bits
  io.tensor.valid := io.in.valid && isTensor && enabled
  io.legacy.valid := io.in.valid && !isTensor && enabled
  io.in.ready := enabled && Mux(isTensor, io.tensor.ready, io.legacy.ready)
}

/** Capture a command BEFORE the legacy unrollers. Hold younger commands until
  * all older work drains, then retain exclusive memory ownership through write
  * visibility AND response acceptance. The caller supplies real memory-idle,
  * including queued writes, rather than a guessed delay or global busy (which
  * includes this very command). Descriptor/profile changes never race a run.
  */
class TensorCommandFrontend(config: BlackwellConfig = BlackwellConfig(),
                            mem: TensorMemoryConfig = TensorMemoryConfig(),
                            xLen: Int = 64) extends Module {
  require(xLen >= 64, "Tensor commands require RV64")
  val io = IO(new Bundle {
    val cmd = Flipped(Decoupled(new TensorCommand(xLen)))
    val response = Decoupled(new TensorCommandResponse(xLen))
    val execute = Decoupled(new TensorExecuteRequest(config))
    val completed = Flipped(Decoupled(UInt(16.W)))
    val legacyIdle = Input(Bool())
    val memoryIdle = Input(Bool())
    val busy = Output(Bool())
    val exclusive = Output(Bool())
  })
  import TensorCommandISA._
  private val idle :: drain :: decode :: launch :: running :: commit :: respond :: Nil = Enum(7)
  private val state = RegInit(idle)
  private val command = Reg(new TensorCommand(xLen))
  private val descriptor = RegInit(0.U.asTypeOf(new TensorExecuteRequest(config)))
  private val haveAB = RegInit(false.B)
  private val haveCD = RegInit(false.B)
  private val haveScales = RegInit(false.B)
  private val status = RegInit(0.U(8.W))
  private val tag = RegInit(0.U(16.W))
  private val enabled = !reset.asBool
  io.cmd.ready := state === idle && enabled
  io.busy := state =/= idle
  io.exclusive := state =/= idle && state =/= drain && enabled
  io.execute.valid := state === launch && enabled
  io.execute.bits := descriptor
  io.completed.ready := state === running && enabled
  io.response.valid := state === respond && command.xd && enabled
  io.response.bits.rd := command.rd
  io.response.bits.data := Cat(tag, 0.U(8.W), status)

  when(io.cmd.fire) {
    command := io.cmd.bits
    status := Success.U
    tag := Mux(io.cmd.bits.funct === Run.U, io.cmd.bits.rs1(31, 16), 0.U)
    state := drain
  }
  when(state === drain && io.legacyIdle && io.memoryIdle) { state := decode }
  when(state === decode) {
    state := respond
    val addressEncoding = command.rs1(xLen-1, 32) === 0.U && command.rs2(xLen-1, 32) === 0.U
    switch(command.funct) {
      is(AddrAB.U) {
        when(addressEncoding) {
          descriptor.a := command.rs1(31, 0); descriptor.b := command.rs2(31, 0); haveAB := true.B
        }.otherwise { status := BadEncoding.U }
      }
      is(AddrCD.U) {
        when(addressEncoding) {
          descriptor.c := command.rs1(31, 0); descriptor.d := command.rs2(31, 0); haveCD := true.B
        }.otherwise { status := BadEncoding.U }
      }
      is(Scales.U) {
        if (config.grouped) {
          when(addressEncoding) {
            descriptor.alpha.get := command.rs1(31, 0)
            descriptor.beta.get := command.rs2(31, 0)
            haveScales := true.B
          }.otherwise { status := BadEncoding.U }
        } else { status := BadEncoding.U }
      }
      is(Run.U) {
        val k = command.rs1(15, 0)
        val zeroC = command.rs1(32)
        def fits(base: UInt, count: UInt, capacity: Int): Bool = (base +& count) <= capacity.U
        val ranges = fits(descriptor.a, k, mem.spRows) && fits(descriptor.b, k, mem.spRows) &&
          fits(descriptor.d, mem.dim.U, mem.accRows) && (zeroC || fits(descriptor.c, mem.dim.U, mem.accRows))
        val scaleRanges = if (config.grouped) {
          val n = k / config.scaleBlockSize.U
          fits(descriptor.alpha.get, n, mem.spRows) && fits(descriptor.beta.get, n, mem.spRows)
        } else true.B
        val encoding = command.rs1(xLen-1, 33) === 0.U && command.rs2 === 0.U &&
          k =/= 0.U && (if (config.grouped) k(5, 0) === 0.U else true.B)
        when(!encoding) { status := BadEncoding.U }
          .elsewhen(!haveAB || !haveCD || (if (config.grouped) !haveScales else false.B)) {
            status := MissingDescriptor.U
          }.elsewhen(!ranges || !scaleRanges) { status := BadRange.U }
          .otherwise {
            descriptor.k := k; descriptor.tag := command.rs1(31, 16); descriptor.zeroC := zeroC
            state := launch
          }
      }
    }
    when(!matches(command.funct)) { status := BadEncoding.U }
  }
  when(io.execute.fire) { state := running }
  when(io.completed.fire) {
    assert(io.completed.bits === descriptor.tag, "Tensor completion tag changed")
    state := commit
  }
  // The engine's local delay is not a substitute for actual bank drain.
  when(state === commit && io.memoryIdle) { state := respond }
  when(state === respond && (!command.xd || io.response.fire)) { state := idle }
  when(io.exclusive) { assert(io.legacyIdle, "Legacy work entered the Tensor memory fence") }
}
