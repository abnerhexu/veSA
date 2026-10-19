package gemmini.tensorcore

import chisel3._
import chisel3.util._
import gemmini.{AccumulatorBankClientIO, ScratchpadReadIO}

/** Raw-row TensorExecute ports to the ORIGINAL accumulator bank protocol.
  * Connect before normalizer/scaler/conversion, not to the narrowed ex response.
  * The surrounding owner must fence legacy/DMA access and account for the
  * bank's actual write visibility delay in TensorMemoryConfig.writeLatency.
  */
class TensorAccumulatorBridge[T <: Data, U <: Data](mem: TensorMemoryConfig,
                                                   rowType: Vec[Vec[T]], scaleType: U) extends Module {
  require(rowType.getWidth == mem.accWidth)
  require(rowType.flatten.forall(_.getWidth == 32), "C/D storage slots must contain raw 32-bit words")
  val io = IO(new Bundle {
    val read = Flipped(new ScratchpadReadIO(mem.accDepth, mem.accWidth))
    val write = Flipped(Decoupled(new TensorAccumulatorWrite(mem)))
    val bank = new AccumulatorBankClientIO(mem.accDepth, rowType, scaleType)
  })
  io.bank.read.req.valid := io.read.req.valid
  io.read.req.ready := io.bank.read.req.ready
  io.bank.read.req.bits := 0.U.asTypeOf(io.bank.read.req.bits)
  io.bank.read.req.bits.addr := io.read.req.bits.addr
  io.bank.read.req.bits.full := true.B
  io.bank.read.req.bits.fromDMA := false.B
  io.read.resp.valid := io.bank.read.resp.valid
  io.read.resp.bits.data := io.bank.read.resp.bits.data.asUInt
  io.read.resp.bits.fromDMA := io.bank.read.resp.bits.fromDMA
  io.bank.read.resp.ready := io.read.resp.ready
  when(io.read.req.fire) { assert(!io.read.req.bits.fromDMA) }
  when(io.bank.read.resp.valid) { assert(!io.bank.read.resp.bits.fromDMA) }

  io.bank.write.valid := io.write.valid
  io.write.ready := io.bank.write.ready
  io.bank.write.bits.addr := io.write.bits.addr
  io.bank.write.bits.data := io.write.bits.data.asTypeOf(rowType)
  io.bank.write.bits.mask := io.write.bits.mask
  io.bank.write.bits.acc := false.B
  when(io.write.valid) { assert(!io.write.bits.accumulate, "Tensor D must overwrite, not accumulate") }
}
