package gemmini

import chisel3._
import chisel3.util._

/** Route full-width bank responses by accepted-request ownership, NOT fromDMA.
  * Both legacy execution and Tensor raw reads may set fromDMA=false. The owner
  * FIFO stays in order under response backpressure and never admits an untagged
  * read. This only arbitrates ports: a matrix-level fence is still required to
  * prevent DMA/legacy writers changing operands during a fused operation.
  */
class AccumulatorBankRouter[T <: Data, U <: Data](n: Int, rowType: Vec[Vec[T]], scaleType: U) extends Module {
  val io = IO(new Bundle {
    val legacy = Flipped(new AccumulatorBankClientIO(n, rowType, scaleType))
    val raw = Flipped(new AccumulatorBankClientIO(n, rowType, scaleType))
    val bank = new AccumulatorBankClientIO(n, rowType, scaleType)
  })
  private val owners = Module(new Queue(Bool(), 2))
  private val enabled = !reset.asBool
  private val reads = Module(new Arbiter(new AccumulatorReadReq(n, rowType.head.head, scaleType), 2))
  private val writes = Module(new Arbiter(new AccumulatorWriteReq(n, rowType), 2))
  reads.io.in(0) <> io.raw.read.req
  reads.io.in(1) <> io.legacy.read.req
  writes.io.in(0) <> io.raw.write
  writes.io.in(1) <> io.legacy.write
  io.bank.write.valid := writes.io.out.valid && enabled
  io.bank.write.bits := writes.io.out.bits
  writes.io.out.ready := io.bank.write.ready && enabled

  // The original bank rejects a simultaneous read and write to the same row.
  // Preserve that contract even when the two requests have different owners.
  private val conflict = writes.io.out.valid &&
    reads.io.out.bits.addr === writes.io.out.bits.addr
  io.bank.read.req.valid := reads.io.out.valid && owners.io.enq.ready && !conflict && enabled
  io.bank.read.req.bits := reads.io.out.bits
  reads.io.out.ready := io.bank.read.req.ready && owners.io.enq.ready && !conflict && enabled
  owners.io.enq.valid := io.bank.read.req.fire
  owners.io.enq.bits := reads.io.chosen === 0.U

  io.raw.read.resp.bits := io.bank.read.resp.bits
  io.legacy.read.resp.bits := io.bank.read.resp.bits
  io.raw.read.resp.valid := io.bank.read.resp.valid && owners.io.deq.valid && owners.io.deq.bits && enabled
  io.legacy.read.resp.valid := io.bank.read.resp.valid && owners.io.deq.valid && !owners.io.deq.bits && enabled
  io.bank.read.resp.ready := owners.io.deq.valid && enabled &&
    Mux(owners.io.deq.bits, io.raw.read.resp.ready, io.legacy.read.resp.ready)
  owners.io.deq.ready := io.bank.read.resp.fire
  when(enabled && io.bank.read.resp.valid) { assert(owners.io.deq.valid, "Accumulator response has no accepted owner") }
  when(io.raw.read.req.valid) {
    assert(io.raw.read.req.bits.full && !io.raw.read.req.bits.fromDMA)
  }
  when(io.raw.write.valid) { assert(!io.raw.write.bits.acc, "Raw Tensor D must not activate the adder") }
}
