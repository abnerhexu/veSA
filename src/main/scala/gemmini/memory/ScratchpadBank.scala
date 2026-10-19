package gemmini

import chisel3._
import chisel3.util._

class ScratchpadReadReq(val n: Int) extends Bundle {
  val addr = UInt(log2Ceil(n).W)
  val fromDMA = Bool()
}

class ScratchpadReadResp(val w: Int) extends Bundle {
  val data = UInt(w.W)
  val fromDMA = Bool()
}

class ScratchpadReadIO(val n: Int, val w: Int) extends Bundle {
  val req = Decoupled(new ScratchpadReadReq(n))
  val resp = Flipped(Decoupled(new ScratchpadReadResp(w)))
}

class ScratchpadWriteIO(val n: Int, val w: Int, val mask_len: Int) extends Bundle {
  val valid = Output(Bool())
  val ready = Input(Bool())
  val addr = Output(UInt(log2Ceil(n).W))
  val mask = Output(Vec(mask_len, Bool()))
  val data = Output(UInt(w.W))
  def fire = valid && ready
}

class ScratchpadBank(n: Int, w: Int, aligned_to: Int, single_ported: Boolean, use_shared_ext_mem: Boolean, is_dummy: Boolean) extends Module {
  // This is essentially a pipelined SRAM with the ability to stall pipeline stages

  require(w % aligned_to == 0 || w < aligned_to)
  val mask_len = (w / (aligned_to * 8)) max 1 // How many mask bits are there?
  val mask_elem = UInt((w min (aligned_to * 8)).W) // What datatype does each mask bit correspond to?

  val io = IO(new Bundle {
    val read = Flipped(new ScratchpadReadIO(n, w))
    val write = Flipped(new ScratchpadWriteIO(n, w, mask_len))
    val ext_mem = if (use_shared_ext_mem) Some(new ExtMemIO) else None
  })

  val ren = io.read.req.fire
  val fromDMA = io.read.req.bits.fromDMA

  // Make a queue which buffers the result of an SRAM read if it can't immediately be consumed
  val q = Module(new Queue(new ScratchpadReadResp(w), 1, true, true))
  val q_will_be_empty = (q.io.count +& q.io.enq.fire) - q.io.deq.fire === 0.U
  // When the scratchpad is single-ported, the writes take precedence
  val singleport_busy_with_write = single_ported.B && io.write.fire

  if (is_dummy) {
    q.io.enq.valid := RegNext(ren)
    q.io.enq.bits.data := 0.U
    q.io.enq.bits.fromDMA := RegNext(fromDMA)
    io.read.req.ready := q_will_be_empty && !singleport_busy_with_write
  } else if (use_shared_ext_mem) { // use ready-valid interface
    val ext_mem = io.ext_mem.get

    /* READ */
    ext_mem.read_req.valid := q_will_be_empty && io.read.req.valid
    ext_mem.read_req.bits := io.read.req.bits.addr
    io.read.req.ready := q_will_be_empty && ext_mem.read_req.ready

    // TODO (richard): the number of entries here should be configurable
    val dma_q = Module(new Queue(Bool(), 4, false, true))
    dma_q.io.enq.valid := ren
    dma_q.io.enq.bits := fromDMA
    dma_q.io.deq.ready := q.io.enq.fire
    assert(dma_q.io.enq.fire === ren, "DMA queue does not have enough entries") // TODO (richard): do backpressure
    assert(dma_q.io.deq.fire === q.io.enq.fire, "fromDMA should be dequeued only when read resp comes back")

    q.io.enq.valid := ext_mem.read_resp.valid
    q.io.enq.bits.data := ext_mem.read_resp.bits
    q.io.enq.bits.fromDMA := dma_q.io.deq.bits
    ext_mem.read_resp.ready := q.io.enq.ready

    /* WRITE */
    val wq = Module(new Queue(ext_mem.write_req.bits.cloneType, 4, pipe=true, flow=true))
    ext_mem.write_req <> wq.io.deq

    wq.io.enq.valid := io.write.valid
    io.write.ready := wq.io.enq.ready
    wq.io.enq.bits.addr := io.write.addr
    wq.io.enq.bits.data := io.write.data
    if (aligned_to >= w) {
      wq.io.enq.bits.mask := VecInit((~(0.U(mask_len.W))).asBools).asUInt
    } else {
      wq.io.enq.bits.mask := io.write.mask.asUInt
    }
    // assert(wq.io.enq.ready || (!io.write.en), "TODO (richard): fix this if triggered")
  } else { // use valid only interface
    val mem = SyncReadMem(n, Vec(mask_len, mask_elem))

    val raddr = io.read.req.bits.addr
    val rdata = if (single_ported) {
      assert(!(ren && io.write.fire))
      mem.read(raddr, ren && !io.write.fire).asUInt
    } else {
      mem.read(raddr, ren).asUInt
    }
    q.io.enq.valid := RegNext(ren)
    q.io.enq.bits.data := rdata
    q.io.enq.bits.fromDMA := RegNext(fromDMA)

    io.read.req.ready := q_will_be_empty && !singleport_busy_with_write

    io.write.ready := true.B
    when(io.write.fire) {
      if (aligned_to >= w)
        mem.write(io.write.addr, io.write.data.asTypeOf(Vec(mask_len, mask_elem)), VecInit((~(0.U(mask_len.W))).asBools))
      else
        mem.write(io.write.addr, io.write.data.asTypeOf(Vec(mask_len, mask_elem)), io.write.mask)
    }
  }

  io.read.resp <> q.io.deq
}
