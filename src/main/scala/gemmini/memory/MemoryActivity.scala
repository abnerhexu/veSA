package gemmini

import chisel3._

/** Track a request through its FINAL local response, including hidden scaler,
  * repeater and response queues. A start in the current cycle is not idle.
  * Reset must be shared with the producer and consumer; it is not a rollback.
  */
class MemoryActivity(countBits: Int = 16) extends Module {
  require(countBits >= 1)
  val io = IO(new Bundle {
    val start = Input(Bool())
    val finish = Input(Bool())
    val idle = Output(Bool())
  })
  private val count = RegInit(0.U(countBits.W))
  io.idle := count === 0.U && !io.start
  when(io.start && !io.finish) {
    assert(!count.andR, "Memory activity counter overflow")
    count := count + 1.U
  }
  when(io.finish && !io.start) {
    assert(count =/= 0.U, "Memory response without an outstanding request")
    count := count - 1.U
  }
}
