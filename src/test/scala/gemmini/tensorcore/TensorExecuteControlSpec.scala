package gemmini.tensorcore

import chisel3._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec

class TensorExecuteControlSpec extends AnyFlatSpec with ChiselScalatestTester {
  "Tensor execution control" should "support one bank and reset an aborted command before writing" in {
    val config = BlackwellConfig()
    val mem = TensorMemoryConfig(spBanks = 1, spDepth = 8, accBanks = 1, accDepth = 4)
    test(new TensorMemoryHarness(config, mem)).withAnnotations(TensorTestBackend.annotations) { dut =>
      dut.io.cmd.valid.poke(false.B)
      dut.io.completed.ready.poke(false.B)
      dut.io.stress.poke(true.B)
      dut.io.spWrite(0).valid.poke(false.B)
      dut.io.spWrite(0).mask.foreach(_.poke(true.B))
      dut.io.accWrite(0).valid.poke(false.B)
      dut.io.accWrite(0).mask.foreach(_.poke(true.B))
      dut.io.accRead(0).req.valid.poke(false.B)
      dut.io.accRead(0).resp.ready.poke(false.B)
      dut.clock.step()
      for ((addr, data) <- Seq(0 -> 0x3838, 1 -> 0x3838)) {
        dut.io.spWrite(0).addr.poke(addr.U)
        dut.io.spWrite(0).data.poke(data.U)
        dut.io.spWrite(0).valid.poke(true.B)
        dut.clock.step()
      }
      dut.io.spWrite(0).valid.poke(false.B)
      dut.io.cmd.bits.a.poke(0.U); dut.io.cmd.bits.b.poke(1.U)
      dut.io.cmd.bits.c.poke("hffffffff".U); dut.io.cmd.bits.d.poke(1.U)
      dut.io.cmd.bits.k.poke(1.U); dut.io.cmd.bits.tag.poke(123.U)
      dut.io.cmd.bits.zeroC.poke(true.B)
      dut.io.cmd.valid.poke(true.B); dut.clock.step(); dut.io.cmd.valid.poke(false.B)
      dut.clock.step(4) // Launch, but no D writes have occurred.
      dut.io.writes.expect(0.U)
      // Both the controller and SRAM queues reset together. Existing memory
      // contents survive, but outstanding response control is discarded.
      dut.reset.poke(true.B); dut.clock.step(3); dut.reset.poke(false.B)
      dut.clock.step(2)
      dut.io.cmd.ready.expect(true.B)
      dut.io.completed.valid.expect(false.B)
      dut.io.cmd.bits.tag.poke(65535.U)
      dut.io.cmd.valid.poke(true.B); dut.clock.step(); dut.io.cmd.valid.poke(false.B)
      var waited = 0
      var lastWriteAt = -1
      while (!dut.io.completed.valid.peek().litToBoolean && waited < 200) {
        if (dut.io.writes.peek().litValue == mem.dim && lastWriteAt < 0) lastWriteAt = waited
        dut.clock.step(); waited += 1
      }
      assert(waited < 200)
      assert(lastWriteAt >= 0 && waited-lastWriteAt >= mem.writeLatency,
        "Completion preceded the last write's visibility delay")
      dut.io.reads.expect(2.U); dut.io.writes.expect(2.U)
      dut.io.completed.bits.expect(65535.U)
      dut.io.completed.ready.poke(true.B); dut.clock.step(); dut.io.completed.ready.poke(false.B)
      for (row <- 1 to 2) {
        val p = dut.io.accRead(0)
        p.req.bits.addr.poke(row.U); p.req.bits.fromDMA.poke(false.B)
        p.req.valid.poke(true.B)
        p.req.ready.expect(true.B); dut.clock.step(); p.req.valid.poke(false.B)
        var wait = 0
        while (!p.resp.valid.peek().litToBoolean && wait < 10) { dut.clock.step(); wait += 1 }
        assert(wait < 10)
        p.resp.bits.data.expect("h3f8000003f800000".U)
        p.resp.ready.poke(true.B); dut.clock.step(); p.resp.ready.poke(false.B)
      }
    }
  }
}
