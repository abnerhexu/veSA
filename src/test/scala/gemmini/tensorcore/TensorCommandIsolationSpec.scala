package gemmini.tensorcore

import chisel3._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec

/** Same queue, dispatch gate, frontend, execution engine, production bank/router
  * and raw bridge as the numeric command tests. Host traffic represents local
  * legacy/DMA access, NOT a simulated TileLink/Rocket system.
  */
class TensorCommandIsolationSpec extends AnyFlatSpec with ChiselScalatestTester {
  "Queued Tensor commands" should "drain late writes and fence younger commands and local writers" in {
    val mem = TensorMemoryConfig()
    test(new TensorMemoryHarness(BlackwellConfig(), mem, realAccumulator = true, fencedCommands = true))
      .withAnnotations(TensorTestBackend.annotations) { dut =>
      dut.clock.setTimeout(4000)
      val p = dut.io.frontend.get
      dut.io.cmd.valid.poke(false.B); dut.io.completed.ready.poke(false.B)
      dut.io.stress.poke(true.B)
      p.cmd.valid.poke(false.B); p.response.ready.poke(false.B); p.legacy.ready.poke(true.B)
      p.olderBusy.poke(false.B); p.commitBlocked.poke(false.B)
      for (w <- dut.io.spWrite ++ dut.io.accWrite) { w.valid.poke(false.B); w.mask.foreach(_.poke(true.B)) }
      dut.io.accRead.foreach { r => r.req.valid.poke(false.B); r.resp.ready.poke(false.B) }
      dut.clock.step()
      def send(funct: Int, rs1: BigInt, rs2: BigInt, rd: Int = 13): Unit = {
        p.cmd.bits.funct.poke(funct.U); p.cmd.bits.rs1.poke(rs1.U); p.cmd.bits.rs2.poke(rs2.U)
        p.cmd.bits.rd.poke(rd.U); p.cmd.bits.xd.poke(true.B)
        p.cmd.valid.poke(true.B); p.cmd.ready.expect(true.B); dut.clock.step(); p.cmd.valid.poke(false.B)
      }
      def waitResponse(expected: BigInt): Unit = {
        var cycles = 0
        while (!p.response.valid.peek().litToBoolean && cycles < 2000) { dut.clock.step(); cycles += 1 }
        assert(cycles < 2000)
        p.response.bits.rd.expect(13.U); p.response.bits.data.expect(expected.U)
      }
      def config(funct: Int, a: Int, b: Int): Unit = {
        send(funct, a, b); waitResponse(0)
        p.response.ready.poke(true.B); dut.clock.step(); p.response.ready.poke(false.B)
      }
      def spWrite(addr: Int): Unit = {
        val w = dut.io.spWrite(addr/mem.spDepth)
        w.addr.poke((addr%mem.spDepth).U); w.data.poke(0x3838.U); w.valid.poke(true.B)
        w.ready.expect(true.B); dut.clock.step(); w.valid.poke(false.B)
      }
      def accWrite(addr: Int, lo: Long, hi: Long): Unit = {
        val w = dut.io.accWrite(addr/mem.accDepth)
        w.addr.poke((addr%mem.accDepth).U)
        w.data.poke(((BigInt(hi) << 32) | lo).U); w.valid.poke(true.B)
        w.ready.expect(true.B); dut.clock.step(); w.valid.poke(false.B)
      }
      spWrite(4); spWrite(260)
      config(TensorCommandISA.AddrAB, 4, 260)
      config(TensorCommandISA.AddrCD, 15, 8)
      p.olderBusy.poke(true.B)
      send(TensorCommandISA.Run, (BigInt(0x6161) << 16) | 1, 0)
      dut.clock.step(4)
      p.exclusive.expect(false.B); dut.io.reads.expect(0.U)
      send(2, 0x1234, 0x5678, 7)
      send(3, 0x9abc, 0xdef0, 8)
      p.cmd.ready.expect(false.B); p.legacy.valid.expect(false.B)
      // Older local writes may still complete while in drain.
      accWrite(15, 0x3f800000L, 0x40000000L) // 1,2
      accWrite(16, 0x40400000L, 0x40800000L) // 3,4
      dut.io.memoryIdle.expect(false.B)
      p.commitBlocked.poke(true.B); p.olderBusy.poke(false.B)
      dut.clock.step(12)
      dut.io.memoryIdle.expect(true.B); p.exclusive.expect(false.B); dut.io.reads.expect(0.U)
      p.commitBlocked.poke(false.B)
      var cycles = 0
      while (!p.exclusive.peek().litToBoolean && cycles < 20) { dut.clock.step(); cycles += 1 }
      assert(cycles < 20)
      p.commitBlocked.poke(true.B) // Also require actual visibility before response.
      val sw = dut.io.spWrite(0)
      sw.valid.poke(true.B); sw.addr.poke(4.U); sw.data.poke(0xffff.U)
      val aw = dut.io.accWrite(0)
      aw.valid.poke(true.B); aw.addr.poke(8.U); aw.data.poke(BigInt("deadbeefdeadbeef",16).U)
      val ar = dut.io.accRead(0)
      ar.req.valid.poke(true.B); ar.req.bits.addr.poke(8.U); ar.req.bits.fromDMA.poke(false.B)
      cycles = 0
      while (dut.io.writes.peek().litValue != 2 && cycles < 2000) {
        p.legacy.valid.expect(false.B); sw.ready.expect(false.B); aw.ready.expect(false.B)
        ar.req.ready.expect(false.B); p.response.valid.expect(false.B)
        dut.clock.step(); cycles += 1
      }
      assert(cycles < 2000)
      dut.clock.step(20)
      dut.io.memoryIdle.expect(true.B); p.response.valid.expect(false.B); p.exclusive.expect(true.B)
      p.commitBlocked.poke(false.B); waitResponse(BigInt(0x6161) << 16)
      dut.clock.step(8)
      p.legacy.valid.expect(false.B); sw.ready.expect(false.B); aw.ready.expect(false.B)
      ar.req.ready.expect(false.B); p.cmd.ready.expect(false.B)
      sw.valid.poke(false.B); aw.valid.poke(false.B); ar.req.valid.poke(false.B)
      p.legacy.ready.poke(false.B)
      p.response.ready.poke(true.B); dut.clock.step(); p.response.ready.poke(false.B)
      p.exclusive.expect(false.B); p.legacy.valid.expect(true.B)
      p.legacy.bits.rd.expect(7.U); p.legacy.bits.rs1.expect(0x1234.U); p.legacy.bits.rs2.expect(0x5678.U)
      dut.clock.step(3); p.legacy.bits.rd.expect(7.U)
      p.legacy.ready.poke(true.B); dut.clock.step()
      p.legacy.bits.rd.expect(8.U); p.legacy.bits.funct.expect(3.U)
      p.legacy.bits.rs1.expect(0x9abc.U); dut.clock.step()
      dut.io.busy.expect(false.B)
      for ((row, expected) <- Seq(8 -> BigInt("4040000040000000",16), 9 -> BigInt("40a0000040800000",16))) {
        ar.req.bits.addr.poke(row.U); ar.req.valid.poke(true.B)
        var wait = 0
        while (!ar.req.ready.peek().litToBoolean && wait < 20) { dut.clock.step(); wait += 1 }
        assert(wait < 20); dut.clock.step(); ar.req.valid.poke(false.B)
        wait = 0
        while (!ar.resp.valid.peek().litToBoolean && wait < 20) { dut.clock.step(); wait += 1 }
        assert(wait < 20); ar.resp.bits.data.expect(expected.U)
        ar.resp.ready.poke(true.B); dut.clock.step(); ar.resp.ready.poke(false.B)
      }
    }
  }
}
