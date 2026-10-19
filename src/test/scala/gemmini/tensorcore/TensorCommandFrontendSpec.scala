package gemmini.tensorcore

import chisel3._
import chiseltest._
import gemmini.MemoryActivity
import org.scalatest.flatspec.AnyFlatSpec

class TensorCommandFrontendSpec extends AnyFlatSpec with ChiselScalatestTester {
  import TensorCommandISA._
  for ((label, config) <- Seq("e4m3" -> BlackwellConfig(), "nvfp4" -> BlackwellConfig.nvfp4)) {
    "Tensor command fence" should s"drain, snapshot, reject and retain ownership for $label" in {
      test(new TensorCommandFrontend(config)).withAnnotations(TensorTestBackend.annotations) { dut =>
        dut.io.cmd.valid.poke(false.B)
        dut.io.response.ready.poke(false.B)
        dut.io.execute.ready.poke(false.B)
        dut.io.completed.valid.poke(false.B); dut.io.completed.bits.poke(0.U)
        dut.io.legacyIdle.poke(true.B); dut.io.memoryIdle.poke(true.B)
        dut.clock.step()
        def send(funct: Int, rs1: BigInt, rs2: BigInt, xd: Boolean = true): Unit = {
          dut.io.cmd.ready.expect(true.B)
          dut.io.cmd.bits.funct.poke(funct.U); dut.io.cmd.bits.rs1.poke(rs1.U); dut.io.cmd.bits.rs2.poke(rs2.U)
          dut.io.cmd.bits.rd.poke(23.U); dut.io.cmd.bits.xd.poke(xd.B)
          dut.io.cmd.valid.poke(true.B); dut.clock.step(); dut.io.cmd.valid.poke(false.B)
        }
        def result(status: Int, tag: Int = 0): Unit = {
          var waited = 0
          while (!dut.io.response.valid.peek().litToBoolean && waited < 12) { dut.clock.step(); waited += 1 }
          assert(waited < 12)
          val expected = (BigInt(tag) << 16) | status
          dut.io.response.bits.rd.expect(23.U); dut.io.response.bits.data.expect(expected.U)
          dut.io.cmd.ready.expect(false.B); dut.io.busy.expect(true.B); dut.io.exclusive.expect(true.B)
          dut.clock.step(5)
          dut.io.response.valid.expect(true.B); dut.io.response.bits.data.expect(expected.U)
          dut.io.response.ready.poke(true.B); dut.clock.step(); dut.io.response.ready.poke(false.B)
          dut.io.busy.expect(false.B); dut.io.exclusive.expect(false.B)
        }
        val k = if (config.grouped) 64 else 17
        val run = BigInt(k) | (BigInt(0xabcd) << 16)
        send(Run, run, 0); result(MissingDescriptor, 0xabcd)
        send(AddrAB, 4, 260); result(Success)
        send(AddrCD, 15, 4); result(Success)
        if (config.grouped) {
          send(Run, run, 0); result(MissingDescriptor, 0xabcd)
          send(Scales, 768, 800); result(Success)
          send(Scales, BigInt(1) << 32, 800); result(BadEncoding)
          send(Scales, 1023, 800); result(Success)
          send(Run, run, 0); result(BadRange, 0xabcd)
          send(Scales, 768, 800); result(Success)
        } else { send(Scales, 768, 800); result(BadEncoding) }
        send(AddrAB, BigInt(1) << 32, 260); result(BadEncoding) // Must not replace valid A.
        send(Run, 0, 0); result(BadEncoding)
        send(Run, run | (BigInt(1) << 33), 0); result(BadEncoding, 0xabcd)
        send(Run, run, 1); result(BadEncoding, 0xabcd)
        if (config.grouped) { send(Run, 63, 0); result(BadEncoding) }
        send(AddrAB, BigInt("ffffffff", 16), 260); result(Success)
        send(Run, run, 0); result(BadRange, 0xabcd) // Carry must not wrap the address.
        send(AddrAB, 4, 260); result(Success)
        send(AddrCD, 31, 4); result(Success)
        send(Run, run, 0); result(BadRange, 0xabcd)
        send(AddrCD, 15, 31); result(Success)
        send(Run, run, 0); result(BadRange, 0xabcd)
        send(AddrCD, BigInt("ffffffff", 16), 4); result(Success)

        // An unretired older instruction and a late write independently hold
        // the fence in drain. Younger commands cannot overwrite the descriptor.
        dut.io.legacyIdle.poke(false.B)
        send(Run, run | (BigInt(1) << 32), 0)
        dut.clock.step(8); dut.io.execute.valid.expect(false.B); dut.io.exclusive.expect(false.B)
        dut.io.memoryIdle.poke(false.B); dut.io.legacyIdle.poke(true.B)
        dut.clock.step(8); dut.io.execute.valid.expect(false.B); dut.io.exclusive.expect(false.B)
        dut.io.memoryIdle.poke(true.B)
        dut.clock.step(2)
        dut.io.execute.valid.expect(true.B)
        dut.io.execute.bits.a.expect(4.U); dut.io.execute.bits.b.expect(260.U)
        dut.io.execute.bits.k.expect(k.U); dut.io.execute.bits.tag.expect(0xabcd.U)
        dut.io.execute.bits.zeroC.expect(true.B)
        dut.io.cmd.bits.funct.poke(AddrAB.U); dut.io.cmd.bits.rs1.poke(999.U)
        dut.io.cmd.valid.poke(true.B)
        dut.clock.step(6)
        dut.io.cmd.ready.expect(false.B); dut.io.execute.valid.expect(true.B)
        dut.io.execute.bits.a.expect(4.U)
        dut.io.cmd.valid.poke(false.B); dut.io.execute.ready.poke(true.B); dut.clock.step()
        dut.io.execute.valid.expect(false.B); dut.io.completed.ready.expect(true.B)
        dut.io.memoryIdle.poke(false.B)
        dut.io.completed.bits.poke(0xabcd.U); dut.io.completed.valid.poke(true.B)
        dut.clock.step(); dut.io.completed.valid.poke(false.B)
        dut.clock.step(10)
        dut.io.response.valid.expect(false.B); dut.io.exclusive.expect(true.B)
        dut.io.memoryIdle.poke(true.B); result(Success, 0xabcd)

        // Fire-and-forget descriptor commands do not manufacture a CPU reply.
        send(AddrAB, 4, 260, false)
        dut.clock.step(5); dut.io.response.valid.expect(false.B); dut.io.busy.expect(false.B)
        dut.io.legacyIdle.poke(false.B)
        send(Run, run, 0)
        dut.clock.step(3); dut.io.busy.expect(true.B); dut.io.execute.valid.expect(false.B)
        dut.reset.poke(true.B)
        dut.io.cmd.valid.poke(true.B); dut.io.cmd.ready.expect(false.B)
        dut.io.response.valid.expect(false.B); dut.clock.step()
        dut.io.cmd.valid.poke(false.B); dut.reset.poke(false.B)
        dut.io.legacyIdle.poke(true.B); dut.clock.step()
        send(Run, run, 0); result(MissingDescriptor, 0xabcd)
      }
    }
  }

  "Memory activity" should "retain late and stalled responses and reset jointly" in {
    test(new MemoryActivity(2)).withAnnotations(TensorTestBackend.annotations) { dut =>
      dut.io.start.poke(false.B); dut.io.finish.poke(false.B); dut.clock.step()
      dut.io.idle.expect(true.B)
      dut.io.start.poke(true.B); dut.io.idle.expect(false.B); dut.clock.step(3)
      dut.io.start.poke(false.B); dut.clock.step(9); dut.io.idle.expect(false.B)
      dut.io.finish.poke(true.B); dut.clock.step(2); dut.io.idle.expect(false.B)
      dut.io.start.poke(true.B); dut.clock.step(); dut.io.idle.expect(false.B)
      dut.io.start.poke(false.B); dut.clock.step(); dut.io.finish.poke(false.B)
      dut.io.idle.expect(true.B)
      dut.io.start.poke(true.B); dut.clock.step(); dut.io.start.poke(false.B)
      dut.reset.poke(true.B); dut.clock.step(); dut.reset.poke(false.B)
      dut.io.idle.expect(true.B)
    }
  }
}
