package gemmini.tensorcore

import chisel3._
import chiseltest._
import gemmini.AccumulatorBankRouter
import org.scalatest.flatspec.AnyFlatSpec

class AccumulatorRouterSpec extends AnyFlatSpec with ChiselScalatestTester {
  "Accumulator bank ownership router" should "route equal DMA tags by owner and serialize conflicting rows" in {
    test(new AccumulatorBankRouter(16, Vec(2, Vec(1, UInt(32.W))), UInt(32.W)))
      .withAnnotations(TensorTestBackend.annotations) { dut =>
      for (client <- Seq(dut.io.raw, dut.io.legacy)) {
        client.read.req.valid.poke(false.B); client.read.resp.ready.poke(false.B)
        client.read.req.bits.addr.poke(3.U)
        client.read.req.bits.scale.poke(0.U)
        client.read.req.bits.igelu_qb.poke(0.U); client.read.req.bits.igelu_qc.poke(0.U)
        client.read.req.bits.iexp_qln2.poke(0.U); client.read.req.bits.iexp_qln2_inv.poke(0.U)
        client.read.req.bits.act.poke(0.U); client.read.req.bits.full.poke(true.B)
        // Crucial: the legacy execute request and raw request have the SAME tag.
        client.read.req.bits.fromDMA.poke(false.B)
        client.write.valid.poke(false.B); client.write.bits.addr.poke(3.U)
        client.write.bits.acc.poke(false.B); client.write.bits.mask.foreach(_.poke(true.B))
        client.write.bits.data.foreach(_.foreach(_.poke(0.U)))
      }
      dut.io.bank.read.req.ready.poke(true.B); dut.io.bank.write.ready.poke(true.B)
      dut.io.bank.read.resp.valid.poke(false.B)
      dut.io.bank.read.resp.bits.fromDMA.poke(false.B)
      dut.io.bank.read.resp.bits.scale.poke(0.U)
      dut.io.bank.read.resp.bits.igelu_qb.poke(0.U); dut.io.bank.read.resp.bits.igelu_qc.poke(0.U)
      dut.io.bank.read.resp.bits.iexp_qln2.poke(0.U); dut.io.bank.read.resp.bits.iexp_qln2_inv.poke(0.U)
      dut.io.bank.read.resp.bits.act.poke(0.U); dut.io.bank.read.resp.bits.acc_bank_id.poke(0.U)
      dut.io.bank.read.resp.bits.data.foreach(_.foreach(_.poke(0.U)))
      dut.clock.step()

      dut.io.raw.read.req.valid.poke(true.B); dut.io.legacy.read.req.valid.poke(true.B)
      dut.io.raw.read.req.ready.expect(true.B); dut.io.legacy.read.req.ready.expect(false.B)
      dut.clock.step() // Accepted raw first.
      dut.io.raw.read.req.valid.poke(false.B)
      dut.io.legacy.read.req.ready.expect(true.B); dut.clock.step()
      dut.io.legacy.read.req.valid.poke(false.B)
      dut.io.raw.read.req.valid.poke(true.B)
      dut.io.raw.read.req.ready.expect(false.B) // Two owner tags are outstanding.
      dut.io.bank.read.req.valid.expect(false.B)

      dut.io.bank.read.resp.bits.data(0)(0).poke("h7fc12345".U)
      dut.io.bank.read.resp.bits.data(1)(0).poke("h80000001".U)
      dut.io.bank.read.resp.valid.poke(true.B)
      dut.io.legacy.read.resp.ready.poke(true.B)
      dut.io.raw.read.resp.valid.expect(true.B); dut.io.legacy.read.resp.valid.expect(false.B)
      dut.io.bank.read.resp.ready.expect(false.B)
      dut.clock.step(3)
      dut.io.raw.read.resp.bits.data(0)(0).expect("h7fc12345".U)
      dut.io.raw.read.resp.ready.poke(true.B); dut.clock.step()
      dut.io.raw.read.req.valid.poke(false.B)
      dut.io.bank.read.resp.valid.poke(false.B)
      dut.io.raw.read.resp.ready.poke(false.B)
      dut.io.bank.read.resp.bits.data(0)(0).poke("h01234567".U)
      dut.io.bank.read.resp.valid.poke(true.B)
      dut.io.raw.read.resp.valid.expect(false.B); dut.io.legacy.read.resp.valid.expect(true.B)
      dut.io.legacy.read.resp.bits.data(0)(0).expect("h01234567".U)
      dut.clock.step(); dut.io.bank.read.resp.valid.poke(false.B)

      // Independent owners cannot read/write the same row on the same edge.
      dut.io.raw.read.req.valid.poke(true.B); dut.io.legacy.write.valid.poke(true.B)
      dut.io.raw.read.req.ready.expect(false.B)
      dut.io.bank.read.req.valid.expect(false.B); dut.io.legacy.write.ready.expect(true.B)
      dut.clock.step(); dut.io.legacy.write.valid.poke(false.B)

      dut.io.raw.read.req.ready.expect(true.B); dut.clock.step()
      dut.io.raw.read.req.valid.poke(false.B)
      dut.io.bank.read.resp.valid.poke(true.B); dut.io.raw.read.resp.ready.poke(true.B)
      dut.io.raw.read.resp.valid.expect(true.B); dut.clock.step()
      dut.io.bank.read.resp.valid.poke(false.B)

      // Raw writes cannot inherit the legacy write's accumulate bit.
      dut.io.raw.write.bits.data(0)(0).poke("h3f800000".U)
      dut.io.legacy.write.bits.acc.poke(true.B)
      dut.io.raw.write.valid.poke(true.B); dut.io.legacy.write.valid.poke(true.B)
      dut.io.raw.write.ready.expect(true.B); dut.io.legacy.write.ready.expect(false.B)
      dut.io.bank.write.bits.acc.expect(false.B)
      dut.io.bank.write.bits.data(0)(0).expect("h3f800000".U)
      dut.clock.step(); dut.io.raw.write.valid.poke(false.B)
      dut.io.bank.write.bits.acc.expect(true.B); dut.io.legacy.write.ready.expect(true.B)
      dut.clock.step(); dut.io.legacy.write.valid.poke(false.B)

      // Reset must not accept new requests even if a producer holds valid.
      dut.io.raw.read.req.valid.poke(true.B); dut.io.raw.write.valid.poke(true.B)
      dut.reset.poke(true.B)
      dut.io.bank.read.req.valid.expect(false.B); dut.io.bank.write.valid.expect(false.B)
      dut.io.raw.read.req.ready.expect(false.B); dut.io.raw.write.ready.expect(false.B)
      dut.clock.step(2)
      dut.io.raw.read.req.valid.poke(false.B); dut.io.raw.write.valid.poke(false.B)
      dut.reset.poke(false.B); dut.clock.step()
      dut.io.legacy.read.req.valid.poke(true.B)
      dut.io.legacy.read.req.ready.expect(true.B); dut.clock.step()
      dut.io.legacy.read.req.valid.poke(false.B)
      dut.io.bank.read.resp.valid.poke(true.B)
      dut.io.legacy.read.resp.valid.expect(true.B); dut.io.raw.read.resp.valid.expect(false.B)
      dut.clock.step(); dut.io.bank.read.resp.valid.poke(false.B)
    }
  }
}
