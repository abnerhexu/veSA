package gemmini.tensorcore

import chisel3._
import chisel3.util._
import chiseltest._
import gemmini.{AccumulatorMemCore, AccumulatorScaleResp}
import org.scalatest.flatspec.AnyFlatSpec

class AccumulatorWidthContract(smallWidth: Int, reversed: Boolean = false) extends Module {
  val io = IO(new Bundle {
    val raw = Input(UInt(64.W))
    val small = Input(UInt((smallWidth*2).W))
    val rawOut = Output(UInt(64.W))
    val smallOut = Output(UInt((smallWidth*2).W))
  })
  val response = Wire(new AccumulatorScaleResp(
    Vec(2, Vec(1, UInt((if (reversed) smallWidth else 32).W))),
    Vec(2, Vec(1, UInt((if (reversed) 32 else smallWidth).W)))))
  require(response.full_data.getWidth == 64)
  require(response.data.getWidth == smallWidth*2)
  response := 0.U.asTypeOf(response)
  response.full_data := io.raw.asTypeOf(response.full_data)
  response.data := io.small.asTypeOf(response.data)
  io.rawOut := response.full_data.asUInt
  io.smallOut := response.data.asUInt
}

/** Exercise the shared production storage with an actual external UInt adder.
  * Tensor mode separately poisons the adder and asserts that it is NEVER used.
  */
class AccumulatorStorageHarness(singlePorted: Boolean, subBanks: Int) extends Module {
  val row = Vec(2, Vec(1, UInt(32.W)))
  val bank = Module(new AccumulatorMemCore(16, row, (x: UInt, _: UInt) => x, UInt(32.W),
    singlePorted, subBanks, false, false, 2, UInt(32.W), false))
  val io = IO(bank.io.cloneType)
  io <> bank.io
  bank.io.adder.sum := VecInit(bank.io.adder.op1.zip(bank.io.adder.op2).map { case (a, b) =>
    VecInit(a.zip(b).map { case (x, y) => RegNext(x+y) })
  })
}

class AccumulatorContractsSpec extends AnyFlatSpec with ChiselScalatestTester {
  for (small <- Seq(8, 16, 32)) {
    "Accumulator response contract" should s"keep full 32-bit slots beside $small-bit converted slots" in {
      test(new AccumulatorWidthContract(small)).withAnnotations(TensorTestBackend.annotations) { dut =>
        for (raw <- Seq(BigInt("7fc1234580000001", 16), BigInt("00000001ff7fffff", 16),
                        BigInt("deadbeef01234567", 16))) {
          val converted = (BigInt(1) << (2*small))-1
          dut.io.raw.poke(raw.U); dut.io.small.poke(converted.U)
          dut.io.rawOut.expect(raw.U); dut.io.smallOut.expect(converted.U)
          dut.clock.step()
        }
      }
    }
  }
  "Accumulator response contract" should "reject reversed full and converted row types" in {
    val error = intercept[IllegalArgumentException] {
      (new chisel3.stage.ChiselStage).emitChirrtl(new AccumulatorWidthContract(8, reversed = true))
    }
    assert(error.getMessage.contains("full_data must not be narrower"))
  }
  for ((single, subBanks) <- Seq(false -> -1, true -> 1, true -> 2)) {
    "Shared production accumulator storage" should s"preserve overwrite masks RMW and stalls single=$single subBanks=$subBanks" in {
      test(new AccumulatorStorageHarness(single, subBanks)).withAnnotations(TensorTestBackend.annotations) { dut =>
        dut.io.read.req.valid.poke(false.B); dut.io.read.resp.ready.poke(false.B)
        dut.io.write.valid.poke(false.B)
        dut.io.adder.sum.foreach(_.foreach(_.poke(0.U)))
        dut.clock.step()
        def write(data: Seq[BigInt], accumulate: Boolean, mask: BigInt = 255): Unit = {
          dut.io.write.bits.addr.poke(3.U)
          for (i <- 0 until 2) dut.io.write.bits.data(i)(0).poke(data(i).U)
          dut.io.write.bits.acc.poke(accumulate.B)
          for (i <- 0 until 8) dut.io.write.bits.mask(i).poke(mask.testBit(i).B)
          dut.io.write.valid.poke(true.B)
          var waited = 0
          while (!dut.io.write.ready.peek().litToBoolean && waited < 30) { dut.clock.step(); waited += 1 }
          assert(waited < 30); dut.io.busy.expect(true.B)
          dut.clock.step(); dut.io.write.valid.poke(false.B)
          dut.io.busy.expect(true.B)
          if (single) {
            // The pipeline has drained here, but its write has only reached
            // the sub-bank queue: busy must not briefly drop before commit.
            dut.clock.step(2); dut.io.busy.expect(true.B); dut.clock.step(6)
          } else { dut.clock.step(8) }
          dut.io.busy.expect(false.B)
        }
        def read(expected: Seq[BigInt]): Unit = {
          dut.io.read.req.bits.addr.poke(3.U)
          dut.io.read.req.bits.fromDMA.poke(true.B)
          dut.io.read.req.bits.full.poke(true.B)
          dut.io.read.req.bits.scale.poke(0x12345678.U)
          dut.io.read.req.bits.igelu_qb.poke(0x87654321L.U)
          dut.io.read.req.bits.igelu_qc.poke(0.U)
          dut.io.read.req.bits.iexp_qln2.poke(0.U)
          dut.io.read.req.bits.iexp_qln2_inv.poke(0.U)
          dut.io.read.req.bits.act.poke(0.U)
          dut.io.read.req.valid.poke(true.B)
          var waited = 0
          while (!dut.io.read.req.ready.peek().litToBoolean && waited < 30) { dut.clock.step(); waited += 1 }
          assert(waited < 30); dut.clock.step(); dut.io.read.req.valid.poke(false.B)
          waited = 0
          while (!dut.io.read.resp.valid.peek().litToBoolean && waited < 30) { dut.clock.step(); waited += 1 }
          assert(waited < 30)
          def check(): Unit = {
            for (i <- 0 until 2) dut.io.read.resp.bits.data(i)(0).expect(expected(i).U)
            dut.io.read.resp.bits.fromDMA.expect(true.B)
            dut.io.read.resp.bits.scale.expect(0x12345678.U)
            dut.io.read.resp.bits.igelu_qb.expect(0x87654321L.U)
          }
          check(); dut.io.busy.expect(true.B)
          dut.clock.step(4); dut.io.read.resp.valid.expect(true.B); check(); dut.io.busy.expect(true.B)
          dut.io.read.resp.ready.poke(true.B); dut.clock.step(); dut.io.read.resp.ready.poke(false.B)
          dut.io.busy.expect(false.B)
        }
        write(Seq(BigInt("89abcdef",16), BigInt("01234567",16)), false)
        read(Seq(BigInt("89abcdef",16), BigInt("01234567",16)))
        write(Seq(BigInt("000000aa",16), BigInt(0)), false, 1)
        read(Seq(BigInt("89abcdaa",16), BigInt("01234567",16)))
        write(Seq(BigInt(5), BigInt(7)), false)
        write(Seq(BigInt(2), BigInt(3)), true)
        read(Seq(BigInt(7), BigInt(10)))
      }
    }
  }
}
