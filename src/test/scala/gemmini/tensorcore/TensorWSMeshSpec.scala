package gemmini.tensorcore

import chisel3._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec

abstract class TensorWSMeshChecks(config: TensorConfig, architecture: String)
    extends AnyFlatSpec with ChiselScalatestTester {
  s"The $architecture weight-stationary Tensor mesh" should
    "load a second bank without a bubble and overlap its exponent wave with replay" in {
    test(new TensorMesh(config)).withAnnotations(TensorTestBackend.annotations) { dut =>
      dut.io.req.valid.poke(false.B)
      dut.io.in.valid.poke(false.B)
      dut.io.out.ready.poke(false.B)
      dut.clock.step()

      dut.io.req.bits.k.poke(64.U)
      dut.io.req.bits.tag.poke(0x5151.U)
      for (r <- 0 until 2; c <- 0 until 2) dut.io.req.bits.c(r)(c).poke(0.U)
      dut.io.req.valid.poke(true.B)
      dut.io.req.ready.expect(true.B)
      dut.clock.step()
      dut.io.req.valid.poke(false.B)

      // Bank 1 is written while bank 0 is already moving through the exponent
      // pipeline. There must be no collection bubble at the bank boundary.
      dut.io.in.valid.poke(true.B)
      for (_ <- 0 until 64) {
        dut.io.in.ready.expect(true.B)
        for (r <- 0 until 2) dut.io.in.bits.a(r).poke(0.U)
        for (c <- 0 until 2) dut.io.in.bits.b(c).poke(0.U)
        dut.clock.step()
      }
      dut.io.in.valid.poke(false.B)

      // A serial prepass-feedback-replay for both 32-term segments needs more
      // than 170 cycles from the final input. The dual datapaths finish below
      // this bound because bank 1 prepass overlaps bank 0 replay.
      var cyclesAfterInput = 0
      while (!dut.io.out.valid.peek().litToBoolean && cyclesAfterInput < 170) {
        dut.clock.step()
        cyclesAfterInput += 1
      }
      assert(cyclesAfterInput < 170, "double-buffered exponent/replay overlap was lost")
      dut.io.out.bits.tag.expect(0x5151.U)
      for (r <- 0 until 2; c <- 0 until 2) dut.io.out.bits.d(r)(c).expect(0.U)
      dut.io.out.ready.poke(true.B)
      dut.clock.step()
      dut.io.req.ready.expect(true.B)
    }
  }
}

class TensorWSMeshSpec extends TensorWSMeshChecks(
  BlackwellConfig(TensorFormat.E4M3), "Blackwell")

class HopperWSMeshSpec extends TensorWSMeshChecks(
  HopperConfig(TensorFormat.E4M3), "Hopper")
