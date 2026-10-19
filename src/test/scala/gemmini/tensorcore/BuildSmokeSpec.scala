package gemmini.tensorcore

import chisel3._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec

class BuildSmoke extends Module {
  val io = IO(new Bundle {
    val a = Input(UInt(8.W))
    val b = Input(UInt(8.W))
    val product = Output(UInt(16.W))
  })
  io.product := io.a * io.b
}

class BuildSmokeSpec extends AnyFlatSpec with ChiselScalatestTester {
  "The standalone Chisel 3.6 build" should "elaborate and simulate without Chipyard" in {
    test(new BuildSmoke) { dut =>
      dut.io.a.poke(255.U)
      dut.io.b.poke(17.U)
      dut.io.product.expect(4335.U)
      dut.clock.step()
    }
  }
}
