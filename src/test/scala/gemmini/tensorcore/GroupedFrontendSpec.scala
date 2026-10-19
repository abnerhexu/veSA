package gemmini.tensorcore

import chisel3._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec

class GroupedFrontendSpec extends AnyFlatSpec with ChiselScalatestTester {
  for ((name, config) <- Seq("mxfp4" -> BlackwellConfig.mxfp4, "nvfp4" -> BlackwellConfig.nvfp4)) {
    "The exact group frontend" should s"hold $name group results under downstream backpressure" in {
      test(new GroupedDotFrontend(config)) { dut =>
        val nv = name == "nvfp4"
        dut.io.start.poke(true.B)
        dut.io.in.valid.poke(false.B)
        dut.io.out.ready.poke(true.B)
        dut.clock.step()
        dut.io.start.poke(false.B)
        for (group <- 0 until 4; element <- 0 until 16) {
          dut.io.in.bits.a.poke(7.U)
          dut.io.in.bits.b.poke((if (group%2 == 0) 7 else 15).U)
          dut.io.in.bits.alpha.get.poke((if (nv) 0xfe else 127).U)
          dut.io.in.bits.beta.get.poke((if (nv) 0xfe else 128).U)
          dut.io.in.valid.poke(true.B)
          if (element == 15) {
            def check(): Unit = {
              dut.io.out.valid.expect(true.B)
              dut.io.out.bits.sign.expect((group%2 != 0).B)
              dut.io.out.bits.magnitude.expect((BigInt(if (nv) 1764 else 576) << 23).U)
              dut.io.out.bits.exponent.expect((if (nv) 16 else 1).S)
              dut.io.out.bits.nan.expect(false.B)
              dut.io.out.bits.infinity.expect(false.B)
            }
            dut.io.out.ready.poke(false.B)
            dut.io.in.ready.expect(false.B)
            check()
            dut.clock.step(3)
            check()
            dut.io.out.ready.poke(true.B)
          } else { dut.io.out.valid.expect(false.B) }
          dut.io.in.ready.expect(true.B)
          dut.clock.step()
        }
        dut.io.in.valid.poke(false.B)
        dut.io.in.ready.expect(false.B)
        dut.io.out.valid.expect(false.B)
      }
    }
  }
}
