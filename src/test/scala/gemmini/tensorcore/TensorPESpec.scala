package gemmini.tensorcore

import chisel3._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec
import scala.io.Source
import scala.util.Random

abstract class TensorPEChecks(hopper: Boolean) extends AnyFlatSpec with ChiselScalatestTester {
  for (format <- Seq(TensorFormat.E4M3, TensorFormat.E5M2, TensorFormat.FP16, TensorFormat.BF16);
       half <- Seq(false, true) if !(format == TensorFormat.BF16 && half)) {
    val label = s"${format.name}-${if (half) "f16" else "f32"}"
    val architecture = if (hopper) "Hopper" else "Blackwell"
    s"$architecture TensorPE" should s"match author-verified vectors for $label under stalls" in {
      val source = Source.fromInputStream(getClass.getResourceAsStream(
        if (hopper) "/hopper-vectors.tsv" else "/vectors.tsv"))
      val vectors = try source.getLines().filterNot(_.startsWith("#")).map(_.split(" ")).filter(
        fields => fields(0) == format.name && fields(1) == (if (half) "f16" else "f32")).toVector
      finally source.close()
      assert(vectors.nonEmpty, s"Missing vectors for $label; run modeling/tensorcore/validate.py")
      val profile: TensorConfig = if (hopper) HopperConfig(format, half) else BlackwellConfig(format, half)
      test(new TensorPE(profile)).withAnnotations(TensorTestBackend.annotations) { dut =>
        val random = new Random(776)
        dut.io.req.valid.poke(false.B)
        dut.io.in.valid.poke(false.B)
        dut.io.out.ready.poke(false.B)
        dut.clock.step()
        for (v <- vectors) {
          val a = v(2).split(",").map(BigInt(_, 16))
          val b = v(3).split(",").map(BigInt(_, 16))
          dut.io.req.ready.expect(true.B)
          dut.io.req.bits.c.poke(BigInt(v(4), 16).U)
          dut.io.req.bits.length.poke(a.length.U)
          dut.io.req.valid.poke(true.B)
          dut.clock.step()
          dut.io.req.valid.poke(false.B)
          for ((av, bv) <- a.zip(b)) {
            dut.io.in.valid.poke(false.B)
            dut.clock.step(random.nextInt(3))
            dut.io.in.bits.a.poke(av.U)
            dut.io.in.bits.b.poke(bv.U)
            dut.io.in.valid.poke(true.B)
            dut.io.in.ready.expect(true.B)
            dut.clock.step()
          }
          dut.io.in.valid.poke(false.B)
          var waited = 0
          while (!dut.io.out.valid.peek().litToBoolean && waited < 100) {
            dut.clock.step()
            waited += 1
          }
          assert(waited < 100, "PE did not complete")
          dut.io.out.bits.expect(BigInt(v(5), 16).U)
          dut.io.req.ready.expect(false.B)
          dut.clock.step(3)
          dut.io.out.valid.expect(true.B)
          dut.io.out.bits.expect(BigInt(v(5), 16).U)
          dut.io.out.ready.poke(true.B)
          dut.clock.step()
          dut.io.out.ready.poke(false.B)
        }
        // Abort a block and verify reset discards its control state.
        dut.io.req.bits.length.poke(2.U)
        dut.io.req.bits.c.poke(0.U)
        dut.io.req.valid.poke(true.B)
        dut.clock.step()
        dut.io.req.valid.poke(false.B)
        dut.io.in.valid.poke(true.B)
        dut.io.in.bits.a.poke(0.U)
        dut.io.in.bits.b.poke(0.U)
        dut.clock.step()
        dut.io.in.valid.poke(false.B)
        dut.reset.poke(true.B)
        dut.clock.step(2)
        dut.reset.poke(false.B)
        dut.io.req.ready.expect(true.B)
        dut.io.out.valid.expect(false.B)
      }
    }
  }
}

class TensorPESpec extends TensorPEChecks(false)
class HopperPESpec extends TensorPEChecks(true)
