package gemmini.tensorcore

import chisel3._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec
import scala.io.Source
import scala.util.Random

class TensorMeshSpec extends AnyFlatSpec with ChiselScalatestTester {
  for (format <- Seq(TensorFormat.E4M3, TensorFormat.E5M2, TensorFormat.FP16, TensorFormat.BF16);
       half <- Seq(false, true) if !(format == TensorFormat.BF16 && half)) {
    val label = s"${format.name}-${if (half) "f16" else "f32"}"
    "Blackwell TensorMesh" should s"preserve block boundaries for $label with K up to 65" in {
      val source = Source.fromInputStream(getClass.getResourceAsStream("/matrices.tsv"))
      val vectors = try source.getLines().filterNot(_.startsWith("#")).map(_.split(" ")).filter(
        fields => fields(0) == format.name && fields(1) == (if (half) "f16" else "f32")).toVector
      finally source.close()
      assert(vectors.nonEmpty)
      test(new TensorMesh(BlackwellConfig(format, half)))
        .withAnnotations(TensorTestBackend.annotations) { dut =>
        val random = new Random(2899)
        dut.io.req.valid.poke(false.B)
        dut.io.in.valid.poke(false.B)
        dut.io.out.ready.poke(false.B)
        dut.clock.step()
        for ((v, tag) <- vectors.zipWithIndex) {
          val k = v(2).toInt
          def bits(field: Int) = v(field).split(",").map(BigInt(_, 16))
          val a = bits(3); val b = bits(4); val c = bits(5); val d = bits(6)
          dut.io.req.ready.expect(true.B)
          dut.io.req.bits.k.poke(k.U)
          dut.io.req.bits.tag.poke(tag.U)
          for (r <- 0 until 2; col <- 0 until 2) dut.io.req.bits.c(r)(col).poke(c(r*2+col).U)
          dut.io.req.valid.poke(true.B)
          dut.clock.step()
          dut.io.req.valid.poke(false.B)
          for (t <- 0 until k) {
            dut.io.in.valid.poke(false.B)
            dut.clock.step(random.nextInt(3))
            for (r <- 0 until 2) dut.io.in.bits.a(r).poke(a(r*k+t).U)
            for (col <- 0 until 2) dut.io.in.bits.b(col).poke(b(t*2+col).U)
            dut.io.in.valid.poke(true.B)
            var waited = 0
            while (!dut.io.in.ready.peek().litToBoolean && waited < 200) {
              dut.clock.step(); waited += 1
            }
            assert(waited < 200, "Mesh input deadlocked")
            dut.clock.step()
          }
          dut.io.in.valid.poke(false.B)
          var waited = 0
          while (!dut.io.out.valid.peek().litToBoolean && waited < 200) {
            dut.clock.step(); waited += 1
          }
          assert(waited < 200, "Mesh output deadlocked")
          def check(): Unit = {
            dut.io.out.bits.tag.expect(tag.U)
            for (r <- 0 until 2; col <- 0 until 2) dut.io.out.bits.d(r)(col).expect(d(r*2+col).U)
          }
          check()
          dut.clock.step(4)
          dut.io.out.valid.expect(true.B)
          check()
          dut.io.out.ready.poke(true.B)
          dut.clock.step()
          dut.io.out.ready.poke(false.B)
        }
      }
    }
  }
}
