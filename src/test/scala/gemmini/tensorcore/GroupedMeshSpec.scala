package gemmini.tensorcore

import chisel3._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec
import scala.io.Source
import scala.util.Random

class GroupedMeshSpec extends AnyFlatSpec with ChiselScalatestTester {
  for ((kind, config) <- Seq("mxfp4" -> BlackwellConfig.mxfp4, "nvfp4" -> BlackwellConfig.nvfp4)) {
    "Blackwell GST-FDPA mesh" should s"route $kind scales and chain 64-element blocks" in {
      val source = Source.fromInputStream(getClass.getResourceAsStream("/grouped-matrices.tsv"))
      val vectors = try source.getLines().filterNot(_.startsWith("#")).map(_.split(" "))
        .filter(_(0) == kind).toVector finally source.close()
      assert(vectors.nonEmpty)
      test(new TensorMesh(config)) { dut =>
        val random = new Random(776)
        dut.io.req.valid.poke(false.B)
        dut.io.in.valid.poke(false.B)
        dut.io.out.ready.poke(false.B)
        dut.clock.step()
        for ((v, tag) <- vectors.zipWithIndex) {
          def bits(field: Int) = v(field).split(",").map(BigInt(_, 16))
          val k = v(1).toInt
          val a = bits(2); val b = bits(3); val c = bits(4)
          val alpha = bits(5); val beta = bits(6); val d = bits(7)
          val blocks = k/config.scaleBlockSize
          dut.io.req.ready.expect(true.B)
          dut.io.req.bits.k.poke(k.U)
          dut.io.req.bits.tag.poke((tag+100).U)
          for (r <- 0 until 2; col <- 0 until 2) dut.io.req.bits.c(r)(col).poke(c(r*2+col).U)
          dut.io.req.valid.poke(true.B)
          dut.clock.step()
          dut.io.req.valid.poke(false.B)
          for (t <- 0 until k) {
            dut.io.in.valid.poke(false.B)
            dut.clock.step(random.nextInt(3))
            for (r <- 0 until 2) {
              dut.io.in.bits.a(r).poke(a(r*k+t).U)
              dut.io.in.bits.alpha.get(r).poke(alpha(r*blocks+t/config.scaleBlockSize).U)
            }
            for (col <- 0 until 2) {
              dut.io.in.bits.b(col).poke(b(t*2+col).U)
              dut.io.in.bits.beta.get(col).poke(beta((t/config.scaleBlockSize)*2+col).U)
            }
            dut.io.in.valid.poke(true.B)
            var waited = 0
            while (!dut.io.in.ready.peek().litToBoolean && waited < 100) {
              dut.clock.step(); waited += 1
            }
            assert(waited < 100)
            dut.clock.step()
          }
          dut.io.in.valid.poke(false.B)
          var waited = 0
          while (!dut.io.out.valid.peek().litToBoolean && waited < 100) {
            dut.clock.step(); waited += 1
          }
          assert(waited < 100)
          def check(): Unit = {
            dut.io.out.bits.tag.expect((tag+100).U)
            for (r <- 0 until 2; col <- 0 until 2) dut.io.out.bits.d(r)(col).expect(d(r*2+col).U)
          }
          check()
          dut.clock.step(3)
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
