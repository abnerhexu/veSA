package gemmini.tensorcore

import chisel3._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec
import scala.io.Source
import scala.util.Random

class GroupedPESpec extends AnyFlatSpec with ChiselScalatestTester {
  for ((kind, config) <- Seq("mxfp4" -> BlackwellConfig.mxfp4, "nvfp4" -> BlackwellConfig.nvfp4)) {
    "Blackwell GST-FDPA PE" should s"match exhaustive and author-verified $kind vectors" in {
      val source = Source.fromInputStream(getClass.getResourceAsStream("/grouped-vectors.tsv"))
      val vectors = try source.getLines().filterNot(_.startsWith("#")).map(_.split(" "))
        .filter(_(0) == kind).toVector finally source.close()
      assert(vectors.nonEmpty)
      test(new TensorPE(config)) { dut =>
        val random = new Random(2899)
        dut.io.req.valid.poke(false.B)
        dut.io.in.valid.poke(false.B)
        dut.io.out.ready.poke(false.B)
        dut.clock.step()
        def send(v: Array[String]): Unit = {
          def bits(field: Int) = v(field).split(",").map(BigInt(_, 16))
          val a = bits(1); val b = bits(2); val alpha = bits(4); val beta = bits(5)
          dut.io.req.ready.expect(true.B)
          dut.io.req.bits.c.poke(BigInt(v(3), 16).U)
          dut.io.req.bits.length.poke(64.U)
          dut.io.req.valid.poke(true.B)
          dut.clock.step()
          dut.io.req.valid.poke(false.B)
          for (i <- 0 until 64) {
            dut.io.in.valid.poke(false.B)
            // Bubbles on both sides of the group and scale-block boundaries.
            if (i % 16 == 0 || i % 16 == 15) dut.clock.step(random.nextInt(3))
            dut.io.in.bits.a.poke(a(i).U)
            dut.io.in.bits.b.poke(b(i).U)
            val av = alpha(i/config.scaleBlockSize)
            val bv = beta(i/config.scaleBlockSize)
            // UE4M3 has no sign: ignored storage bits may vary within a block.
            dut.io.in.bits.alpha.get.poke((if (kind == "nvfp4" && i%2 == 1) av ^ 128 else av).U)
            dut.io.in.bits.beta.get.poke((if (kind == "nvfp4" && i%3 == 1) bv ^ 128 else bv).U)
            dut.io.in.valid.poke(true.B)
            dut.io.in.ready.expect(true.B)
            dut.clock.step()
          }
          dut.io.in.valid.poke(false.B)
          var waited = 0
          while (!dut.io.out.valid.peek().litToBoolean && waited < 30) {
            dut.clock.step(); waited += 1
          }
          assert(waited < 30)
          dut.io.out.bits.expect(BigInt(v(6), 16).U)
          dut.clock.step(2)
          dut.io.out.valid.expect(true.B)
          dut.io.out.bits.expect(BigInt(v(6), 16).U)
          dut.io.req.ready.expect(false.B)
          dut.io.out.ready.poke(true.B)
          dut.clock.step()
          dut.io.out.ready.poke(false.B)
        }
        vectors.foreach(send)
        dut.io.req.bits.c.poke(0.U)
        dut.io.req.bits.length.poke(64.U)
        dut.io.req.valid.poke(true.B)
        dut.clock.step()
        dut.io.req.valid.poke(false.B)
        dut.io.in.bits.a.poke(7.U)
        dut.io.in.bits.b.poke(7.U)
        dut.io.in.bits.alpha.get.poke((if (kind == "mxfp4") 127 else 0x38).U)
        dut.io.in.bits.beta.get.poke((if (kind == "mxfp4") 127 else 0x38).U)
        dut.io.in.valid.poke(true.B)
        dut.clock.step(8)
        dut.io.in.valid.poke(false.B)
        dut.reset.poke(true.B)
        dut.clock.step(2)
        dut.reset.poke(false.B)
        dut.io.req.ready.expect(true.B)
        dut.io.out.valid.expect(false.B)
        send(vectors.head) // aborted group state must not leak into a fresh request
      }
    }
  }
}
