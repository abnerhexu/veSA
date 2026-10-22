package gemmini.wsexperiments

import chisel3._
import chiseltest._
import gemmini.tensorcore._
import org.scalatest.flatspec.AnyFlatSpec
import scala.io.Source

class WSReuseExperimentSpec extends AnyFlatSpec with ChiselScalatestTester {
  "A 16-deep WS experiment mesh" should
    "reuse its physical PEs across exponent and significand passes" in {
    val source = Source.fromInputStream(getClass.getResourceAsStream("/matrices.tsv"))
    val candidates = try source.getLines().filterNot(_.startsWith("#")).map(_.split(" "))
      .filter(fields => fields(0) == "e4m3" && fields(1) == "f32" &&
        Set("32", "64").contains(fields(2))).toVector
    finally source.close()
    def parsed(fields: Array[String]) = {
      def bits(index: Int) = fields(index).split(",").map(BigInt(_, 16))
      (fields(2).toInt, bits(3), bits(4), bits(5), bits(6))
    }
    val published = Seq("32", "64").map(k => parsed(candidates.find(_(2) == k).get))
    val long = Seq(128 -> BigInt("43000000", 16), 256 -> BigInt("43800000", 16)).map {
      case (k, positive) =>
        val a = Array.fill(k)(BigInt(0x38)) ++ Array.tabulate(k)(i => BigInt(if (i % 2 == 0) 0x38 else 0xb8))
        val b = Array.tabulate(k * 2) { i =>
          val index = i / 2; val column = i % 2
          BigInt(if (column == 0 || index % 2 == 0) 0x38 else 0xb8)
        }
        (k, a, b, Array.fill(4)(BigInt(0)), Array(positive, BigInt(0), BigInt(0), positive))
    }
    val vectors = published ++ long
    assert(vectors.map(_._1).toSet == Set(32, 64, 128, 256))

    val config = BlackwellConfig(TensorFormat.E4M3)
    test(new WSReuseExperimentMesh(config, rows = 2, columns = 2,
      physicalDepth = 16, maxK = 256)).withAnnotations(TensorTestBackend.annotations) { dut =>
      dut.io.req.valid.poke(false.B)
      dut.io.in.valid.poke(false.B)
      dut.io.out.ready.poke(false.B)
      dut.clock.step()
      for (((k, a, b, c, d), caseIndex) <- vectors.zipWithIndex) {
        dut.io.req.bits.k.poke(k.U)
        dut.io.req.bits.tag.poke((0x6000 + caseIndex).U)
        for (row <- 0 until 2; column <- 0 until 2) {
          dut.io.req.bits.c(row)(column).poke(c(row * 2 + column).U)
        }
        dut.io.req.valid.poke(true.B)
        dut.io.req.ready.expect(true.B)
        dut.clock.step()
        dut.io.req.valid.poke(false.B)

        var accepted = 0
        var overlapCycles = 0
        var drivenCycles = 0
        while (!dut.io.out.valid.peek().litToBoolean && drivenCycles < 5000) {
          if (dut.io.in.ready.peek().litToBoolean) {
            val index = dut.io.expectedIndex.peek().litValue.toInt
            assert(index < k)
            for (row <- 0 until 2) dut.io.in.bits.a(row).poke(a(row * k + index).U)
            for (column <- 0 until 2) dut.io.in.bits.b(column).poke(b(index * 2 + column).U)
            dut.io.in.valid.poke(true.B)
            accepted += 1
          } else {
            dut.io.in.valid.poke(false.B)
          }
          if (dut.io.overlap.peek().litToBoolean) overlapCycles += 1
          dut.clock.step()
          drivenCycles += 1
        }
        dut.io.in.valid.poke(false.B)
        assert(drivenCycles < 5000, s"K=$k result timed out while driving")
        assert(accepted == k, s"expected one buffered raw-operand pass for K=$k")
        var waited = 0
        while (!dut.io.out.valid.peek().litToBoolean && waited < 500) {
          dut.clock.step(); waited += 1
        }
        assert(waited < 500, s"K=$k result timed out")
        if (k > 32) assert(overlapCycles > 0,
          s"K=$k did not overlap exponent and significand waves")
        dut.io.out.bits.tag.expect((0x6000 + caseIndex).U)
        for (row <- 0 until 2; column <- 0 until 2) {
          dut.io.out.bits.d(row)(column).expect(d(row * 2 + column).U)
        }
        dut.io.out.ready.poke(true.B)
        dut.clock.step()
        dut.io.out.ready.poke(false.B)
      }
    }
  }

  "A 16x4 WS experiment point" should "produce all four output columns" in {
    test(new WSReuseExperimentMesh(rows = 2, columns = 4,
      physicalDepth = 16, maxK = 256)).withAnnotations(TensorTestBackend.annotations) { dut =>
      dut.io.req.valid.poke(false.B)
      dut.io.in.valid.poke(false.B)
      dut.io.out.ready.poke(false.B)
      dut.clock.step()
      dut.io.req.bits.k.poke(32.U)
      dut.io.req.bits.tag.poke(0x6404.U)
      for (row <- 0 until 2; column <- 0 until 4) {
        dut.io.req.bits.c(row)(column).poke(0.U)
      }
      dut.io.req.valid.poke(true.B)
      dut.clock.step()
      dut.io.req.valid.poke(false.B)
      var accepted = 0
      var cycles = 0
      while (!dut.io.out.valid.peek().litToBoolean && cycles < 2000) {
        if (dut.io.in.ready.peek().litToBoolean) {
          for (row <- 0 until 2) dut.io.in.bits.a(row).poke(0x38.U)
          for (column <- 0 until 4) dut.io.in.bits.b(column).poke(0x38.U)
          dut.io.in.valid.poke(true.B)
          accepted += 1
        } else dut.io.in.valid.poke(false.B)
        dut.clock.step(); cycles += 1
      }
      assert(accepted == 32)
      assert(cycles < 2000)
      dut.io.out.bits.tag.expect(0x6404.U)
      for (row <- 0 until 2; column <- 0 until 4) {
        dut.io.out.bits.d(row)(column).expect(BigInt("42000000", 16).U)
      }
    }
  }
}

class WSReuseStructureSpec extends AnyFlatSpec {
  "The experimental WS mesh" should "scale only physical PE count with width" in {
    import chisel3.stage.ChiselStage
    def instances(columns: Int): Int = (new ChiselStage).emitChirrtl(
      new WSReuseExperimentMesh(rows = 2, columns = columns,
        physicalDepth = 16, maxK = 256)).linesIterator
      .count(_.trim.startsWith("inst pes_"))
    for (columns <- Seq(2, 4, 8, 16, 32)) {
      assert(instances(columns) == 16 * columns)
    }
  }
}
