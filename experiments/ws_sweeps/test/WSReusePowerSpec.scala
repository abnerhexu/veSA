package gemmini.wsexperiments

import chisel3._
import chiseltest._
import chiseltest.simulator.WriteVcdAnnotation
import gemmini.tensorcore._
import org.scalatest.flatspec.AnyFlatSpec
import java.nio.file.{Files, Paths}

class WSReusePowerSpec extends AnyFlatSpec with ChiselScalatestTester {
  "The WS area-sweep point" should "execute its checked activity workload" in {
    val columns = sys.env("WS_SWEEP_COLUMNS").toInt
    val k = sys.env("WS_SWEEP_K").toInt
    val vectorPath = Paths.get(sys.env("WS_SWEEP_VECTOR"))
    val outputPath = Paths.get(sys.env("WS_SWEEP_RUNTIME"))
    val fields = Files.readAllLines(vectorPath).toArray(new Array[String](0))
      .filterNot(_.startsWith("#")).map(_.split(" ")).map(x => x(0) -> x(1)).toMap
    def bits(name: String) = fields(name).split(",").map(BigInt(_, 16))
    val a = bits("a"); val b = bits("b"); val c = bits("c"); val d = bits("d")
    val rows = 16
    assert(a.length == rows * k && b.length == k * columns)
    assert(c.length == rows * columns && d.length == rows * columns)

    test(new WSReuseExperimentMesh(rows = rows, columns = columns,
      physicalDepth = 16, maxK = 256)).withAnnotations(
        TensorTestBackend.annotations :+ WriteVcdAnnotation) { dut =>
      var cycle = 0L
      def step(): Unit = { dut.clock.step(); cycle += 1 }
      dut.io.req.valid.poke(false.B)
      dut.io.in.valid.poke(false.B)
      dut.io.out.ready.poke(false.B)
      dut.reset.poke(true.B); step(); step(); dut.reset.poke(false.B); step()
      val begin = cycle
      dut.io.req.bits.k.poke(k.U)
      dut.io.req.bits.tag.poke(0x7171.U)
      for (row <- 0 until rows; column <- 0 until columns) {
        dut.io.req.bits.c(row)(column).poke(c(row * columns + column).U)
      }
      dut.io.req.valid.poke(true.B); dut.io.req.ready.expect(true.B); step()
      dut.io.req.valid.poke(false.B)
      var accepted = 0
      while (!dut.io.out.valid.peek().litToBoolean && cycle - begin < 20000) {
        if (dut.io.in.ready.peek().litToBoolean) {
          val index = dut.io.expectedIndex.peek().litValue.toInt
          for (row <- 0 until rows) dut.io.in.bits.a(row).poke(a(row * k + index).U)
          for (column <- 0 until columns) dut.io.in.bits.b(column).poke(b(index * columns + column).U)
          dut.io.in.valid.poke(true.B)
          accepted += 1
        } else dut.io.in.valid.poke(false.B)
        step()
      }
      dut.io.in.valid.poke(false.B)
      assert(accepted == 2 * k)
      dut.io.out.valid.expect(true.B)
      dut.io.out.bits.tag.expect(0x7171.U)
      for (row <- 0 until rows; column <- 0 until columns) {
        dut.io.out.bits.d(row)(column).expect(d(row * columns + column).U)
      }
      val resultCycle = cycle
      dut.io.out.ready.poke(true.B); step(); dut.io.out.ready.poke(false.B)
      Files.createDirectories(outputPath.getParent)
      Files.writeString(outputPath,
        "physical_depth,rows,columns,k,begin_cycle,result_cycle,cycles,accepted_slices,operations,result_check\n" +
        s"16,$rows,$columns,$k,$begin,$resultCycle,${resultCycle-begin},$accepted,${2L*rows*columns*k},passed\n")
    }
  }
}
