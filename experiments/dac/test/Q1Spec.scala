package gemmini.dac

import chisel3._
import chiseltest._
import gemmini.tensorcore.TensorTestBackend
import org.scalatest.flatspec.AnyFlatSpec
import scala.io.Source
import java.io.{File, PrintWriter}

class Q1Spec extends AnyFlatSpec with ChiselScalatestTester {
  for ((design, resultField) <- Seq("ieee" -> 5, "blackwell" -> 6, "hopper" -> 7,
                                   "blackwell-overlap" -> 6, "hopper-overlap" -> 7,
                                   "blackwell-reuse-overlap" -> 6, "hopper-reuse-overlap" -> 7)
       if sys.env.get("Q1_DESIGNS").forall(_.split(",").contains(design))) {
    "Q1" should s"measure and check $design" in {
      val source = Source.fromFile(sys.env("Q1_VECTORS"))
      val vectors = try source.getLines().filterNot(_.startsWith("#"))
        .map(_.split(" ")).toVector finally source.close()
      val csv = new PrintWriter(new File(sys.env("Q1_OUTPUT"), s"$design-runtime.csv"))
      csv.println("design,workload,M,N,K,memory_scope,cycles,input_cycles,non_input_cycles,effective_bytes,result_check,next_block_input_cycle")
      try test(new Q1Mesh(design)).withAnnotations(TensorTestBackend.annotations) { dut =>
        dut.io.req.valid.poke(false.B)
        dut.io.in.valid.poke(false.B)
        dut.io.out.ready.poke(false.B)
        dut.clock.step()
        var cycle = 0
        def tick(): Unit = { dut.clock.step(); cycle += 1 }
        def runVector(v: Array[String], tag: Int, bubbles: Boolean = false, hold: Int = 0,
                      suffix: String = ""): Unit = {
          val k = v(1).toInt
          def bits(field: Int) = v(field).split(",").map(BigInt(_, 16))
          val a = bits(2); val b = bits(3); val c = bits(4); val d = bits(resultField)
          dut.io.req.ready.expect(true.B)
          dut.io.req.bits.k.poke(k.U)
          dut.io.req.bits.tag.poke(tag.U)
          for (r <- 0 until 2; col <- 0 until 2) dut.io.req.bits.c(r)(col).poke(c(r*2+col).U)
          dut.io.req.valid.poke(true.B)
          tick()
          dut.io.req.valid.poke(false.B)
          val start = cycle
          var sent = 0
          var nextBlockCycle = 0
          while (!dut.io.out.valid.peek().litToBoolean) {
            val sending = sent < k && (!bubbles || (cycle-start) % 5 != 0)
            dut.io.in.valid.poke(sending.B)
            if (sent < k) {
              for (r <- 0 until 2) dut.io.in.bits.a(r).poke(a(r*k+sent).U)
              for (col <- 0 until 2) dut.io.in.bits.b(col).poke(b(sent*2+col).U)
              if (sending && dut.io.in.ready.peek().litToBoolean) {
                sent += 1
                if (sent == 33) nextBlockCycle = cycle-start+1
              }
            }
            tick()
          }
          dut.io.in.valid.poke(false.B)
          assert(sent == k)
          dut.io.out.bits.tag.expect(tag.U)
          for (r <- 0 until 2; col <- 0 until 2) dut.io.out.bits.d(r)(col).expect(d(r*2+col).U)
          val elapsed = cycle - start
          if (design.endsWith("overlap") && k >= 64 && !bubbles) {
            assert(nextBlockCycle <= 34, "Next block was not collected while the first block replayed")
          }
          for (_ <- 0 until hold) {
            tick()
            dut.io.out.valid.expect(true.B)
            for (r <- 0 until 2; col <- 0 until 2) dut.io.out.bits.d(r)(col).expect(d(r*2+col).U)
          }
          csv.println(s"$design,${v(0)}$suffix,2,2,$k,core-only,$elapsed,$k,${elapsed-k},${4*k+32},passed,$nextBlockCycle")
          csv.flush()
          dut.io.out.ready.poke(true.B)
          tick()
          dut.io.out.ready.poke(false.B)
        }
        for ((v, tag) <- vectors.zipWithIndex) runVector(v, tag)
        if (design.endsWith("overlap")) {
          val v = vectors.find(_(1) == "65").get
          runVector(v, 1000, bubbles = true, hold = 4, suffix = "_bubbles")
          // Abort with block two partly collected while block one is replaying.
          dut.io.req.bits.k.poke(65.U)
          dut.io.req.valid.poke(true.B)
          tick()
          dut.io.req.valid.poke(false.B)
          dut.io.in.valid.poke(true.B)
          var sent = 0
          while (sent < 37) {
            if (dut.io.in.ready.peek().litToBoolean) sent += 1
            tick()
          }
          dut.reset.poke(true.B)
          tick()
          dut.reset.poke(false.B)
          dut.io.in.valid.poke(false.B)
          runVector(v, 1001, suffix = "_reset")
        }
      } finally csv.close()
    }
  }
}
