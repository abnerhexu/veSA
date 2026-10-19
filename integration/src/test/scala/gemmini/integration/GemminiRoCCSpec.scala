package gemmini.integration

import chisel3._
import chiseltest._
import freechips.rocketchip.diplomacy.LazyModule
import gemmini.tensorcore._
import org.chipsalliance.cde.config.Parameters
import org.scalatest.flatspec.AnyFlatSpec
import scala.io.Source

class GemminiRoCCSpec extends AnyFlatSpec with ChiselScalatestTester {
  val profiles = for (f <- Seq(TensorFormat.E4M3, TensorFormat.E5M2, TensorFormat.FP16, TensorFormat.BF16);
                     half <- Seq(false, true) if !(f == TensorFormat.BF16 && half))
    yield s"${f.name}-${if (half) "f16" else "f32"}" -> BlackwellConfig(f, half)
  for ((label, profile) <- profiles ++ Seq("mxfp4" -> BlackwellConfig.mxfp4, "nvfp4" -> BlackwellConfig.nvfp4)) {
    "Real Gemmini RoCC and DMA" should s"load fuse and store $label through TileLink" in {
      implicit val p: Parameters = IntegrationParameters()
      val source = Source.fromInputStream(getClass.getResourceAsStream(
        if (profile.grouped) "/grouped-matrices.tsv" else "/matrices.tsv"))
      val vectors = try source.getLines().filterNot(_.startsWith("#")).map(_.split(" ")).filter { v =>
        if (profile.grouped) v(0) == label
        else v(0) == profile.format.name && v(1) == (if (profile.halfOutput) "f16" else "f32")
      }.toVector finally source.close()
      assert(vectors.nonEmpty)
      test(LazyModule(new GemminiIntegrationHarness(profile)).module)
        .withAnnotations(TensorTestBackend.annotations) { dut =>
        dut.clock.setTimeout(300000)
        dut.io.cmd.valid.poke(false.B); dut.io.response.ready.poke(false.B)
        dut.io.memory.a.valid.poke(false.B); dut.io.memory.d.ready.poke(false.B)
        dut.clock.step(5)
        def waitFor(test: => Boolean, limit: Int = 50000): Unit = {
          var cycles = 0
          while (!test && cycles < limit) { dut.clock.step(); cycles += 1 }
          assert(cycles < limit, s"$label stalled after $cycles cycles")
          dut.io.interrupt.expect(false.B)
        }
        def memory(addr: Int, write: Option[BigInt]): BigInt = {
          val a = dut.io.memory.a
          a.bits.opcode.poke((if (write.nonEmpty) 0 else 4).U)
          a.bits.param.poke(0.U); a.bits.size.poke(3.U); a.bits.source.poke(0.U)
          a.bits.address.poke(addr.U); a.bits.mask.poke(255.U)
          a.bits.data.poke(write.getOrElse(BigInt(0)).U); a.bits.corrupt.poke(false.B)
          a.valid.poke(true.B); waitFor(a.ready.peek().litToBoolean)
          dut.clock.step(); a.valid.poke(false.B)
          val d = dut.io.memory.d
          waitFor(d.valid.peek().litToBoolean)
          d.bits.denied.expect(false.B); d.bits.corrupt.expect(false.B)
          d.bits.opcode.expect((if (write.nonEmpty) 0 else 1).U)
          val result = d.bits.data.peek().litValue
          d.ready.poke(true.B); dut.clock.step(); d.ready.poke(false.B)
          result
        }
        def writeWords(addr: Int, values: Seq[BigInt], width: Int): Unit = {
          val bytes = values.flatMap(v => (0 until width/8).map(i => (v >> (8*i)) & 255))
          for ((chunk, index) <- bytes.grouped(8).zipWithIndex) {
            val packed = chunk.zipWithIndex.map { case (b, i) => b << (8*i) }.sum
            memory(addr+index*8, Some(packed))
          }
        }
        def command(funct: Int, rs1: BigInt, rs2: BigInt, xd: Boolean = false, rd: Int = 17): Unit = {
          dut.io.cmd.bits.funct.poke(funct.U); dut.io.cmd.bits.rs1.poke(rs1.U); dut.io.cmd.bits.rs2.poke(rs2.U)
          dut.io.cmd.bits.rd.poke(rd.U); dut.io.cmd.bits.xd.poke(xd.B)
          dut.io.cmd.valid.poke(true.B); waitFor(dut.io.cmd.ready.peek().litToBoolean)
          dut.clock.step(); dut.io.cmd.valid.poke(false.B)
        }
        def reply(expected: BigInt): Unit = {
          waitFor(dut.io.response.valid.peek().litToBoolean)
          dut.io.response.bits.rd.expect(17.U); dut.io.response.bits.data.expect(expected.U)
          dut.io.response.ready.poke(true.B); dut.clock.step(); dut.io.response.ready.poke(false.B)
        }
        def descriptor(funct: Int, a: Int, b: Int): Unit = { command(funct, a, b, true); reply(0) }
        def load(system: Int, local: BigInt, rows: Int, rowBytes: Int): Unit = {
          // CONFIG_LOAD state 0, no shrinking, pixel repeat 1, block stride 2.
          command(0, BigInt(1) | (BigInt(1) << 8) | (BigInt(2) << 16), rowBytes)
          for (first <- 0 until rows by 2) {
            val n = math.min(2, rows-first)
            command(2, system+first*rowBytes,
              (BigInt(n) << 48) | (BigInt(2) << 32) | (local+first))
          }
        }
        val sentinel = BigInt("5a5a5a5a5a5a5a5a", 16)
        for ((v, index) <- vectors.zipWithIndex) {
          val shift = if (profile.grouped) 0 else 1
          def bits(i: Int) = v(i+shift).split(",").map(BigInt(_,16)).toVector
          val k = v(1+shift).toInt
          val a = bits(2); val b = bits(3); val c = bits(4)
          val d = bits(if (profile.grouped) 7 else 5)
          val slot = profile.format.width max 8
          val dirty = if (profile.grouped) BigInt(0xa0) else BigInt(0)
          writeWords(0x1000, (0 until k).flatMap(t => (0 until 2).map(r => a(r*k+t) | dirty)), slot)
          writeWords(0x2000, b.map(_ | dirty), slot)
          val dirtyHalf = if (profile.halfOutput) BigInt("dead0000",16) else BigInt(0)
          writeWords(0x3000, c.map(_ | dirtyHalf), 32)
          for (offset <- -8 to 16 by 8) memory(0x3800+offset, Some(sentinel))
          val cBase = 15
          val dBase = if (index%2 == 0) cBase else 4
          descriptor(TensorCommandISA.AddrAB, 4, 253)
          descriptor(TensorCommandISA.AddrCD, cBase, dBase)
          if (profile.grouped) descriptor(TensorCommandISA.Scales, 768, 800)
          // No idle wait between legacy DMA and RUN: the real frontend must
          // drain unrollers, ROB, controllers, scalers and pending bank writes.
          load(0x1000, 4, k, 2*slot/8)
          load(0x2000, 253, k, 2*slot/8)
          load(0x3000, (BigInt(1) << 31) | cBase, 2, 8)
          if (profile.grouped) {
            val blocks = k/profile.scaleBlockSize
            val alpha = bits(5); val beta = bits(6)
            writeWords(0x4000, (0 until blocks).flatMap(s => (0 until 2).map(r => alpha(r*blocks+s))), 8)
            writeWords(0x4800, beta, 8)
            load(0x4000, 768, blocks, 2); load(0x4800, 800, blocks, 2)
          }
          val tag = index+300
          command(TensorCommandISA.Run, (BigInt(tag) << 16) | k, 0, true)
          // Queue a real MVOUT behind RUN. It must not touch system memory
          // until the CPU accepts the Tensor response, even after compute ends.
          command(0, 2, 8) // CONFIG_STORE, no activation, full-width stride 8.
          command(3, 0x3800, (BigInt(2) << 48) | (BigInt(2) << 32) |
            (BigInt(1) << 31) | (BigInt(1) << 29) | dBase)
          waitFor(dut.io.response.valid.peek().litToBoolean)
          dut.io.response.bits.data.expect((BigInt(tag) << 16).U)
          dut.clock.step(7)
          assert(memory(0x3800, None) == sentinel, "MVOUT crossed a backpressured Tensor fence")
          reply(BigInt(tag) << 16)
          waitFor(!dut.io.busy.peek().litToBoolean)
          for (r <- 0 until 2) {
            val expected = d(r*2) | (d(r*2+1) << 32)
            assert(memory(0x3800+r*8, None) == expected, s"$label K=$k row=$r differs from paper model")
          }
          assert(memory(0x37f8, None) == sentinel && memory(0x3810, None) == sentinel,
            "DMA store corrupted neighboring system-memory words")
          info(s"Verified real RoCC/DMA $label K=$k tag=$tag")
        }
        if (label == "e4m3-f32") {
          // A blocked legacy counter reply must drain before the next Tensor
          // command can take ownership; exercise the real response arbiter.
          command(126, 1, 0, true, rd = 7)
          command(TensorCommandISA.AddrAB, 4, 253, true)
          waitFor(dut.io.response.valid.peek().litToBoolean)
          dut.io.response.bits.rd.expect(7.U)
          dut.clock.step(8); dut.io.response.bits.rd.expect(7.U)
          dut.io.busy.expect(true.B)
          dut.io.response.ready.poke(true.B); dut.clock.step(); dut.io.response.ready.poke(false.B)
          reply(0)
          info("Verified real counter/Tensor response arbitration under backpressure")

          val before = Seq(memory(0x3800, None), memory(0x3808, None))
          descriptor(TensorCommandISA.AddrCD, 15, 31)
          command(TensorCommandISA.Run, (BigInt(0xbeef) << 16) | 1, 0, true)
          reply((BigInt(0xbeef) << 16) | TensorCommandISA.BadRange)
          command(0, 2, 8)
          command(3, 0x3800, (BigInt(2) << 48) | (BigInt(2) << 32) |
            (BigInt(1) << 31) | (BigInt(1) << 29) | 4)
          waitFor(!dut.io.busy.peek().litToBoolean)
          assert(Seq(memory(0x3800, None), memory(0x3808, None)) == before,
            "Invalid Tensor request modified a previous D tile")
          info("Verified malformed real RoCC RUN has no local-memory side effect")

          def zeroTask(tag: Int, legacyExecute: Boolean = false): Unit = {
            descriptor(TensorCommandISA.AddrAB, 4, 253)
            command(TensorCommandISA.AddrCD, BigInt("ffffffff",16), 4, true); reply(0)
            // Address zero is the existing legacy zero-fill operation: this
            // exercises the actual ZeroWriter and its drain scoreboard too.
            load(0, 4, 2, 2); load(0, 253, 2, 2)
            if (legacyExecute) {
              writeWords(0x5000, Seq.fill(4)(BigInt("41414141",16)), 32)
              load(0x5000, (BigInt(1) << 31) | 8, 2, 8)
              // Original output-stationary integer mesh: zero A/B, zero C,
              // overwrite a sentinel tile at acc8. RUN follows without idle.
              command(0, BigInt(1) << 16, BigInt(1) << 48)
              val shape = (BigInt(2) << 48) | (BigInt(2) << 32)
              command(6, shape | BigInt("ffffffff",16), shape | (BigInt(1) << 31) | 8)
              command(4, shape | 4, shape | 253)
            }
            command(TensorCommandISA.Run, (BigInt(1) << 32) | (BigInt(tag) << 16) | 1, 0, true)
            command(0, 2, 8)
            command(3, 0x3800, (BigInt(2) << 48) | (BigInt(2) << 32) |
              (BigInt(1) << 31) | (BigInt(1) << 29) | 4)
            reply(BigInt(tag) << 16)
            waitFor(!dut.io.busy.peek().litToBoolean)
            assert(memory(0x3800, None) == 0 && memory(0x3808, None) == 0)
            if (legacyExecute) {
              command(3, 0x5000, (BigInt(2) << 48) | (BigInt(2) << 32) |
                (BigInt(1) << 31) | (BigInt(1) << 29) | 8)
              waitFor(!dut.io.busy.peek().litToBoolean)
              assert(memory(0x5000, None) == 0 && memory(0x5008, None) == 0,
                "Older legacy OS execution did not overwrite its sentinel tile")
            }
          }
          zeroTask(0xcece, legacyExecute = true)
          info("Verified zero-C, DMA zero-fill and older real legacy OS execution drain")

          command(TensorCommandISA.Run, (BigInt(1) << 32) | (BigInt(0xface) << 16) | 65, 0, true)
          dut.clock.step(12); dut.io.busy.expect(true.B)
          dut.reset.poke(true.B); dut.clock.step(5); dut.reset.poke(false.B); dut.clock.step(8)
          dut.io.response.valid.expect(false.B)
          waitFor(!dut.io.busy.peek().litToBoolean)
          command(TensorCommandISA.Run, (BigInt(0xabab) << 16) | 1, 0, true)
          reply((BigInt(0xabab) << 16) | TensorCommandISA.MissingDescriptor)
          zeroTask(0xdede)
          info("Verified in-flight full-system reset clears commands, owners and descriptors and can restart")
        }
      }
    }
  }
}
