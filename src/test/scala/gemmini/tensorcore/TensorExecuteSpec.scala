package gemmini.tensorcore

import chisel3._
import chisel3.util._
import chiseltest._
import gemmini.{AccumulatorBankRouter, AccumulatorMemCore, MemoryActivity, ScratchpadBank, ScratchpadReadIO, ScratchpadWriteIO}
import org.scalatest.flatspec.AnyFlatSpec
import scala.io.Source

/** The baseline uses ScratchpadBank for both spaces. realAccumulator selects
  * the original production AccumulatorMemCore, via the same bank ownership
  * router and raw bridge used by the optional Scratchpad integration path.
  * Neither mode claims full Controller/DMA/normalizer integration.
  */
class TensorMemoryHarness(config: BlackwellConfig, mem: TensorMemoryConfig,
                          realAccumulator: Boolean = false, singlePorted: Boolean = false,
                          fencedCommands: Boolean = false) extends Module {
  val io = IO(new Bundle {
    val cmd = Flipped(Decoupled(new TensorExecuteRequest(config)))
    val completed = Decoupled(UInt(16.W))
    val busy = Output(Bool())
    val memoryIdle = Output(Bool())
    val frontend = if (fencedCommands) Some(new Bundle {
      val cmd = Flipped(Decoupled(new TensorCommand()))
      val response = Decoupled(new TensorCommandResponse())
      val legacy = Decoupled(new TensorCommand())
      val olderBusy = Input(Bool())
      val commitBlocked = Input(Bool())
      val exclusive = Output(Bool())
    }) else None
    val stress = Input(Bool())
    val spWrite = Vec(mem.spBanks, Flipped(new ScratchpadWriteIO(mem.spDepth, mem.spWidth(config), mem.spWidth(config)/8)))
    val accWrite = Vec(mem.accBanks, Flipped(new ScratchpadWriteIO(mem.accDepth, mem.accWidth, mem.accWidth/8)))
    val accRead = Vec(mem.accBanks, Flipped(new ScratchpadReadIO(mem.accDepth, mem.accWidth)))
    val reads = Output(UInt(32.W))
    val writes = Output(UInt(32.W))
  })
  private val engine = Module(new TensorExecute(config, mem))
  private val frontend = if (fencedCommands) Some(Module(new TensorCommandFrontend(config, mem))) else None
  private val idleChecks = scala.collection.mutable.ArrayBuffer.empty[Bool]
  private val hostAllowed = !frontend.map(_.io.exclusive).getOrElse(false.B)
  frontend match {
    case Some(front) =>
      val port = io.frontend.get
      val commands = Module(new Queue(new TensorCommand(), 2))
      val gate = Module(new TensorCommandGate(new TensorCommand(), (c: TensorCommand) => c.funct))
      commands.io.enq <> port.cmd
      gate.io.in <> commands.io.deq
      gate.io.hold := front.io.busy
      front.io.cmd <> gate.io.tensor
      port.legacy <> gate.io.legacy
      port.response <> front.io.response
      front.io.legacyIdle := !port.olderBusy
      front.io.memoryIdle := io.memoryIdle && !port.commitBlocked
      port.exclusive := front.io.exclusive
      engine.io.cmd <> front.io.execute
      front.io.completed <> engine.io.completed
      io.cmd.ready := false.B
      io.completed.valid := false.B
      io.completed.bits := 0.U
      io.busy := front.io.busy || commands.io.deq.valid
    case None =>
      engine.io.cmd <> io.cmd
      io.completed <> engine.io.completed
      io.busy := engine.io.busy
  }
  private val cycle = RegInit(0.U(8.W))
  cycle := cycle + 1.U
  private val requestGate = !io.stress || cycle(1, 0) =/= 0.U
  private val responseGate = !io.stress || cycle(2, 0) >= 3.U
  private val writeGate = !io.stress || cycle(2, 0) === 6.U
  private val reads = RegInit(0.U(32.W))
  private val writes = RegInit(0.U(32.W))
  io.reads := reads
  io.writes := writes
  when(engine.io.cmd.fire) { reads := 0.U; writes := 0.U }
  when(engine.io.spad.map(_.req.fire).reduce(_ || _) || engine.io.accRead.map(_.req.fire).reduce(_ || _)) {
    reads := reads + 1.U
  }
  when(engine.io.accWrite.map(_.fire).reduce(_ || _)) { writes := writes + 1.U }
  for (i <- 0 until mem.spBanks) {
    val bank = Module(new ScratchpadBank(mem.spDepth, mem.spWidth(config), 1, true, false, false))
    val port = engine.io.spad(i)
    bank.io.read.req.valid := port.req.valid && requestGate
    bank.io.read.req.bits := port.req.bits
    port.req.ready := bank.io.read.req.ready && requestGate
    port.resp.valid := bank.io.read.resp.valid && responseGate
    port.resp.bits := bank.io.read.resp.bits
    bank.io.read.resp.ready := port.resp.ready && responseGate
    bank.io.write.valid := io.spWrite(i).valid && hostAllowed
    bank.io.write.addr := io.spWrite(i).addr
    bank.io.write.data := io.spWrite(i).data
    bank.io.write.mask := io.spWrite(i).mask
    io.spWrite(i).ready := bank.io.write.ready && hostAllowed
    val activity = Module(new MemoryActivity())
    activity.io.start := bank.io.read.req.fire
    activity.io.finish := bank.io.read.resp.fire
    idleChecks += (activity.io.idle && !bank.io.write.valid)
    when(engine.io.busy) { assert(!bank.io.write.valid, "Host wrote operands while busy") }
  }
  for (i <- 0 until mem.accBanks) {
    val (bankRead, bankWrite) = if (realAccumulator) {
      val rowType = Vec(mem.dim, Vec(1, UInt(32.W)))
      val bank = Module(new AccumulatorMemCore(mem.accDepth, rowType,
        (x: UInt, _: UInt) => x, UInt(32.W), singlePorted, if (singlePorted) 2 else -1,
        false, false, 2, UInt(32.W), false))
      val bridge = Module(new TensorAccumulatorBridge(mem, rowType, UInt(32.W)))
      val router = Module(new AccumulatorBankRouter(mem.accDepth, rowType, UInt(32.W)))
      bank.io.read <> router.io.bank.read
      bank.io.write <> router.io.bank.write
      bridge.io.bank <> router.io.raw
      router.io.legacy.read.req.valid := false.B
      router.io.legacy.read.req.bits := 0.U.asTypeOf(router.io.legacy.read.req.bits)
      router.io.legacy.read.resp.ready := false.B
      router.io.legacy.write.valid := false.B
      router.io.legacy.write.bits := 0.U.asTypeOf(router.io.legacy.write.bits)
      // A deliberately wrong adder output detects accidental use of the adder.
      bank.io.adder.sum := Fill(mem.accWidth, 1.U).asTypeOf(rowType)
      idleChecks += !bank.io.busy
      assert(!bank.io.adder.valid, "Tensor/host overwrite unexpectedly activated the shared adder")
      val write = Wire(new ScratchpadWriteIO(mem.accDepth, mem.accWidth, mem.accWidth/8))
      bridge.io.write.valid := write.valid
      bridge.io.write.bits.addr := write.addr
      bridge.io.write.bits.data := write.data
      bridge.io.write.bits.mask := write.mask
      bridge.io.write.bits.accumulate := false.B
      write.ready := bridge.io.write.ready
      (bridge.io.read, write)
    } else {
      val bank = Module(new ScratchpadBank(mem.accDepth, mem.accWidth, 1, true, false, false))
      (bank.io.read, bank.io.write)
    }
    val port = engine.io.accRead(i)
    bankRead.req.valid := Mux(engine.io.busy, port.req.valid && requestGate, io.accRead(i).req.valid && hostAllowed)
    bankRead.req.bits := Mux(engine.io.busy, port.req.bits, io.accRead(i).req.bits)
    port.req.ready := engine.io.busy && bankRead.req.ready && requestGate
    io.accRead(i).req.ready := !engine.io.busy && bankRead.req.ready && hostAllowed
    port.resp.valid := engine.io.busy && bankRead.resp.valid && responseGate
    port.resp.bits := bankRead.resp.bits
    io.accRead(i).resp.valid := !engine.io.busy && bankRead.resp.valid && hostAllowed
    io.accRead(i).resp.bits := bankRead.resp.bits
    bankRead.resp.ready := Mux(engine.io.busy, port.resp.ready && responseGate, io.accRead(i).resp.ready && hostAllowed)
    val write = engine.io.accWrite(i)
    bankWrite.valid := Mux(engine.io.busy, write.valid && writeGate, io.accWrite(i).valid && hostAllowed)
    bankWrite.addr := Mux(engine.io.busy, write.bits.addr, io.accWrite(i).addr)
    bankWrite.data := Mux(engine.io.busy, write.bits.data, io.accWrite(i).data)
    bankWrite.mask := Mux(engine.io.busy, write.bits.mask, io.accWrite(i).mask)
    write.ready := engine.io.busy && bankWrite.ready && writeGate
    io.accWrite(i).ready := !engine.io.busy && bankWrite.ready && hostAllowed
    val activity = Module(new MemoryActivity())
    activity.io.start := bankRead.req.fire
    activity.io.finish := bankRead.resp.fire
    idleChecks += (activity.io.idle && !bankWrite.valid)
    when(write.valid) {
      assert(!write.bits.accumulate, "Store-side add would double-count fused C")
      assert(write.bits.mask.asUInt.andR)
    }
    if (!fencedCommands) {
      when(engine.io.busy) { assert(!io.accWrite(i).valid && !io.accRead(i).req.valid) }
    }
  }
  io.memoryIdle := idleChecks.reduce(_ && _)
}

abstract class TensorExecuteChecks(realAccumulator: Boolean, singlePorted: Boolean = false,
                                   fencedCommands: Boolean = false)
  extends AnyFlatSpec with ChiselScalatestTester {
  val unscaled = for (f <- Seq(TensorFormat.E4M3, TensorFormat.E5M2, TensorFormat.FP16, TensorFormat.BF16);
                     half <- Seq(false, true) if !(f == TensorFormat.BF16 && half))
    yield s"${f.name}-${if (half) "f16" else "f32"}" -> BlackwellConfig(f, half)
  for ((label, config) <- unscaled ++ Seq("mxfp4" -> BlackwellConfig.mxfp4, "nvfp4" -> BlackwellConfig.nvfp4)) {
    val backend = if (fencedCommands) "fenced command" else if (realAccumulator) {
      if (singlePorted) "single-port accumulator" else "real accumulator"
    } else "local-memory"
    s"Tensor $backend execution" should s"load fuse and overwrite $label without conversion" in {
      val file = if (config.grouped) "/grouped-matrices.tsv" else "/matrices.tsv"
      val source = Source.fromInputStream(getClass.getResourceAsStream(file))
      val vectors = try source.getLines().filterNot(_.startsWith("#")).map(_.split(" ")).filter { v =>
        if (config.grouped) v(0) == label
        else v(0) == config.format.name && v(1) == (if (config.halfOutput) "f16" else "f32")
      }.toVector finally source.close()
      // All existing matrices: boundary K, finite cases and arbitrary raw bits.
      assert(vectors.nonEmpty)
      // Fenced completion also checks the actual bank queues. Deliberately do
      // not compensate for single-port queues with a longer engine delay.
      val mem = TensorMemoryConfig(writeLatency = if (singlePorted && !fencedCommands) 4 else 2)
      test(new TensorMemoryHarness(config, mem, realAccumulator, singlePorted, fencedCommands))
        .withAnnotations(TensorTestBackend.annotations) { dut =>
        // A complete memory-driven K=128/192 operation intentionally takes
        // more than Chiseltest's default 1000 idle cycles without host pokes.
        // The per-request bounded waits below still detect real deadlocks.
        dut.clock.setTimeout(12000)
        dut.io.cmd.valid.poke(false.B)
        dut.io.completed.ready.poke(false.B)
        dut.io.stress.poke(true.B)
        dut.io.frontend.foreach { p =>
          p.cmd.valid.poke(false.B); p.response.ready.poke(false.B)
          p.legacy.ready.poke(true.B)
          p.olderBusy.poke(false.B); p.commitBlocked.poke(false.B)
        }
        for (p <- dut.io.spWrite) { p.valid.poke(false.B); p.mask.foreach(_.poke(true.B)) }
        for (p <- dut.io.accWrite) { p.valid.poke(false.B); p.mask.foreach(_.poke(true.B)) }
        for (p <- dut.io.accRead) { p.req.valid.poke(false.B); p.resp.ready.poke(false.B) }
        dut.clock.step()
        def command(funct: Int, rs1: BigInt, rs2: BigInt): Unit = {
          val p = dut.io.frontend.get
          p.cmd.bits.funct.poke(funct.U); p.cmd.bits.rs1.poke(rs1.U); p.cmd.bits.rs2.poke(rs2.U)
          p.cmd.bits.rd.poke(13.U); p.cmd.bits.xd.poke(true.B)
          p.cmd.valid.poke(true.B); p.cmd.ready.expect(true.B)
          dut.clock.step(); p.cmd.valid.poke(false.B)
        }
        def response(expected: BigInt): Unit = {
          val p = dut.io.frontend.get
          var waited = 0
          while (!p.response.valid.peek().litToBoolean && waited < 9000) { dut.clock.step(); waited += 1 }
          assert(waited < 9000, "Fenced command failed to complete")
          p.response.bits.rd.expect(13.U); p.response.bits.data.expect(expected.U)
          p.exclusive.expect(true.B); dut.io.memoryIdle.expect(true.B)
          dut.clock.step(5)
          // The outer queue may buffer younger commands; none may dispatch.
          p.response.bits.data.expect(expected.U); p.legacy.valid.expect(false.B)
          p.response.ready.poke(true.B); dut.clock.step(); p.response.ready.poke(false.B)
          dut.io.busy.expect(false.B)
        }
        def start(a: Int, b: Int, c: Int, d: Int, k: Int, tag: Int, zeroC: Boolean): Unit = {
          if (fencedCommands) {
            command(TensorCommandISA.AddrAB, a, b); response(0)
            command(TensorCommandISA.AddrCD, BigInt(c & 0xffffffffL), d); response(0)
            if (config.grouped) {
              command(TensorCommandISA.Scales, dut.io.cmd.bits.alpha.get.peek().litValue,
                dut.io.cmd.bits.beta.get.peek().litValue); response(0)
            }
            command(TensorCommandISA.Run, BigInt(k) | (BigInt(tag) << 16) |
              (if (zeroC) BigInt(1) << 32 else BigInt(0)), 0)
          } else {
            dut.io.cmd.valid.poke(true.B); dut.clock.step(); dut.io.cmd.valid.poke(false.B)
          }
        }
        def packed(xs: Seq[BigInt], width: Int): BigInt =
          xs.zipWithIndex.map { case (x, i) => x << (i*width) }.sum
        def writeSp(addr: Int, data: BigInt): Unit = {
          val p = dut.io.spWrite(addr/mem.spDepth)
          p.addr.poke((addr%mem.spDepth).U); p.data.poke(data.U); p.valid.poke(true.B)
          p.ready.expect(true.B); dut.clock.step(); p.valid.poke(false.B)
        }
        def writeAcc(addr: Int, data: BigInt): Unit = {
          val p = dut.io.accWrite(addr/mem.accDepth)
          p.addr.poke((addr%mem.accDepth).U); p.data.poke(data.U); p.valid.poke(true.B)
          p.ready.expect(true.B); dut.clock.step(); p.valid.poke(false.B)
        }
        def readAcc(addr: Int): BigInt = {
          val p = dut.io.accRead(addr/mem.accDepth)
          p.req.bits.addr.poke((addr%mem.accDepth).U); p.req.bits.fromDMA.poke(false.B)
          p.req.valid.poke(true.B)
          var wait = 0
          while (!p.req.ready.peek().litToBoolean && wait < 20) { dut.clock.step(); wait += 1 }
          assert(wait < 20); dut.clock.step(); p.req.valid.poke(false.B)
          wait = 0
          while (!p.resp.valid.peek().litToBoolean && wait < 20) { dut.clock.step(); wait += 1 }
          assert(wait < 20)
          val result = p.resp.bits.data.peek().litValue
          p.resp.ready.poke(true.B); dut.clock.step(); p.resp.ready.poke(false.B)
          result
        }
        for ((v, tag) <- vectors.zipWithIndex) {
          val shift = if (config.grouped) 0 else 1
          def bits(i: Int) = v(i+shift).split(",").map(BigInt(_, 16)).toVector
          val k = v(1+shift).toInt
          val a = bits(2); val b = bits(3); val c = bits(4)
          val d = bits(if (config.grouped) 7 else 5)
          // A and B begin in the same physical bank; B crosses a bank boundary.
          val aBase = 4
          val bBase = mem.spDepth-3
          val alphaBase = 3*mem.spDepth
          val betaBase = alphaBase+32
          val cBase = mem.accDepth-1 // C crosses the accumulator-bank boundary.
          val dBase = if (tag%2 == 0) cBase else 4 // Exercise in-place C/D too.
          for (t <- 0 until k) {
            // Dirty upper FP4 storage nibbles must not enter arithmetic.
            val dirty = if (config.grouped) BigInt(0xa0) else BigInt(0)
            writeSp(aBase+t, packed((0 until 2).map(r => a(r*k+t) | dirty), mem.slotWidth(config)))
            writeSp(bBase+t, packed((0 until 2).map(col => b(t*2+col) | dirty), mem.slotWidth(config)))
          }
          if (config.grouped) {
            val alpha = bits(5); val beta = bits(6)
            val blocks = k/config.scaleBlockSize
            for (s <- 0 until blocks) {
              writeSp(alphaBase+s, packed((0 until 2).map(r => alpha(r*blocks+s)), 8))
              writeSp(betaBase+s, packed((0 until 2).map(col => beta(s*2+col)), 8))
            }
          }
          val sentinel = BigInt("89abcdef01234567", 16)
          writeAcc(dBase-1, sentinel); writeAcc(dBase+2, sentinel)
          if (dBase != cBase) for (r <- 0 until 2) writeAcc(dBase+r, sentinel)
          for (r <- 0 until 2) {
            // Upper half-word is deliberately dirty in FP16-output storage.
            val dirty = if (config.halfOutput) BigInt("dead0000", 16) else BigInt(0)
            writeAcc(cBase+r, packed((0 until 2).map(col => c(r*2+col) | dirty), 32))
          }
          if (!fencedCommands) dut.io.cmd.ready.expect(true.B)
          dut.io.cmd.bits.a.poke(aBase.U); dut.io.cmd.bits.b.poke(bBase.U)
          dut.io.cmd.bits.c.poke(cBase.U); dut.io.cmd.bits.d.poke(dBase.U)
          dut.io.cmd.bits.k.poke(k.U); dut.io.cmd.bits.tag.poke((tag+200).U)
          dut.io.cmd.bits.zeroC.poke(false.B)
          if (config.grouped) {
            dut.io.cmd.bits.alpha.get.poke(alphaBase.U); dut.io.cmd.bits.beta.get.poke(betaBase.U)
          }
          start(aBase, bBase, cBase, dBase, k, tag+200, false)
          if (fencedCommands) { response(BigInt(tag+200) << 16) } else {
            var waited = 0
            while (!dut.io.completed.valid.peek().litToBoolean && waited < k*40+500) {
              dut.io.cmd.ready.expect(false.B); dut.clock.step(); waited += 1
            }
            assert(waited < k*40+500, s"$label K=$k deadlock")
            dut.io.completed.bits.expect((tag+200).U)
            dut.clock.step(5)
            dut.io.completed.valid.expect(true.B)
            dut.io.completed.bits.expect((tag+200).U)
            dut.io.completed.ready.poke(true.B); dut.clock.step(); dut.io.completed.ready.poke(false.B)
            dut.io.busy.expect(false.B)
          }
          dut.io.writes.expect(2.U)
          val expectedScales = if (config.grouped) 2*k/config.scaleBlockSize else 0
          dut.io.reads.expect((2+2*k+expectedScales).U)
          for (r <- 0 until 2) assert(readAcc(dBase+r) == packed(d.slice(r*2,r*2+2), 32), s"$label K=$k row=$r")
          assert(readAcc(dBase-1) == sentinel && readAcc(dBase+2) == sentinel, "Output write corrupted adjacent rows")
          info(s"Verified $label K=$k tag=${tag+200}")
        }
        // Zero-C path must not issue any accumulator read, even with invalid C.
        val k = if (config.grouped) 64 else 1
        for (t <- 0 until k) { writeSp(4+t, 0); writeSp(256+t, 0) }
        if (config.grouped) {
          for (s <- 0 until k/config.scaleBlockSize) { writeSp(768+s, 0); writeSp(800+s, 0) }
          dut.io.cmd.bits.alpha.get.poke(768.U); dut.io.cmd.bits.beta.get.poke(800.U)
        }
        dut.io.cmd.bits.a.poke(4.U); dut.io.cmd.bits.b.poke(256.U)
        dut.io.cmd.bits.c.poke("hffffffff".U); dut.io.cmd.bits.d.poke(4.U)
        dut.io.cmd.bits.k.poke(k.U); dut.io.cmd.bits.tag.poke(65535.U); dut.io.cmd.bits.zeroC.poke(true.B)
        start(4, 256, -1, 4, k, 65535, true)
        if (fencedCommands) { response(BigInt(65535) << 16) } else {
          var waited = 0
          while (!dut.io.completed.valid.peek().litToBoolean && waited < 4000) { dut.clock.step(); waited += 1 }
          assert(waited < 4000)
          dut.io.completed.bits.expect(65535.U)
          dut.io.completed.ready.poke(true.B); dut.clock.step(); dut.io.completed.ready.poke(false.B)
        }
        dut.io.reads.expect((2*k+(if (config.grouped) 2*k/config.scaleBlockSize else 0)).U)
        assert(readAcc(4) == 0 && readAcc(5) == 0)
      }
    }
  }
}

class TensorExecuteSpec extends TensorExecuteChecks(false)
class TensorAccumulatorExecuteSpec extends TensorExecuteChecks(true)
class TensorAccumulatorSinglePortSpec extends TensorExecuteChecks(true, true)
class TensorCommandExecuteSpec extends TensorExecuteChecks(true, false, true)
class TensorCommandSinglePortSpec extends TensorExecuteChecks(true, true, true)
