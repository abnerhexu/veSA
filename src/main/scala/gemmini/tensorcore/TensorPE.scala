package gemmini.tensorcore

import chisel3._
import chisel3.util._

class TensorRequest(config: TensorConfig) extends Bundle {
  val c = UInt(config.outputWidth.W)
  val length = UInt(log2Ceil(config.fusionLength + 1).W)
}

class TensorOperands(config: TensorConfig) extends Bundle {
  val a = UInt(config.format.width.W)
  val b = UInt(config.format.width.W)
  val alpha = if (config.grouped) Some(UInt(8.W)) else None
  val beta = if (config.grouped) Some(UInt(8.W)) else None
}

/** Low-precision front end to the shared exact fused accumulator. */
class TensorPE(val config: TensorConfig = BlackwellConfig()) extends Module {
  val io = IO(new Bundle {
    val req = Flipped(Decoupled(new TensorRequest(config)))
    val in = Flipped(Decoupled(new TensorOperands(config)))
    val out = Decoupled(UInt(config.outputWidth.W))
  })
  private val accumulator = Module(new FusedAccumulator(config,
    if (config.grouped) 4 else config.fusionLength))
  accumulator.io.req.valid := io.req.valid
  accumulator.io.req.bits.c := io.req.bits.c
  accumulator.io.req.bits.length := (if (config.grouped) 4.U else io.req.bits.length)
  io.req.ready := accumulator.io.req.ready
  io.out <> accumulator.io.out

  if (!config.grouped) {
    accumulator.io.in.valid := io.in.valid
    accumulator.io.in.bits := TensorArithmetic.product(io.in.bits.a, io.in.bits.b,
      config.format, config.zeroExponent)
    io.in.ready := accumulator.io.in.ready
  } else {
    val frontend = Module(new GroupedDotFrontend(config))
    frontend.io.start := io.req.fire
    frontend.io.in <> io.in
    accumulator.io.in <> frontend.io.out
    when(io.req.fire) { assert(io.req.bits.length === 64.U, "GST-FDPA requires 64 elements") }
  }
}
