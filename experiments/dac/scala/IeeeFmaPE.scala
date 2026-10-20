package gemmini.dac

import chisel3._
import chisel3.util._
import gemmini.tensorcore._
import hardfloat._

/** E4M3 inputs, exact conversion, and FP32 RNE FMA after every product.
  * One combinational FMA per PE. Cycle throughput does not imply a verified clock.
  */
class IeeeFmaPE extends Module {
  private val config = BlackwellConfig()
  val io = IO(new Bundle {
    val req = Flipped(Decoupled(new TensorRequest(config)))
    val in = Flipped(Decoupled(new TensorOperands(config)))
    val out = Decoupled(UInt(32.W))
  })

  private def fp32(bits: UInt): UInt = {
    val exp = bits(6, 3)
    val frac = bits(2, 0)
    val sub = MuxLookup(frac, 0.U(31.W))(Seq(
      1.U -> "h3b000000".U, 2.U -> "h3b800000".U,
      3.U -> "h3bc00000".U, 4.U -> "h3c000000".U,
      5.U -> "h3c200000".U, 6.U -> "h3c400000".U,
      7.U -> "h3c600000".U))
    val normal = Cat((exp +& 120.U)(7, 0), frac, 0.U(20.W))
    Mux(bits(6, 0) === 127.U, "h7fc00000".U,
      Cat(bits(7), Mux(exp === 0.U, sub, normal)))
  }

  private val idle :: active :: respond :: Nil = Enum(3)
  private val state = RegInit(idle)
  private val left = Reg(UInt(6.W))
  private val acc = Reg(UInt(32.W))
  private val fma = Module(new MulAddRecFN(8, 24))
  fma.io.op := 0.U
  fma.io.a := recFNFromFN(8, 24, fp32(io.in.bits.a))
  fma.io.b := recFNFromFN(8, 24, fp32(io.in.bits.b))
  fma.io.c := recFNFromFN(8, 24, acc)
  fma.io.roundingMode := consts.round_near_even
  fma.io.detectTininess := consts.tininess_afterRounding
  private val result = fNFromRecFN(8, 24, fma.io.out)
  private val nan = result(30, 23).andR && result(22, 0).orR

  io.req.ready := state === idle
  io.in.ready := state === active
  io.out.valid := state === respond
  io.out.bits := acc
  when(io.req.fire) {
    acc := io.req.bits.c
    left := io.req.bits.length
    state := active
  }
  when(io.in.fire) {
    acc := Mux(nan, "h7fc00000".U, result)
    left := left - 1.U
    when(left === 1.U) { state := respond }
  }
  when(io.out.fire) { state := idle }
}
