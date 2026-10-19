package gemmini

import chisel3._
import chisel3.util._

import Util._

class AccPipe[T <: Data : Arithmetic](latency: Int, t: T)(implicit ev: Arithmetic[T]) extends Module {
  val io = IO(new Bundle {
    val op1 = Input(t.cloneType)
    val op2 = Input(t.cloneType)
    val sum = Output(t.cloneType)
  })
  import ev._
  io.sum := ShiftRegister(io.op1 + io.op2, latency)
}

class AccPipeShared[T <: Data : Arithmetic](latency: Int, t: Vec[Vec[T]], banks: Int) extends Module {
  val io = IO(new Bundle {
    val in_sel = Input(Vec(banks, Bool()))
    val ina = Input(Vec(banks, t.cloneType))
    val inb = Input(Vec(banks, t.cloneType))
    val out = Output(t.cloneType)
  })
  val ina = Mux1H(io.in_sel, io.ina)
  val inb = Mux1H(io.in_sel, io.inb)
  io.out := VecInit((ina zip inb).map { case (rv, wv) =>
    VecInit((rv zip wv).map { case (re, we) =>
      val m = Module(new AccPipe(latency, t.head.head.cloneType))
      m.io.op1 := re
      m.io.op2 := we
      m.io.sum
    })
  })
}

/** Compatibility wrapper: storage lives in the independently tested core.
  * The original implicit arithmetic API is retained for existing Gemmini users.
  * Arithmetic is only used by the external shared adder, not by SRAM storage.
  */
class AccumulatorMem[T <: Data, U <: Data](
  n: Int, t: Vec[Vec[T]], scale_func: (T, U) => T, scale_t: U,
  acc_singleported: Boolean, acc_sub_banks: Int,
  use_shared_ext_mem: Boolean, use_tl_ext_ram: Boolean,
  acc_latency: Int, acc_type: T, is_dummy: Boolean
)(implicit ev: Arithmetic[T]) extends AccumulatorMemCore[T, U](
  n, t, scale_func, scale_t, acc_singleported, acc_sub_banks,
  use_shared_ext_mem, use_tl_ext_ram, acc_latency, acc_type, is_dummy
)
