package gemmini

import chisel3._
import chisel3.util._

class AccumulatorReadReq[T <: Data, U <: Data](n: Int, acc_t: T, scale_t: U) extends Bundle {
  val addr = UInt(log2Ceil(n).W)
  val scale = scale_t
  val igelu_qb = acc_t.cloneType
  val igelu_qc = acc_t.cloneType
  val iexp_qln2 = acc_t.cloneType
  val iexp_qln2_inv = acc_t.cloneType
  val act = UInt(Activation.bitwidth.W) // TODO magic number
  val full = Bool() // Whether or not we return the full bitwidth output

  val fromDMA = Bool()

}

class AccumulatorReadResp[T <: Data, U <: Data](fullDataType: Vec[Vec[T]], scale_t: U) extends Bundle {
  val data = fullDataType.cloneType
  val fromDMA = Bool()
  val scale = scale_t.cloneType
  val igelu_qb = fullDataType.head.head.cloneType
  val igelu_qc = fullDataType.head.head.cloneType
  val iexp_qln2 = fullDataType.head.head.cloneType
  val iexp_qln2_inv = fullDataType.head.head.cloneType
  val act = UInt(Activation.bitwidth.W) // TODO magic number
  val acc_bank_id = UInt(2.W) // TODO magic number
}

class AccumulatorReadIO[T <: Data, U <: Data](n: Int, fullDataType: Vec[Vec[T]], scale_t: U) extends Bundle {
  val req = Decoupled(new AccumulatorReadReq[T, U](n, fullDataType.head.head.cloneType, scale_t))
  val resp = Flipped(Decoupled(new AccumulatorReadResp[T, U](fullDataType, scale_t)))
}

/** Client-side bank protocol shared by the legacy and raw execution paths. */
class AccumulatorBankClientIO[T <: Data, U <: Data](n: Int, rowType: Vec[Vec[T]], scaleType: U) extends Bundle {
  val read = new AccumulatorReadIO(n, rowType, scaleType)
  val write = Decoupled(new AccumulatorWriteReq(n, rowType))
}

class AccumulatorWriteReq[T <: Data](n: Int, t: Vec[Vec[T]]) extends Bundle {
  val addr = UInt(log2Up(n).W)
  val data = t.cloneType
  val acc = Bool()
  val mask = Vec(t.getWidth / 8, Bool()) // TODO Use aligned_to here
}


class AccumulatorMemIO [T <: Data, U <: Data](n: Int, t: Vec[Vec[T]], scale_t: U,
  acc_sub_banks: Int, use_shared_ext_mem: Boolean
) extends Bundle {
  val read = Flipped(new AccumulatorReadIO(n, t, scale_t))
  val write = Flipped(Decoupled(new AccumulatorWriteReq(n, t)))

  // Includes physical write queues, not merely accepted port handshakes.
  val busy = Output(Bool())

  val ext_mem = if (use_shared_ext_mem) Some(Vec(acc_sub_banks, new ExtMemIO)) else None

  val adder = new Bundle {
    val valid = Output(Bool())
    val op1 = Output(t.cloneType)
    val op2 = Output(t.cloneType)
    val sum = Input(t.cloneType)
  }
}

/** Full-width raw row and independently narrowed/scaled row. Never swap them. */
class AccumulatorScaleResp[T <: Data](fullDataType: Vec[Vec[T]], rDataType: Vec[Vec[T]]) extends Bundle {
  require(fullDataType.size == rDataType.size &&
    fullDataType.zip(rDataType).forall { case (a, b) => a.size == b.size })
  require(fullDataType.head.head.getWidth >= rDataType.head.head.getWidth,
    "Accumulator full_data must not be narrower than the converted data row")
  val full_data = fullDataType.cloneType
  val data = rDataType.cloneType
  val acc_bank_id = UInt(2.W)
  val fromDMA = Bool()
}
