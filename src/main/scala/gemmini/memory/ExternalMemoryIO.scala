package gemmini

import chisel3._
import chisel3.util._

class ExtMemIO extends Bundle {
  val read_req = DecoupledIO(UInt())
  val read_resp = Flipped(DecoupledIO(UInt()))
  val write_req = DecoupledIO(new Bundle {
    val addr = Output(UInt())
    val data = Output(UInt())
    val mask = Output(UInt())
  })
}
