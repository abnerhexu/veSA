package gemmini.dac

import chisel3._
import chisel3.util._
import gemmini.tensorcore._

class Q1MeshIO extends Bundle {
  private val config = BlackwellConfig()
  val req = Flipped(Decoupled(new TensorMatrixRequest(config, 2, 2)))
  val in = Flipped(Decoupled(new TensorMatrixOperands(config, 2, 2)))
  val out = Decoupled(new TensorMatrixResult(config, 2, 2))
}

/** Identical top-level widths for B0 and the two unmodified release meshes. */
class Q1Mesh(design: String) extends Module {
  val io = IO(new Q1MeshIO)
  design match {
    case "ieee" => io <> Module(new IeeeFmaMesh).io
    case "blackwell" => io <> Module(new TensorMesh(BlackwellConfig())).io
    case "hopper" => io <> Module(new TensorMesh(HopperConfig())).io
    case "hopper-overlap" => io <> Module(new OverlapTensorMesh(HopperConfig())).io
    case "blackwell-overlap" => io <> Module(new OverlapTensorMesh(BlackwellConfig())).io
    case "hopper-reuse-overlap" => io <> Module(new OverlapTensorMesh(HopperConfig(), reuseSlots = true)).io
    case "blackwell-reuse-overlap" => io <> Module(new OverlapTensorMesh(BlackwellConfig(), reuseSlots = true)).io
  }
}

object GenerateQ1 extends App {
  (new chisel3.stage.ChiselStage).emitSystemVerilog(new Q1Mesh(args(0)),
    Array("--target-dir", args(1)))
}
