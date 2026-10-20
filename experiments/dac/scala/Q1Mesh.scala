package gemmini.dac

import chisel3._
import chisel3.util._
import gemmini.tensorcore._

class Q1MeshIO(rows: Int = 2, columns: Int = 2) extends Bundle {
  private val config = BlackwellConfig()
  val req = Flipped(Decoupled(new TensorMatrixRequest(config, rows, columns)))
  val in = Flipped(Decoupled(new TensorMatrixOperands(config, rows, columns)))
  val out = Decoupled(new TensorMatrixResult(config, rows, columns))
}

/** Identical top-level widths for B0 and the two unmodified release meshes. */
class Q1Mesh(design: String, dim: Int = 2) extends Module {
  val io = IO(new Q1MeshIO(dim, dim))
  design match {
    case "ieee" => io <> Module(new IeeeFmaMesh(dim, dim)).io
    case "blackwell" => io <> Module(new TensorMesh(BlackwellConfig(), dim, dim)).io
    case "hopper" => io <> Module(new TensorMesh(HopperConfig(), dim, dim)).io
    case "hopper-overlap" => io <> Module(new OverlapTensorMesh(HopperConfig(), dim, dim)).io
    case "blackwell-overlap" => io <> Module(new OverlapTensorMesh(BlackwellConfig(), dim, dim)).io
    case "hopper-reuse-overlap" => io <> Module(new OverlapTensorMesh(HopperConfig(), dim, dim, reuseSlots = true)).io
    case "blackwell-reuse-overlap" => io <> Module(new OverlapTensorMesh(BlackwellConfig(), dim, dim, reuseSlots = true)).io
  }
}

object GenerateQ1 extends App {
  (new chisel3.stage.ChiselStage).emitSystemVerilog(new Q1Mesh(args(0), args.lift(2).map(_.toInt).getOrElse(2)),
    Array("--target-dir", args(1)))
}
