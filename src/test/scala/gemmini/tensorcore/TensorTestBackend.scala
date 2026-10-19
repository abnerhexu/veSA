package gemmini.tensorcore

import chiseltest.simulator.{VerilatorBackendAnnotation, VerilatorCFlags, VerilatorFlags}
import firrtl.annotations.Annotation

object TensorTestBackend {
  val annotations: Seq[Annotation] = if (sys.env.get("TENSORCORE_SIM").contains("verilator"))
    // Chiseltest 0.6 names 32-bit wide-signal words WData. Verilator 5.052
    // renamed that alias EData; this changes only the generated C++ harness.
    // Keep this small-array regression in a single generated translation unit.
    // Verilator 5.052's split-build PCH is incompatible with the old harness's
    // forced -include on this GCC; this also avoids extra PCH memory overhead.
    Seq(VerilatorBackendAnnotation, VerilatorCFlags(Seq("-DWData=EData")),
      VerilatorFlags(Seq("--output-split", "0"))) else Seq.empty
}
