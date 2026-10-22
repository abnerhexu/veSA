"""Restore ABC port IDs from its authoritative output BLIF files.

Some Yosys/ABC builds corrupt non-builtin Liberty port names during ABC result
reintegration.  Cells are matched by the importer ordinal and their net
bijection is reconstructed from unaffected pins.  No pin ordering is guessed.
"""
from collections import Counter
import json
from pathlib import Path
import re
import subprocess


def repair(raw_json, log, output, yosys, workdir=None):
    design = json.loads(Path(raw_json).read_text())
    log_text = Path(log).read_text()
    partitions = re.findall(
        r"Extracting gate netlist of module `\\([^']+)' to `([^']+)/input.blif'", log_text)
    total = 0
    records = []
    for module_name, temporary in partitions:
        path = ((Path(workdir) if workdir else Path.cwd()) / temporary).resolve() / "output.blif"
        if not path.exists():
            continue
        reference_path = Path(output).parent / f"{module_name}-abc-reference.json"
        subprocess.run([str(yosys), "-Q", "-T", "-p",
                        f"read_blif {path}; write_json {reference_path}"],
                       check=True, stdout=subprocess.DEVNULL)
        reference = json.loads(reference_path.read_text())["modules"]["netlist"]
        cells = design["modules"][module_name]["cells"]
        constants = {}
        for cell in reference["cells"].values():
            if cell["type"] in ("_const0_", "_const1_"):
                for bits in cell["connections"].values():
                    for bit in bits:
                        constants[bit] = "0" if cell["type"] == "_const0_" else "1"
        expected = {int(name.rsplit("$", 1)[1]): cell
                    for name, cell in reference["cells"].items()
                    if cell["type"] not in ("_const0_", "_const1_")}
        actual = {int(name.rsplit("$", 1)[1]): cell for name, cell in cells.items()
                  if re.match(r"\$abc\$\d+\$blif\$\d+\$\d+$", name)}
        if expected.keys() != actual.keys():
            raise RuntimeError(f"ABC cell ordinals differ in {module_name}")
        bit_map = dict(constants)
        bad = 0
        for index, cell in actual.items():
            ref = expected[index]
            if cell["type"] != ref["type"] or len(cell["connections"]) != len(ref["connections"]):
                raise RuntimeError(f"ABC cell identity differs in {module_name} ordinal {index}")
            for port, bits in cell["connections"].items():
                if port not in ref["connections"]:
                    bad += 1
                    continue
                for old, new in zip(ref["connections"][port], bits, strict=True):
                    if old in bit_map and bit_map[old] != new:
                        raise RuntimeError(f"ABC net map conflict in {module_name} ordinal {index}")
                    bit_map[old] = new

        aliases = {}
        for index, cell in actual.items():
            ref = expected[index]
            missing = set(ref["connections"]) - set(cell["connections"])
            for port in set(cell["connections"]) - set(ref["connections"]):
                aliases[port] = aliases.get(port, missing) & missing
        for index, cell in actual.items():
            ref = expected[index]
            for port, bits in cell["connections"].items():
                if port not in ref["connections"] and len(aliases[port]) == 1:
                    correct = next(iter(aliases[port]))
                    for old, new in zip(ref["connections"][correct], bits, strict=True):
                        if old in bit_map and bit_map[old] != new:
                            raise RuntimeError(f"ABC alias conflict in {module_name} ordinal {index}")
                        bit_map[old] = new

        while True:
            domains = {}
            for index, cell in actual.items():
                reference_bits = [bit for bits in expected[index]["connections"].values()
                                  for bit in bits]
                remaining = Counter(bit for bits in cell["connections"].values() for bit in bits)
                for bit in reference_bits:
                    if bit in bit_map:
                        remaining[bit_map[bit]] -= 1
                if any(count < 0 for count in remaining.values()):
                    raise RuntimeError(f"ABC endpoint multiplicity failed in {module_name}")
                choices = {bit for bit, count in remaining.items() if count > 0}
                for bit in set(reference_bits) - bit_map.keys():
                    domains[bit] = domains.get(bit, choices) & choices
            if any(not domain for domain in domains.values()):
                raise RuntimeError(f"ABC endpoint solution failed in {module_name}")
            singles = {bit: next(iter(domain)) for bit, domain in domains.items()
                       if len(domain) == 1}
            if not singles:
                break
            bit_map.update(singles)

        for index, cell in actual.items():
            ref = expected[index]
            try:
                connections = {port: [bit_map[bit] for bit in bits]
                               for port, bits in ref["connections"].items()}
            except KeyError as error:
                raise RuntimeError(
                    f"ABC endpoint remained ambiguous in {module_name} ordinal {index}") from error
            for port, bits in cell["connections"].items():
                if port in connections and connections[port] != bits:
                    raise RuntimeError(f"ABC preserved pin changed in {module_name} ordinal {index}")
            old_bits = sorted((bit for bits in cell["connections"].values() for bit in bits), key=str)
            new_bits = sorted((bit for bits in connections.values() for bit in bits), key=str)
            if old_bits != new_bits:
                raise RuntimeError(f"ABC endpoint set changed in {module_name} ordinal {index}")
            cell["connections"] = connections
        records.append(dict(module=module_name, cells=len(actual), restored_pins=bad,
                            abc_blif=str(path)))
        total += bad
    Path(output).write_text(json.dumps(design) + "\n")
    print(f"Restored {total} corrupted port names from exact ABC BLIF correspondence",
          flush=True)
    return records
