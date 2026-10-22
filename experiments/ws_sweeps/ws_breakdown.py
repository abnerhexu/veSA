"""Auditable area/power ownership for the WS width/K experiment.

The partition is created before arithmetic lowering and ABC.  It is therefore
the measured implementation, rather than a proportional estimate applied after
mapping.  All reports are checked against the physical leaf-cell totals.
"""
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent
RULES_PATH = HERE / "breakdown-rules.json"
TOP = "WSReuseExperimentMesh"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_table(path, rows, fields=None, delimiter=","):
    rows = list(rows)
    if fields is None:
        fields = list(dict.fromkeys(key for row in rows for key in row))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter=delimiter,
                                lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def read_table(path, delimiter=","):
    with Path(path).open() as stream:
        return list(csv.DictReader(stream, delimiter=delimiter))


def rules():
    return json.loads(RULES_PATH.read_text())


def yosys_quote(path):
    return json.dumps(str(Path(path).resolve()))


class SourceAnnotations:
    """Map Yosys source spans through emitted Chisel @[Scala line] comments."""

    def __init__(self, configuration):
        self.configuration = configuration
        self.files = {}
        self.cache = {}

    def text_categories(self, text):
        categories = set()
        for comment in re.findall(r"@\[([^]]+)\]", text):
            for item in re.finditer(
                    r"([\w/.-]+\.scala)\s+(.+?)(?=\s+[\w/.-]+\.scala|$)", comment):
                name = Path(item[1]).name
                for line in map(int, re.findall(r"(\d+):(?:\d+|\{[\d,]+\})", item[2])):
                    for first, last, category in self.configuration["source_ranges"].get(name, []):
                        if first <= line <= last:
                            categories.add(category)
        return categories

    def categories(self, attributes):
        source = attributes.get("src", "")
        if source in self.cache:
            return self.cache[source].copy()
        categories = set()
        for match in re.finditer(r"([^|\s]+\.sv):(\d+)\.\d+-(\d+)\.\d+", source):
            path = Path(match[1])
            if path not in self.files:
                self.files[path] = path.read_text().splitlines()
            first, last = int(match[2]) - 1, int(match[3])
            snippet = "\n".join(self.files[path][first:last])
            if "@[" not in snippet and ";" not in snippet:
                end = last
                while end < len(self.files[path]) and end < last + 16:
                    snippet += "\n" + self.files[path][end]
                    end += 1
                    if ";" in self.files[path][end - 1] or "@[" in self.files[path][end - 1]:
                        break
            categories.update(self.text_categories(snippet))
        self.cache[source] = categories
        return categories.copy()


def generic_sequential(cell_type):
    return bool(re.match(r"^\$(?:[as]?dff|[as]?dffe|sdffce|dffsr|dlatch)", cell_type))


def storage_detail(module_name, cell_name, cell, module):
    qbits = set(cell.get("connections", {}).get("Q", []))
    names = [name for name, net in module.get("netnames", {}).items()
             if qbits.intersection(bit for bit in net.get("bits", []) if isinstance(bit, int))]
    text = " ".join([module_name, cell_name, *names,
                     cell.get("attributes", {}).get("src", "")])
    if re.search(r"\bweights(?:[._\[]|$)", text):
        return "stationary_weight_storage"
    if re.search(r"\bactivations(?:[._\[]|$)", text):
        return "activation_buffers"
    if re.search(r"(?:accumulators|prepassMaximum|replayMaximum|fixedSums|"
                 r"positiveInfinity|negativeInfinity|(?:^|[._])nan(?:[._\[]|$))", text):
        return "accumulator_state"
    if "WSReuseExperimentMesh.scala 28" in text or "feedback" in text.lower():
        return "feedback_pipeline"
    if module_name.startswith("WSReuseExperimentPE"):
        return "pipeline_interconnect"
    return "control_state"


def classify_operators(design, emitted_rtl, configuration=None):
    """Assign each generic RTL operator to one disjoint canonical category."""
    configuration = configuration or rules()
    sources = SourceAnnotations(configuration)
    ledger = []
    for module_name, module in design["modules"].items():
        if module.get("attributes", {}).get("blackbox") == "0" * 31 + "1":
            continue
        cells = module.get("cells", {})
        state_bits = set()
        for cell in cells.values():
            if generic_sequential(cell["type"]):
                state_bits.update(bit for bit in cell.get("connections", {}).get("Q", [])
                                  if isinstance(bit, int))

        # A mux whose data arms contain only state-derived bits is storage
        # addressing/routing, even if Chisel attributed it to a downstream decode.
        routing_bits = set(state_bits)
        routing = set()
        changed = True
        while changed:
            changed = False
            for name, cell in cells.items():
                if name in routing or cell["type"] not in ("$mux", "$pmux"):
                    continue
                data = cell.get("connections", {}).get("A", []) + cell.get("connections", {}).get("B", [])
                integer = [bit for bit in data if isinstance(bit, int)]
                if integer and any(bit in routing_bits for bit in integer) and all(
                        bit in routing_bits for bit in integer):
                    routing.add(name)
                    routing_bits.update(bit for bit in cell.get("connections", {}).get("Y", [])
                                        if isinstance(bit, int))
                    changed = True

        producer_categories = defaultdict(set)
        for net in module.get("netnames", {}).values():
            categories = sources.categories(net.get("attributes", {}))
            for bit in net.get("bits", []):
                if isinstance(bit, int):
                    producer_categories[bit].update(categories)

        for name, cell in cells.items():
            if cell["type"] in design["modules"] or cell["type"] == "$scopeinfo":
                continue
            attributes = cell.setdefault("attributes", {})
            source_categories = sources.categories(attributes)
            output_categories = set()
            for port, bits in cell.get("connections", {}).items():
                if cell.get("port_directions", {}).get(port) == "output":
                    for bit in bits:
                        if isinstance(bit, int):
                            output_categories.update(producer_categories.get(bit, set()))

            detail = ""
            if generic_sequential(cell["type"]):
                category = "storage_interconnect"
                detail = storage_detail(module_name, name, cell, module)
                reason = "physical state cell"
            elif name in routing:
                category, detail = "storage_interconnect", "data_routing"
                reason = "register-data routing"
            elif len(output_categories) == 1:
                category = next(iter(output_categories))
                reason = "single source-defined producer"
            elif len(source_categories) == 1:
                category = next(iter(source_categories))
                reason = "single source-defined operation"
            elif cell["type"] in ("$mux", "$pmux"):
                category, detail = "storage_interconnect", "data_routing"
                reason = "cross-stage data selection"
            elif name.startswith("$procmux") or cell["type"] in (
                    "$eq", "$eqx", "$ne", "$nex", "$logic_and", "$logic_or", "$logic_not"):
                category = "control"
                reason = "process or state control"
            else:
                category = "shared_other"
                reason = "ambiguous or unresolved ownership"

            if category not in configuration["categories"]:
                raise ValueError(f"unknown category {category} for {module_name}.{name}")
            suffix = category if not detail else f"{category}__{detail}"
            attributes["submod"] = "ws_stage__" + suffix
            attributes["ws_category"] = category
            if detail:
                attributes["ws_storage_detail"] = detail
            ledger.append(dict(module=module_name, operator=name, type=cell["type"],
                               category=category, storage_detail=detail, reason=reason,
                               source_categories="|".join(sorted(source_categories)),
                               output_categories="|".join(sorted(output_categories)),
                               src=attributes.get("src", "")))
    return ledger


def partition_rtl(rtl, output, yosys, run_command):
    """Create and audit the pre-map category hierarchy."""
    output = Path(output)
    configuration = rules()
    operators_script = output / "operators.ys"
    operators_script.write_text("\n".join((
        f"read_verilog -sv {yosys_quote(rtl)}",
        f"hierarchy -check -top {TOP}",
        "chformal -remove", "proc -noopt", "opt_merge", "memory_map", "opt_clean",
        f"write_json {yosys_quote(output / 'operators.json')}", "")))
    run_command((yosys, "-Q", "-T", "-s", operators_script))
    design = json.loads((output / "operators.json").read_text())
    ledger = classify_operators(design, rtl, configuration)
    write_table(output / "operator-ownership.csv", ledger)
    (output / "tagged-operators.json").write_text(json.dumps(design) + "\n")

    partition_script = output / "partition.ys"
    partition_script.write_text("\n".join((
        f"read_json {yosys_quote(output / 'tagged-operators.json')}",
        "submod", f"write_json {yosys_quote(output / 'partitioned-raw.json')}", "")))
    run_command((yosys, "-Q", "-T", "-s", partition_script))
    partitioned = json.loads((output / "partitioned-raw.json").read_text())
    module_categories = {}
    module_storage_details = {}
    for name, module in partitioned["modules"].items():
        leaf_cells = [cell for cell in module.get("cells", {}).values()
                      if cell["type"] not in partitioned["modules"]]
        categories = {cell.get("attributes", {}).get("ws_category") for cell in leaf_cells}
        categories.discard(None)
        if categories:
            if len(categories) != 1:
                raise RuntimeError(f"partition module {name} has categories {categories}")
            category = next(iter(categories))
            module_categories[name] = category
            details = {cell.get("attributes", {}).get("ws_storage_detail")
                       for cell in leaf_cells if cell.get("attributes", {}).get("ws_storage_detail")}
            if len(details) > 1:
                raise RuntimeError(f"partition module {name} has storage details {details}")
            detail = next(iter(details)) if details else ""
            if detail:
                module_storage_details[name] = detail
            module.setdefault("attributes", {}).update(
                ws_category=category, keep_hierarchy="0" * 31 + "1")
            if detail:
                module["attributes"]["ws_storage_detail"] = detail
    (output / "partitioned.json").write_text(json.dumps(partitioned) + "\n")
    ownership = dict(
        version=1, categories=configuration["categories"],
        storage_details=configuration["storage_details"],
        policy=configuration["policy"], module_categories=module_categories,
        module_storage_details=module_storage_details,
        operator_counts=dict(Counter(row["category"] for row in ledger)),
        storage_operator_counts=dict(Counter(row["storage_detail"] for row in ledger
                                             if row["storage_detail"])),
        rules_sha256=sha(RULES_PATH), rtl_sha256=sha(rtl))
    (output / "stage-ownership.json").write_text(json.dumps(ownership, indent=2) + "\n")
    return output / "partitioned.json"


def read_liberty(path):
    text = Path(path).read_text()
    matches = list(re.finditer(r"\bcell\s*\(\s*\"?([^\"\s)]+)\"?\s*\)\s*\{", text))
    cells = {}
    for index, match in enumerate(matches):
        block = text[match.end():matches[index + 1].start() if index + 1 < len(matches) else len(text)]
        area_match = re.search(r"\barea\s*:\s*([\d.eE+-]+)", block)
        if not area_match:
            continue
        cells[match[1]] = dict(area=float(area_match[1]),
                              sequential=bool(re.search(r"\b(?:ff|latch)\s*\(", block)))
    return cells


def attributed_names(design, output, yosys, liberty, run_command):
    tagged = json.loads(json.dumps(design))
    for module in tagged["modules"].values():
        for name, cell in module.get("cells", {}).items():
            cell.setdefault("attributes", {})["breakdown_original_id"] = name
    tagged_path = Path(output) / "mapped-tagged.json"
    tagged_path.write_text(json.dumps(tagged) + "\n")
    top = next(name for name, module in design["modules"].items()
               if module.get("attributes", {}).get("top") == "0" * 31 + "1")
    script = Path(output) / "mapped-names.ys"
    script.write_text("\n".join((
        f"read_json {yosys_quote(tagged_path)}", f"read_liberty -lib {yosys_quote(liberty)}",
        f"hierarchy -check -top {top}",
        f"write_verilog -noattr {yosys_quote(Path(output) / 'mapped.v')}",
        f"write_verilog -attr2comment {yosys_quote(Path(output) / 'mapped-attributed.v')}", "")))
    run_command((yosys, "-Q", "-T", "-s", script))
    names = {}
    module_name = None
    original = None
    for line in (Path(output) / "mapped-attributed.v").read_text().splitlines():
        match = re.match(r"module (\\?[^\s(]+)", line)
        if match:
            module_name = match[1].rstrip("(").lstrip("\\")
            names[module_name] = {}
        match = re.search(r'breakdown_original_id = "(.*)"', line)
        if match:
            original = match[1].replace("\\\\", "\\").replace('\\"', '"')
        if original is not None:
            match = re.match(r"\s+(\\?\S+)\s+(\\?\S+)\s*\(", line)
            if match:
                names[module_name][original] = match[2].lstrip("\\")
                original = None
    for name, module in design["modules"].items():
        expected = {cell_name for cell_name, cell in module.get("cells", {}).items()
                    if cell["type"] != "$scopeinfo"}
        if set(names.get(name, {})) != expected:
            raise RuntimeError(f"incomplete mapped instance-name ledger for {name}")
    return names, top


def aggregate_area(cells, configuration=None):
    configuration = configuration or rules()
    total = sum(float(cell["area_um2"]) for cell in cells)
    rows = []
    for category in configuration["categories"]:
        selected = [cell for cell in cells if cell["category"] == category]
        area = sum(float(cell["area_um2"]) for cell in selected)
        rows.append(dict(category=category, cells=len(selected), area_um2=area,
                         area_percent=100.0 * area / total if total else 0.0))
    storage_rows = []
    storage_total = sum(float(cell["area_um2"]) for cell in cells
                        if cell["category"] == "storage_interconnect")
    for detail in configuration["storage_details"]:
        selected = [cell for cell in cells if cell.get("storage_detail") == detail]
        area = sum(float(cell["area_um2"]) for cell in selected)
        storage_rows.append(dict(storage_detail=detail, cells=len(selected), area_um2=area,
                                 storage_percent=100.0 * area / storage_total if storage_total else 0.0,
                                 design_percent=100.0 * area / total if total else 0.0))
    return rows, storage_rows


def prepare_cell_ledger(output, yosys, liberty, run_command):
    output = Path(output)
    design = json.loads((output / "mapped.json").read_text())
    library = read_liberty(liberty)
    names, top = attributed_names(design, output, yosys, liberty, run_command)
    cells = []

    def walk(module_name, prefix=""):
        module = design["modules"][module_name]
        category = module.get("attributes", {}).get("ws_category", "shared_other")
        detail = module.get("attributes", {}).get("ws_storage_detail", "")
        for original, cell in module.get("cells", {}).items():
            if cell["type"] == "$scopeinfo":
                continue
            instance = prefix + names[module_name][original]
            if cell["type"] in design["modules"]:
                walk(cell["type"], instance + "/")
                continue
            if cell["type"] not in library:
                raise RuntimeError(f"mapped cell {cell['type']} absent from Liberty")
            lib = library[cell["type"]]
            leaf_category = "storage_interconnect" if lib["sequential"] else category
            leaf_detail = detail if leaf_category == "storage_interconnect" else ""
            if leaf_category == "storage_interconnect" and not leaf_detail:
                leaf_detail = "unclassified_storage"
            cells.append(dict(instance=instance, module=module_name,
                              original_cell=original, cell_type=cell["type"],
                              category=leaf_category, storage_detail=leaf_detail,
                              area_um2=lib["area"], sequential=int(lib["sequential"])))

    walk(top)
    if len({cell["instance"] for cell in cells}) != len(cells):
        raise RuntimeError("mapped leaf instance names are not unique")
    stat_area = float(json.loads((output / "stat.json").read_text())["design"]["area"])
    leaf_area = sum(cell["area_um2"] for cell in cells)
    if abs(stat_area - leaf_area) > max(1e-5, stat_area * 1e-8):
        raise RuntimeError(f"area closure failed: stat={stat_area}, leaves={leaf_area}")
    write_table(output / "cells.csv", cells)
    area, storage = aggregate_area(cells)
    write_table(output / "area-breakdown.csv", area)
    write_table(output / "storage-area-breakdown.csv", storage)
    return cells


def power_tcl(output, cell_ledger):
    """Tcl fragment for total plus every physical leaf-cell power."""
    output = Path(output)
    return "\n".join((
        f"report_power -digits 12 -format json > {{{output / 'power.json'}}}",
        f"report_power -digits 12 > {{{output / 'power.txt'}}}",
        f"set stream [open {{{output / 'cell-power.tsv'}}} w]",
        'puts $stream "instance\\tinternal_w\\tswitching_w\\tleakage_w\\ttotal_w"',
        "set scene [sta::cmd_scene]",
        "foreach inst [sta::network_leaf_instances] {",
        '  puts $stream "[sta::get_full_name $inst]\\t[join [sta::instance_power $inst $scene] \\t]"',
        "}", "close $stream", "exit", ""))


def aggregate_power(cell_ledger, cell_power_path, total_power_path, configuration=None):
    configuration = configuration or rules()
    power_rows = read_table(cell_power_path, delimiter="\t")
    power = {row["instance"]: row for row in power_rows}
    expected = {cell["instance"] for cell in cell_ledger}
    if set(power) != expected:
        missing = sorted(expected - set(power))[:5]
        extra = sorted(set(power) - expected)[:5]
        raise RuntimeError(f"power/area leaf identity mismatch: missing={missing}, extra={extra}")
    combined = []
    for cell in cell_ledger:
        row = dict(cell)
        row.update({key: float(value) for key, value in power[cell["instance"]].items()
                    if key != "instance"})
        combined.append(row)
    total = json.loads(Path(total_power_path).read_text())["Total"]
    for column, key in (("internal_w", "internal"), ("switching_w", "switching"),
                        ("leakage_w", "leakage"), ("total_w", "total")):
        leaf_sum = sum(cell[column] for cell in combined)
        reference = float(total[key])
        if abs(leaf_sum - reference) > max(1e-11, abs(reference) * 1e-4):
            raise RuntimeError(f"power closure failed for {key}: total={reference}, leaves={leaf_sum}")

    def rows_for(field, values):
        result = []
        design_power = sum(cell["total_w"] for cell in combined)
        storage_power = sum(cell["total_w"] for cell in combined
                            if cell["category"] == "storage_interconnect")
        for value in values:
            selected = [cell for cell in combined if cell[field] == value]
            sums = {key: sum(cell[key] for cell in selected)
                    for key in ("internal_w", "switching_w", "leakage_w", "total_w")}
            row = {field: value, "cells": len(selected),
                   "area_um2": sum(float(cell["area_um2"]) for cell in selected), **sums}
            denominator = storage_power if field == "storage_detail" else design_power
            row["power_percent"] = 100.0 * sums["total_w"] / denominator if denominator else 0.0
            if field == "storage_detail":
                row["design_power_percent"] = (100.0 * sums["total_w"] / design_power
                                                 if design_power else 0.0)
            result.append(row)
        return result

    return (rows_for("category", configuration["categories"]),
            rows_for("storage_detail", configuration["storage_details"]), combined)
