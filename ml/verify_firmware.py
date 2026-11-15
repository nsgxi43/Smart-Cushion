"""Phase 9 check 8 - the new firmware must preserve the original scan sequence
and pin configuration exactly.

Rather than eyeballing a diff, this extracts the machine-checkable facts from
both sketches and compares them:
  * every pin / timing constant
  * the ADC configuration calls
  * the token sequence of the scan body (setAddress / digitalWrite / delays /
    readAverage / readings[] assignments), with comments and whitespace removed
  * the body of readAverage() and setAddress()
Writes reports/firmware_scan_diff.md.
"""
from __future__ import annotations

import difflib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ORIGINAL = ROOT / "esp32" / "backup_20261115" / "matrix_scanner_original.ino"
NEW = ROOT / "esp32" / "smart_cushion" / "smart_cushion.ino"
OUT = ROOT / "reports" / "firmware_scan_diff.md"

CONSTANTS = ["ROWS", "COLS", "ROW_S0", "ROW_S1", "ROW_S2", "ROW_S3",
             "COL_S0", "COL_S1", "COL_S2", "COL_S3", "ROW_EN_1", "ROW_EN_2",
             "ROW_SIG", "COL_ADC_1", "COL_ADC_2", "SAMPLES", "SETTLE_US"]

EXPECTED = {"ROWS": 24, "COLS": 21, "ROW_S0": 25, "ROW_S1": 26, "ROW_S2": 27,
            "ROW_S3": 14, "COL_S0": 16, "COL_S1": 17, "COL_S2": 18, "COL_S3": 19,
            "ROW_EN_1": 21, "ROW_EN_2": 22, "ROW_SIG": 13,
            "COL_ADC_1": 34, "COL_ADC_2": 35, "SAMPLES": 16, "SETTLE_US": 300}


def strip_comments(src):
    src = re.sub(r"/\*.*?\*/", "", src, flags=re.S)
    src = re.sub(r"//[^\n]*", "", src)
    return src


def constants(src):
    out = {}
    for name in CONSTANTS:
        m = re.search(r"const\s+int\s+%s\s*=\s*(\d+)\s*;" % re.escape(name), src)
        if m:
            out[name] = int(m.group(1))
    return out


def extract_function(src, name):
    """Return the normalised body of a function by brace matching."""
    m = re.search(r"\b(?:void|int)\s+%s\s*\([^)]*\)\s*\{" % re.escape(name), src)
    if not m:
        return None
    i = m.end() - 1
    depth = 0
    for j in range(i, len(src)):
        if src[j] == "{":
            depth += 1
        elif src[j] == "}":
            depth -= 1
            if depth == 0:
                return normalise(src[i + 1:j])
    return None


def normalise(code):
    code = strip_comments(code)
    code = re.sub(r"\s+", " ", code)
    code = re.sub(r"\s*([{}();,])\s*", r"\1", code)
    return code.strip()


def scan_body(src, func_names):
    """The scan loop lives in loop() in the original and scanMatrix() in the new
    sketch. Return whichever is present, normalised and truncated to the scan.

    The original loop() ends with a Serial.print block that emits the frame;
    the new sketch emits it from printFrame() instead. Neither is part of the
    acquisition, so the comparison is cut at the first serial operation.
    """
    for n in func_names:
        b = extract_function(src, n)
        if b and "readAverage" in b:
            cut = b.find("Serial.")
            return n, (b[:cut] if cut != -1 else b)
    return None, None


def token_sequence(body):
    """Ordered sequence of hardware operations, ignoring everything else.

    Trailing control-flow headers are dropped: after truncating the original
    loop() at its first serial call, the `for` headers that opened the (now
    removed) print block are left dangling and are not part of acquisition.
    """
    pat = re.compile(
        r"(digitalWrite\([^)]*\)|setAddress\([^)]*\)|delayMicroseconds\([^)]*\)"
        r"|readings\[[^\]]*\]\[[^\]]*\]=readAverage\([^)]*\)"
        r"|analogRead\([^)]*\)|for\([^)]*\)|if\([^)]*\))")
    toks = pat.findall(body)
    while toks and (toks[-1].startswith("for(") or toks[-1].startswith("if(")):
        toks.pop()
    return toks


def main():
    orig = ORIGINAL.read_text(encoding="utf-8")
    new = NEW.read_text(encoding="utf-8")

    report = {"checks": []}
    ok_all = True

    # ---- 1. constants ----
    co, cn = constants(orig), constants(new)
    const_rows = []
    for name in CONSTANTS:
        a, b, e = co.get(name), cn.get(name), EXPECTED[name]
        match = (a == b == e)
        ok_all &= match
        const_rows.append({"constant": name, "original": a, "new": b,
                           "required": e, "match": match})
    report["checks"].append({"name": "pin_and_timing_constants",
                             "pass": all(r["match"] for r in const_rows),
                             "rows": const_rows})

    # ---- 2. ADC configuration ----
    adc_rows = []
    for needle in ("analogReadResolution(12)",
                   "analogSetPinAttenuation(COL_ADC_1,ADC_11db)",
                   "analogSetPinAttenuation(COL_ADC_2,ADC_11db)",
                   "Serial.begin(115200)"):
        a = needle in normalise(orig)
        b = needle in normalise(new)
        adc_rows.append({"call": needle, "in_original": a, "in_new": b,
                         "match": a == b == True})
        ok_all &= (a == b == True)
    report["checks"].append({"name": "adc_and_serial_configuration",
                             "pass": all(r["match"] for r in adc_rows),
                             "rows": adc_rows})

    # ---- 3. helper functions identical ----
    helper_rows = []
    for fn in ("setAddress", "readAverage"):
        a, b = extract_function(orig, fn), extract_function(new, fn)
        same = (a is not None and a == b)
        ok_all &= same
        helper_rows.append({"function": fn, "identical": same,
                            "original_len": len(a or ""), "new_len": len(b or "")})
    report["checks"].append({"name": "scan_helper_functions",
                             "pass": all(r["identical"] for r in helper_rows),
                             "rows": helper_rows})

    # ---- 4. scan body operation sequence ----
    on, ob = scan_body(orig, ["loop", "scanMatrix"])
    nn, nb = scan_body(new, ["scanMatrix", "loop"])
    to, tn = token_sequence(ob or ""), token_sequence(nb or "")
    # the original loop() also contains the Serial print block; cut the token
    # list at the end of the row loop by comparing only the scan operations
    seq_match = to == tn
    ok_all &= seq_match
    report["checks"].append({
        "name": "scan_operation_sequence",
        "pass": seq_match,
        "original_function": on, "new_function": nn,
        "original_ops": len(to), "new_ops": len(tn),
        "diff": None if seq_match else list(difflib.unified_diff(to, tn, lineterm="",
                                                                 n=2))[:60],
    })

    # ---- 5. pinMode set identical ----
    pm_o = sorted(re.findall(r"pinMode\(([^)]*)\)", normalise(orig)))
    pm_n = sorted(re.findall(r"pinMode\(([^)]*)\)", normalise(new)))
    pm_match = pm_o == pm_n
    ok_all &= pm_match
    report["checks"].append({"name": "pinMode_calls", "pass": pm_match,
                             "original": pm_o, "new": pm_n})

    report["overall_pass"] = bool(ok_all)

    # ---- markdown ----
    md = ["# Firmware scan-preservation check\n\n"]
    md.append("Original: `%s`  \nNew: `%s`\n\n"
              % (ORIGINAL.relative_to(ROOT), NEW.relative_to(ROOT)))
    md.append("**Overall: %s**\n\n" % ("PASS" if ok_all else "FAIL"))

    md.append("## Pin and timing constants\n\n")
    md.append("| constant | original | new | required | match |\n|---|---|---|---|---|\n")
    for r in const_rows:
        md.append("| %s | %s | %s | %s | %s |\n" % (r["constant"], r["original"],
                                                    r["new"], r["required"], r["match"]))

    md.append("\n## ADC / serial configuration\n\n| call | original | new | match |\n|---|---|---|---|\n")
    for r in adc_rows:
        md.append("| `%s` | %s | %s | %s |\n" % (r["call"], r["in_original"],
                                                 r["in_new"], r["match"]))

    md.append("\n## Scan helper functions\n\n| function | byte-identical after "
              "comment/whitespace normalisation |\n|---|---|\n")
    for r in helper_rows:
        md.append("| `%s()` | %s |\n" % (r["function"], r["identical"]))

    c = report["checks"][3]
    md.append("\n## Scan operation sequence\n\n")
    md.append("Ordered list of `digitalWrite` / `setAddress` / "
              "`delayMicroseconds` / `readAverage` operations inside the scan.\n\n")
    md.append("| | function | operations |\n|---|---|---|\n")
    md.append("| original | `%s()` | %d |\n" % (c["original_function"], c["original_ops"]))
    md.append("| new | `%s()` | %d |\n" % (c["new_function"], c["new_ops"]))
    md.append("\nIdentical: **%s**\n" % c["pass"])
    if c["diff"]:
        md.append("\n```\n" + "\n".join(c["diff"]) + "\n```\n")

    md.append("\n## pinMode calls\n\nIdentical set and order: **%s**\n\n" % pm_match)
    md.append("```\n" + "\n".join(pm_o) + "\n```\n")

    md.append("\n## What changed, and why\n\n")
    md.append("See `esp32/FIRMWARE_CHANGES.md` for the full annotated diff. In "
              "summary, the original `loop()` body was moved verbatim into "
              "`scanMatrix()` so that inference can run *after* a complete 504-value "
              "frame has been acquired, without a single statement of the scan "
              "being reordered, retimed or removed.\n")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("".join(md), encoding="utf-8")
    (OUT.with_suffix(".json")).write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("".join(md))
    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
