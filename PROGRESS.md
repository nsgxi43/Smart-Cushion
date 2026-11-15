# PROGRESS LOG — Smart Cushion posture classifier

Running log. Newest phase at the bottom. Every claim here must be backed by a
file under `reports/`. Anything not measurable without the board is labelled
`PENDING_HARDWARE`.

Project root: `C:\Users\Niranjana Gopinath\Projects\Smart-Cushion`
(extracted from `Downloads\Smart-Cushion-main.zip` on 2026-11-15)

---

## Phase 0 — Environment check — DONE

| Item | Result |
|---|---|
| System Python | 3.12.10 (also 3.11 available via `py -3.11`) |
| Chosen interpreter | Python 3.12 (TensorFlow 2.16+ supports 3.12) |
| Virtualenv | `venv/` created with `py -3.12 -m venv venv` |
| PlatformIO | **NOT INSTALLED** (`pio` not on PATH) |
| arduino-cli | **NOT INSTALLED** (`arduino-cli` not on PATH) |
| git | 2.56.0.windows.4 — repo initialised in project root |
| Board connected | none detected — all on-device numbers are PENDING_HARDWARE |

Pre-existing packages in system Python: boto3, pytest, colorama (no numpy /
pandas / sklearn / tensorflow). A clean venv was therefore created rather than
polluting the system interpreter.

---

## Phase 1 — Inspection — see `reports/dataset_report/`

Artifacts:
- `reports/checksums_phase1.csv` — SHA-256 of all 69 original DATASET files
  plus the 7 original root scripts (re-verified in Phase 9).
- `esp32/backup_20261115/matrix_scanner_original.ino` — untouched copy of the
  firmware supplied by the user
  (sha256 `c669202b66a7f8bad5d502b2a703e112f578e5112fe354819732139a9a92f5fd`).
- `reports/dataset_report/README.md` — full findings.

(Findings summary is appended below once the inspection run completes.)
