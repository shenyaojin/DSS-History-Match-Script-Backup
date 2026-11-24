# Coordinate generation and comparison

Only new files in the isolated task directory were created. No existing project files were read or modified. The saved script uses only the Python standard library; no dependencies, installs or network access were needed.

Input coordinates (metres): (0, 0), (10, 0), (20, 0).
Change: add 2 metres to the second point's x coordinate, giving (12, 0). All other coordinates remain identical.

Actual command, run synchronously in the task directory:
`python3 -B generate_compare.py > run.log 2>&1`

Exit status: 0. Runtime: Python 3.6.8. The -B option prevents bytecode cache writes. The script creates CSV files exclusively and refuses to overwrite existing files.

Verification: the script read both saved CSV files back and checked all three point IDs and coordinates against the required values. The saved files and log were also inspected after execution. Changed points: 1. Maximum Euclidean movement: 2 metres. Individual distances: 0, 2, 0 metres.

File inventory and SHA-256:

- generate_compare.py — actual generation and comparison script; f5f92b4f8a83d0b97a9fdf6a1815b101a043baf08763957a0df2218f37440bfb
- before.csv — initial coordinates; f9cea95d66a2ee83c2c8a92d7cbc854a2f026550b34b793254ec6ca47f37dd54
- after.csv — changed coordinates; 85c2e8d48cdfa81da711c43d38c3640aa074b0c09c64de27b812dbce282eb3b9
- run.log — actual standard output and error; 9c6121bec8ff2ec82483bd4932d7ad9c8deb9ae2825f5737b6ff87c43d2a1fe3
- report.md — this execution report.

Limitations: this is the requested three-point functional test only; it makes no claims about existing research data or larger datasets.
