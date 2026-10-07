"""The command line: ``lattice-ds <command> [options]``.

Commands:

    run               independent chains on a list of points (n_max, lambda, m^2), in parallel
    run-constrained   one chain with the zero mode held fixed (mean force and curvature)
    run-with-force         one unconstrained chain that also records the force on the zero mode

``lattice-ds <command> --help`` describes the options. Every chain is written as
an .npz file, in the format described in reproduction/README.md.

Each process uses one BLAS thread, because the parallelism is across chains. The
number of threads is fixed when NumPy is loaded, which happens before this
module runs, so on the command line the program sets the thread variables and
restarts itself once. Variables already set by the user are left as they are.
"""

from __future__ import annotations

import importlib
import os
import subprocess
import sys

COMMANDS = {
    "run": ("lattice_ds.cli.run",
            "independent chains on a list of points (n_max, lambda, m^2), in parallel"),
    "run-constrained": ("lattice_ds.cli.run_constrained",
                        "one chain with the zero mode held fixed (mean force and curvature)"),
    "run-with-force": ("lattice_ds.cli.run_with_force",
                  "one unconstrained chain that also records the force on the zero mode"),
}
THREAD_VARIABLES = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")


def usage() -> str:
    lines = ["usage: lattice-ds <command> [options]", "", "commands:"]
    lines += [f"  {name:17s} {doc}" for name, (_, doc) in COMMANDS.items()]
    lines += ["", "'lattice-ds <command> --help' describes the options of a command."]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    from_shell = argv is None
    argv = sys.argv[1:] if from_shell else list(argv)
    if not argv or argv[0] in ("-h", "--help"):
        print(usage())
        return 0
    if argv[0] in ("-V", "--version"):
        from lattice_ds import __version__
        print(__version__)
        return 0
    if argv[0] not in COMMANDS:
        print(f"lattice-ds: unknown command '{argv[0]}'\n\n{usage()}", file=sys.stderr)
        return 2
    missing = [v for v in THREAD_VARIABLES if v not in os.environ]
    if missing:
        os.environ.update({v: "1" for v in missing})
        if from_shell:                       # restart, so that NumPy starts with one thread
            cmd = [sys.executable, "-m", "lattice_ds.cli", *argv]
            if os.name == "posix":
                os.execv(sys.executable, cmd)
            return subprocess.call(cmd)
    code = importlib.import_module(COMMANDS[argv[0]][0]).main(argv[1:])
    return int(code or 0)
