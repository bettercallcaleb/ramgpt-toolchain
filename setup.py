from pathlib import Path
from setuptools import setup
from setuptools.command.build_py import build_py

ENGINE = (
    "adaptive.py", "audit_parity.py", "autotune_worker.py", "doctor.py",
    "evaluation.py", "evaluator_worker.py", "evidence.py", "inference.py",
    "joint.py", "ledger.py", "optimize.py", "passport.py", "planner.py", "ramgpt.py",
    "registry.py", "release_info.py", "router_worker.py", "runtime.py",
    "runtime_opt.py", "telemetry.py", "weight.py",
)

class BuildEngine(build_py):
    def run(self):
        super().run()
        root = Path(__file__).resolve().parent
        dest = Path(self.build_lib) / "ramgpt_toolchain" / "_engine"
        dest.mkdir(parents=True, exist_ok=True)
        for name in ENGINE:
            self.copy_file(str(root / name), str(dest / name))

    def get_outputs(self, include_bytecode=1):
        dest = Path(self.build_lib) / "ramgpt_toolchain" / "_engine"
        return super().get_outputs(include_bytecode) + [str(dest / name) for name in ENGINE]

setup(cmdclass={"build_py": BuildEngine})
