import subprocess
import sys
from pathlib import Path

def main():
    engine = Path(__file__).resolve().parent / "_engine"
    if not (engine / "ramgpt.py").is_file():
        print("RAMGPT: missing installed engine; install a built wheel", file=sys.stderr)
        return 1
    code = "import runpy,sys; p=sys.argv.pop(1); sys.path.insert(0,p); runpy.run_path(p+'/ramgpt.py',run_name='__main__')"
    try:
        return subprocess.call([sys.executable, "-I", "-c", code, str(engine), *sys.argv[1:]])
    except KeyboardInterrupt:
        return 130

if __name__ == "__main__":
    raise SystemExit(main())
