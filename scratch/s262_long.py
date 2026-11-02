"""§259 BUNDLE: queue line after a TRAIN line (the runner appends --device <d> --out <dir>; <dir> = the run dir):
  <dir> scratch.s262_long --kprime_list 8,64,128 --long_ds 1,2,3,4,5,6 --long_name eval_long_k128.json
boxes.inject_aw_eval --run <dir> --kprime_list ... --long_ds ... --long_name ... --eval_bs 250 (the §239 run_long on the run's final model.pt), then host_shas.json
(s257_guarded.manifest; refreshed over the whole dir). Exit = the eval's rc."""
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); os.chdir(ROOT); sys.path.insert(0, ROOT)
import scratch.s257_guarded as SG  # noqa: E402


def main():
    argv = sys.argv[1:]; dev = SG.take(argv, "--device"); out = SG.take(argv, "--out"); g = lambda f: argv[argv.index(f) + 1]  # noqa: E731
    assert os.path.exists(out + "/model.pt"), ("no trained model.pt", out)
    rc = subprocess.call([sys.executable, "-u", "-B", "-m", "boxes.inject_aw_eval", "--run", out, "--kprime_list", g("--kprime_list"), "--long_ds", g("--long_ds"),
                          "--long_name", g("--long_name"), "--eval_bs", "250", "--device", dev]); SG.manifest(out); return rc


if __name__ == "__main__":
    sys.exit(main())
