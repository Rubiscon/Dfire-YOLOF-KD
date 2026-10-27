"""Run scripts/eval_dfire.py against this tree.

Same reason as run_train.py: a stale editable install can shadow the tree, and the eval
script unpickles ``YOLOFDistillationModel`` by qualified name, so the import has to resolve
to this copy.

usage: python run_eval.py --data ultralytics/cfg/datasets/dfire.yaml --out eval/kd ...
"""
import os
import runpy
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.meta_path = [
    f
    for f in sys.meta_path
    if "editable" not in (getattr(f, "__module__", "") or "").lower()
    and "editable" not in type(f).__name__.lower()
]
sys.path.insert(0, HERE)

import ultralytics  # noqa: E402

resolved = os.path.abspath(ultralytics.__file__)
assert resolved.startswith(HERE), "ultralytics resolved to %s, expected under %s" % (resolved, HERE)

sys.argv = ["eval_dfire.py"] + sys.argv[1:]
runpy.run_path(os.path.join(HERE, "scripts", "eval_dfire.py"), run_name="__main__")
