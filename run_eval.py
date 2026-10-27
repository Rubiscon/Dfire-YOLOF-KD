"""Run scripts/eval_dfire.py against the tree this file lives in.

Same problem as run_train.py: a site-packages / editable `ultralytics` can shadow the tree, and
the eval script torch.loads a checkpoint that pickles `YOLOFDistillationModel` by qualified
name -- so it must import the tree's package or the load fails. This pins the tree, asserts the
resolved package, then hands over to scripts/eval_dfire.py unchanged.
"""
import os
import runpy
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.meta_path = [
    f for f in sys.meta_path
    if "editable" not in (getattr(f, "__module__", "") or "").lower()
    and "editable" not in type(f).__name__.lower()
]
sys.path = [p for p in sys.path if os.path.abspath(p or ".") != "/root/Ultra"]
sys.path.insert(0, HERE)

import ultralytics  # noqa: E402

resolved = os.path.abspath(ultralytics.__file__)
assert resolved.startswith(HERE), "ultralytics resolved to %s, expected under %s" % (resolved, HERE)
print("run_eval: ultralytics =", resolved, flush=True)

sys.argv = ["eval_dfire.py"] + sys.argv[1:]
runpy.run_path(os.path.join(HERE, "scripts", "eval_dfire.py"), run_name="__main__")
