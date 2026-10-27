"""Launch a training run from the ISOLATED tree this file lives in.

The host has an editable (PEP-660) install of /root/Ultra whose meta-path finder takes
precedence over sys.path, so `import ultralytics` would silently load the campaign's copy of
the code instead of the tree next to this file. That is exactly the kind of drift the
experiment must not have, so strip the finder, pin the tree, assert the resolved package, then
hand over to scripts/train_dfs.py unchanged.

usage: python run_train.py --dataset dfire --baseline dcn-kd ... (all train_dfs.py args)
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
print("run_train: ultralytics =", resolved, flush=True)

sys.argv = ["train_dfs.py"] + sys.argv[1:]
runpy.run_path(os.path.join(HERE, "scripts", "train_dfs.py"), run_name="__main__")
