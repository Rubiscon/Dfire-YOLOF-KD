"""Run scripts/train_dfs.py against this tree.

An editable install elsewhere on the machine can shadow this tree on sys.path, so drop the
editable finder and pin the tree before handing over.

usage: python run_train.py --dataset dfire --baseline dcn-kd ... (any train_dfs.py args)
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

sys.argv = ["train_dfs.py"] + sys.argv[1:]
runpy.run_path(os.path.join(HERE, "scripts", "train_dfs.py"), run_name="__main__")
