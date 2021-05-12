import os
import sys
import sly_globals as g


def init_script_arguments():
    sys.argv.extend(["--config", os.path.join(g.root_source_path, "configs/resnet/resnet34_b16x8_cifar10.py")])
    sys.argv.extend(["--work-dir", "/data"])
