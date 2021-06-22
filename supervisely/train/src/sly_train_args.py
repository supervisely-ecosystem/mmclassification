import sys
import sly_globals as g
import train_config


def init_script_arguments(state):
    sys.argv.append(train_config.main_config_path)
    #sys.argv.append(os.path.join(g.root_source_dir, "configs/resnet/resnet18_b16x8_cifar10.py"))

    sys.argv.extend(["--work-dir", g.checkpoints_dir])
    #sys.argv.extend(["--device", "cuda"])
    sys.argv.extend(["--gpu-ids", state["gpusId"]])
