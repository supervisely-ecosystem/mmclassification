import os
import sly_globals as g


def init(data, state):
    state["epochs"] = 10
    state["gpusIds"] = '0'

    state["imgSize"] = 256
    state["batchSizePerGPU"] = 32
    state["workersPerGPU"] = 2  # 0 - for debug @TODO: for debug
    state["valInterval"] = 1
    #state["metricsPeriod"] = 1
    state["checkpointInterval"] = 1
    state["maxKeepCkptsEnabled"] = False
    state["maxKeepCkpts"] = 3
    state["saveLast"] = True
    state["saveBest"] = True

    state["optimizer"] = "SGD"
    state["lr"] = 0.001
    state["momentum"] = 0.9
    state["weightDecay"] = 0.0001
    state["nesterov"] = False
    state["gradClipEnabled"] = False
    state["maxNorm"] = 1

    state["lrPolicyEnabled"] = False

    file_path = os.path.join(g.root_source_dir, "configs/_base_/schedules/supervisely_lr_policy.py")
    with open(file_path) as f:
        state["lrPolicyPyConfig"] = f.read()

    state["metricsPeriod"] = 1
    state["valInterval"] = 1

