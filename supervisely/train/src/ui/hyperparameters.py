def init(state):
    state["epochs"] =  10
    state["gpusIds"] = '0'

    state["imgSize"] = 256
    state["batchSizePerGPU"] = 32
    state["workersPerGPU"] = 2  # 0 - for debug @TODO: for debug
    state["valInterval"] = 1
    state["metricsPeriod"] = 1

    state["optimizer"] = "SGD"
    state["lr"] = 0.001
    state["momentum"] = 0.9
    state["weightDecay"] = 0.0001
    state["nesterov"] = False
    state["gradClip"] = False
    state["maxNorm"] = 1

    state["activeTabName"] = "General"
    # state["hyp"] = {
    #     "scratch": g.scratch_str,
    #     "finetune": g.finetune_str,
    # }
    # state["hypRadio"] = "scratch"
    state["optimizer"] = "SGD"
    state["metricsPeriod"] = 1
    state["valInterval"] = 1