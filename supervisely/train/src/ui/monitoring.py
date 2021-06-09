import supervisely_lib as sly
#import sly_metrics as metrics


def init(data, state):
    _init_start_state(state)
    #_init_galleries(data)
    _init_progress(data)
    _init_output(data)
    #metrics.init(data, state)

    state["collapsed9"] = True
    state["disabled9"] = True
    state["done9"] = False


def restart(data, state):
    data["done9"] = False


def _init_start_state(state):
    state["started"] = False
    state["activeNames"] = []


def _init_galleries(data):
    pass
    #data["vis"] = empty_gallery
    #data["labelsVis"] = empty_gallery
    #data["predVis"] = empty_gallery
    #data["syncBindings"] = []


def _init_progress(data):
    data["progressName"] = ""
    data["currentProgress"] = 0
    data["totalProgress"] = 0
    data["currentProgressLabel"] = ""
    data["totalProgressLabel"] = ""


def _init_output(data):
    data["outputUrl"] = ""
    data["outputName"] = ""