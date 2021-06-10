import supervisely_lib as sly
#import sly_metrics as metrics
from sly_train_progress import init_progress

def init(data, state):
    #metrics.init(data, state)

    init_progress("Epoch", data)
    init_progress("Iter", data)
    data["eta"] = None

    state["collapsed9"] = True
    state["disabled9"] = True
    state["done9"] = False



def restart(data, state):
    data["done9"] = False


def init_chart(title, names, xs, ys, smoothing=None):
    series = []
    for name, x, y in zip(names, xs, ys):
        series.append({
            "name": name,
            "data": [[px, py] for px, py in zip(x, y)]
        })
    result = {
        "options": {
            "title": title,
            #"groupKey": "my-synced-charts",
        },
        "series": series
    }
    if smoothing is not None:
        result["options"]["smoothingWeight"] = smoothing
    return result


def init(data, state):
    demo_x = [[], []] #[[1, 2, 3, 4], [2, 4, 6, 8]]
    demo_y = [[], []] #[[10, 15, 13, 17], [16, 5, 11, 9]]
    data["mGIoU"] = init_chart("GIoU",
                               names=["train", "val"],
                               xs=demo_x,
                               ys=demo_y,
                               smoothing=0.6)

    data["mObjectness"] = init_chart("Objectness",
                                     names=["train", "val"],
                                     xs=demo_x,
                                     ys=demo_y,
                                     smoothing=0.6)

    data["mClassification"] = init_chart("Classification",
                                         names=["train", "val"],
                                         xs=demo_x,
                                         ys=demo_y,
                                         smoothing=0.6)

    data["mPR"] = init_chart("Pr + Rec",
                             names=["precision", "recall"],
                             xs=demo_x,
                             ys=demo_y)

    data["mMAP"] = init_chart("mAP",
                              names=["mAP@0.5", "mAP@0.5:0.95"],
                              xs=demo_x,
                              ys=demo_y)
    state["smoothing"] = 0.6
