import supervisely_lib as sly
from sly_train_progress import init_progress


def init(data, state):
    init_progress("Epoch", data)
    init_progress("Iter", data)
    init_progress("UploadDir", data)
    data["eta"] = None

    init_charts(data, state)

    state["collapsed9"] = True
    state["disabled9"] = True
    state["done9"] = False

    state["started"] = False


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


def init_charts(data, state):
    #demo_x = [[], []] #[[1, 2, 3, 4], [2, 4, 6, 8]]
    #demo_y = [[], []] #[[10, 15, 13, 17], [16, 5, 11, 9]]
    data["chartLR"] = init_chart("LR", names=["train"], xs=[[]], ys=[[]], smoothing=0.6)
    data["chartTrainLoss"] = init_chart("Loss", names=["train"], xs=[[]], ys=[[]], smoothing=0.6)
    data["chartValAccuracy"] = init_chart("Val Acc", names=["top-1", "top-5"], xs=[[], []], ys=[[], []], smoothing=0.6)

    data["chartTime"] = init_chart("Time", names=["time"], xs=[[]], ys=[[]])
    data["chartDataTime"] = init_chart("Data Time", names=["data_time"], xs=[[]], ys=[[]])
    data["chartMemory"] = init_chart("Memory", names=["memory"], xs=[[]], ys=[[]])
    state["smoothing"] = 0.6

