import os
import sys
import supervisely_lib as sly
import sly_globals as g
import train_config


def init_script_arguments(state, project_dir):
    sys.argv.append(train_config)
    #sys.argv.append(os.path.join(g.root_source_dir, "configs/resnet/resnet18_b16x8_cifar10.py"))
    sys.argv.extend(["--work-dir", "/data"])


    # global local_artifacts_dir, remote_artifacts_dir
    # sys.argv.append("--sly")

    # data_path = os.path.join(yolov5_format_dir, 'data_config.yaml')
    # sys.argv.extend(["--data", data_path])
    #
    # hyp_content = yaml.safe_load(state["hyp"][state["hypRadio"]])
    # hyp = os.path.join(my_app.data_dir, 'hyp.custom.yaml')
    # with open(hyp, 'w') as f:
    #     f.write(state["hyp"][state["hypRadio"]])
    # sys.argv.extend(["--hyp", hyp])
    #
    # if state["weightsInitialization"] == "coco":
    #     model_name = state['selectedModel'].lower()
    #     _sub_path = "models/hub" if model_name.endswith('6') else "models"
    #     cfg = os.path.join(g.root_source_dir, _sub_path, f"{model_name}.yaml")
    #     sys.argv.extend(["--cfg", cfg])
    # sys.argv.extend(["--weights", state["weightsPath"]])
    #
    # sys.argv.extend(["--epochs", str(state["epochs"])])
    # sys.argv.extend(["--batch-size", str(state["batchSize"])])
    # sys.argv.extend(["--img-size", str(state["imgSize"]), str(state["imgSize"])])
    # if state["multiScale"]:
    #     sys.argv.append("--multi-scale")
    # if state["singleClass"]:
    #     sys.argv.append("--single-cls")
    # sys.argv.extend(["--device", state["device"]])
    #
    # if "workers" in state:
    #     sys.argv.extend(["--workers", str(state["workers"])])
    # if state["optimizer"] == "Adam":
    #     sys.argv.append("--adam")
    #
    # sys.argv.extend(["--metrics_period", str(state["metricsPeriod"])])
    # sys.argv.extend(["--project", g.runs_dir])
    # sys.argv.extend(["--name", g.experiment_name])
