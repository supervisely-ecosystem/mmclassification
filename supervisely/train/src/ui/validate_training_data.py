import supervisely_lib as sly
import sly_globals as g
import splits


@g.my_app.callback("validate_data")
@sly.timeit
@g.my_app.ignore_errors_and_show_dialog_window()
def validate_data(api: sly.Api, task_id, context, state, app_logger):
    data_report = [
        {
            "type": "info",
            "name": "total images",
            "count": 8886263,
        },
        {
            "type": "info",
            "count": 245,
            "name": "total tags",
        },
        {
            "type": "accept",
            "count": 245,
            "name": "training tags",
            "description": "number of selected tags for training"
        },
        {
            "type": "warning",
            "count": 0,
            "name": "unavailable tags",
            "description": "tags that can not be used in training",
        },
        {
            "type": "warning",
            "count": 0,
            "name": "unavailable tags",
            "description": "tags that can not be used in training",
        }
    ]


    total_images = g.project_info.items_count

    final_train_set_size = len(splits.train_set)
    final_val_set_size = len(splits.val_set)
    train_val_intersection = 0
    images_not_in_train_val_splits = 0

    tags_in_project = len(g.project_meta.tag_metas)
    training_tags = len(state["selectedTags"])

    tags_without_training_samples = 0
    tags_without_validation_samples = 0

    images_without_target_tags = 0
    images_with_collisions = 0



