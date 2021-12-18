# config.py


# YOLOv2 with darknet-19
yolov2_d19_cfg = {
    # network
    'backbone': 'd19',
    # for multi-scale trick
    'size': 640,
    'random_size_range': [10, 19],
    # anchor size
    'anchor_size_voc': [[2,2], [3,3], [4,4], [6,6], [11,11]],
    # train
    'max_epoch': 40,
    'ignore_thresh': 0.5
}

# YOLOv2 with resnet-50
yolov2_r50_cfg = {
    # network
    'backbone': 'r50',
    # for multi-scale trick
    'size': 640,
    'random_size_range': [10, 19],
    # anchor size
    'anchor_size': [[2,2], [3,3], [4,4], [6,6], [11,11]],
    # train
    'max_epoch': 40,
    'ignore_thresh': 0.5
}

# YOLOv2Slim
yolov2_slim_cfg = {
    # network
    'backbone': 'dtiny',
    # for multi-scale trick
    'size': 640,
    'random_size_range': [10, 19],
    # anchor size
    'anchor_size': [[2,2], [3,3], [4,4], [6,6], [11,11]],
    # train
    'max_epoch': 40,
    'ignore_thresh': 0.5
}

# YOLOv3 / YOLOv3Spp
yolov3_d53_cfg = {
    # network
    'backbone': 'd53',
    # for multi-scale trick
    'size': 640,
    'random_size_range': [10, 19],
    # anchor size
    'anchor_size': [[10, 10], [30, 30], [60, 60],
                    [100, 100], [150, 150], [200, 200]],
    # train
    'max_epoch': 40,
    'ignore_thresh': 0.5
}

# YOLOv3Tiny
yolov3_tiny_cfg = {
    # network
    'backbone': 'd-light',
    # for multi-scale trick
    'size': 640,
    'random_size_range': [10, 19],
    # anchor size
    'anchor_size': [[10, 10], [30, 30], [60, 60],
                    [100, 100], [150, 150], [200, 200]],
    # train
    'max_epoch': 10,
    'ignore_thresh': 0.5
}
