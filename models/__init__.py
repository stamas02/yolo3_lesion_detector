from models import yolov2_d19, yolov2_r50, yolov2_slim, yolov3, yolov3_spp, yolov3_tiny, config

model_dict = {'yolov2_d19': yolov2_d19.YOLOv2D19,
              'yolov2_r50': yolov2_r50.YOLOv2R50,
              'yolov2_slim': yolov2_slim.YOLOv2Slim,
              'yolov3': yolov3.YOLOv3,
              'yolov3_spp': yolov3_spp.YOLOv3Spp,
              'yolov3_tiny': yolov3_tiny.YOLOv3tiny}

model_cfg_dict = {'yolov2_d19': config.yolov2_d19_cfg,
                  'yolov2_r50': config.yolov2_r50_cfg,
                  'yolov2_slim': config.yolov2_slim_cfg,
                  'yolov3': config.yolov3_d53_cfg,
                  'yolov3_spp': config.yolov3_d53_cfg,
                  'yolov3_tiny': config.yolov3_tiny_cfg}
