- Use softmax output for reranking


- Ensemble of 9 models:
    + 3x B7 (3 crop, yolov5 crop, backfin crop)
    + 2x B6 (3 crop, backfin crop)
    + 1x V2-m
    + 1x B5
    + 1x resnet like