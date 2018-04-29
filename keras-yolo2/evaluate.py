
yolo = YOLO(...)  # Create model
yolo.load_weights(weights_path)  # Load weights
mAP = DetectionMAP(num_classes)  # Initialise metric
for image in images:
    boxes = yolo.predict(image)
    # prepare objects pred_bb, pred_classes, pred_conf, gt_bb and gt_classes
    mAP.evaluate(pred_bb, pred_classes, pred_conf, gt_bb, gt_classes)  # Update the metric

mAP.plot()  # Get the value of the metric and precision-recall plot for each class
