def nms(boxes, scores, iou_threshold):
    """
    Apply Non-Maximum Suppression to filter overlapping bounding boxes.
    Returns a list of indices for the kept boxes in order of selection.
    """
    if not boxes:
        return []

    # 1. Helper function for IoU (Intersection over Union)
    def compute_iou(boxA, boxB):
        # Calculate intersection coordinates
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        # Compute area of intersection
        inter_width = max(0, xB - xA)
        inter_height = max(0, yB - yA)
        inter_area = inter_width * inter_height

        if inter_area == 0:
            return 0.0

        # Compute area of both boxes
        areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

        # IoU = Area of Intersection / Area of Union
        iou = inter_area / float(areaA + areaB - inter_area)
        return iou

    # 2. Sort indices of boxes by scores in descending order
    # indices = [i for i, score in sorted(enumerate(scores), key=lambda x: x[1], reverse=True)]
    indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    
    kept_indices = []
    
    # 3. Greedy selection and suppression
    while indices:
        # Pick the box with the highest score
        current = indices.pop(0)
        kept_indices.append(current)
        
        # Filter remaining indices: keep only those with IoU < threshold
        remaining_indices = []
        for i in indices:
            if compute_iou(boxes[current], boxes[i]) < iou_threshold:
                remaining_indices.append(i)
        
        indices = remaining_indices
        
    return kept_indices