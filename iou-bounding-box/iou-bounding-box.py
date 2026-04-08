def iou(box_a, box_b):
    """
    Compute Intersection over Union of two bounding boxes.
    box_a, box_b: [x1, y1, x2, y2]
    Returns: float (0.0 to 1.0)
    """
    # 1. Calculate the coordinates of the intersection rectangle
    inter_x1 = max(box_a[0], box_b[0])
    inter_y1 = max(box_a[1], box_b[1])
    inter_x2 = min(box_a[2], box_b[2])
    inter_y2 = min(box_a[3], box_b[3])
    
    # 2. Compute the width and height of the intersection
    # Use max(0, ...) to handle cases where the boxes do not overlap
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    
    inter_area = inter_w * inter_h
    
    # 3. Compute the area of both individual bounding boxes
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    
    # 4. Compute the Union area
    # Formula: Area(A) + Area(B) - Intersection
    union_area = area_a + area_b - inter_area
    
    # 5. Handle the zero-area edge case
    if union_area == 0:
        return 0.0
        
    return float(inter_area / union_area)