
import cv2
import numpy as np

COLORS = {
    0:(0,0,255),
    1:(0,255,0),
    2:(255,0,0),
    3:(255,255,0),
    4:(255,0,255),
    5:(0,255,255),
}

def counts_and_draw(image, multiclass_mask, label_defs, vis_imgf):
    counts = {}
    for idx,(label, deff) in enumerate(label_defs.items()):
        binary_mask = (multiclass_mask==label).astype('uint8')
        num_instances, instance_mask = cv2.connectedComponents(binary_mask)
        # num_instances, instance_mask = countours_v2(binary_mask)
        count = np.sum(np.unique(instance_mask)>0)
        counts[deff] = count
        if vis_imgf:
            image = draw_contour_from_mask( image = image, 
                                mask = instance_mask.astype('uint8'),
                                text=deff,
                                color=COLORS[idx])
    if vis_imgf:
        cv2.imwrite(vis_imgf, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    
    return counts

def countours_v2(binary_mask):
    distance = cv2.distanceTransform(binary_mask, cv2.DIST_L2, 5)

    # Threshold the distance transform to identify separate regions
    _, markers = cv2.threshold(distance, 0.5 * distance.max(), 255, 0)
    markers = np.uint8(markers)

    # Use connected components on the refined markers
    num_components, labels = cv2.connectedComponents(markers)
    return num_components, labels

def draw_contour_from_mask(image, mask, text, color):
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 
    image = draw_image(image, contours, [text]*len(contours), color=color)
    return image

def draw_image(image, pts, texts, color = (0,0,255), line_thickness=2, down_scale=1, draw_text=False):
    pts = [(np.array(pt)/down_scale).astype('int32') for pt in pts]
    image = cv2.drawContours(image, pts, -1, color, line_thickness, lineType=cv2.LINE_8)
    if draw_text:
        fontScale=2
        textColor = color
        text_thickness = 2
        for pt, text in zip(pts, texts):
            pt = np.squeeze(pt)
            bot_right_idx = pt.sum(axis=-1).argmax()
            org = pt[bot_right_idx]
            # draw text at bottom right
            cv2.putText(image, text, org, cv2.FONT_HERSHEY_PLAIN, fontScale, 
                    textColor, text_thickness, cv2.LINE_AA) 
    return image
