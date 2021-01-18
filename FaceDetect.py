import argparse
import sys
import cv2
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "-img",
        dest = "image",
        help = "Path to the image",
        default = None,
        type = str
    )
    if len(sys.argv) == 1:
        ap.print_help()
        sys.exit(1)
    return ap.parse_args()

args = parse_args()

def get_output_layers(net):
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] -1] for i in net.getUnconnectedOutLayers()]
    return output_layers

face_config = "tiny-yolo.cfg"
face_weights = "tiny-yolo_final.weights"

image = cv2.imread(args.image)
height, width, _ = image.shape

scale = 00.00392

blob = cv2.dnn.blobFromImage(image, scale, 
    (416, 416), (0,0,0), True, crop = False)

face_net = cv2.dnn.readNet(face_weights, face_config)
face_net.setInput(blob)
face_net_outputs = face_net.forward(get_output_layers(face_net))

faces = []

for face_detection in face_net_outputs:
    for detection in face_detection:
        if detection[5] > 0.1:
            center_x = int(detection[0]*width)
            center_y = int(detection[1]*height)
            w = int(detection[2]*width)
            h = int(detection[3]*height)
            x = int(center_x - (w/2))
            y = int(center_y - (h/2))
            if x < 0:
                x = 0
            if y < 0:
                y = 0
            faces.append([x, y, w, h])

face = image[y:y+h, x:x+w]
cv2.imshow("Face", face)
cv2.waitKey(0)
