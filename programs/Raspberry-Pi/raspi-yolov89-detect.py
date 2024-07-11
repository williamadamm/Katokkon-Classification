
import cv2
import argparse
import time
import serial


from ultralytics import YOLO
import supervision as sv

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLOv8 Live")
    parser.add_argument(
        "--webcam-resolution",
        default=[1280, 720],
        nargs=2,
        type=int
    )
    parser.add_argument(
        "--modeldir",
        default="yolov8n.pt",
        help='Folder the .pt file is located in'
        )
    parser.add_argument(
        "--source",
        default=0,
        type=int
    )
    args = parser.parse_args()
    return args

def write_list_to_file(my_list, filename):
  """
  Writes a list to a text file with each element on a new line.

  Args:
    my_list: The list to be written to the file.
    filename: The name of the text file.
  """
  with open(filename, 'w') as f:
    for item in my_list:
      f.seek(0, 2)
      f.write(str(item) + '\n')  # Convert each item to string and add newline


results_list = []

def main():
    args = parse_arguments()
    frame_width, frame_height = args.webcam_resolution
    MODEL_NAME = args.modeldir
    CAMERA_SOURCE = args.source
    
    cap = cv2.VideoCapture(CAMERA_SOURCE)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
    
    model = YOLO(MODEL_NAME)
    
    box_annotator = sv.BoxAnnotator(
        thickness=2,
        text_thickness=2,
        text_scale=1
    )
    
    #calculating fps
    prev_frame_time = 0
    new_frame_time = 0
    
    #loop through image from frame
    while True:

        ret, frame = cap.read()
        
        result = model(frame, agnostic_nms=True)
        object_classes = result[0].boxes.cls.to('cpu').tolist()
        speed_dict = result[0].speed['inference']
        results_list.append(speed_dict)

        #print out detection results 
        for i in object_classes:
            print(i)
                
            
        detections = sv.Detections.from_ultralytics(result[0])

        labels = [
            f"{model.model.names[class_id]} {confidence:.2f}"
            for class_id, confidence
            in zip(detections.class_id, detections.confidence)
        ]

        frame = box_annotator.annotate(
            scene=frame,
            detections=detections,
            labels=labels
        )
        
        font = cv2.FONT_HERSHEY_SIMPLEX 
        new_frame_time = time.time()
        
        #calculating fps
        fps = 1/(new_frame_time-prev_frame_time)
        prev_frame_time = new_frame_time
        fps = float("{:.2f}".format(fps))
        fps = str(fps)
        
        cv2.putText(frame,fps,(7,35), font, 1, (100, 255, 0), 1, cv2.LINE_AA)
        
        cv2.imshow("yolo", frame)

        if (cv2.waitKey(30) == 27):
            break

    #write_list_to_file(results_list, "yolov8-inference.txt")
        
if __name__ == "__main__":
    main()
            