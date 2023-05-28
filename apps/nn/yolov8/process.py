import os


def process_video(
        model,
        input_video: str,
        output_video: str,
):
    import torch
    import cv2


    results = model.track(
        input_video, 
            augment=True, 
        tracker=os.path.join(os.path.dirname(__file__), 'trackers', 'tracker.yaml'),
        device=(0 if torch.cuda.is_available() else 'cpu'),
    )

    cap = cv2.VideoCapture(input_video)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
    
    i = 0
    while cap.isOpened():
        # Read a frame from the video
        success, _ = cap.read()
        if success:
            # Run YOLOv8 inference on the frame
            # Visualize the results on the frame
            annotated_frame = results[i].plot()

            # Display the annotated frame
            # Write the padded frame with overlay to the output video
            out.write(annotated_frame)

            # Break the loop if 'q' is pressed
            i += 1
        else:
            # Break the loop if the end of the video is reached
            break

    # Release the video capture object and close the display window
    cap.release()
    out.release()
    # print path of file of VideoWriter 



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('input_video', type=str)
    parser.add_argument('output_video', type=str)
    parser.add_argument('-W', type=str, required=True)

    args = parser.parse_args()
    
    import ultralytics
    model = ultralytics.YOLO(args.W, task='detect')
    process_video(model, args.input_video, args.output_video)
    print('Done')