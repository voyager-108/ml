import cv2
import os
import logging

def split_video_by_frames(
        video_file_path: str, 
        take_each_n: int, 
        output_collection: str,
        verbose: bool
):
    if verbose:
        logger = logging.getLogger("utility:split")
        logger.addHandler(
            logging.StreamHandler()
        )
    # Open the video file
    video = cv2.VideoCapture(video_file_path)
    
    # Create the output directory if it doesn't exist
    os.makedirs(output_collection, exist_ok=True)
    
    # Get the total number of frames in the video
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Calculate the number of trailing zeros needed based on the total frames
    num_zeros = len(str(total_frames))
    
    if verbose:
        logger.info(f"File loaded: {video_file_path}")
        logger.info(f"File {video_file_path}. Total frames: {total_frames}")

    # Initialize variables
    frame_count = 0
    frame_num = 0
    
    while True:
        # Read the next frame
        ret, frame = video.read()
        
        # Break the loop if no frame is available
        if not ret:
            break
        
        # Increment the frame count
        frame_count += 1
        
        # Process every take_each_n frames
        if frame_count % take_each_n == 0:
            # Convert the frame number to a string with trailing zeros
            frame_num_str = str(frame_num).zfill(num_zeros)
                
            # Save the frame to a file
            output_file = f"{output_collection}/video__f{frame_num_str}.jpg"
            cv2.imwrite(output_file, frame)
            if verbose:
                logger.info(f"File {output_file}. Frame {frame_count} --> {output_file}")
            # Increment the frame number
            frame_num += 1
    
    # Release the video capture
    video.release()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s\t%(name)s\t::\t%(message)s"
    )
    logging.root.handlers.clear()


    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "video_file_path",
        help="Path to the video file to split"
    )

    parser.add_argument(
        "-n",
        default=1,
        type=int,
        help="Take each n frames"
    )

    parser.add_argument(
        "-o",
        default="output",
        help="Output directory"
    )

    parser.add_argument(
        '--as-folder',
        action='store_true',
        help="Save the frames in a folder with the same name as the video"
    )

    args = parser.parse_args()
    collection = args.o or f"{args.video_file_path}_frames"

    path = args.video_file_path
    if args.as_folder:
        path = os.path.join(
            collection,
            os.path.splitext(
                os.path.basename(args.video_file_path)
            )[0]
        )


    print(f"Saving frames in {path}")

    split_video_by_frames(
        args.video_file_path,
        args.n,
        path,
        verbose=True
    )

 