import logging
import tempfile
import shutil
import uuid

import numpy as np
import apps.nn.run as nn

from fastapi import APIRouter, File, UploadFile
from ...nn.yolov8.trackers.bytetrack import embed_tracking_into_results
from ...nn.yolov8.score import postprocess_results, assign_to_rooms
from ..stats import derive_statistics
from ...utility.split_video import split_video_by_frames



# Number of frames to skip when splitting the video
skip = 5


# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s :: %(message)s",
)

# Server logger
logger = logging.getLogger('server')
logger.setLevel(logging.INFO)
streamHandler = logging.StreamHandler()
streamHandler.setFormatter(logging.Formatter("%(asctime)s :: %(message)s"))
logger.addHandler(streamHandler)

# Score card router
score_card_router = APIRouter(
    prefix='/score-card'
)



@score_card_router.post('/v2/video')
def process_video_for_score_card_v2(
    embeddings: list[str] = None,
    yolo_results: list[str] = None,
    isLast: list[str] = False,
    video: UploadFile = File(...), 
):
    video_format = video.filename.split('.')[-1]
    with tempfile.NamedTemporaryFile(prefix=video.filename, suffix=f'.{video_format.lower()}') as video_temp_file:
        shutil.copyfileobj(video.file, video_temp_file)
        video_path = video_temp_file.name
        frames = split_video_by_frames(video_path, skip, return_arrays=True)
        logger.info(f"[v2] video={video.filename}, temp={video_path}, frames={len(frames)}")

        logger.info(f"[v2] video={video_path[:20]+'...'+video_path[-20:]}, task=embedder, run ( frames={len(frames)} )")
        embeddings_vector = nn.run_embedder(frames)     
        logger.info(f"[v2] video={video_path[:20]+'...'+video_path[-20:]}, task=embedder, completed ( frames={len(frames)} , embeddings.shape={embeddings_vector.shape} )")

        if embeddings and isLast:
            loaded_embeddings = list(map(nn.storage.get, embeddings))
            loaded_embeddings = np.concatenate(loaded_embeddings)
            logger.info(f"[v2] video={video_path[:20] + '...' + video_path[-20:]}, inject=embeddings, embeddings.shape={loaded_embeddings.shape}")
            embeddings_vector = np.concatenate((loaded_embeddings, embeddings_vector,))
            logger.info(f"[v2] video={video_path[:20] + '...' + video_path[-20:]}, inject=embeddings, done. embeddings_vector.shape={embeddings_vector.shape}")

        if not embeddings:
            embeddings_id = uuid.uuid4().hex
            nn.storage.put(embeddings_id, embeddings_vector)
            logger.info(f"[v2] store=embeddings, embeddings_id={embeddings_id}, embeddings.shape={embeddings_vector.shape}")

        logger.info(f"[v2] video={video_path[:20]+'...'+video_path[-20:]}, task=yolo, run ( frames={len(frames)} )")
        # yolo_outputs, yolo_vectors = nn.run_yolo(video_path)
        yolo_outputs = nn.run_yolo(video_path, raw=True)
        logger.info(f"[v2] video={video_path[:20]+'...'+video_path[-20:]}, task=yolo, completed ( frames={len(frames)} , len(yolo_outputs)={len(yolo_outputs)} )")

        if not yolo_results:
            yolo_id = uuid.uuid4().hex
            nn.storage.put(yolo_id, yolo_outputs)
            logger.info(f"[v2] store=yolo, yolo_id={yolo_id}, len(yolo_results)={len(yolo_outputs)}")
    

        if yolo_results and isLast:
            logger.info(f"[v2] video={video_path[:20] + '...' + video_path[-20:]}, inject=yolo")
            loaded_yolo_results = list(map(nn.storage.get, yolo_results))
            loaded_yolo_results = sum(loaded_yolo_results, [])
            yolo_outputs = loaded_yolo_results + yolo_outputs
            logger.info(f"[v2] video={video_path[:20] + '...' + video_path[-20:]}, inject=yolo, len(yolo_outputs)={len(yolo_outputs)}")


        if isLast:
            logger.info(f"[v2] video={video_path[:20] + '...' + video_path[-20:]}, task=embed_tracking_into_results, len(yolo_outputs)={len(yolo_outputs)}")
            yolo_outputs = embed_tracking_into_results(yolo_outputs)
            logger.info(f"[v2] video={video_path[:20] + '...' + video_path[-20:]}, task=embed_tracking_into_results, completed, len(yolo_outputs)={len(yolo_outputs)}")

            logger.info(f"[v2] video={video_path[:20] + '...' + video_path[-20:]}, task=postprocess, len(yolo_outputs)={len(yolo_outputs)}")
            yolo_out, yolo_vectors = postprocess_results(yolo_outputs)
            yolo_vectors = np.array(yolo_vectors)
            logger.info(f"[v2] video={video_path[:20] + '...' + video_path[-20:]}, task=postprocess, completed, unique={len(yolo_out)}, yolo_vectors.shape={yolo_vectors.shape}")
        
            logger.info(f"[v2] isLast=true => task=classifier, run ( embeddings.shape={embeddings_vector.shape}, yolo_vectors.shape={yolo_vectors.shape} )")
            logits = nn.run_classifier(embeddings_vector, yolo_vectors)
            logger.info(f"[v2] task=classifier, completed ( embeddings.shape={embeddings_vector.shape}, yolo_vectors.shape={yolo_vectors.shape}, logits.shape={logits.shape} )")
            
            logger.info(f"[v2] task=predict, run ( embeddings.shape={embeddings_vector.shape}, logits.shape={logits.shape} )")
            classification = nn.predict(embeddings_vector, logits, logits=True)
            logger.info(f"[v2] task=predict, completed ( embeddings.shape={embeddings_vector.shape}, logits.shape={logits.shape}, len(classification)={len(classification)} )")
            
            logger.info(f"[v2] task=assign_to_rooms, run ( len(classification)={len(classification)}, yolo_vectors.shape={yolo_vectors.shape} )")
            yolo_out = assign_to_rooms(classification, yolo_out)
            logger.info(f"[v2] task=assign_to_rooms, completed ( len(classification)={len(classification)}, yolo_vectors.shape={yolo_vectors.shape} )")

            logger.info(f"[v2] finished ( embeddings.shape={embeddings_vector.shape}, yolo_vectors.shape={yolo_vectors.shape}, logits.shape={logits.shape}, len(classification)={len(classification)}, yolo_vectors.shape={yolo_vectors.shape} )")

            return {
                'stats': derive_statistics(list(yolo_out.values())),
                'output': yolo_out,
            }
        
        else:
            logger.info(f"[v2] isLast=false => response = ( embeddings.shape={embeddings_vector.shape}, len(yolo_outputs={len(yolo_outputs)} )"
                        f" ( embeddings_id={embeddings_id}, yolo_id={yolo_id}")

            return {
                'embeddings': embeddings_id,
                'yolo': yolo_id,
            }
            

        
