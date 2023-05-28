import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import hypercorn.asyncio as hypercorn

from .routes.score_card import score_card_router

# from ..nn.yolov8.fastapi.routes import router as yolov8_router, serve_yolo

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logging.getLogger().handlers.clear()

app = FastAPI()

app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
)

app.include_router(score_card_router)



# @hydra.main(config_path="config", config_name="default", version_base="1.1")
# def main(config: DictConfig):

#     # CORS
#     # Allow all origins

   

#     config['yolo']['pt_path'] = hydra.utils.to_absolute_path(config['yolo']['pt_path'])
    
    
#     # uvicorn.run("apps.fastapi.app:app", **config['uvicorn'])
#     # serve(app, **config['server'])
#     hypercorn_config = hypercorn.Config().from_mapping(**config['server'])
#     hypercorn_config.from_mapping(**config['server'])
#     asyncio.run(
#         hypercorn.serve(app, hypercorn_config)
#     )

# if __name__ == "__main__":
#     main()