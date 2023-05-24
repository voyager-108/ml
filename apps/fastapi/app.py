import asyncio
import os
from fastapi import FastAPI
import hydra
from fastapi.middleware.cors import CORSMiddleware
from omegaconf import DictConfig
from waitress import serve # WSGI
import hypercorn.asyncio as hypercorn
from ..nn.yolov8.fastapi.routes import router as yolov8_router, serve_yolo

app = FastAPI()

app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
)

serve_yolo(os.environ['YOLO_CONFIG'])
app.include_router(yolov8_router)


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