import asyncio
from fastapi import FastAPI
import hydra
from fastapi.middleware.cors import CORSMiddleware
from omegaconf import DictConfig
from waitress import serve
import hypercorn.asyncio as hypercorn


@hydra.main(config_path="config", config_name="default", version_base="1.1")
def main(config: DictConfig):
    import uvicorn
    from ..nn.yolov8.fastapi.routes import router as yolov8_router, serve_yolo
    app = FastAPI()

    # CORS
    # Allow all origins

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    config['yolo']['pt_path'] = hydra.utils.to_absolute_path(config['yolo']['pt_path'])
    serve_yolo(config['yolo'])


    app.include_router(yolov8_router)
    
    # uvicorn.run("apps.fastapi.app:app", **config['uvicorn'])
    # serve(app, **config['server'])
    hypercorn_config = hypercorn.Config()
    hypercorn_config.from_mapping(config['server'])
    asyncio.run(
        hypercorn.serve(app, hypercorn_config)
    )

if __name__ == "__main__":
    main()