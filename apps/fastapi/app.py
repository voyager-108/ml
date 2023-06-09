import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .routes.score_card import score_card_router

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
