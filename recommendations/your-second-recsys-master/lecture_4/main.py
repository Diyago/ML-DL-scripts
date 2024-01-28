import pickle
from typing import List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from scipy.spatial.distance import cdist

from ann.recommender import AnnoyRecommender
from config.config import recommender_conf, path_conf

def load_object(path):
    with open(path, "rb") as fh:
        obj = pickle.load(fh)
    return obj


def read_vectors_and_mappings(
    user_vectors_path, item_vectors_path, user_map_path, item_map_path
):
    return (
        load_object(user_vectors_path),
        load_object(item_vectors_path),
        load_object(user_map_path),
        load_object(item_map_path),
    )


class Response(BaseModel):
    user_id: int
    item_ids: List[int]


class Request(BaseModel):
    user_id: int
    item_whitelist: List[int]


app = FastAPI(docs_url="/docs", redoc_url="/redoc")


@app.on_event("startup")
async def startup():
    user_vectors, item_vectors, user_map, item_map = read_vectors_and_mappings(**path_conf)
    app.state.recommender = AnnoyRecommender(
        item_vectors=item_vectors,
        user_vectors=user_vectors,
        user_id_user_index_id_mapping=user_map,
        item_id_item_index_id_mapping=item_map,
        sim_function=lambda x, y: 1 - cdist(x, y, metric='cosine'),
        **recommender_conf
    )
    app.state.recommender.fit()

@app.post("/api/v1/recommend_for_user", response_model=Response)
async def recommend_for_user(request: Request):
    try:
        recommendations = app.state.recommender.recommend_single_user(
            request.user_id, request.item_whitelist
        )
    except KeyError:
        raise HTTPException(status_code=404, detail="Item or user not found")
    return Response(user_id=request.user_id, item_ids=recommendations)

@app.post("/api/v1/recommend_bruteforce", response_model=Response)
async def recommend_bruteforce(request: Request):
    recommendations = app.state.recommender.recommend_bruteforce_single_user(
        request.user_id, request.item_whitelist
    )
    return Response(user_id=request.user_id, item_ids=recommendations)
