import numpy as np
import uvicorn
from fastapi import FastAPI

app = FastAPI()


recomendations_for_offers = {}
click_offer_pairs = {}
offers_reward = {}
conversions = {}


@app.put("/feedback/")
def feedback(click_id: int, reward: float) -> dict:
    """Get feedback for particular click"""
    # Response body consists of click ID
    # and accepted click status (True/False)
    response = {
        "click_id": click_id,
        "offer_id": ...,
        "is_conversion": ...,
        "reward": ...
    }
    return response


@app.get("/offer_ids/{offer_id}/stats/")
def stats(offer_id: int) -> dict:
    """Return offer's statistics"""
    response = {
        "offer_id": offer_id,
        "clicks": ...,
        "conversions": ...,
        "reward": ...,
        "cr": ...,
        "rpc": ...
    }
    return response


@app.get("/sample/")
def sample(click_id: int, offer_ids: str) -> dict:
    """Greedy sampling"""
    # Parse offer IDs
    offers_ids = [int(offer) for offer in offer_ids.split(",")]

    # Sample top offer ID
    ...

    # Prepare response
    response = {
        "click_id": click_id,
        "offer_id": offer_id,
        "sampler": "random",
    }

    return response


def main() -> None:
    """Run application"""
    uvicorn.run("app:app", host="localhost")


if __name__ == "__main__":
    main()