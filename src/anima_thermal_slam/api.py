from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel


class PredictRequest(BaseModel):
    frame_u16_flat: list[float]
    height: int
    width: int


app = FastAPI(title="ANIMA Thermal-SLAM API", version="0.1.0")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/ready")
def ready() -> dict[str, str]:
    return {"status": "ready"}


@app.post("/predict")
def predict(_req: PredictRequest) -> dict[str, str]:
    # Placeholder endpoint; wire to runtime inference service in next iteration.
    return {"status": "not_implemented", "message": "Use CLI inference for now."}


def main() -> None:
    import uvicorn

    uvicorn.run("anima_thermal_slam.api:app", host="0.0.0.0", port=8080, reload=False)


if __name__ == "__main__":
    main()
