from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, Response, status

import inference


MODEL = None


@asynccontextmanager
async def lifespan(_: FastAPI):
    global MODEL
    MODEL = inference.initialize()
    yield
    MODEL = None


app = FastAPI(lifespan=lifespan)


@app.get("/health")
async def health():
    code = status.HTTP_200_OK if MODEL is not None else status.HTTP_503_SERVICE_UNAVAILABLE
    return Response(status_code=code)


@app.post("/invoke")
async def invoke():
    inference.run(MODEL)
    return Response(status_code=status.HTTP_201_CREATED)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=4743)
