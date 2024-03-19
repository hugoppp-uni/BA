from fastapi import FastAPI
from starlette.requests import Request

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.post("/upload")
async def upload(request: Request):
    body = await request.body()
    body_str = body.decode('utf-8')
    writeToFile(body_str)
    return


def writeToFile(content: str):
    with open("data.ply", "w") as f:
        f.write(content)
