from fastapi import FastAPI, HTTPException
from mangum import Mangum
from engine.predict import classify_news
from pydantic import BaseModel
import os

app = FastAPI()


class News(BaseModel):
    title: str

    class Config:
        schema_extra = {
            "example": {
                "title": "女性を潤す新たな注目ワードはアミノ酸",
            }
        }


@app.get("/", status_code=200)
async def root():
    return {"message": "Hello World"}


@app.post("/v1/news/classify-type/")
async def classify_news_type(news: News):
    # try:
    #     return test(news.title)
    # except Exception as e:
    #     raise HTTPException(status_code=500, detail=str(e))

    try:
        predict = classify_news(news.title)
        return {
            "label": predict[0],
            "label_name": predict[1],
            "score": float(predict[3]),
            "prediction_scores": predict[2],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


lambda_handler = Mangum(app)
