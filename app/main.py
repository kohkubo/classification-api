from fastapi import FastAPI, HTTPException
from engine.predict import classify_news
from pydantic import BaseModel


class News(BaseModel):
    title: str

    class Config:
        schema_extra = {
            "example": {
                "title": "女性を潤す新たな注目ワードはアミノ酸",
            }
        }


app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.post("/v1/classify_news_type/")
async def classify_news_type(news: News):
    try:
        predict = classify_news(news.title)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {
        "label": predict[0],
        "label_name": predict[1],
        "score": float(predict[3]),
        "prediction_scores": predict[2],
    }
