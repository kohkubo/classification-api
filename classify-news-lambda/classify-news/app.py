from fastapi import FastAPI, HTTPException
from mangum import Mangum
from engine.news_classifier import NewsClassifier
from pydantic import BaseModel

app = FastAPI()
classifier = NewsClassifier()


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
    try:
        predict = classifier.classify_news(news.title)
        return {
            "label": predict[0],
            "label_name": predict[1],
            "score": float(predict[3]),
            "prediction_scores": predict[2],
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=f"RuntimeError: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")


lambda_handler = Mangum(app)
