import os
import json
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

# Basic Hari's project for recommending alternate phrases.
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

model = ChatGroq(model="Gemma2-9b-It", groq_api_key=groq_api_key)

app = FastAPI()

system_message = SystemMessage(
    content="""
ou are an expert assistant specializing in crafting clear, engaging, and appealing product descriptions.
 When provided with a product description, your task is to:
Enhance the description to make it cozier, more appealing, and customer-focused.Give the response in a array of strings.
Give me atleast 3 recommendations for the product description. the length of the array should be 3 or more.
I dont need other details from you.Just the product result. The response should be in json. Array of strings
    """
)


class RecommendationRequest(BaseModel):
    user_product_description: str


class RecommendationResponse(BaseModel):
    result: list[str]


@app.post("/rephrase")
async def rephrase_product_description(request: RecommendationRequest):
    if not request.user_product_description:
        raise HTTPException(
            status_code=400, detail="Please provide a product description."
        )

    user_message = HumanMessage(content=request.user_product_description)

    # Invoke the model and check response structure
    response = model.invoke([system_message, user_message])
    actual_result = response.content

    json_result = json.dumps(actual_result)
    print(actual_result)
    return {"result": json_result}
