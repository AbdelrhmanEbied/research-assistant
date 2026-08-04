import os

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

from agent.agent_schemas import RouteQuery

load_dotenv()

llm = init_chat_model(
    model = "gemini-3.5-flash-lite",
    model_provider= "google_genai",
)

classifier_llm = init_chat_model(
    model = "gemini-3.5-flash-lite",
    model_provider= "google_genai",
).with_structured_output(RouteQuery)

