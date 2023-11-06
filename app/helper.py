from langchain.chat_models import AzureChatOpenAI
from langchain.chat_models.base import BaseChatModel
import config


def make_chat_model() -> BaseChatModel:
    chat_model = AzureChatOpenAI(
        openai_api_type=config.OPENAI_API_TYPE,
        openai_api_key=config.OPENAI_API_KEY,
        openai_api_base=config.OPENAI_API_BASE,
        openai_api_version=config.OPENAI_API_VERSION,
        deployment_name=config.AZURE_DEPLOYMENT,
        model_version=config.AZURE_CHAT_MODEL,
    )
    return chat_model
