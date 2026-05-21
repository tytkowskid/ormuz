from langchain_openai import ChatOpenAI

def get_standard_model():

    return ChatOpenAI(
        model="gpt-4.1-mini",
        temperature=0
    )