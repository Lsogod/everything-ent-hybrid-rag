from pydantic import BaseModel


class SessionsResponse(BaseModel):
    items: list[dict]


class MessagesResponse(BaseModel):
    items: list[dict]
