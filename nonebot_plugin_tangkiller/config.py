from pydantic import BaseModel


class Config(BaseModel):
    # 是否撤回消息
    tangkiller_is_withdraw: bool = False
    # 置信度阈值
    tangkiller_confidence_threshold: float = 0.95
