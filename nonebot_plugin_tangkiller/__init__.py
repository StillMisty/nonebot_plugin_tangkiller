from nonebot import on_message
from nonebot.adapters.onebot.v11 import MessageEvent, Bot, MessageSegment
from nonebot.plugin import PluginMetadata
import ultralytics
from typing import Literal
import httpx
from PIL import Image
from PIL.Image import Image as PILImage
from io import BytesIO
from pathlib import Path

from .config import Config,config

PluginMetadata(
    name="nonebot_plugin_tangkiller",
    description="检测用户所发的唐菲图片",
    usage="发送图片即可",
    type="application",
    homepage="https://github.com/StillMisty/nonebot_plugin_tangkiller",
    supported_adapters=("~.onebot_v11",),
    config=Config,
)

msg = on_message(priority=5, block=True)

# 是否撤回消息
is_withdraw = config.tangkiller_is_withdraw
confidence_threshold = config.tangkiller_confidence_threshold

# 加载模型
path = Path(__file__).parent
model = ultralytics.YOLO(path / "tang.pt")


async def process_image(img: PILImage) -> list[PILImage]:
    """处理图片,将图片转换为图片列表"""
    result = []
    if getattr(img, "is_animated", False):
        for i in range(img.n_frames):
            img.seek(i)
            frame = img.convert("RGB")
            result.append(frame)
    else:
        result.append(img.convert("RGB"))
    return result


async def detect_image(
    image: list[PILImage], confidence_threshold: float = 0.95
) -> float | Literal[False]:
    """检测图片是否包含目标对象，置信度高于confidence_threshold则返回置信度，否则返回False"""
    for frame in image:
        results = model(frame)
        if any(
            result.probs.top1 == 1
            and (top1conf := result.probs.top1conf.item()) >= confidence_threshold
            for result in results
        ):
            return top1conf
    return False


@msg.handle()
async def _(bot: Bot, event: MessageEvent):
    for seg in event.message:
        if seg.type == "image":
            url = seg.data["url"]
            async with httpx.AsyncClient(verify=False, timeout=10) as client:
                res = await client.get(url)
                if res.status_code != 200:
                    await msg.finish("图片下载失败", at_sender=True)
                    return

            # 处理图片
            img = Image.open(BytesIO(res.content))
            processed_img = await process_image(img)

            if conf := await detect_image(processed_img, confidence_threshold):
                # 是否撤回消息，撤回消息失败则不发送消息
                if is_withdraw:
                    try:
                        await bot.delete_msg(message_id=event.message_id)
                        return
                    except Exception:
                        pass

                await msg.finish(
                    MessageSegment.reply(event.message_id)
                    + f"唐菲出现了, 可信度: {int(conf*100) }%"
                )
