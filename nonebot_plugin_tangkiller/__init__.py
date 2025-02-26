from nonebot import on_message
from nonebot.adapters.onebot.v11 import MessageEvent, Bot, MessageSegment
from nonebot.plugin import PluginMetadata
from nonebot.log import logger
import ultralytics
from typing import Literal
import httpx
import ssl
from PIL import Image
from PIL.Image import Image as PILImage
from io import BytesIO
from pathlib import Path

from .config import Config, config

__plugin_meta__ = PluginMetadata(
    name="唐菲检测",
    description="检测用户所发的唐菲图片",
    usage="发送图片即可",
    type="application",
    homepage="https://github.com/StillMisty/nonebot_plugin_tangkiller",
    supported_adapters={"~onebot.v11"},
    config=Config,
)

msg = on_message(priority=5, block=True)

# 是否撤回消息
is_withdraw = config.tangkiller_is_withdraw
confidence_threshold = config.tangkiller_confidence_threshold

# 忽略ssl证书
ssl_context = ssl.create_default_context()
ssl_context.check_hostname = False
ssl_context.verify_mode = ssl.CERT_NONE
ssl_context.set_ciphers("DEFAULT@SECLEVEL=2")

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
        if seg.type != "image":
            continue

        url = seg.data["url"]
        try:
            async with httpx.AsyncClient(verify=ssl_context) as client:
                res = await client.get(url, timeout=10)
                res.raise_for_status()
        except Exception as e:
            logger.warning(f"下载图片失败: {e}")
            continue

        img = Image.open(BytesIO(res.content))
        processed_img = await process_image(img)

        if not (conf := await detect_image(processed_img, confidence_threshold)):
            continue

        if is_withdraw:
            try:
                await bot.delete_msg(message_id=event.message_id)
                return
            except Exception as e:
                logger.warning(f"撤回图片失败: {e}")

        reply = (
            MessageSegment.reply(event.message_id)
            + f"唐菲出现了, 可信度: {int(conf * 100)}%"
        )
        await msg.finish(reply)
