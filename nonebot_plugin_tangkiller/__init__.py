from typing import Literal
import httpx
from nonebot import on_message
from nonebot.adapters.onebot.v11 import MessageEvent, Bot, MessageSegment
import ultralytics
from PIL import Image
from io import BytesIO
from pathlib import Path
from PIL.Image import Image as PILImage

msg = on_message(priority=5, block=True)

# 是否撤回消息
is_withdraw = False

# 加载模型
path = Path(__file__).parent
model = ultralytics.YOLO(path / "tang.pt")


async def process_image(img: PILImage) -> list[PILImage] | PILImage:
    """处理图片,将GIF转换为图片列表或返回单张图片"""
    if getattr(img, "is_animated", False):
        frames = []
        for i in range(img.n_frames):
            img.seek(i)
            frame = img.convert("RGB")
            frames.append(frame)
        return frames
    return img.convert("RGB")


async def detect_image(
    image: list[PILImage] | PILImage, confidence_threshold: float = 0.95
) -> float | Literal[False]:
    """检测图片是否包含目标对象"""
    if isinstance(image, list):
        # 处理GIF的每一帧
        for frame in image:
            results = model(frame)
            if any(
                result.probs.top1 == 1
                and (top1conf := result.probs.top1conf.item()) > confidence_threshold
                for result in results
            ):
                return top1conf
    else:
        # 处理单张图片
        results = model(image)
        if any(
            result.probs.top1 == 1
            and (top1conf := result.probs.top1conf.item()) > confidence_threshold
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

            img = Image.open(BytesIO(res.content))
            processed_img = await process_image(img)

            # 置信度大于0.95则认为是唐菲，可以根据需要调整
            if conf := await detect_image(processed_img):
                # 是否撤回消息
                if is_withdraw:
                    await bot.delete_msg(message_id=event.message_id)
                else:
                    await msg.finish(
                        MessageSegment.reply(event.message_id)
                        + f"唐菲出现了, 可信度: {int(conf*100) }%"
                    )
                return
