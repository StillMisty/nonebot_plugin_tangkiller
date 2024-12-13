# nonebot_plugin_tangkiller

## 项目介绍

基于nonebot2、Yolo11的识别唐菲插件

## 使用说明

1. 依赖

```shell
pip install ultralytics pillow httpx pydantic
```

2. 可选配置(在.env中配置,以下为默认配置)

```shell
# 是否撤回消息
tangkiller_is_withdraw = False
# 置信度阈值
tangkiller_confidence_threshold = 0.95
```

3. 使用：发送表情包即可
