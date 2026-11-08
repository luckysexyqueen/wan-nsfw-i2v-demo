---
title: Wan 2.2 NSFW I2V
emoji: 🎬
colorFrom: purple
colorTo: pink
sdk: gradio
sdk_version: 6.0.1
python_version: "3.12"
app_file: app.py
pinned: false
short_description: Adult image-to-video with Wan 2.2 and NSFW LoRA
startup_duration_timeout: 1h
---

# Wan 2.2 NSFW Image-to-Video

Upload an image and a motion prompt to generate a short adult video. There is no built-in safety filter.

This Space runs a lightning Wan 2.2 14B image-to-video pipeline and, by default, applies the [Wan 2.2 General NSFW LoRA](https://huggingface.co/lopi999/Wan2.2-I2V_General-NSFW-LoRA) (trigger word: `nsfwsks`).

## Notes

- For adults only. Do not generate sexual content involving minors.
- You are responsible for what you generate and share.
- First startup downloads a large video model, so it can take a while.
- Base pipeline: Wan 2.2 I2V (Apache 2.0). NSFW LoRA: community adapter mirrored from Civitai.

