import os; os.system('pip install --upgrade --no-deps spaces')
import spaces
import shutil
import subprocess
import sys
import copy
import random
import tempfile
import warnings
import time
import gc
import uuid
from tqdm import tqdm
import cv2
import numpy as np
import torch
import torch._dynamo
from torch.nn import functional as F
from PIL import Image

import gradio as gr
from diffusers import (
    FlowMatchEulerDiscreteScheduler,
    SASolverScheduler,
    DEISMultistepScheduler,
    DPMSolverMultistepInverseScheduler,
    UniPCMultistepScheduler,
    DPMSolverMultistepScheduler,
    DPMSolverSinglestepScheduler,
)
from diffusers.pipelines.wan.pipeline_wan_i2v import WanImageToVideoPipeline
from diffusers.utils.export_utils import export_to_video

from torchao.quantization import quantize_, Float8DynamicActivationFloat8WeightConfig, Int8WeightOnlyConfig
import aoti
import lora_loader

os.environ["TOKENIZERS_PARALLELISM"] = "true"
warnings.filterwarnings("ignore")
IS_ZERO_GPU = bool(os.getenv("SPACES_ZERO_GPU"))

# --- FRAME EXTRACTION JS & LOGIC ---

get_timestamp_js = """
function() {
    const video = document.querySelector('#generated-video video');
    
    if (video) {
        console.log("Video found! Time: " + video.currentTime);
        return video.currentTime;
    } else {
        console.log("No video element found.");
        return 0;
    }
}
"""


def extract_frame(video_path, timestamp):
    if not video_path:
        return None
    
    print(f"Extracting frame at timestamp: {timestamp}") 
    
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    target_frame_num = int(float(timestamp) * fps)
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if target_frame_num >= total_frames:
        target_frame_num = total_frames - 1
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame_num)
    ret, frame = cap.read()
    cap.release()
    
    if ret:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    return None

# --- END FRAME EXTRACTION LOGIC ---


def clear_vram():
    gc.collect()
    torch.cuda.empty_cache()


# RIFE
if not os.path.exists("RIFEv4.26_0921.zip"):
    print("Downloading RIFE Model...")
    subprocess.run([
        "wget", "-q",
        "https://huggingface.co/thornmaze/RIFE/resolve/main/RIFEv4.26_0921.zip",
        "-O", "RIFEv4.26_0921.zip"
    ], check=True)
    subprocess.run(["unzip", "-o", "RIFEv4.26_0921.zip"], check=True)

from train_log.RIFE_HDv3 import Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
rife_model = Model()
rife_model.load_model("train_log", -1)
rife_model.eval()


@torch.no_grad()
def interpolate_bits(frames_np, multiplier=2, scale=1.0):
    if isinstance(frames_np, list):
        T = len(frames_np)
        H, W, C = frames_np[0].shape
    else:
        T, H, W, C = frames_np.shape

    if multiplier < 2:
        if isinstance(frames_np, np.ndarray):
            return list(frames_np)
        return frames_np

    n_interp = multiplier - 1
    
    tmp = max(128, int(128 / scale))
    ph = ((H - 1) // tmp + 1) * tmp
    pw = ((W - 1) // tmp + 1) * tmp
    padding = (0, pw - W, 0, ph - H)

    def to_tensor(frame_np):
        t = torch.from_numpy(frame_np).to(device)
        t = t.permute(2, 0, 1).unsqueeze(0)
        return F.pad(t, padding).half()

    def from_tensor(tensor):
        t = tensor[0, :, :H, :W]
        t = t.permute(1, 2, 0)
        return t.float().cpu().numpy()

    def make_inference(I0, I1, n):
        if rife_model.version >= 3.9:
            res = []
            for i in range(n):
                res.append(rife_model.inference(I0, I1, (i+1) * 1. / (n+1), scale))
            return res
        else:
            middle = rife_model.inference(I0, I1, scale)
            if n == 1:
                return [middle]
            first_half = make_inference(I0, middle, n=n//2)
            second_half = make_inference(middle, I1, n=n//2)
            if n % 2:
                return [*first_half, middle, *second_half]
            else:
                return [*first_half, *second_half]

    output_frames = []
    I1 = to_tensor(frames_np[0])
    total_steps = T - 1

    with tqdm(total=total_steps, desc="Interpolating", unit="frame") as pbar:
        for i in range(total_steps):
            I0 = I1
            output_frames.append(from_tensor(I0))
            I1 = to_tensor(frames_np[i+1])
            mid_tensors = make_inference(I0, I1, n_interp)
            for mid in mid_tensors:
                output_frames.append(from_tensor(mid))
            if (i + 1) % 50 == 0:
                pbar.update(50)
        pbar.update(total_steps % 50)
        output_frames.append(from_tensor(I1))
    
    del I0, I1, mid_tensors
    torch.cuda.empty_cache()
    return output_frames


# WAN

MODEL_ID = "thornmaze/WAMU_v3_WAN2.2_I2V_LIGHTNING"

LORA_MODELS = []

MAX_DIM = 832
MIN_DIM = 480
SQUARE_DIM = 640
MULTIPLE_OF = 16
MAX_SEED = np.iinfo(np.int32).max

FIXED_FPS = 16
MIN_FRAMES_MODEL = 8
MAX_FRAMES_MODEL = 321

MIN_DURATION = round(MIN_FRAMES_MODEL / FIXED_FPS, 1)
MAX_DURATION = round(MAX_FRAMES_MODEL / FIXED_FPS, 1)

SCHEDULER_MAP = {
    "FlowMatchEulerDiscrete": FlowMatchEulerDiscreteScheduler,
    "SASolver": SASolverScheduler,
    "DEISMultistep": DEISMultistepScheduler,
    "DPMSolverMultistepInverse": DPMSolverMultistepInverseScheduler,
    "UniPCMultistep": UniPCMultistepScheduler,
    "DPMSolverMultistep": DPMSolverMultistepScheduler,
    "DPMSolverSinglestep": DPMSolverSinglestepScheduler,
}

pipe = WanImageToVideoPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
).to('cuda')
original_scheduler = copy.deepcopy(pipe.scheduler)

for i, lora in enumerate(LORA_MODELS):
    name_high_tr = lora["high_tr"].split(".")[0].split("/")[-1] + "Hh"
    name_low_tr = lora["low_tr"].split(".")[0].split("/")[-1] + "Ll"
    
    try: 
        pipe.load_lora_weights(
            lora["repo_id"],
            weight_name=lora["high_tr"],
            adapter_name=name_high_tr
        )
    
        kwargs_lora = {"load_into_transformer_2": True}
        pipe.load_lora_weights(
            lora["repo_id"],
            weight_name=lora["low_tr"],
            adapter_name=name_low_tr,
            **kwargs_lora
        )
    
        pipe.set_adapters([name_high_tr, name_low_tr], adapter_weights=[1.0, 1.0])
    
        pipe.fuse_lora(adapter_names=[name_high_tr], lora_scale=lora["high_scale"], components=["transformer"])
        pipe.fuse_lora(adapter_names=[name_low_tr], lora_scale=lora["low_scale"], components=["transformer_2"])
    
        pipe.unload_lora_weights()

        print(f"Applied: {lora['high_tr']}, hs={lora['high_scale']}/ls={lora['low_scale']}, {i+1}/{len(LORA_MODELS)}") 
    except Exception as e:
        print("Error:", str(e))
        print("Failed LoRA:", name_high_tr)
        pipe.unload_lora_weights()

quantize_(pipe.text_encoder, Int8WeightOnlyConfig())
torch._dynamo.reset()
quantize_(pipe.transformer, Float8DynamicActivationFloat8WeightConfig())
torch._dynamo.reset()
quantize_(pipe.transformer_2, Float8DynamicActivationFloat8WeightConfig())
torch._dynamo.reset()

spaces.aoti_load(
    module=pipe.transformer,
    repo_id='thornmaze/WanTransformer3DModel-sm120-cu130-raa',
)
spaces.aoti_load(
    module=pipe.transformer_2,
    repo_id='thornmaze/WanTransformer3DModel-sm120-cu130-raa',
)

default_prompt_i2v = "nsfwsks, make this image come alive, cinematic motion, smooth animation"
default_negative_prompt = "色调艳丽, 过曝, 静态, 细节模糊不清, 字幕, 风格, 作品, 画作, 画面, 静止, 整体发灰, 最差质量, 低质量, JPEG压缩残留, 丑陋的, 残缺的, 多余的手指, 画得不好的手部, 画得不好的脸部, 畸形的, 毁容的, 形态畸形的肢体, 手指融合, 静止不动的画面, 杂乱的背景, 三条腿, 背景人很多, 倒着走"

LORA_CHOICES = lora_loader.get_lora_choices()


def model_title():
    return "## Wan 2.2 NSFW Image-to-Video"


def resize_image(image: Image.Image) -> Image.Image:
    width, height = image.size
    if width == height:
        return image.resize((SQUARE_DIM, SQUARE_DIM), Image.LANCZOS)
    
    aspect_ratio = width / height
    MAX_ASPECT_RATIO = MAX_DIM / MIN_DIM
    MIN_ASPECT_RATIO = MIN_DIM / MAX_DIM

    image_to_resize = image
    if aspect_ratio > MAX_ASPECT_RATIO:
        target_w, target_h = MAX_DIM, MIN_DIM
        crop_width = int(round(height * MAX_ASPECT_RATIO))
        left = (width - crop_width) // 2
        image_to_resize = image.crop((left, 0, left + crop_width, height))
    elif aspect_ratio < MIN_ASPECT_RATIO:
        target_w, target_h = MIN_DIM, MAX_DIM
        crop_height = int(round(width / MIN_ASPECT_RATIO))
        top = (height - crop_height) // 2
        image_to_resize = image.crop((0, top, width, top + crop_height))
    else:
        if width > height:
            target_w = MAX_DIM
            target_h = int(round(target_w / aspect_ratio))
        else:
            target_h = MAX_DIM
            target_w = int(round(target_h * aspect_ratio))

    final_w = round(target_w / MULTIPLE_OF) * MULTIPLE_OF
    final_h = round(target_h / MULTIPLE_OF) * MULTIPLE_OF
    final_w = max(MIN_DIM, min(MAX_DIM, final_w))
    final_h = max(MIN_DIM, min(MAX_DIM, final_h))
    return image_to_resize.resize((final_w, final_h), Image.LANCZOS)


def resize_and_crop_to_match(target_image, reference_image):
    ref_width, ref_height = reference_image.size
    target_width, target_height = target_image.size
    scale = max(ref_width / target_width, ref_height / target_height)
    new_width, new_height = int(target_width * scale), int(target_height * scale)
    resized = target_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    left, top = (new_width - ref_width) // 2, (new_height - ref_height) // 2
    return resized.crop((left, top, left + ref_width, top + ref_height))


def get_num_frames(duration_seconds: float):
    raw = int(round(duration_seconds * FIXED_FPS))
    raw = max(MIN_FRAMES_MODEL, min(MAX_FRAMES_MODEL, raw))
    return ((raw - 1) // 4) * 4 + 1


def get_inference_duration(
    resized_image,
    processed_last_image,
    prompt,
    steps,
    negative_prompt,
    num_frames,
    guidance_scale,
    guidance_scale_2,
    current_seed,
    scheduler_name,
    flow_shift,
    frame_multiplier,
    quality,
    duration_seconds,
    safe_mode,
    lora_groups,
    progress
):
    BASE_FRAMES_HEIGHT_WIDTH = 81 * 832 * 624
    BASE_STEP_DURATION = 5.
    width, height = resized_image.size
    factor = num_frames * width * height / BASE_FRAMES_HEIGHT_WIDTH
    step_duration = BASE_STEP_DURATION * factor ** 1.5
    gen_time = int(steps) * step_duration

    if guidance_scale > 1:
        gen_time = gen_time * 2.4

    frame_factor = frame_multiplier // FIXED_FPS
    if frame_factor > 1:
        total_out_frames = (num_frames * frame_factor) - num_frames
        inter_time = (total_out_frames * 0.02)
        gen_time += inter_time

    total_time = 15 + gen_time
    if safe_mode:
        total_time = total_time * 1.30

    return total_time


@spaces.GPU(duration=get_inference_duration, size='xlarge')
def run_inference(
    resized_image,
    processed_last_image,
    prompt,
    steps,
    negative_prompt,
    num_frames,
    guidance_scale,
    guidance_scale_2,
    current_seed,
    scheduler_name,
    flow_shift,
    frame_multiplier,
    quality,
    duration_seconds,
    safe_mode=False,
    lora_groups=None,
    progress=gr.Progress(track_tqdm=True),
):
    scheduler_class = SCHEDULER_MAP.get(scheduler_name)
    if scheduler_class.__name__ != pipe.scheduler.config._class_name or flow_shift != pipe.scheduler.config.get("flow_shift", "shift"):
        config = copy.deepcopy(original_scheduler.config)
        if scheduler_class == FlowMatchEulerDiscreteScheduler:
            config['shift'] = flow_shift
        else:
            config['flow_shift'] = flow_shift
        pipe.scheduler = scheduler_class.from_config(config)

    clear_vram()

    task_name = str(uuid.uuid4())[:8]
    print(f"Generating {num_frames} frames, task: {task_name}, {duration_seconds}, {resized_image.size}, lora={lora_groups}")
    start = time.time()

    lora_loaded = False
    if lora_groups:
        try:
            lora_loaded = lora_loader.load_lora_slots(pipe, lora_groups)
            print(f"LoRA loaded: {lora_groups}")
        except Exception as e:
            print(f"LoRA warning: {e}")
            try:
                lora_loader.unload_lora(pipe)
            except Exception:
                pass

    result = pipe(
        image=resized_image,
        last_image=processed_last_image,
        prompt=prompt,
        negative_prompt=negative_prompt,
        height=resized_image.height,
        width=resized_image.width,
        num_frames=num_frames,
        guidance_scale=float(guidance_scale),
        guidance_scale_2=float(guidance_scale_2),
        num_inference_steps=int(steps),
        generator=torch.Generator(device="cuda").manual_seed(current_seed),
        output_type="np"
    )

    if lora_loaded:
        lora_loader.unload_lora(pipe)

    print("gen time passed:", time.time() - start)

    raw_frames_np = result.frames[0]
    pipe.scheduler = original_scheduler

    frame_factor = frame_multiplier // FIXED_FPS
    if frame_factor > 1:
        start = time.time()
        print(f"Processing frames (RIFE Multiplier: {frame_factor}x)...")
        rife_model.device()
        rife_model.flownet = rife_model.flownet.half()
        final_frames = interpolate_bits(raw_frames_np, multiplier=int(frame_factor))
        print("Interpolation time passed:", time.time() - start)
    else:
        final_frames = list(raw_frames_np)

    final_fps = FIXED_FPS * int(frame_factor)

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmpfile:
        video_path = tmpfile.name

    start = time.time()
    with tqdm(total=3, desc="Rendering Media", unit="clip") as pbar:
        pbar.update(2)
        export_to_video(final_frames, video_path, fps=final_fps, quality=quality)
        pbar.update(1)
    print(f"Export time passed, {final_fps} FPS:", time.time() - start)

    return video_path, task_name


def generate_video(
    input_image,
    last_image,
    prompt,
    steps=4,
    negative_prompt=default_negative_prompt,
    duration_seconds=MAX_DURATION,
    guidance_scale=1,
    guidance_scale_2=1,
    seed=42,
    randomize_seed=False,
    quality=5,
    scheduler="UniPCMultistep",
    flow_shift=6.0,
    frame_multiplier=16,
    safe_mode=False,
    # original 3
    lora1_preset="General NSFW Booster",
    lora1_high_scale=1.0,
    lora1_low_scale=1.0,
    lora2_preset="(None)",
    lora2_high_scale=1.0,
    lora2_low_scale=1.0,
    lora3_preset="(None)",
    lora3_high_scale=1.0,
    lora3_low_scale=1.0,
    # my 5
    my1_repo="", my1_high="(None)", my1_hs=1.0, my1_low="(None)", my1_ls=1.0,
    my2_repo="", my2_high="(None)", my2_hs=1.0, my2_low="(None)", my2_ls=1.0,
    my3_repo="", my3_high="(None)", my3_hs=1.0, my3_low="(None)", my3_ls=1.0,
    my4_repo="", my4_high="(None)", my4_hs=1.0, my4_low="(None)", my4_ls=1.0,
    my5_repo="", my5_high="(None)", my5_hs=1.0, my5_low="(None)", my5_ls=1.0,
    video_component=True,
    progress=gr.Progress(track_tqdm=True),
):
    if input_image is None:
        raise gr.Error("Please upload an input image.")

    num_frames = get_num_frames(duration_seconds)
    current_seed = random.randint(0, MAX_SEED) if randomize_seed else int(seed)
    resized_image = resize_image(input_image)

    processed_last_image = None
    if last_image:
        processed_last_image = resize_and_crop_to_match(last_image, resized_image)

    lora_groups = [
        {"preset": lora1_preset, "high_scale": lora1_high_scale, "low_scale": lora1_low_scale},
        {"preset": lora2_preset, "high_scale": lora2_high_scale, "low_scale": lora2_low_scale},
        {"preset": lora3_preset, "high_scale": lora3_high_scale, "low_scale": lora3_low_scale},
        {"repo": my1_repo, "file_high": my1_high, "file_low": my1_low, "high_scale": my1_hs, "low_scale": my1_ls},
        {"repo": my2_repo, "file_high": my2_high, "file_low": my2_low, "high_scale": my2_hs, "low_scale": my2_ls},
        {"repo": my3_repo, "file_high": my3_high, "file_low": my3_low, "high_scale": my3_hs, "low_scale": my3_ls},
        {"repo": my4_repo, "file_high": my4_high, "file_low": my4_low, "high_scale": my4_hs, "low_scale": my4_ls},
        {"repo": my5_repo, "file_high": my5_high, "file_low": my5_low, "high_scale": my5_hs, "low_scale": my5_ls},
    ]

    video_path, task_n = run_inference(
        resized_image,
        processed_last_image,
        prompt,
        steps,
        negative_prompt,
        num_frames,
        guidance_scale,
        guidance_scale_2,
        current_seed,
        scheduler,
        flow_shift,
        frame_multiplier,
        quality,
        duration_seconds,
        safe_mode,
        lora_groups,
        progress,
    )
    print(f"GPU complete: {task_n}")

    return (video_path if video_component else None), video_path, current_seed


def _load_repo_files(repo_link):
    files = lora_loader.list_loras_from_repo(repo_link)
    return (
        gr.update(choices=files, value="(None)"),
        gr.update(choices=files, value="(None)"),
    )


CSS = """
#hidden-timestamp {
    opacity: 0;
    height: 0px;
    width: 0px;
    margin: 0px;
    padding: 0px;
    overflow: hidden;
    position: absolute;
    pointer-events: none;
}
"""


with gr.Blocks(delete_cache=(3600, 10800)) as demo:
    gr.Markdown(model_title())
    gr.Markdown("Adult image-to-video using Wan 2.2 with the [general NSFW LoRA](https://huggingface.co/lopi999/Wan2.2-I2V_General-NSFW-LoRA). Upload a still and describe the motion. Add the trigger word `nsfwsks` to the prompt. For adults only.")

    with gr.Row():
        with gr.Column():
            input_image_component = gr.Image(type="pil", label="Input Image", sources=["upload", "clipboard"])
            prompt_input = gr.Textbox(label="Prompt", value=default_prompt_i2v)
            duration_seconds_input = gr.Slider(minimum=MIN_DURATION, maximum=MAX_DURATION, step=0.1, value=3.5, label="Duration (seconds)", info=f"Clamped to model's {MIN_FRAMES_MODEL}-{MAX_FRAMES_MODEL} frames at {FIXED_FPS}fps.")
            frame_multi = gr.Dropdown(
                choices=[FIXED_FPS, FIXED_FPS*2, FIXED_FPS*4, FIXED_FPS*8],
                value=FIXED_FPS,
                label="Video Fluidity (Frames per Second)",
                info="Extra frames will be generated using flow estimation, which estimates motion between frames to make the video smoother."
            )
            safe_mode_checkbox = gr.Checkbox(
                label="🛠️ Safe Mode",
                value=True,
                info="Requests 30% extra processing time to try to prevent unfinished tasks when the server is busy."
            )
            with gr.Accordion("Advanced Settings", open=False):
                last_image_component = gr.Image(type="pil", label="Last Image (Optional)", sources=["upload", "clipboard"])
                negative_prompt_input = gr.Textbox(label="Negative Prompt", value=default_negative_prompt, info="Used if any Guidance Scale > 1.", lines=3)
                quality_slider = gr.Slider(minimum=1, maximum=10, step=1, value=6, label="Video Quality", info="If set to 10, the generated video may be too large and won't play in the Gradio preview.")
                seed_input = gr.Slider(label="Seed", minimum=0, maximum=MAX_SEED, step=1, value=42, interactive=True)
                randomize_seed_checkbox = gr.Checkbox(label="Randomize seed", value=True, interactive=True)
                steps_slider = gr.Slider(minimum=1, maximum=30, step=1, value=6, label="Inference Steps")
                guidance_scale_input = gr.Slider(minimum=0.0, maximum=10.0, step=0.5, value=1, label="Guidance Scale - high noise stage", info="Values above 1 increase GPU usage and may take longer to process.")
                guidance_scale_2_input = gr.Slider(minimum=0.0, maximum=10.0, step=0.5, value=1, label="Guidance Scale 2 - low noise stage")
                scheduler_dropdown = gr.Dropdown(
                    label="Scheduler",
                    choices=list(SCHEDULER_MAP.keys()),
                    value="UniPCMultistep",
                    info="Select a custom scheduler."
                )
                flow_shift_slider = gr.Slider(minimum=0.5, maximum=15.0, step=0.1, value=3.0, label="Flow Shift")

                gr.Markdown("### 1) Original LoRA x3 + strength")
                with gr.Row():
                    lora1_preset = gr.Dropdown(choices=LORA_CHOICES, value="General NSFW Booster", label="Original LoRA 1", filterable=True)
                    lora1_high_scale = gr.Slider(0.0, 2.0, value=1.0, step=0.05, label="High strength")
                    lora1_low_scale = gr.Slider(0.0, 2.0, value=1.0, step=0.05, label="Low strength")
                with gr.Row():
                    lora2_preset = gr.Dropdown(choices=LORA_CHOICES, value="(None)", label="Original LoRA 2", filterable=True)
                    lora2_high_scale = gr.Slider(0.0, 2.0, value=1.0, step=0.05, label="High strength")
                    lora2_low_scale = gr.Slider(0.0, 2.0, value=1.0, step=0.05, label="Low strength")
                with gr.Row():
                    lora3_preset = gr.Dropdown(choices=LORA_CHOICES, value="(None)", label="Original LoRA 3", filterable=True)
                    lora3_high_scale = gr.Slider(0.0, 2.0, value=1.0, step=0.05, label="High strength")
                    lora3_low_scale = gr.Slider(0.0, 2.0, value=1.0, step=0.05, label="Low strength")

                gr.Markdown("### 2) My LoRA x5 from HF repo + strength")
                gr.Markdown("For each slot: paste repo → Load files → pick High/Low → set strength. Example: `username/my-loras`")

                # --- My LoRA 1 ---
                gr.Markdown("**My LoRA 1**")
                with gr.Row():
                    my1_repo = gr.Textbox(label="Repo link", placeholder="username/repo", lines=1, scale=4)
                    my1_load = gr.Button("Load files", scale=1)
                with gr.Row():
                    my1_high = gr.Dropdown(choices=["(None)"], value="(None)", label="High file", filterable=True)
                    my1_hs = gr.Slider(0.0, 2.0, value=1.0, step=0.05, label="High strength")
                with gr.Row():
                    my1_low = gr.Dropdown(choices=["(None)"], value="(None)", label="Low file", filterable=True)
                    my1_ls = gr.Slider(0.0, 2.0, value=1.0, step=0.05, label="Low strength")
                my1_load.click(fn=_load_repo_files, inputs=[my1_repo], outputs=[my1_high, my1_low])

                # --- My LoRA 2 ---
                gr.Markdown("**My LoRA 2**")
                with gr.Row():
                    my2_repo = gr.Textbox(label="Repo link", placeholder="username/repo", lines=1, scale=4)
                    my2_load = gr.Button("Load files", scale=1)
                with gr.Row():
                    my2_high = gr.Dropdown(choices=["(None)"], value="(None)", label="High file", filterable=True)
                    my2_hs = gr.Slider(0.0, 2.0, value=1.0, step=0.05, label="High strength")
                with gr.Row():
                    my2_low = gr.Dropdown(choices=["(None)"], value="(None)", label="Low file", filterable=True)
                    my2_ls = gr.Slider(0.0, 2.0, value=1.0, step=0.05, label="Low strength")
                my2_load.click(fn=_load_repo_files, inputs=[my2_repo], outputs=[my2_high, my2_low])

                # --- My LoRA 3 ---
                gr.Markdown("**My LoRA 3**")
                with gr.Row():
                    my3_repo = gr.Textbox(label="Repo link", placeholder="username/repo", lines=1, scale=4)
                    my3_load = gr.Button("Load files", scale=1)
                with gr.Row():
                    my3_high = gr.Dropdown(choices=["(None)"], value="(None)", label="High file", filterable=True)
                    my3_hs = gr.Slider(0.0, 2.0, value=1.0, step=0.05, label="High strength")
                with gr.Row():
                    my3_low = gr.Dropdown(choices=["(None)"], value="(None)", label="Low file", filterable=True)
                    my3_ls = gr.Slider(0.0, 2.0, value=1.0, step=0.05, label="Low strength")
                my3_load.click(fn=_load_repo_files, inputs=[my3_repo], outputs=[my3_high, my3_low])

                # --- My LoRA 4 ---
                gr.Markdown("**My LoRA 4**")
                with gr.Row():
                    my4_repo = gr.Textbox(label="Repo link", placeholder="username/repo", lines=1, scale=4)
                    my4_load = gr.Button("Load files", scale=1)
                with gr.Row():
                    my4_high = gr.Dropdown(choices=["(None)"], value="(None)", label="High file", filterable=True)
                    my4_hs = gr.Slider(0.0, 2.0, value=1.0, step=0.05, label="High strength")
                with gr.Row():
                    my4_low = gr.Dropdown(choices=["(None)"], value="(None)", label="Low file", filterable=True)
                    my4_ls = gr.Slider(0.0, 2.0, value=1.0, step=0.05, label="Low strength")
                my4_load.click(fn=_load_repo_files, inputs=[my4_repo], outputs=[my4_high, my4_low])

                # --- My LoRA 5 ---
                gr.Markdown("**My LoRA 5**")
                with gr.Row():
                    my5_repo = gr.Textbox(label="Repo link", placeholder="username/repo", lines=1, scale=4)
                    my5_load = gr.Button("Load files", scale=1)
                with gr.Row():
                    my5_high = gr.Dropdown(choices=["(None)"], value="(None)", label="High file", filterable=True)
                    my5_hs = gr.Slider(0.0, 2.0, value=1.0, step=0.05, label="High strength")
                with gr.Row():
                    my5_low = gr.Dropdown(choices=["(None)"], value="(None)", label="Low file", filterable=True)
                    my5_ls = gr.Slider(0.0, 2.0, value=1.0, step=0.05, label="Low strength")
                my5_load.click(fn=_load_repo_files, inputs=[my5_repo], outputs=[my5_high, my5_low])

                play_result_video = gr.Checkbox(label="Display result", value=True, interactive=True)

            generate_button = gr.Button("Generate Video", variant="primary")

        with gr.Column():
            video_output = gr.Video(label="Generated Video", autoplay=True, sources=["upload"], buttons=["download", "share"], interactive=True, elem_id="generated-video")
            
            with gr.Row():
                grab_frame_btn = gr.Button("Use Current Frame as Input", variant="secondary")
                timestamp_box = gr.Number(value=0, label="Timestamp", visible=True, elem_id="hidden-timestamp")
            
            file_output = gr.File(label="Download Video")

    ui_inputs = [
        input_image_component, last_image_component, prompt_input, steps_slider,
        negative_prompt_input, duration_seconds_input,
        guidance_scale_input, guidance_scale_2_input, seed_input, randomize_seed_checkbox,
        quality_slider, scheduler_dropdown, flow_shift_slider, frame_multi,
        safe_mode_checkbox,
        # original 3
        lora1_preset, lora1_high_scale, lora1_low_scale,
        lora2_preset, lora2_high_scale, lora2_low_scale,
        lora3_preset, lora3_high_scale, lora3_low_scale,
        # my 5
        my1_repo, my1_high, my1_hs, my1_low, my1_ls,
        my2_repo, my2_high, my2_hs, my2_low, my2_ls,
        my3_repo, my3_high, my3_hs, my3_low, my3_ls,
        my4_repo, my4_high, my4_hs, my4_low, my4_ls,
        my5_repo, my5_high, my5_hs, my5_low, my5_ls,
        play_result_video,
    ]
    
    generate_button.click(
        fn=generate_video, 
        inputs=ui_inputs, 
        outputs=[video_output, file_output, seed_input]
    )
    
    grab_frame_btn.click(
        fn=None,
        inputs=None,
        outputs=[timestamp_box],
        js=get_timestamp_js
    )
    
    timestamp_box.change(
        fn=extract_frame,
        inputs=[video_output, timestamp_box],
        outputs=[input_image_component]
    )

if __name__ == "__main__":
    demo.queue().launch(
        mcp_server=True,
        css=CSS,
        show_error=True,
    )