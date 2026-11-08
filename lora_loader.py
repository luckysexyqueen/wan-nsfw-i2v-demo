"""LoRA loader for Wan 2.2 I2V with the general NSFW booster enabled."""
import os
import urllib.parse
import re
from huggingface_hub import hf_hub_download, list_repo_files

HF_TOKEN = os.environ.get("HF_TOKEN")
NSFW_REPO = "lopi999/Wan2.2-I2V_General-NSFW-LoRA"

PINNED_REVISIONS = {
    NSFW_REPO: "aeef17d7fa51",
}

LORA_PAIRS = {
    "General NSFW Booster": {
        "HIGH": (NSFW_REPO, "NSFW-22-H-e8.safetensors"),
        "LOW": (NSFW_REPO, "NSFW-22-L-e8.safetensors"),
    }
}

def get_lora_choices():
    choices = ["(None)"]
    for group in sorted(LORA_PAIRS.keys()):
        p = LORA_PAIRS[group]
        if p["HIGH"] and p["LOW"]:
            choices.append(group)
        elif p["HIGH"]:
            choices.append(f"{group} (HIGH only)")
        elif p["LOW"]:
            choices.append(f"{group} (LOW only)")
    return choices


def parse_repo_link(link: str):
    if not link:
        return None
    link = link.strip().strip('"').strip("'")
    m = re.match(r"https?://huggingface\.co/(?:models/)?([^/]+)/([^/]+)", link)
    if m:
        return f"{m.group(1)}/{m.group(2)}"
    if not link.startswith("http") and link.count("/") == 1:
        return link
    return None


def parse_hf_file_link(link: str):
    if not link:
        return None, None
    link = link.strip().strip('"').strip("'")
    m = re.match(
        r"https?://huggingface\.co/([^/]+)/([^/]+)/(?:resolve|blob)/[^/]+/(.+)",
        link,
    )
    if m:
        return f"{m.group(1)}/{m.group(2)}", urllib.parse.unquote(m.group(3).split("?")[0])
    if not link.startswith("http") and link.count("/") >= 2:
        parts = link.split("/")
        return f"{parts[0]}/{parts[1]}", "/".join(parts[2:])
    return None, None


def list_loras_from_repo(repo_link: str):
    repo_id = parse_repo_link(repo_link)
    if not repo_id:
        return ["(None)"]
    try:
        files = list_repo_files(repo_id, token=HF_TOKEN, revision=PINNED_REVISIONS.get(repo_id))
        safes = sorted([f for f in files if f.lower().endswith(".safetensors")])
        return ["(None)"] + safes if safes else ["(None)"]
    except Exception as e:
        print(f"list_loras_from_repo error: {e}")
        return ["(None)"]


def download_lora(group_name):
    if not group_name:
        return None, None
    clean_name = re.sub(r'\s*$$HIGH only$$|\s*$$LOW only$$', '', group_name)
    if clean_name not in LORA_PAIRS:
        return None, None
    pair = LORA_PAIRS[clean_name]
    high_path, low_path = None, None
    if pair["HIGH"]:
        repo, fn = pair["HIGH"]
        high_path = hf_hub_download(repo, fn, token=HF_TOKEN, revision=PINNED_REVISIONS.get(repo))
    if pair["LOW"]:
        repo, fn = pair["LOW"]
        low_path = hf_hub_download(repo, fn, token=HF_TOKEN, revision=PINNED_REVISIONS.get(repo))
    return high_path, low_path


def download_repo_file(repo_link: str, filename: str):
    if not filename or filename == "(None)":
        return None
    repo_from_file, weight = parse_hf_file_link(filename)
    if repo_from_file and weight:
        return hf_hub_download(
            repo_from_file, weight, token=HF_TOKEN,
            revision=PINNED_REVISIONS.get(repo_from_file),
        )
    repo_id = parse_repo_link(repo_link)
    if not repo_id:
        raise ValueError(f"Invalid repo/file: repo={repo_link}, file={filename}")
    return hf_hub_download(
        repo_id, filename, token=HF_TOKEN,
        revision=PINNED_REVISIONS.get(repo_id),
    )


def _load_path(pipe, path, adapter_name, into_t2=False):
    kwargs = {"adapter_name": adapter_name}
    if into_t2:
        kwargs["load_into_transformer_2"] = True
    pipe.load_lora_weights(path, **kwargs)
    return adapter_name


def load_lora_to_pipe(pipe, group_name, adapter_name="lora", high_scale=1.0, low_scale=1.0):
    if not group_name or group_name == "(None)":
        return [], []
    high_path, low_path = download_lora(group_name)
    names, weights = [], []
    if high_path:
        n = f"{adapter_name}_high"
        _load_path(pipe, high_path, n, into_t2=False)
        names.append(n)
        weights.append(float(high_scale))
    if low_path:
        n = f"{adapter_name}_low"
        _load_path(pipe, low_path, n, into_t2=True)
        names.append(n)
        weights.append(float(low_scale))
    if names:
        print(f"Loaded preset LoRA: {group_name} scales={weights}")
    return names, weights


def load_lora_slots(pipe, slots):
    all_names, all_weights = [], []
    for idx, slot in enumerate(slots or []):
        if not slot:
            continue
        preset = slot.get("preset")
        if preset and preset != "(None)":
            names, weights = load_lora_to_pipe(
                pipe, preset, adapter_name=f"lora_{idx}",
                high_scale=slot.get("high_scale", 1.0),
                low_scale=slot.get("low_scale", 1.0),
            )
            all_names.extend(names)
            all_weights.extend(weights)
            continue
        repo = slot.get("repo")
        fh = slot.get("file_high")
        fl = slot.get("file_low")
        hs = float(slot.get("high_scale", 1.0) or 1.0)
        ls = float(slot.get("low_scale", 1.0) or 1.0)
        if fh and fh != "(None)":
            path = download_repo_file(repo, fh)
            n = f"link_{idx}_high"
            _load_path(pipe, path, n, into_t2=False)
            all_names.append(n)
            all_weights.append(hs)
            print(f"Loaded link HIGH: {fh} scale={hs}")
        if fl and fl != "(None)":
            path = download_repo_file(repo, fl)
            n = f"link_{idx}_low"
            _load_path(pipe, path, n, into_t2=True)
            all_names.append(n)
            all_weights.append(ls)
            print(f"Loaded link LOW: {fl} scale={ls}")
    if all_names:
        pipe.set_adapters(all_names, adapter_weights=all_weights)
        print(f"Active adapters: {list(zip(all_names, all_weights))}")
        return True
    return False


def unload_lora(pipe):
    try:
        pipe.unload_lora_weights()
    except Exception:
        pass
        