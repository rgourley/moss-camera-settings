"""Depth Lens FX: lens blur and vignette for finished images, with no diffusion, so the picture itself doesn't change.

Depth Lens Blur: estimates depth with Depth Anything V2 Small, keeps the point you pick sharp, and blurs what's behind it more the further back it sits.
Radial Vignette: a smooth oval falloff, darkest in the corners.

Install: copy this folder into ComfyUI/custom_nodes and restart ComfyUI. No extra packages are needed. Depth Lens Blur looks for the model in a
depth_anything_v2_small folder inside your models folder. If it isn't there, the first run downloads about 100 MB from the Hugging Face repo
depth-anything/Depth-Anything-V2-Small-hf. Depth runs on the CPU, so it doesn't use VRAM.
The subject's edge comes from BiRefNet Lite (MIT), which downloads about 180 MB to a birefnet_lite folder on first use and also runs on the CPU.
If it can't load, or the point you click isn't on what it finds, the edge comes from the depth map instead.
"""
import hashlib
import importlib
import json
import os
import sys
import types

import torch
import torch.nn.functional as F

import folder_paths

REPO = "depth-anything/Depth-Anything-V2-Small-hf"
FOLDER = "depth_anything_v2_small"
LUM = (0.2126, 0.7152, 0.0722)
_model = None


def model_dir():
    roots = [folder_paths.models_dir]
    for path in folder_paths.get_folder_paths("vae"):  # picks up extra_model_paths.yaml
        roots.append(os.path.dirname(path.rstrip("/\\")))
    for root in roots:
        path = os.path.join(root, FOLDER)
        if os.path.exists(os.path.join(path, "model.safetensors")):
            return path
    from huggingface_hub import snapshot_download
    target = os.path.join(folder_paths.models_dir, FOLDER)
    print(f"[Depth Lens FX] downloading {REPO} (about 100 MB) to {target}")
    snapshot_download(REPO, local_dir=target, allow_patterns=["*.safetensors", "*.json"])
    return target


def load_model():
    global _model
    if _model is None:
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        path = model_dir()
        _model = (AutoImageProcessor.from_pretrained(path, local_files_only=True),
                  AutoModelForDepthEstimation.from_pretrained(path, local_files_only=True).eval().to("cpu"))
    return _model


CUTOUT_REPO = "ZhengPeng7/BiRefNet_lite"
CUTOUT_REVISION = "aa62cd87ea"  # the exact version whose code was reviewed
CUTOUT_FOLDER = "birefnet_lite"
_cutout_net = None
_cutout_last = (None, None)


def cutout_dir():
    roots = [folder_paths.models_dir]
    for path in folder_paths.get_folder_paths("vae"):
        roots.append(os.path.dirname(path.rstrip("/\\")))
    for root in roots:
        path = os.path.join(root, CUTOUT_FOLDER)
        if os.path.exists(os.path.join(path, "model.safetensors")):
            return path
    from huggingface_hub import snapshot_download
    target = os.path.join(folder_paths.models_dir, CUTOUT_FOLDER)
    print(f"[Depth Lens FX] downloading {CUTOUT_REPO} (about 180 MB) to {target}")
    snapshot_download(CUTOUT_REPO, revision=CUTOUT_REVISION, local_dir=target, allow_patterns=["*.safetensors", "config.json", "birefnet.py", "BiRefNet_config.py"])
    return target


def load_cutout():
    """Returns the model, or False if it can't load. Its code imports three small helpers from timm, which ComfyUI doesn't include, so they exist only during this import."""
    global _cutout_net
    if _cutout_net is not None:
        return _cutout_net
    added = []
    try:
        path = cutout_dir()
        import transformers  # noqa: F401  # loaded first, so it doesn't see the temporary helpers
        if importlib.util.find_spec("timm") is None:
            class DropPath(torch.nn.Module):
                def __init__(self, drop_prob=0.0, *a, **k):
                    super().__init__()

                def forward(self, x):
                    return x  # only does anything during training
            timm, layers = types.ModuleType("timm"), types.ModuleType("timm.layers")
            layers.DropPath = DropPath
            layers.to_2tuple = lambda x: tuple(x) if isinstance(x, (tuple, list)) else (x, x)
            layers.trunc_normal_ = lambda tensor, *a, **k: tensor
            timm.layers = layers
            sys.modules["timm"], sys.modules["timm.layers"] = timm, layers
            added = ["timm", "timm.layers"]
        package = types.ModuleType("depth_lens_fx_birefnet")
        package.__path__ = [path]
        sys.modules[package.__name__] = package
        module = importlib.import_module(package.__name__ + ".birefnet")
        _cutout_net = module.BiRefNet.from_pretrained(path, local_files_only=True).eval().to("cpu")
    except Exception as e:
        print(f"[Depth Lens FX] the cutout model didn't load, so the edge comes from the depth map: {e}")
        _cutout_net = False
    finally:
        for name in added:
            sys.modules.pop(name, None)
    return _cutout_net


def find_subject(rgb):
    """rgb is H, W, 3 in 0-1. Returns H, W in 0-1, or None if the model isn't available. Remembers the last image, since menu changes reuse it."""
    global _cutout_last
    key = hashlib.md5(rgb[::8, ::8].numpy().tobytes()).hexdigest()
    if _cutout_last[0] == key:
        return _cutout_last[1]
    net = load_cutout()
    if net is False:
        return None
    h, w = rgb.shape[:2]
    x = F.interpolate(rgb.permute(2, 0, 1)[None], size=(1024, 1024), mode="bilinear", align_corners=False)[0]
    mean, std = torch.tensor([.485, .456, .406])[:, None, None], torch.tensor([.229, .224, .225])[:, None, None]
    with torch.inference_mode():
        p = net(((x - mean) / std)[None])[-1].sigmoid()
    result = F.interpolate(p, size=(h, w), mode="bilinear", align_corners=False)[0, 0].clamp(0, 1)
    _cutout_last = (key, result)
    return result


def box(x, r):
    """Mean filter with edge padding. x is 2D."""
    k = 2 * r + 1
    return F.avg_pool2d(F.pad(x[None, None], (r, r, r, r), mode="replicate"), k, stride=1)[0, 0]


def guided(guide, src, r=8, eps=1e-4):
    """Fits the soft depth map to the edges in the image, so hair and shoulders don't get a halo."""
    mg, ms = box(guide, r), box(src, r)
    a = (box(guide * src, r) - mg * ms) / (box(guide * guide, r) - mg * mg + eps)
    return box(a, r) * guide + box(ms - a * mg, r)


def estimate_depth(rgb):
    """rgb is H, W, 3 in 0-1. Returns H, W in 0-1, where 1 is near."""
    proc, net = load_model()
    h, w = rgb.shape[:2]
    arr = (rgb.numpy() * 255).round().astype("uint8")
    with torch.inference_mode():
        pred = net(**proc(images=arr, return_tensors="pt")).predicted_depth
    d = F.interpolate(pred[None], size=(h, w), mode="bicubic", align_corners=False)[0, 0]
    d = (d - d.min()) / (d.max() - d.min()).clamp(min=1e-6)
    return guided(rgb @ torch.tensor(LUM), d).clamp(0, 1)


def smooth(t):
    t = t.clamp(0, 1)
    return t * t * (3 - 2 * t)


def disc(r):
    n = int(r) + 2
    ax = torch.arange(-n, n + 1, dtype=torch.float32)
    y, x = torch.meshgrid(ax, ax, indexing="ij")
    k = (r + 0.5 - torch.hypot(x, y)).clamp(0, 1)
    return k / k.sum()


def convolve(x, kernel):
    """FFT convolution, same size as x, with edge padding. x is H, W or H, W, C."""
    n = kernel.shape[0] // 2
    chan = x[None] if x.dim() == 2 else x.permute(2, 0, 1)
    pad = F.pad(chan[None], (n, n, n, n), mode="replicate")[0]
    h, w = pad.shape[-2:]
    size = (h + 2 * n, w + 2 * n)
    out = torch.fft.irfft2(torch.fft.rfft2(pad, s=size) * torch.fft.rfft2(kernel, s=size), s=size)
    out = out[..., 2 * n:2 * n + chan.shape[-2], 2 * n:2 * n + chan.shape[-1]]
    return out[0] if x.dim() == 2 else out.permute(1, 2, 0)


def line(r, angle):
    """A straight streak, r pixels each way from the center."""
    n = int(r) + 2
    ax = torch.arange(-n, n + 1, dtype=torch.float32)
    y, x = torch.meshgrid(ax, ax, indexing="ij")
    a = torch.tensor(angle * 3.14159265 / 180)
    along, across = x * a.cos() - y * a.sin(), x * a.sin() + y * a.cos()
    k = (1.5 - across.abs()).clamp(0, 1) * (r + 1 - along.abs()).clamp(0, 1)
    return k / k.sum()


def subject_matte(rgb, amount):
    """How much of each pixel belongs to the sharp subject. The depth map is too coarse for hair, so this fits its edge to the fine lines in the image itself. Nothing here is set per image."""
    h, w = amount.shape
    lum = rgb @ torch.tensor(LUM)
    near = 1 - smooth(amount / 0.25)
    matte = guided(lum, near, r=max(w // 100, 4), eps=1e-4)
    matte = guided(lum, matte.clamp(0, 1), r=max(w // 300, 2), eps=1e-5)
    return smooth((matte - 0.15) / 0.7)


def strand_matte(rgb, matte, amount):
    """Fine strands just outside the subject's edge, like flyaway hair, which the depth map can't see. A pixel counts if it stands out from the background's own texture nearby,
    so a plain background gives the strands back and a busy one doesn't turn into sharp patches. It also has to connect back to the subject, which drops loose specks of background. Nothing here is set per image."""
    w = rgb.shape[1]
    lum = rgb @ torch.tensor(LUM)
    line_detail = (lum - box(lum, max(w // 400, 2))).abs()
    bg = (amount > 0.5).float()
    r = max(int(0.04 * w), 4)
    texture = box(line_detail * bg, r) / box(bg, r).clamp(min=1e-3)
    strand = smooth((line_detail - 2.5 * texture - 0.008) / 0.04)
    candidate = strand > 0.25
    reached = matte > 0.5
    for _ in range(max(int(0.025 * w), 3)):  # grows one pixel at a time, only along strand pixels
        reached = reached | (F.max_pool2d(reached.float()[None, None], 3, stride=1, padding=1)[0, 0] > 0) & candidate
    return strand * box(reached.float(), 1)


def eased_dim(blurred, amount, dim):
    """Darkening eases off when most of the background is near white: a blank sky or wall has nothing to darken and just turns into a flat grey card. It's one decision for the whole image, so it can't leave blotches."""
    background = amount > 0.5
    if dim <= 0 or not background.any():
        return dim
    white = ((blurred @ torch.tensor(LUM))[background] > 0.85).float().mean()
    return dim * float(1 - smooth((white - 0.25) / 0.35))


EDGE_LOW = 0.0  # raising this pulls the subject's edge inward


def lens_blur(rgb, amount, rmax, bloom, steps=14, kernel=disc, dim=0.0, edge_refine=1.0, edge_soft=0.0, cutout=None):
    """Blurs at several radii, leaving out the sharp subject so it can't bleed in, then blends per pixel between the two nearest radii."""
    if cutout is not None:  # the depth map reaches a little past the subject's real edge: just outside it, take the blur of the background nearby
        around = max(rgb.shape[1] // 90, 3)
        ring = (F.max_pool2d((cutout > 0.5).float()[None, None], 2 * around + 1, stride=1, padding=around)[0, 0] > 0) & (cutout < 0.5)
        raised = F.max_pool2d(amount[None, None], 2 * around + 1, stride=1, padding=around)[0, 0]
        fade = box(ring.float(), 2)
        amount = amount * (1 - fade) + raised * fade
    plain = rgb ** 2.2
    lit = plain @ torch.tensor(LUM)
    lin = plain * (1 + bloom * ((lit - 0.85) / 0.15).clamp(0, 1))[..., None]  # bright points spread like real bokeh
    # the blur only draws from pixels clearly behind the subject: a soft depth edge would otherwise leak the subject's color and make a halo
    grow = int(rmax / 3) + 2
    held = (amount < 0.05).float() if cutout is None else torch.maximum((amount < 0.05).float(), (cutout > 0.1).float())
    subject = F.max_pool2d(held[None, None], 2 * grow + 1, stride=1, padding=grow)[0, 0]
    weight = 1 - box(subject, 2)
    pos = box(amount.clamp(0, 1), 2) * steps  # feathered, so edges between sharp and blurred aren't jagged
    low = pos.long().clamp(max=steps - 1)
    t = (pos - low)[..., None]
    out = torch.zeros_like(plain)
    prev = plain  # radius zero is the untouched image
    for i in range(steps):
        k = kernel(rmax * (i + 1) / steps)
        reach = convolve(weight, k)
        cur = convolve(lin * weight[..., None], k).clamp(min=0) / reach.clamp(min=1e-4)[..., None]
        cur = torch.where((reach < 0.02)[..., None], prev, cur)  # nothing usable in range: keep the previous radius
        mask = low == i
        out[mask] = (prev * (1 - t) + cur * t)[mask]
        prev = cur
    out = out.clamp(0, 1) ** (1 / 2.2)
    if cutout is not None:
        w = rgb.shape[1]
        e = max(w // 600, 1)
        inner = -F.max_pool2d(-cutout[None, None], 2 * e + 1, stride=1, padding=e)[0, 0]  # just inside the edge
        reach_in = box(box(inner, 2 * e + 1), 2 * e + 1)
        own = torch.stack([box(box(rgb[..., i] * inner, 2 * e + 1), 2 * e + 1) / reach_in.clamp(min=1e-3) for i in range(3)], dim=-1)  # edge pixels hold some background, so use the subject's own colors there
        keep = smooth((inner - 0.5) / 0.45)[..., None]
        clean = torch.where((reach_in < 0.02)[..., None], rgb, rgb * keep + own * (1 - keep))
        alpha = smooth((cutout - EDGE_LOW) / (1 - 2 * EDGE_LOW))
        if edge_soft > 0:
            alpha = box(alpha, max(int(round(edge_soft * w / 1216)), 1))
        dim = eased_dim(out, amount, dim)
        out = out * (1 - dim * box(amount.clamp(0, 1), 1))[..., None]
        return out * (1 - alpha[..., None]) + clean * alpha[..., None]
    if edge_refine > 0:
        lum = rgb @ torch.tensor(LUM)
        w = rgb.shape[1]
        # the real silhouette: the depth edge reaches slightly past it, which would keep a ring of background
        fit = guided(lum, 1 - smooth(amount / 0.25), r=max(w // 150, 3), eps=1e-5)
        reach = max(w // 50, 3)
        edge_band = F.max_pool2d((amount > 0.25).float()[None, None], 2 * reach + 1, stride=1, padding=reach)[0, 0]
        outside = edge_band * (1 - smooth((fit - 0.5) / 0.3))
        solid = subject_matte(rgb, amount) * (1 - outside)
        strands = strand_matte(rgb, solid, amount)
        subject = rgb
        edge_soft = edge_soft + rmax / w * 1216 / 20 if edge_soft > 0 else 0.0  # heavier blur gets a softer edge, so a sharp subject doesn't look cut out
        if edge_soft > 0:  # focus falls off toward the edge, as with a real lens, so the subject doesn't look cut out. Sizes are set for a 1216-wide image and scale with it.
            r, band = max(int(round(edge_soft * w / 1216)), 1), max(int(edge_soft * 4.5 * w / 1216), 2)
            rim = smooth((1 - box(solid, band)) * 2)
            rim = torch.where(solid < 0.5, torch.ones_like(rim), rim)
            weight = box(box(solid, r), r).clamp(min=1e-3)
            soft = torch.stack([box(box(rgb[..., i] * solid, r), r) / weight for i in range(3)], dim=-1)  # only the subject's own colors, so nothing bleeds in
            subject = rgb * (1 - rim[..., None]) + soft * rim[..., None]
            solid = smooth((box(box(solid, max(r // 2, 1)), max(r // 2, 1)) - 0.5) * 2)  # softens inward from the real edge, so the subject's color can't spread past it
            strands = box(strands, 1) * max(1 - edge_soft / 6, 0.2)
        dim = eased_dim(out, amount, dim)
        darken = dim * box(torch.maximum(amount.clamp(0, 1), outside), 1)  # as if the light fell off behind the subject
        out = out * (1 - darken)[..., None]
        strands = strands * edge_refine
        out = out * (1 - strands[..., None]) + rgb * (1 - dim * outside)[..., None] * strands[..., None]  # a strand darkens with what's around it, so the ring of bright background can't come back as a line
        matte = solid * edge_refine
    else:
        matte, subject = torch.zeros_like(amount), rgb
        out = out * (1 - eased_dim(out, amount, dim) * box(amount.clamp(0, 1), 1))[..., None]
    return out * (1 - matte[..., None]) + subject * matte[..., None]


class DepthLensBlur:
    """Keeps the focus point sharp and blurs by distance from it. Doesn't run diffusion, so the picture doesn't change."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "image": ("IMAGE",),
            "focus_x": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Across the image: 0 is the left edge, 1 is the right edge."}),
            "focus_y": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Down the image: 0 is the top edge, 1 is the bottom edge."}),
            "blur": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 8.0, "step": 0.1, "tooltip": "Largest blur radius, as a percent of the image width. About 1 is light, 2 is heavy, 3 is very heavy."}),
            "sharp_range": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "How much stays sharp around the focus point, as a share of the distance to the furthest part of the image. Raise it if parts of your subject go soft."}),
            "foreground_blur": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip": "Also blur what's nearer than the focus point. 0 leaves it alone."}),
            "highlight_bloom": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 6.0, "step": 0.1, "tooltip": "How strongly bright points spread into soft shapes."}),
            "background_dim": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip": "Darkens the background by depth. 0 leaves its brightness alone, 0.5 is half as bright, 1 is black."}),
            "edge_refine": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip": "Fits the subject's edge to fine lines like hair. Lower it if the background shows sharp patches around the subject."}),
        }, "optional": {
            "edge_softness": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 8.0, "step": 0.5, "tooltip": "Softens the subject's edge so it doesn't look cut out. 1.5 is natural, 2.5 is soft, 4 is very soft. High values blur features near the edge."}),
        }}

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("image", "depth_map")
    FUNCTION = "run"
    CATEGORY = "image/lens"

    def run(self, image, focus_x, focus_y, blur, sharp_range, foreground_blur, highlight_bloom, background_dim=0.0, edge_refine=1.0, edge_softness=0.0):
        images, depths = [], []
        for rgb in image.cpu():
            rgb = rgb[..., :3]
            h, w = rgb.shape[:2]
            d = estimate_depth(rgb)
            depths.append(d[..., None].repeat(1, 1, 3))
            rmax = blur / 100 * w
            if rmax < 0.5 and background_dim <= 0:
                images.append(rgb)
                continue
            cx, cy = int(focus_x * (w - 1)), int(focus_y * (h - 1))
            r = max(int(0.015 * w), 2)
            focus = d[max(cy - r, 0):cy + r + 1, max(cx - r, 0):cx + r + 1].median()
            far = torch.quantile(F.interpolate(d[None, None], scale_factor=0.25, mode="area").flatten(), 0.02)
            near = torch.quantile(F.interpolate(d[None, None], scale_factor=0.25, mode="area").flatten(), 0.98)
            behind_edge = focus - sharp_range * (focus - far)
            amount = smooth((behind_edge - d) / (behind_edge - far).clamp(min=1e-3))
            if foreground_blur > 0:
                front_edge = focus + sharp_range * (near - focus)
                amount = amount + foreground_blur * smooth((d - front_edge) / (near - front_edge).clamp(min=1e-3))
            found = find_subject(rgb) if edge_refine > 0 else None
            on_it = found is not None and float(found[max(cy - r, 0):cy + r + 1, max(cx - r, 0):cx + r + 1].median()) > 0.5  # only when the click is on what the model found
            images.append(lens_blur(rgb, amount.clamp(0, 1), rmax, highlight_bloom, dim=background_dim, edge_refine=edge_refine, edge_soft=edge_softness, cutout=found if on_it else None))
        return (torch.stack(images), torch.stack(depths))


class RadialVignette:
    """A smooth round falloff, darkest in the corners. Strength is the brightness you see: at 0.5, the corners are half as bright."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "image": ("IMAGE",),
            "strength": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "How dark the corners get. 0.3 is light, 0.6 is clear, 0.9 is heavy, 1 is black."}),
            "start": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 0.95, "step": 0.01, "tooltip": "Where the darkening begins: 0 is the center, 1 is the corner. Lower reaches further in."}),
            "roundness": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip": "1 is a true circle. 0 is an oval that follows the frame."}),
        }}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run"
    CATEGORY = "image/lens"

    def run(self, image, strength, start, roundness):
        h, w = image.shape[1:3]
        y, x = torch.meshgrid(torch.linspace(-1, 1, h), torch.linspace(-1, 1, w), indexing="ij")
        # measured so the far corner is always 1; a circle stretches the long side to its true length
        long_side = max(h, w)
        sx = 1 + roundness * (w / long_side - 1)
        sy = 1 + roundness * (h / long_side - 1)
        r = torch.hypot(x * sx, y * sy) / torch.hypot(torch.tensor(sx), torch.tensor(sy))
        t = smooth((r - start) / (1 - start))
        return ((image[..., :3].cpu() * (1 - strength * t)[None, ..., None]).clamp(0, 1),)


class LensCrop:
    """A longer lens from the same spot: crops, scales back to the original size, and adds the background blur a longer lens would give. It can't go wider.
    To add detail, run sharp_crop through a low-denoise re-render, then into Depth Lens Blur with this node's focus_x, focus_y and blur outputs wired in."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "image": ("IMAGE",),
            "from_mm": ("FLOAT", {"default": 35.0, "min": 8.0, "max": 400.0, "step": 1.0, "tooltip": "The lens you imagine the shot was taken on. Generated images have no real lens, so this only sets the amount of crop."}),
            "to_mm": ("FLOAT", {"default": 50.0, "min": 8.0, "max": 800.0, "step": 1.0, "tooltip": "The lens you want. It must be the same or longer: a wider view would need pixels the image doesn't have."}),
            "focus_x": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "The point to keep sharp, in the original image: 0 is the left edge, 1 is the right edge. The crop centers here."}),
            "focus_y": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "The point to keep sharp, in the original image: 0 is the top edge, 1 is the bottom edge. The crop centers here."}),
            "shift_x": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01, "tooltip": "Moves the crop sideways from the focus point, as a share of the original width."}),
            "shift_y": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01, "tooltip": "Moves the crop up or down from the focus point, as a share of the original height."}),
            "lens_blur": ("FLOAT", {"default": 2.2, "min": 0.0, "max": 6.0, "step": 0.1, "tooltip": "How much background blur the longer lens adds. The default is close to f/1.8 with the subject 2 m away and a distant background. 0 gives a plain crop."}),
            "background_dim": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip": "Darkens the background by depth. A longer lens doesn't do this by itself; it's a separate look."}),
        }}

    RETURN_TYPES = ("IMAGE", "IMAGE", "FLOAT", "FLOAT", "FLOAT")
    RETURN_NAMES = ("image", "sharp_crop", "focus_x", "focus_y", "blur")
    FUNCTION = "run"
    CATEGORY = "image/lens"

    def run(self, image, from_mm, to_mm, focus_x, focus_y, shift_x, shift_y, lens_blur, background_dim=0.0):
        if to_mm < from_mm:
            raise ValueError("Lens Crop can only go to a longer lens: to_mm must be at least from_mm. A wider view would need pixels the image doesn't have.")
        h, w = image.shape[1:3]
        scale = from_mm / to_mm
        ch, cw = max(int(round(h * scale)), 1), max(int(round(w * scale)), 1)
        y0 = min(max(int((focus_y + shift_y) * h - ch / 2), 0), h - ch)
        x0 = min(max(int((focus_x + shift_x) * w - cw / 2), 0), w - cw)
        crop = image[:, y0:y0 + ch, x0:x0 + cw, :3].cpu().permute(0, 3, 1, 2)
        sharp = F.interpolate(crop, size=(h, w), mode="bicubic", align_corners=False, antialias=True).clamp(0, 1).permute(0, 2, 3, 1)
        fx = min(max((focus_x * w - x0) / cw, 0.0), 1.0)  # where the focus point lands in the crop
        fy = min(max((focus_y * h - y0) / ch, 0.0), 1.0)
        blur = lens_blur * (to_mm / from_mm - 1)  # a real lens's blur grows faster than the crop enlarges it
        result = sharp
        if blur > 0:
            result = DepthLensBlur().run(sharp, fx, fy, min(blur, 8.0), 0.3, 0.0, 1.0, background_dim)[0]
        return (result, sharp, fx, fy, min(blur, 8.0))


class MotionBlur:
    """Camera shake smears the whole frame. Panning keeps the focus point sharp and streaks everything at other depths. Both are straight streaks: the image holds no real motion."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "image": ("IMAGE",),
            "mode": (["camera shake", "panning"], {"default": "panning"}),
            "length": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 10.0, "step": 0.1, "tooltip": "Streak length each way, as a percent of the image width."}),
            "angle": ("FLOAT", {"default": 0.0, "min": -180.0, "max": 180.0, "step": 1.0, "tooltip": "Direction of the streak in degrees. 0 is level."}),
            "focus_x": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Panning only: the point that stays sharp, across the image."}),
            "focus_y": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Panning only: the point that stays sharp, down the image."}),
            "sharp_range": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Panning only: how much stays sharp around the focus point, as a share of the image's depth range."}),
        }}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run"
    CATEGORY = "image/lens"

    def run(self, image, mode, length, angle, focus_x, focus_y, sharp_range):
        out = []
        for rgb in image.cpu():
            rgb = rgb[..., :3]
            h, w = rgb.shape[:2]
            rmax = length / 100 * w
            if rmax < 0.5:
                out.append(rgb)
                continue
            if mode == "camera shake":
                amount = torch.ones((h, w))
            else:
                d = estimate_depth(rgb)
                cx, cy, r = int(focus_x * (w - 1)), int(focus_y * (h - 1)), max(int(0.015 * w), 2)
                focus = d[max(cy - r, 0):cy + r + 1, max(cx - r, 0):cx + r + 1].median()
                span = (d - focus).abs().flatten().quantile(0.9).clamp(min=1e-3)
                keep = sharp_range * span
                amount = smooth(((d - focus).abs() - keep) / (span - keep).clamp(min=1e-3))
            out.append(lens_blur(rgb, amount, rmax, 0.0, kernel=lambda rad: line(rad, angle)))
        return (torch.stack(out),)


def srgb_to_linear(x):
    return torch.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(x):
    return torch.where(x <= 0.0031308, x * 12.92, 1.055 * x.clamp(min=1e-8) ** (1 / 2.4) - 0.055)


class Exposure:
    """Under or over exposes in stops, in linear light, the way a camera does: one stop is twice or half the light. It can't recover detail the image doesn't have,
    so highlights that are already white stay white. Highlight rolloff eases bright areas into white instead of cutting them off."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "image": ("IMAGE",),
            "stops": ("FLOAT", {"default": 0.0, "min": -5.0, "max": 5.0, "step": 0.1, "tooltip": "-1 is half the light, +1 is twice the light. 0 leaves the image alone."}),
            "highlight_rolloff": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip": "When overexposing, how gently bright areas reach white. 0 is a hard cut, 1 is the most gradual."}),
        }}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run"
    CATEGORY = "image/lens"

    def run(self, image, stops, highlight_rolloff):
        if stops == 0:
            return (image[..., :3].cpu(),)
        lin = srgb_to_linear(image[..., :3].cpu().clamp(0, 1)) * 2 ** stops
        hard = lin.clamp(0, 1)
        if stops > 0 and highlight_rolloff > 0:
            knee = 1 - 0.6 * highlight_rolloff
            soft = torch.where(lin <= knee, lin, knee + (1 - knee) * (1 - torch.exp(-(lin - knee) / (1 - knee))))
            hard = torch.lerp(hard, soft, min(stops, 1.0))  # eases in, so a tiny push can't dim the highlights
        return (linear_to_srgb(hard).clamp(0, 1),)


LENSES = {"35mm (whole shot)": 35.0, "50mm": 50.0, "85mm": 85.0}
BACKGROUNDS = {"Sharp": 0.0, "Soft": 1.0, "Blurred": 2.0, "Very blurred": 3.0}
BACKGROUND_LIGHT = {"Normal": 0.0, "Dim": 0.4, "Dark": 0.7}
EXPOSURES = {"-2 stops": -2.0, "-1 stop": -1.0, "Normal": 0.0, "+1 stop": 1.0, "+2 stops": 2.0}
VIGNETTES = {"Off": 0.0, "Light": 0.3, "Medium": 0.5, "Heavy": 0.8}
FACE_DETAIL = {"Off": 0.0, "Safe": 0.35, "Strong": 0.5}
KEEP_SHARP = {"Narrow": 0.15, "Normal": 0.3, "Wide": 0.5}
SUBJECT_EDGE = {"Crisp": 0.0, "Natural": 1.5, "Soft": 2.5, "Very soft": 4.0}
FOREGROUND = {"Sharp": 0.0, "Blurred": 1.0}


def seen_path():
    return os.path.join(folder_paths.get_user_directory(), "camera_settings_seen.json")  # kept on disk so the check still works right after a restart


def load_seen():
    try:
        with open(seen_path(), encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def save_seen(seen):
    try:
        with open(seen_path(), "w", encoding="utf-8") as f:
            json.dump(seen, f)
    except OSError:
        pass  # the check is a convenience: it should never stop a render


def send_picker_image(image, node_id, signature="", reset_focus=False):
    """Shows the image on the node even when the run stops here. A stopped run never reaches the usual preview step."""
    if node_id is None:
        return
    from PIL import Image
    from server import PromptServer
    name = f"camera_settings_{node_id}.png".replace(":", "_")
    Image.fromarray((image[0, ..., :3].cpu().numpy().clip(0, 1) * 255).round().astype("uint8")).save(os.path.join(folder_paths.get_temp_directory(), name), compress_level=1)
    PromptServer.instance.send_sync("depth_lens_fx.picker", {"node": str(node_id), "filename": name, "signature": signature, "reset_focus": reset_focus}, PromptServer.instance.client_id)


class CameraSettings:
    """Pick the look from menus and click the subject on the image. The first run stops here so you can click. Press Run again to finish: the render isn't repeated while the seed stays the same."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "image": ("IMAGE",),
            "lens": (list(LENSES), {"default": "85mm", "tooltip": "Longer lenses crop in tighter. This is a crop: it can't change perspective."}),
            "aperture": (list(BACKGROUNDS), {"default": "Blurred", "tooltip": "Like opening a lens: how blurred things get away from the point you click. Longer lenses add more."}),
            "background_light": (list(BACKGROUND_LIGHT), {"default": "Dim", "tooltip": "Darkens the background by distance. It eases off when most of the background is near white, like a blank sky, which would otherwise turn flat grey and make the subject look cut out."}),
            "exposure": (list(EXPOSURES), {"default": "Normal", "tooltip": "-1 stop is half the light, +1 stop is double. It can't bring back detail the image doesn't have."}),
            "vignette": (list(VIGNETTES), {"default": "Medium", "tooltip": "Darkens the corners."}),
            "face_detail": (list(FACE_DETAIL), {"default": "Off", "tooltip": "Redraws the crop to sharpen skin and hair. Only use it when the face is large in the frame: it damages small faces, mouths and eyes first. The detail is invented, and Strong can change the face."}),
            "keep_sharp": (list(KEEP_SHARP), {"default": "Normal", "tooltip": "How much stays sharp around the point you click. Use Wide if part of the subject goes soft."}),
            "focus_x": ("FLOAT", {"default": -1.0, "min": -1.0, "max": 1.0, "step": 0.01, "tooltip": "Set by clicking the image. -1 means nothing is picked yet."}),
            "focus_y": ("FLOAT", {"default": -1.0, "min": -1.0, "max": 1.0, "step": 0.01, "tooltip": "Set by clicking the image. -1 means nothing is picked yet."}),
        }, "optional": {
            "focus_image": ("STRING", {"default": "", "tooltip": "Set by clicking: which image the click was made on."}),
            "subject_edge": (list(SUBJECT_EDGE), {"default": "Natural", "tooltip": "How soft the subject's edge is. Soften it if the subject looks cut out. Very soft can blur features near the edge, like a face in profile."}),
            "foreground": (list(FOREGROUND), {"default": "Sharp", "tooltip": "Whether things nearer than the point you click also go out of focus, as with a real lens."}),
        }, "hidden": {"unique_id": "UNIQUE_ID"}}

    RETURN_TYPES = ("IMAGE", "CAMERA_SETTINGS", "FLOAT")
    RETURN_NAMES = ("crop", "settings", "denoise")
    FUNCTION = "run"
    CATEGORY = "image/lens"

    def run(self, image, lens, aperture, background_light, exposure, vignette, face_detail, keep_sharp, focus_x, focus_y, focus_image="", subject_edge="Natural", foreground="Sharp", unique_id=None):
        # a new image with the old focus point would keep the wrong thing sharp
        signature = hashlib.md5(image[0, ::8, ::8, :3].cpu().numpy().tobytes()).hexdigest()
        seen = load_seen()
        previous = seen.get(str(unique_id))
        if focus_image:  # the click says which image it was made on
            stale = focus_x >= 0 and focus_image != signature
        else:  # typed numbers or an older workflow: all we can tell is that the image changed and the point didn't
            stale = focus_x >= 0 and previous is not None and previous[0] != signature and previous[1] == [focus_x, focus_y]
        seen[str(unique_id)] = [signature, [focus_x, focus_y]]
        save_seen(seen)
        send_picker_image(image, unique_id, signature, stale)
        if stale or focus_x < 0 or focus_y < 0:  # waits for a click: the nodes after this one simply don't run, and it isn't reported as a failure
            from comfy_execution.graph_utils import ExecutionBlocker
            print("[Depth Lens FX] Camera Settings: click the subject on the image, then press Run again.")
            return tuple(ExecutionBlocker(None) for _ in self.RETURN_TYPES)
        mm = LENSES[lens]
        _, crop, fx, fy, _ = LensCrop().run(image, 35.0, mm, focus_x, focus_y, 0.0, 0.05, 0.0)
        blur = min(BACKGROUNDS[aperture] * (mm / 35.0) ** 0.5, 8.0)
        denoise = FACE_DETAIL[face_detail]
        settings = {"focus_x": fx, "focus_y": fy, "blur": blur, "dim": BACKGROUND_LIGHT[background_light], "stops": EXPOSURES[exposure], "vignette": VIGNETTES[vignette], "denoise": denoise, "sharp_range": KEEP_SHARP[keep_sharp], "edge_soft": SUBJECT_EDGE[subject_edge], "foreground": FOREGROUND[foreground]}
        return (crop, settings, max(denoise, 0.01))


class CameraFinish:
    """Applies what Camera Settings picked: background blur, exposure and vignette. With Face detail off, the detail pass doesn't run at all."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"crop": ("IMAGE",), "settings": ("CAMERA_SETTINGS",)},
                "optional": {"detail": ("IMAGE", {"lazy": True, "tooltip": "The detail pass's result. Only used when Face detail is on."})}}

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("image", "depth_map")
    FUNCTION = "run"
    CATEGORY = "image/lens"

    def check_lazy_status(self, crop, settings, detail=None):
        return ["detail"] if settings["denoise"] > 0 and detail is None else []

    def run(self, crop, settings, detail=None):
        s = settings
        base = detail if s["denoise"] > 0 and detail is not None else crop
        image, depth = DepthLensBlur().run(base, s["focus_x"], s["focus_y"], s["blur"], s["sharp_range"], s["foreground"], 1.0, s["dim"], 1.0, s["edge_soft"])
        image = Exposure().run(image, s["stops"], 0.5)[0]
        if s["vignette"] > 0:
            image = RadialVignette().run(image, s["vignette"], 0.35, 1.0)[0]
        return (image, depth)


NODE_CLASS_MAPPINGS = {"LensCrop": LensCrop, "MotionBlur": MotionBlur, "DepthLensBlur": DepthLensBlur, "RadialVignette": RadialVignette, "Exposure": Exposure, "CameraSettings": CameraSettings, "CameraFinish": CameraFinish}
NODE_DISPLAY_NAME_MAPPINGS = {"LensCrop": "Lens Crop", "MotionBlur": "Motion Blur", "DepthLensBlur": "Depth Lens Blur", "RadialVignette": "Radial Vignette", "Exposure": "Exposure", "CameraSettings": "Camera Settings", "CameraFinish": "Camera Finish"}
WEB_DIRECTORY = "./web"
