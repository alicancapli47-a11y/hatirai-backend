from fastapi import FastAPI, APIRouter, HTTPException, Depends, Header, Request
from fastapi.responses import HTMLResponse, RedirectResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
import asyncio
import uuid
import jwt
import base64
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import List, Optional, Literal
from pydantic import BaseModel, Field
from passlib.context import CryptContext
from collections import defaultdict
import time

import google.genai as genai
from google.genai import types as genai_types
import anthropic as _anthropic
import iyzipay
import json as _json
import fal_client
import base64 as _b64
import subprocess
import tempfile
import shutil
import requests as _requests
from io import BytesIO
from PIL import Image as _PILImage, ImageDraw as _PILDraw, ImageFont as _PILFont, ImageFilter as _PILFilter

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# MongoDB
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

# Auth config
JWT_SECRET = os.environ.get("JWT_SECRET", "change_me")
JWT_ALGO = "HS256"
ADMIN_USERNAME = os.environ.get("ADMIN_USERNAME", "admin")
ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "admin")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
FAL_KEY = os.environ.get("FAL_KEY")

# Iyzico config
IYZICO_OPTIONS = {
    "api_key": os.environ.get("IYZICO_API_KEY", ""),
    "secret_key": os.environ.get("IYZICO_SECRET_KEY", ""),
    "base_url": os.environ.get("IYZICO_BASE_URL", "https://sandbox-api.iyzipay.com"),
}
IYZICO_MODE = os.environ.get("IYZICO_MODE", "sandbox")
PUBLIC_BACKEND_URL = os.environ.get("PUBLIC_BACKEND_URL", "http://localhost:8001")

# App
app = FastAPI(title="HatırAI Backend")
api_router = APIRouter(prefix="/api")


# ===================== MODELS =====================
EraId = Literal["1950s", "80s", "modern"]


class PhotoTransformRequest(BaseModel):
    image_base64: str  # raw base64 string (no data URL prefix)
    era: EraId = "modern"


class PhotoRecord(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    original_b64: str
    era: EraId = "modern"
    noir_b64: Optional[str] = None
    status: Literal["processing", "ready", "failed"] = "processing"
    error: Optional[str] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class PhotoPublic(BaseModel):
    id: str
    noir_b64: Optional[str] = None  # ALWAYS the watermarked preview (deprecated name kept for compat)
    status: str
    error: Optional[str] = None


class AdminLoginRequest(BaseModel):
    username: str
    password: str


class AdminLoginResponse(BaseModel):
    token: str
    expires_at: datetime


class VideoJob(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    photo_id: str
    user_email: Optional[str] = None
    status: str = "pending_payment"  # free-form for backward compat; new jobs use pending_payment|ready|failed
    payment_status: str = "unpaid"
    media_url: Optional[str] = None
    kind: Optional[str] = "image"
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class VideoRequestBody(BaseModel):
    photo_id: str
    user_email: Optional[str] = None


class MemoryFormBody(BaseModel):
    photo_id: str
    name: str  # Sender's name (the user)
    relationship: str  # e.g. "Dede", "Anne"
    last_memory: Optional[str] = None


class MemoryForm(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    photo_id: str
    name: str
    relationship: str
    last_memory: Optional[str] = None
    ai_sentence: Optional[str] = None
    full_script: Optional[str] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class PaymentInitiateBody(BaseModel):
    job_id: str


class AdminDecisionBody(BaseModel):
    decision: Literal["approve", "reject"]
    note: Optional[str] = None


class VideoJobPublic(BaseModel):
    id: str
    status: str
    payment_status: str
    media_url: Optional[str] = None
    kind: str = "image"
    progress: Optional[int] = 0


# ===================== AUTH =====================
def create_admin_token() -> tuple[str, datetime]:
    expires = datetime.now(timezone.utc) + timedelta(hours=12)
    payload = {"sub": ADMIN_USERNAME, "role": "admin", "exp": expires}
    token = jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGO)
    return token, expires


async def require_admin(authorization: Optional[str] = Header(None)) -> str:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing token")
    token = authorization.split(" ", 1)[1]
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGO])
        if payload.get("role") != "admin":
            raise HTTPException(status_code=403, detail="Forbidden")
        return payload.get("sub", "")
    except jwt.PyJWTError as e:
        raise HTTPException(status_code=401, detail=f"Invalid token: {e}")


# Optional user auth helper (used by /memory/form + history) — defined early so routes can Depends on it
async def current_user(authorization: Optional[str] = Header(None)) -> Optional[dict]:
    if not authorization or not authorization.startswith("Bearer "):
        return None
    token = authorization.split(" ", 1)[1]
    if token.count(".") == 2:  # admin JWT — skip
        return None
    sess = await db.user_sessions.find_one({"session_token": token}, {"_id": 0})
    if not sess:
        return None
    exp = sess.get("expires_at")
    if isinstance(exp, str):
        try: exp = datetime.fromisoformat(exp)
        except Exception: exp = None
    if exp and getattr(exp, "tzinfo", None) is None:
        exp = exp.replace(tzinfo=timezone.utc)
    if exp and exp < datetime.now(timezone.utc):
        return None
    return await db.users.find_one({"user_id": sess["user_id"]}, {"_id": 0})


async def require_user(user: Optional[dict] = Depends(current_user)) -> dict:
    if not user:
        raise HTTPException(status_code=401, detail="Giriş gerekli")
    return user


# ===================== ROUTES =====================
@api_router.get("/")
async def root():
    return {"app": "HatırAI", "status": "ok"}


@api_router.post("/admin/login", response_model=AdminLoginResponse)
async def admin_login(body: AdminLoginRequest):
    if body.username != ADMIN_USERNAME or body.password != ADMIN_PASSWORD:
        raise HTTPException(status_code=401, detail="Geçersiz kullanıcı adı veya şifre")
    token, expires = create_admin_token()
    return AdminLoginResponse(token=token, expires_at=expires)


# ---------- Photo: cinematic transform ----------
ERA_PROMPTS: dict[str, str] = {
    "1950s": (
        "Restore and dramatize this old portrait. PRIORITY 1 — fix quality: heavily denoise, "
        "sharpen the face, fix any blur, recover natural skin texture, clear sharp eyes "
        "looking straight at the lens. PRIORITY 2 — REFRAME so the subject is facing the "
        "camera with a frontal headshot angle (head straight, eyes meeting the camera, "
        "shoulders centered). Even if the original was a side profile, redraw to a frontal "
        "view while preserving the person's identity, age, hair, clothing and 1950s-70s era. "
        "PRIORITY 3 — Apply 1950s Hollywood B&W noir lighting: strong key light, soft rim, "
        "deep chiaroscuro, silver-warm grayscale tones, fine film grain. "
        "Output ONLY the final image, sharp and ready for further animation."
    ),
    "80s": (
        "Restore and dramatize this old portrait. PRIORITY 1 — fix quality: heavily denoise, "
        "sharpen the face, fix any blur, restore natural skin texture, clear sharp eyes. "
        "PRIORITY 2 — REFRAME so the subject is facing the camera with a frontal headshot "
        "angle, eyes meeting the lens. Preserve identity, age, hair, clothing and the late "
        "1980s – early 1990s era. PRIORITY 3 — Color-grade like warm VHS / analog film: "
        "faded pastel tones, soft halation around highlights, warm film grain, slight "
        "chromatic aberration, gentle golden rim light. Keep COLORS — not B&W. "
        "Output ONLY the final image, sharp and ready for animation."
    ),
    "modern": (
        "Restore and dramatize this portrait. PRIORITY 1 — fix quality: heavily denoise, "
        "sharpen the face, fix any blur, restore skin texture, clear sharp eyes meeting "
        "the lens. PRIORITY 2 — REFRAME so the subject is facing the camera with a frontal "
        "headshot angle, head straight, shoulders centered. Preserve identity, age, hair, "
        "clothing and the photograph's original era. PRIORITY 3 — Apply a refined modern "
        "high-contrast cinema grade: deep true blacks, rich shadows, clean neutral midtones, "
        "subtle warm golden highlight rim, delicate film grain. Keep colors natural. "
        "Output ONLY the final image, sharp and ready for animation."
    ),
}


def _make_watermarked_preview_b64(clean_b64: str) -> str:
    """Render a noir-styled watermark across the image: gold 'HatırAI · ÖNİZLEME'
    diagonal repeat + corner mark. Returns new base64 PNG."""
    try:
        img = _PILImage.open(BytesIO(_b64.b64decode(clean_b64))).convert("RGBA")
        W, H = img.size

        # Try to load a serif font, fallback to default
        font_main = None
        font_corner = None
        for path in [
            "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        ]:
            try:
                font_main = _PILFont.truetype(path, max(28, W // 22))
                font_corner = _PILFont.truetype(path, max(16, W // 40))
                break
            except Exception:
                continue
        if font_main is None:
            font_main = _PILFont.load_default()
            font_corner = _PILFont.load_default()

        # Diagonal repeating watermark layer
        layer = _PILImage.new("RGBA", img.size, (0, 0, 0, 0))
        draw = _PILDraw.Draw(layer)
        text = "HatırAI · ÖNİZLEME"
        # Approximate text size for tiling
        try:
            bbox = draw.textbbox((0, 0), text, font=font_main)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        except Exception:
            tw, th = (W // 4, H // 20)
        step_x = int(tw * 1.6)
        step_y = int(th * 4.5)
        for y in range(-H, H * 2, step_y):
            for x in range(-W, W * 2, step_x):
                draw.text((x, y), text, font=font_main, fill=(201, 169, 97, 70))
        # Rotate diagonal
        layer = layer.rotate(-22, resample=_PILImage.BICUBIC, expand=False)
        out = _PILImage.alpha_composite(img, layer)

        # Corner mark
        draw2 = _PILDraw.Draw(out)
        corner = "HatırAI · ÖNİZLEME · ÖDEME GEREKLİ"
        try:
            bb = draw2.textbbox((0, 0), corner, font=font_corner)
            cw, ch = bb[2] - bb[0], bb[3] - bb[1]
        except Exception:
            cw, ch = (W // 3, H // 30)
        pad = 16
        # background slab
        draw2.rectangle(
            [(W - cw - pad * 2, H - ch - pad * 2), (W, H)],
            fill=(0, 0, 0, 180),
        )
        draw2.text(
            (W - cw - pad, H - ch - pad - 2),
            corner,
            font=font_corner,
            fill=(201, 169, 97, 255),
        )

        buf = BytesIO()
        out.convert("RGB").save(buf, format="PNG", optimize=True)
        return _b64.b64encode(buf.getvalue()).decode("ascii")
    except Exception as e:
        logger.warning(f"[watermark] failed, returning clean: {e}")
        return clean_b64


async def _run_noir_transform(photo_id: str, image_b64: str, era: str = "modern"):
    try:
        import base64 as _base64
        prompt = ERA_PROMPTS.get(era, ERA_PROMPTS["modern"])

        # fal.ai'ye base64 görseli upload et, URL al
        image_bytes = _base64.b64decode(image_b64)
        tmp_path = f"/tmp/upload_{photo_id}.jpg"
        with open(tmp_path, "wb") as f:
            f.write(image_bytes)
        image_url = await fal_client.upload_file_async(tmp_path)

        # fal.ai nano-banana-2 (Gemini 3.1 Flash Image) ile görsel dönüşümü
        handle = await fal_client.submit_async(
            "fal-ai/nano-banana-2/edit",
            arguments={
                "prompt": prompt,
                "image_urls": [image_url],
                "num_images": 1,
                "aspect_ratio": "auto",
                "output_format": "jpeg",
                "resolution": "1K",
                "safety_tolerance": "4",
            },
        )
        result = await handle.get()

        images = []
        text = result.get("description", "")
        if result.get("images"):
            for img in result["images"]:
                img_url = img.get("url", "")
                if img_url:
                    import requests as _req
                    img_bytes = _req.get(img_url, timeout=60).content
                    images.append({"data": _base64.b64encode(img_bytes).decode()})

        if not images:
            await db.photos.update_one(
                {"id": photo_id},
                {"$set": {"status": "failed", "error": "No image returned by model"}},
            )
            logger.error(f"[cinema] No image returned for {photo_id}. Text: {text[:200] if text else ''}")
            return

        noir_b64 = images[0].get("data")
        preview_b64 = await asyncio.to_thread(_make_watermarked_preview_b64, noir_b64) if noir_b64 else None
        await db.photos.update_one(
            {"id": photo_id},
            {"$set": {"status": "ready", "noir_b64": noir_b64, "preview_b64": preview_b64, "error": None}},
        )
        logger.info(f"[cinema] Photo {photo_id} ready ({len(noir_b64) if noir_b64 else 0} b64 chars, era={era}, watermark={bool(preview_b64)})")
    except Exception as e:
        logger.exception(f"[cinema] Transform failed for {photo_id}")
        await db.photos.update_one(
            {"id": photo_id},
            {"$set": {"status": "failed", "error": str(e)[:500]}},
        )


@api_router.post("/photo/transform", response_model=PhotoPublic)
async def transform_photo(body: PhotoTransformRequest):
    if not FAL_KEY:
        raise HTTPException(status_code=500, detail="FAL_KEY missing")
    img_b64 = body.image_base64.split(",", 1)[-1] if body.image_base64.startswith("data:") else body.image_base64
    record = PhotoRecord(original_b64=img_b64, era=body.era)
    await db.photos.insert_one(record.model_dump())
    asyncio.create_task(_run_noir_transform(record.id, img_b64, body.era))
    return PhotoPublic(id=record.id, status=record.status, noir_b64=None)


@api_router.get("/photo/{photo_id}", response_model=PhotoPublic)
async def get_photo(photo_id: str):
    doc = await db.photos.find_one({"id": photo_id}, {"_id": 0})
    if not doc:
        raise HTTPException(status_code=404, detail="Fotoğraf bulunamadı")
    # Always return the watermarked preview to the public — clean version stays server-side
    public_b64 = doc.get("preview_b64") or doc.get("noir_b64")
    return PhotoPublic(
        id=doc["id"],
        noir_b64=public_b64,
        status=doc.get("status", "processing"),
        error=doc.get("error"),
    )


# ---------- Memory form + AI sentence ----------
async def _generate_ai_sentence(name: str, relationship: str, last_memory: Optional[str]) -> str:
    if not last_memory or not last_memory.strip():
        return "Birlikte geçirdiğimiz o güzel günleri unutamıyorum"
    try:
        ac = _anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        response = await asyncio.to_thread(
            ac.messages.create,
            model="claude-sonnet-4-5-20250929",
            max_tokens=100,
            system=(
                "Sen Türkçe konuşan, sıcak, sinematik, hafif duygusal bir senaryo yazarısın. "
                "Verilen 'paylaşılan anı'yı, kayıp bir yakının hayalete benzeyen bir "
                "selamlama videosunda söyleyeceği TEK bir cümleye dönüştür. "
                "En fazla 18 kelime. Klişeden kaçın. 'Hatırlıyor musun', 'O günler' gibi "
                "dokunaklı, kişisel bir ton kullan. Sadece cümleyi döndür, başka hiçbir şey yazma."
            ),
            messages=[{
                "role": "user",
                "content": (
                    f"İlişki: {relationship}\n"
                    f"Karşı tarafa söyleyenin adı: {name}\n"
                    f"Paylaşılan anı: {last_memory}\n\n"
                    "Bu anıya atıfta bulunan 1 cümle yaz."
                )
            }]
        )
        line = (response.content[0].text or "").strip().strip('"').strip("'").splitlines()[0]
        return line[:240] if line else "Birlikte geçirdiğimiz o güzel günleri unutamıyorum"
    except Exception as e:
        logger.warning(f"[ai-sentence] fallback ({e})")
        return "Birlikte geçirdiğimiz o güzel günleri unutamıyorum"


def _build_full_script(name: str, ai_sentence: str) -> str:
    """Final Turkish script the figure 'speaks' in the video."""
    return (
        f"Selam {name}. {ai_sentence}. Biliyorum beni çok özledin, "
        f"ama ben de seni çok özledim. Umarım yakında kavuşuruz. "
        f"Seni çok seviyorum {name}. Kendine iyi bak, hayata tutunmaya çalış."
    )


@api_router.post("/memory/form")
async def submit_memory_form(body: MemoryFormBody, user: Optional[dict] = Depends(current_user)):
    photo = await db.photos.find_one({"id": body.photo_id}, {"_id": 0})
    if not photo:
        raise HTTPException(status_code=404, detail="Fotoğraf bulunamadı")
    if photo.get("status") != "ready":
        raise HTTPException(status_code=400, detail="Fotoğraf henüz hazır değil")

    name = body.name.strip()[:40] or "Sevgilim"
    rel = body.relationship.strip()[:30] or "Sevdiğim"
    memory = (body.last_memory or "").strip()[:400]
    ai_sentence = await _generate_ai_sentence(name, rel, memory)
    full = _build_full_script(name, ai_sentence)
    rec = MemoryForm(
        photo_id=body.photo_id, name=name, relationship=rel,
        last_memory=memory or None, ai_sentence=ai_sentence, full_script=full,
    )
    await db.memory_forms.insert_one(rec.model_dump())

    job = VideoJob(photo_id=body.photo_id, status="awaiting_payment", payment_status="unpaid",
                   user_email=user.get("email") if user else None)
    job_doc = job.model_dump()
    if user:
        job_doc["user_id"] = user["user_id"]
    await db.video_jobs.insert_one(job_doc)
    # NOTE: Veo video generation is intentionally deferred until Iyzico payment
    # completes (or dev-skip fires). This prevents wasted fal.ai credits on
    # users who drop off before paying. See _finalize_job_by_token and
    # payment_dev_skip below — they kick off _run_veo_pipeline on paid.
    logger.info(f"[memory/form] photo={body.photo_id} job={job.id} user={user.get('email') if user else 'anon'} → awaiting payment")

    return {"form": rec.model_dump(), "job_id": job.id}


@api_router.get("/memory/form/{photo_id}", response_model=Optional[MemoryForm])
async def get_memory_form(photo_id: str):
    doc = await db.memory_forms.find_one({"photo_id": photo_id}, {"_id": 0}, sort=[("created_at", -1)])
    return MemoryForm(**doc) if doc else None


# ---------- Veo 3.1 sequential video generation ----------
NUM_CLIPS = 2  # 2 × 8s = 16s total — fewer transitions = less audio drift


def _ffmpeg_last_frame(video_path: str, frame_path: str) -> None:
    cmd = ["ffmpeg", "-y", "-sseof", "-1", "-i", video_path, "-vframes", "1", "-q:v", "2", frame_path]
    subprocess.run(cmd, check=True, capture_output=True, timeout=60)


def _ffmpeg_concat(video_paths: List[str], out_path: str) -> None:
    """Concat clips and burn a small HatırAI corner watermark onto the final video."""
    tmp = tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False)
    try:
        for p in video_paths:
            tmp.write(f"file '{os.path.abspath(p)}'\n")
        tmp.flush(); tmp.close()

        # Try to use a bundled font; fall back to DejaVuSans which ships with Debian.
        font_candidates = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        ]
        font_path = next((f for f in font_candidates if os.path.exists(f)), None)

        # Bottom-right watermark: "HatırAI" with a subtle gold tint and soft box.
        if font_path:
            draw = (
                f"drawtext=fontfile='{font_path}':text='HatırAI':"
                f"fontcolor=0xE6C36A@0.85:fontsize=h/26:"
                f"x=w-tw-24:y=h-th-24:"
                f"box=1:boxcolor=black@0.35:boxborderw=10"
            )
            vf_args = ["-vf", draw]
        else:
            vf_args = []
            logger.warning("[watermark] no usable font found, writing unmarked final video")

        cmd = [
            "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", tmp.name,
            *vf_args,
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "22",
            "-c:a", "aac", "-b:a", "128k", out_path,
        ]
        subprocess.run(cmd, check=True, capture_output=True, timeout=300)
    finally:
        try: os.unlink(tmp.name)
        except Exception: pass


def _http_download(url: str, dest: str) -> None:
    r = _requests.get(url, timeout=120)
    r.raise_for_status()
    with open(dest, "wb") as f:
        f.write(r.content)


async def _veo_clip(image_url: str, prompt: str) -> str:
    """Submit a single Veo 3.1 LITE image-to-video request and return the video URL."""
    handle = await fal_client.submit_async(
        "fal-ai/veo3.1/lite/image-to-video",
        arguments={
            "prompt": prompt,
            "image_url": image_url,
            "duration": "8s",
            "resolution": "720p",
            "aspect_ratio": "9:16",
            "generate_audio": True,
        },
    )
    result = await handle.get()
    return result["video"]["url"]


async def _run_veo_pipeline(job_id: str):
    """Background task: build full prompt, run sequential Veo 3.1 clips,
    concat with ffmpeg, upload via fal storage, update job."""
    workdir = tempfile.mkdtemp(prefix=f"veo-{job_id}-")
    try:
        await db.video_jobs.update_one({"id": job_id}, {"$set": {"status": "generating", "progress": 0}})

        job = await db.video_jobs.find_one({"id": job_id}, {"_id": 0})
        if not job:
            return
        photo = await db.photos.find_one({"id": job["photo_id"]}, {"_id": 0})
        if not photo or not photo.get("noir_b64"):
            await db.video_jobs.update_one({"id": job_id}, {"$set": {"status": "failed", "error": "photo missing"}})
            return
        form = await db.memory_forms.find_one({"photo_id": job["photo_id"]}, {"_id": 0}, sort=[("created_at", -1)])
        full_script = (form or {}).get("full_script") or "Merhaba, seni çok özledim."

        # Upload starting frame to fal storage
        start_path = os.path.join(workdir, "start.png")
        with open(start_path, "wb") as f:
            f.write(_b64.b64decode(photo["noir_b64"]))
        ref_url = await fal_client.upload_file_async(start_path)
        logger.info(f"[veo {job_id}] start frame uploaded {ref_url}")

        # Split script into NUM_CLIPS roughly equal parts by sentences
        sentences = [s.strip() for s in full_script.replace("…", ".").split(".") if s.strip()]
        per = max(1, len(sentences) // NUM_CLIPS)
        chunks = [". ".join(sentences[i*per:(i+1)*per]) + "." for i in range(NUM_CLIPS - 1)]
        chunks.append(". ".join(sentences[(NUM_CLIPS-1)*per:]) + ".")

        clip_paths: List[str] = []
        current_ref = ref_url

        import random
        ATMOSPHERES = [
            "Cinematic, dark gallery atmosphere with subtle warm golden rim light. ",
            "Cinematic, soft candlelight in a dimly lit room, warm amber tones. ",
            "Cinematic, gentle window light from the side, soft shadows, intimate. ",
            "Cinematic, warm late afternoon golden hour light, soft and nostalgic. ",
            "Cinematic, soft overhead lamp light, dark background, theatrical. ",
        ]
        EXPRESSIONS = [
            "The person looks straight at the camera with a gentle, loving, warm expression. ",
            "The person gazes softly at the camera with a tender, nostalgic expression. ",
            "The person looks into the camera with calm, warm, deeply affectionate eyes. ",
            "The person faces the camera with a quiet, loving, emotional expression. ",
        ]
        MOVEMENTS = [
            "Subtle natural lip sync matching the spoken line, slow blinks, intimate medium close-up, soft film grain, shallow depth. ",
            "Gentle lip movement, soft natural blinks, warm close-up, cinematic film grain, bokeh background. ",
            "Natural speech movement, slow blinks, medium close-up, soft focus background, vintage film texture. ",
            "Subtle mouth movement matching the words, intimate close-up, slow blinks, shallow depth of field. ",
        ]
        atmosphere = random.choice(ATMOSPHERES)
        expression = random.choice(EXPRESSIONS)
        movement = random.choice(MOVEMENTS)

        for i, line in enumerate(chunks):
            prompt = (
                atmosphere +
                "CRITICAL — preserve the exact age, face, identity, hair, skin and "
                "clothing of the person shown in the reference image. DO NOT age, "
                "rejuvenate, stylize or alter their appearance in any way; they must "
                "look identical in age to the reference. " +
                expression +
                movement +
                f"Speaks softly in Turkish: \"{line.strip()}\". "
                "Static camera; no zoom, no pan; preserve the reference face one-to-one."
            )
            video_url = await _veo_clip(current_ref, prompt)
            clip_local = os.path.join(workdir, f"c{i}.mp4")
            await asyncio.to_thread(_http_download, video_url, clip_local)
            clip_paths.append(clip_local)
            logger.info(f"[veo {job_id}] clip {i+1}/{NUM_CLIPS} ok")

            await db.video_jobs.update_one(
                {"id": job_id},
                {"$set": {"progress": int((i + 1) / NUM_CLIPS * 90)}},
            )

            if i < NUM_CLIPS - 1:
                next_frame = os.path.join(workdir, f"f{i}.png")
                await asyncio.to_thread(_ffmpeg_last_frame, clip_local, next_frame)
                current_ref = await fal_client.upload_file_async(next_frame)

        # Concat
        final_local = os.path.join(workdir, "final.mp4")
        await asyncio.to_thread(_ffmpeg_concat, clip_paths, final_local)
        final_url = await fal_client.upload_file_async(final_local)
        logger.info(f"[veo {job_id}] final uploaded {final_url}")

        await db.video_jobs.update_one(
            {"id": job_id},
            {"$set": {"status": "ready", "media_url": final_url, "kind": "video", "progress": 100}},
        )
    except Exception as e:
        logger.exception(f"[veo {job_id}] failed")
        await db.video_jobs.update_one(
            {"id": job_id}, {"$set": {"status": "failed", "error": str(e)[:500]}}
        )
    finally:
        try: shutil.rmtree(workdir, ignore_errors=True)
        except Exception: pass


# ---------- Video job lifecycle ----------
@api_router.post("/video/request", response_model=VideoJob)
async def request_video(body: VideoRequestBody):
    photo = await db.photos.find_one({"id": body.photo_id}, {"_id": 0})
    if not photo:
        raise HTTPException(status_code=404, detail="Fotoğraf bulunamadı")
    if photo.get("status") != "ready":
        raise HTTPException(status_code=400, detail="Fotoğraf henüz hazır değil")
    job = VideoJob(photo_id=body.photo_id, user_email=body.user_email)
    await db.video_jobs.insert_one(job.model_dump())
    return job


@api_router.post("/payment/dev-skip/{job_id}", response_model=VideoJobPublic)
async def payment_dev_skip(job_id: str):
    """TEST-ONLY: skip Iyzico entirely, mark job paid.
    In the new post-paywall flow, Veo video generation only starts AFTER
    payment — so dev-skip also has to kick off the pipeline (unless a
    video is already generated from a previous run). Refuses to run if
    IYZICO_MODE == 'production'."""
    if IYZICO_MODE == "production":
        raise HTTPException(status_code=403, detail="Dev-skip is disabled in production")
    job = await db.video_jobs.find_one({"id": job_id}, {"_id": 0})
    if not job:
        raise HTTPException(status_code=404, detail="İş bulunamadı")

    # Idempotent: if already paid, just echo state.
    if job.get("payment_status") == "paid":
        return VideoJobPublic(
            id=job_id, status=job.get("status", "generating"),
            payment_status="paid", media_url=job.get("media_url"),
            kind=job.get("kind", "video"), progress=int(job.get("progress") or 0),
        )

    # Require that the cinematic photo is ready — Veo cannot start otherwise.
    photo = await db.photos.find_one({"id": job["photo_id"]}, {"_id": 0})
    if not photo or not photo.get("noir_b64"):
        raise HTTPException(status_code=400, detail="Fotoğraf hazır değil")

    await db.video_jobs.update_one(
        {"id": job_id},
        {"$set": {
            "payment_status": "paid",
            "iyzico_payment_id": "DEV_SKIP",
            "iyzico_paid_price": "0.00",
        }},
    )

    already_ready = job.get("status") == "ready" and job.get("media_url")
    if not already_ready:
        asyncio.create_task(_run_veo_pipeline(job_id))
        logger.info(f"[dev-skip] job {job_id} unlocked — Veo pipeline STARTED")
        return VideoJobPublic(
            id=job_id, status="generating",
            payment_status="paid", media_url=None,
            kind="video", progress=0,
        )
    logger.info(f"[dev-skip] job {job_id} unlocked — video already ready, reusing")
    return VideoJobPublic(
        id=job_id, status=job.get("status", "ready"),
        payment_status="paid", media_url=job.get("media_url"),
        kind=job.get("kind", "video"), progress=int(job.get("progress") or 100),
    )


@api_router.post("/payment/initiate")
async def payment_initiate(body: PaymentInitiateBody, request: Request):
    """Create an Iyzico hosted Checkout Form session for this job.
    Returns the `payment_page_url` (open in browser/WebView) and our `callback_return`
    URL (where client polls status after the user returns)."""
    job = await db.video_jobs.find_one({"id": body.job_id}, {"_id": 0})
    if not job:
        raise HTTPException(status_code=404, detail="İş bulunamadı")
    if job.get("payment_status") == "paid":
        raise HTTPException(status_code=400, detail="Zaten ödenmiş")

    photo = await db.photos.find_one({"id": job["photo_id"]}, {"_id": 0})
    if not photo or not photo.get("noir_b64"):
        raise HTTPException(status_code=400, detail="Fotoğraf hazır değil")

    conversation_id = str(uuid.uuid4())
    buyer_ip = (request.client.host if request.client else None) or "85.34.78.112"

    # Derive callback URL dynamically from incoming request so the SAME backend
    # works for preview, deployed, and local dev without env reconfiguration.
    # Kubernetes ingress sets X-Forwarded-Proto / X-Forwarded-Host; fall back
    # to request.url and finally PUBLIC_BACKEND_URL env var.
    fwd_proto = request.headers.get("x-forwarded-proto")
    fwd_host = request.headers.get("x-forwarded-host") or request.headers.get("host")
    if fwd_host:
        scheme = fwd_proto or ("https" if "emergent" in fwd_host or "emergentagent" in fwd_host else request.url.scheme)
        callback_url = f"{scheme}://{fwd_host}/api/payment/callback"
    else:
        callback_url = f"{PUBLIC_BACKEND_URL}/api/payment/callback"
    logger.info(f"[iyzico] job={body.job_id} callbackUrl={callback_url}")

    checkout_request = {
        "locale": "tr",
        "conversationId": conversation_id,
        "price": "100.00",
        "paidPrice": "100.00",
        "currency": "TRY",
        "basketId": f"hatirai-{body.job_id}",
        "paymentGroup": "PRODUCT",
        "callbackUrl": callback_url,
        "enabledInstallments": ["1", "2", "3", "6", "9"],
        "buyer": {
            "id": body.job_id,
            "name": "HatirAI",
            "surname": "Musteri",
            "gsmNumber": "+905350000000",
            "email": "musteri@hatirai.app",
            "identityNumber": "11111111111",
            "lastLoginDate": "2024-01-01 00:00:00",
            "registrationDate": "2024-01-01 00:00:00",
            "registrationAddress": "Istanbul",
            "ip": buyer_ip,
            "city": "Istanbul",
            "country": "Turkey",
            "zipCode": "34000",
        },
        "shippingAddress": {
            "contactName": "HatirAI Musteri",
            "city": "Istanbul",
            "country": "Turkey",
            "address": "Dijital teslimat",
            "zipCode": "34000",
        },
        "billingAddress": {
            "contactName": "HatirAI Musteri",
            "city": "Istanbul",
            "country": "Turkey",
            "address": "Dijital teslimat",
            "zipCode": "34000",
        },
        "basketItems": [
            {
                "id": body.job_id,
                "name": "HatırAI Sinematik Canlandırma",
                "category1": "Dijital Icerik",
                "itemType": "VIRTUAL",
                "price": "100.00",
            }
        ],
    }

    try:
        resp = iyzipay.CheckoutFormInitialize().create(checkout_request, IYZICO_OPTIONS)
        data = _json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        logger.exception("[iyzico] initialize failed")
        raise HTTPException(status_code=502, detail=f"Iyzico hatası: {e}")

    if data.get("status") != "success":
        err = data.get("errorMessage") or "Bilinmeyen Iyzico hatası"
        logger.error(f"[iyzico] initialize non-success: {data}")
        raise HTTPException(status_code=400, detail=err)

    await db.video_jobs.update_one(
        {"id": body.job_id},
        {"$set": {
            "iyzico_token": data.get("token"),
            "iyzico_conversation_id": conversation_id,
            "iyzico_page_url": data.get("paymentPageUrl"),
            "iyzico_mode": IYZICO_MODE,
        }},
    )

    return {
        "job_id": body.job_id,
        "payment_page_url": data.get("paymentPageUrl"),
        "token": data.get("token"),
        "conversation_id": conversation_id,
        "mode": IYZICO_MODE,
    }


async def _finalize_job_by_token(token: str) -> Optional[dict]:
    """Verify a checkout token with Iyzico and mark job paid if successful."""
    if not token:
        return None
    try:
        resp = iyzipay.CheckoutForm().retrieve(
            {"locale": "tr", "conversationId": "", "token": token},
            IYZICO_OPTIONS,
        )
        data = _json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        logger.exception("[iyzico] retrieve failed")
        return None

    if data.get("status") != "success" or data.get("paymentStatus") != "SUCCESS":
        logger.warning(f"[iyzico] payment not successful: status={data.get('status')} paymentStatus={data.get('paymentStatus')}")
        job = await db.video_jobs.find_one({"iyzico_token": token}, {"_id": 0})
        if job:
            await db.video_jobs.update_one(
                {"id": job["id"]},
                {"$set": {"status": "failed", "iyzico_error": data.get("errorMessage")}},
            )
        return data

    # Success — find job and deliver content
    job = await db.video_jobs.find_one({"iyzico_token": token}, {"_id": 0})
    if not job:
        logger.error("[iyzico] job not found for token")
        return data

    photo = await db.photos.find_one({"id": job["photo_id"]}, {"_id": 0})
    if not photo or not photo.get("noir_b64"):
        return data

    await db.video_jobs.update_one(
        {"id": job["id"]},
        {"$set": {
            "payment_status": "paid",
            "iyzico_payment_id": data.get("paymentId"),
            "iyzico_paid_price": data.get("paidPrice"),
        }},
    )
    # Video generation is deferred until payment — kick it off NOW.
    already_ready = job.get("status") == "ready" and job.get("media_url")
    if not already_ready:
        asyncio.create_task(_run_veo_pipeline(job["id"]))
        logger.info(f"[iyzico] job {job['id']} paid (paymentId={data.get('paymentId')}) — Veo pipeline STARTED")
    else:
        logger.info(f"[iyzico] job {job['id']} paid (paymentId={data.get('paymentId')}) — video already ready")
    return data


@api_router.post("/payment/callback")
async def payment_callback_post(request: Request):
    """Iyzico POSTs to this URL after the user finishes the payment form.
    Form-encoded body with `token`. We verify, mark paid, and return a tiny HTML
    that the front-end polls via /api/video/{job_id}."""
    form = await request.form()
    token = form.get("token")
    result = await _finalize_job_by_token(token) if token else None

    ok = bool(result and result.get("paymentStatus") == "SUCCESS")
    title = "Ödeme Başarılı" if ok else "Ödeme Başarısız"
    color = "#C9A961" if ok else "#E59A9A"
    body_msg = (
        "Uygulamaya geri dönebilirsiniz. İçeriğiniz hazır."
        if ok else
        "Ödeme tamamlanamadı. Uygulamaya dönün ve tekrar deneyin."
    )
    html = f"""<!doctype html><html lang="tr"><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>HatırAI · {title}</title>
<style>
 body{{margin:0;background:#000;color:#F4F1EA;font-family:Georgia,serif;display:flex;align-items:center;justify-content:center;min-height:100vh;padding:24px;text-align:center}}
 .c{{max-width:420px}}
 h1{{font-size:40px;letter-spacing:-1px;color:{color};margin:0 0 16px}}
 p{{opacity:.8;line-height:1.6}}
 .mono{{font-family:'Courier New',monospace;font-size:11px;letter-spacing:3px;opacity:.6;margin-top:24px}}
</style></head>
<body><div class="c">
 <h1>{title}</h1>
 <p>{body_msg}</p>
 <p class="mono">HATIRAI · SİNEMATİK GALERİ</p>
</div>
<script>setTimeout(()=>{{try{{window.close()}}catch(e){{}}}},4000)</script>
</body></html>"""
    return HTMLResponse(content=html, status_code=200)


@api_router.get("/payment/callback")
async def payment_callback_get(token: Optional[str] = None):
    """Some Iyzico redirects come in as GET with token as query param."""
    result = await _finalize_job_by_token(token) if token else None
    ok = bool(result and result.get("paymentStatus") == "SUCCESS")
    color = "#C9A961" if ok else "#E59A9A"
    title = "Ödeme Başarılı" if ok else "Ödeme Başarısız"
    html = (
        "<html><body style='background:#000;color:#fff;font-family:serif;text-align:center;padding:80px'>"
        f"<h1 style='color:{color}'>{title}</h1>"
        "<p>Uygulamaya dönebilirsiniz.</p></body></html>"
    )
    return HTMLResponse(content=html)


@api_router.post("/payment/webhook")
async def payment_webhook(request: Request):
    """Iyzico webhook — server-side confirmation. Verifies signature
    and finalizes the job if we missed the callback."""
    body = await request.body()
    try:
        payload = _json.loads(body.decode("utf-8")) if body else {}
    except Exception:
        raise HTTPException(status_code=400, detail="Bad JSON")

    # Iyzico v3 signature: HMAC-SHA256(secret, secret + iyziEventType + paymentId + paymentConversationId + status)
    sig_header = request.headers.get("X-IYZ-SIGNATURE-V3", "")
    secret = IYZICO_OPTIONS["secret_key"]
    ev = payload.get("iyziEventType", "")
    pid = str(payload.get("paymentId", ""))
    cid = str(payload.get("paymentConversationId", ""))
    st = payload.get("status", "")
    try:
        import hmac as _hmac, hashlib as _hashlib
        expected = _hmac.new(
            secret.encode("utf-8"),
            f"{secret}{ev}{pid}{cid}{st}".encode("utf-8"),
            _hashlib.sha256,
        ).hexdigest()
        if sig_header and sig_header.lower() != expected.lower():
            logger.warning("[iyzico-webhook] signature mismatch")
            raise HTTPException(status_code=401, detail="Invalid signature")
    except HTTPException:
        raise
    except Exception:
        logger.exception("[iyzico-webhook] signature check error")

    # Finalize by conversation_id if we have the job
    job = await db.video_jobs.find_one({"iyzico_conversation_id": cid}, {"_id": 0})
    if job and st == "SUCCESS":
        token = job.get("iyzico_token")
        await _finalize_job_by_token(token)
    return {"ok": True}


@api_router.get("/video/{job_id}", response_model=VideoJobPublic)
async def get_video(job_id: str):
    """Client polls this after payment to receive media_url."""
    job = await db.video_jobs.find_one({"id": job_id}, {"_id": 0})
    if not job:
        raise HTTPException(status_code=404, detail="İş bulunamadı")
    return VideoJobPublic(
        id=job["id"],
        status=job.get("status", "pending_payment"),
        payment_status=job.get("payment_status", "unpaid"),
        media_url=job.get("media_url"),
        kind=job.get("kind", "image"),
        progress=int(job.get("progress") or 0),
    )


# ---------- User Auth (Emergent Google OAuth) ----------
EMERGENT_AUTH_SESSION_URL = "https://demobackend.emergentagent.com/auth/v1/env/oauth/session-data"


class EmailRegisterBody(BaseModel):
    email: str
    password: str
    name: str


class EmailLoginBody(BaseModel):
    email: str
    password: str


RESEND_API_KEY = os.environ.get("RESEND_API_KEY")

# Bcrypt
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Rate limiting — IP başına login denemesi
_login_attempts: dict = defaultdict(list)
_RATE_LIMIT_MAX = 10  # 10 deneme
_RATE_LIMIT_WINDOW = 300  # 5 dakika

def _check_rate_limit(ip: str):
    now = time.time()
    attempts = [t for t in _login_attempts[ip] if now - t < _RATE_LIMIT_WINDOW]
    _login_attempts[ip] = attempts
    if len(attempts) >= _RATE_LIMIT_MAX:
        raise HTTPException(status_code=429, detail="Çok fazla deneme. 5 dakika bekleyin.")
    _login_attempts[ip].append(now)


class ForgotPasswordBody(BaseModel):
    email: str


class ResetPasswordBody(BaseModel):
    token: str
    password: str


@api_router.post("/auth/forgot-password")
async def forgot_password(body: ForgotPasswordBody):
    import hashlib, secrets
    email = body.email.lower().strip()
    user = await db.users.find_one({"email": email}, {"_id": 0})
    if not user:
        return {"message": "Eğer bu email kayıtlıysa sıfırlama linki gönderildi."}
    token = secrets.token_urlsafe(32)
    expires = datetime.now(timezone.utc) + timedelta(hours=1)
    await db.password_resets.insert_one({
        "email": email,
        "token": token,
        "expires_at": expires,
        "used": False,
        "created_at": datetime.now(timezone.utc),
    })
    reset_url = f"{PUBLIC_BACKEND_URL}/reset-password?token={token}"
    if RESEND_API_KEY:
        try:
            import httpx
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.post(
                    "https://api.resend.com/emails",
                    headers={"Authorization": f"Bearer {RESEND_API_KEY}", "Content-Type": "application/json"},
                    json={
                        "from": "HatırAI <noreply@hatirai.com>",
                        "to": [email],
                        "subject": "Şifre Sıfırlama — HatırAI",
                        "html": f"""
                        <div style="background:#080808;color:#E8E0D0;font-family:Georgia,serif;padding:48px;max-width:480px;margin:0 auto;">
                          <h1 style="color:#C9A961;font-size:32px;letter-spacing:4px;margin-bottom:8px;">HATIR<span style="color:#E8E0D0;">AI</span></h1>
                          <hr style="border-color:#1E1C18;margin:24px 0;">
                          <p style="font-size:16px;line-height:1.8;color:#9C9A93;">Şifrenizi sıfırlamak için aşağıdaki butona tıklayın. Link 1 saat geçerlidir.</p>
                          <a href="{reset_url}" style="display:inline-block;background:#C9A961;color:#080808;font-family:monospace;font-size:12px;letter-spacing:3px;padding:16px 32px;text-decoration:none;margin:32px 0;">ŞİFREYİ SIFIRLA</a>
                          <p style="font-size:11px;color:#4A4540;">Bu isteği siz yapmadıysanız bu emaili görmezden gelin.</p>
                        </div>
                        """
                    }
                )
                logger.info(f"[resend] status={resp.status_code} body={resp.text[:200]}")
        except Exception as e:
            logger.error(f"[resend] Email gönderilemedi: {e}")
    return {"message": "Eğer bu email kayıtlıysa sıfırlama linki gönderildi."}


@api_router.post("/auth/reset-password")
async def reset_password(body: ResetPasswordBody):
    import hashlib
    record = await db.password_resets.find_one({"token": body.token, "used": False})
    if not record:
        raise HTTPException(status_code=400, detail="Geçersiz veya süresi dolmuş link.")
    if record["expires_at"].replace(tzinfo=timezone.utc) < datetime.now(timezone.utc):
        raise HTTPException(status_code=400, detail="Link süresi dolmuş.")
    pw_hash = pwd_context.hash(body.password)
    await db.users.update_one({"email": record["email"]}, {"$set": {"pw_hash": pw_hash}})
    await db.password_resets.update_one({"token": body.token}, {"$set": {"used": True}})
    return {"message": "Şifreniz başarıyla güncellendi."}


@api_router.post("/auth/register")
async def auth_register(body: EmailRegisterBody, request: Request):
    _check_rate_limit(request.client.host)
    email = body.email.lower().strip()
    if not email or not body.password:
        raise HTTPException(status_code=400, detail="Email ve şifre gerekli")
    if len(body.password) < 6:
        raise HTTPException(status_code=400, detail="Şifre en az 6 karakter olmalı")
    existing = await db.users.find_one({"email": email}, {"_id": 0})
    if existing:
        raise HTTPException(status_code=400, detail="Bu email zaten kayıtlı")
    pw_hash = pwd_context.hash(body.password)
    user_id = f"user_{uuid.uuid4().hex[:12]}"
    await db.users.insert_one({
        "user_id": user_id, "email": email, "name": body.name.strip(),
        "picture": None, "pw_hash": pw_hash,
        "created_at": datetime.now(timezone.utc),
    })
    session_token = uuid.uuid4().hex
    expires = datetime.now(timezone.utc) + timedelta(days=7)
    await db.user_sessions.insert_one({
        "user_id": user_id, "session_token": session_token,
        "expires_at": expires, "created_at": datetime.now(timezone.utc),
    })
    # Hoşgeldin maili gönder
    if RESEND_API_KEY:
        try:
            import httpx
            async with httpx.AsyncClient(timeout=10) as client:
                await client.post(
                    "https://api.resend.com/emails",
                    headers={"Authorization": f"Bearer {RESEND_API_KEY}", "Content-Type": "application/json"},
                    json={
                        "from": "HatırAI <noreply@hatirai.com>",
                        "to": [email],
                        "subject": "HatırAI'ya Hoş Geldiniz",
                        "html": f"""
                        <div style="background:#080808;color:#E8E0D0;font-family:Georgia,serif;padding:48px;max-width:480px;margin:0 auto;">
                          <h1 style="color:#C9A961;font-size:32px;letter-spacing:4px;margin-bottom:8px;">HATIR<span style="color:#E8E0D0;">AI</span></h1>
                          <hr style="border-color:#1E1C18;margin:24px 0;">
                          <p style="font-size:16px;line-height:1.8;color:#9C9A93;">Merhaba {body.name.strip()},</p>
                          <p style="font-size:16px;line-height:1.8;color:#9C9A93;">HatırAI'ya hoş geldiniz. Artık yakınlarınızın anılarını sinematik videolara dönüştürebilirsiniz.</p>
                          <a href="https://hatirai.com" style="display:inline-block;background:#C9A961;color:#080808;font-family:monospace;font-size:12px;letter-spacing:3px;padding:16px 32px;text-decoration:none;margin:32px 0;">UYGULAMAYA GİT</a>
                        </div>
                        """
                    }
                )
        except Exception as e:
            logger.warning(f"[register] Hoşgeldin maili gönderilemedi: {e}")
    return {
        "session_token": session_token,
        "user": {"user_id": user_id, "email": email, "name": body.name.strip(), "picture": None},
        "expires_at": expires,
    }


@api_router.post("/auth/login")
async def auth_login(body: EmailLoginBody, request: Request):
    _check_rate_limit(request.client.host)
    email = body.email.lower().strip()
    user = await db.users.find_one({"email": email}, {"_id": 0})
    if not user or not pwd_context.verify(body.password, user.get("pw_hash", "")):
        raise HTTPException(status_code=401, detail="Email veya şifre hatalı")
    session_token = uuid.uuid4().hex
    expires = datetime.now(timezone.utc) + timedelta(days=7)
    await db.user_sessions.insert_one({
        "user_id": user["user_id"], "session_token": session_token,
        "expires_at": expires, "created_at": datetime.now(timezone.utc),
    })
    return {
        "session_token": session_token,
        "user": {"user_id": user["user_id"], "email": email, "name": user.get("name", ""), "picture": user.get("picture")},
        "expires_at": expires,
    }


class User(BaseModel):
    user_id: str
    email: str
    name: str
    picture: Optional[str] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class AuthSessionBody(BaseModel):
    session_id: str


async def current_user(authorization: Optional[str] = Header(None)) -> Optional[dict]:
    """Optional user: extracts Bearer <session_token> if present; returns user dict or None."""
    if not authorization or not authorization.startswith("Bearer "):
        return None
    token = authorization.split(" ", 1)[1]
    # admin JWTs are also "Bearer"; keep them separate
    if token.count(".") == 2:  # looks like JWT — skip
        return None
    sess = await db.user_sessions.find_one({"session_token": token}, {"_id": 0})
    if not sess:
        return None
    exp = sess.get("expires_at")
    if isinstance(exp, str):
        try: exp = datetime.fromisoformat(exp)
        except Exception: exp = None
    if exp and exp.tzinfo is None:
        exp = exp.replace(tzinfo=timezone.utc)
    if exp and exp < datetime.now(timezone.utc):
        return None
    user = await db.users.find_one({"user_id": sess["user_id"]}, {"_id": 0})
    return user


async def require_user(user: Optional[dict] = Depends(current_user)) -> dict:
    if not user:
        raise HTTPException(status_code=401, detail="Giriş gerekli")
    return user


@api_router.post("/auth/session")
async def auth_session(body: AuthSessionBody):
    """Exchange Emergent OAuth session_id for our own session_token + user."""
    try:
        r = _requests.get(
            EMERGENT_AUTH_SESSION_URL,
            headers={"X-Session-ID": body.session_id},
            timeout=10,
        )
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Oturum doğrulanamadı: {e}")

    email = (data.get("email") or "").lower().strip()
    name = data.get("name") or "Anonim"
    picture = data.get("picture")
    session_token = data.get("session_token") or str(uuid.uuid4())
    if not email:
        raise HTTPException(status_code=401, detail="Email alınamadı")

    # Upsert user
    existing = await db.users.find_one({"email": email}, {"_id": 0})
    if existing:
        user_id = existing["user_id"]
        await db.users.update_one(
            {"user_id": user_id},
            {"$set": {"name": name, "picture": picture}},
        )
    else:
        user_id = f"user_{uuid.uuid4().hex[:12]}"
        await db.users.insert_one({
            "user_id": user_id, "email": email, "name": name,
            "picture": picture, "created_at": datetime.now(timezone.utc),
        })

    # Create session (7 days)
    expires = datetime.now(timezone.utc) + timedelta(days=7)
    await db.user_sessions.insert_one({
        "user_id": user_id, "session_token": session_token,
        "expires_at": expires, "created_at": datetime.now(timezone.utc),
    })
    return {
        "session_token": session_token,
        "user": {"user_id": user_id, "email": email, "name": name, "picture": picture},
        "expires_at": expires,
    }


@api_router.get("/auth/me")
async def auth_me(user: dict = Depends(require_user)):
    return {"user_id": user["user_id"], "email": user["email"],
            "name": user["name"], "picture": user.get("picture")}


@api_router.post("/auth/logout")
async def auth_logout(authorization: Optional[str] = Header(None)):
    if authorization and authorization.startswith("Bearer "):
        token = authorization.split(" ", 1)[1]
        await db.user_sessions.delete_many({"session_token": token})
    return {"ok": True}


@api_router.get("/user/history")
async def user_history(user: dict = Depends(require_user)):
    cursor = db.video_jobs.find(
        {"user_id": user["user_id"]}, {"_id": 0}
    ).sort("created_at", -1).limit(50)
    items = await cursor.to_list(length=50)
    result = []
    for i in items:
        photo = await db.photos.find_one({"id": i.get("photo_id")}, {"_id": 0, "noir_b64": 1})
        noir_b64 = (photo or {}).get("noir_b64")
        result.append({
            "job_id": i.get("id"),
            "photo_id": i.get("photo_id"),
            "status": i.get("status"),
            "payment_status": i.get("payment_status"),
            "kind": i.get("kind"),
            "media_url": i.get("media_url"),
            "created_at": i.get("created_at"),
            "noir_b64": noir_b64,
        })
    return result


# ---------- End User Auth ----------

@api_router.get("/admin/jobs", response_model=List[VideoJob])
async def admin_all(_: str = Depends(require_admin)):
    cursor = db.video_jobs.find({}, {"_id": 0}).sort("created_at", -1)
    items = await cursor.to_list(length=500)
    return [VideoJob(**i) for i in items]


# Mount router
app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Expo web static bundle — serves the frontend when running as a single-pod
# deployment (e.g. Emergent production). Harmless in preview because the
# Emergent preview ingress routes / to the Expo dev server (port 3000) and
# only hits the backend for /api/*, so this catch-all is never exercised.
#
# Requires `npx expo export -p web` to have been run (produces dist/).
# If dist/ is missing, the handlers gracefully fall back to 404.
# ---------------------------------------------------------------------------
DIST_DIR = Path("/app/frontend/dist")

if DIST_DIR.is_dir():
    # Static asset folders produced by expo export.
    for sub in ("_expo", "assets"):
        sub_path = DIST_DIR / sub
        if sub_path.is_dir():
            app.mount(f"/{sub}", StaticFiles(directory=str(sub_path)), name=f"expo-{sub}")

    @app.get("/favicon.ico", include_in_schema=False)
    async def _favicon():
        ico = DIST_DIR / "favicon.ico"
        if ico.is_file():
            return FileResponse(ico)
        raise HTTPException(status_code=404)

    @app.get("/{full_path:path}", include_in_schema=False)
    async def spa_fallback(full_path: str):
        """Serve Expo Router's per-route HTML files; fall back to index.html."""
        # Never intercept /api/* (registered above; FastAPI matches in order).
        if full_path.startswith("api/") or full_path.startswith("api"):
            raise HTTPException(status_code=404)

        # Direct file hit (css/js/images inside dist/)
        direct = DIST_DIR / full_path
        if direct.is_file():
            return FileResponse(direct)

        # Per-route HTML produced by expo export
        if full_path:
            html = DIST_DIR / f"{full_path}.html"
            if html.is_file():
                return FileResponse(html)

        # Root
        index = DIST_DIR / "index.html"
        if index.is_file():
            return FileResponse(index)

        # Expo's generated not-found page
        not_found = DIST_DIR / "+not-found.html"
        if not_found.is_file():
            return FileResponse(not_found, status_code=404)

        raise HTTPException(status_code=404)

    logger.info(f"[static] Serving Expo web bundle from {DIST_DIR}")
else:
    logger.info(f"[static] {DIST_DIR} not present — frontend served externally (preview mode)")


@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()
