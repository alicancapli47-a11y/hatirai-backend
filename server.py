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
ELEVENLABS_API_KEY = os.environ.get("ELEVENLABS_API_KEY", "")

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
    noir_b64: Optional[str] = None
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
    status: str = "pending_payment"
    payment_status: str = "unpaid"
    media_url: Optional[str] = None
    kind: Optional[str] = "image"
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class VideoRequestBody(BaseModel):
    photo_id: str
    user_email: Optional[str] = None


class MemoryFormBody(BaseModel):
    photo_id: str
    name: str
    relationship: str
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
    retry_count: Optional[int] = 0


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


async def current_user(authorization: Optional[str] = Header(None)) -> Optional[dict]:
    if not authorization or not authorization.startswith("Bearer "):
        return None
    token = authorization.split(" ", 1)[1]
    if token.count(".") == 2:
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
        raise HTTPException(status_code=401, detail="Giris gerekli")
    return user


# ===================== ROUTES =====================
@api_router.get("/")
async def root():
    return {"app": "HatirAI", "status": "ok"}


@api_router.post("/admin/login", response_model=AdminLoginResponse)
async def admin_login(body: AdminLoginRequest):
    if body.username != ADMIN_USERNAME or body.password != ADMIN_PASSWORD:
        raise HTTPException(status_code=401, detail="Gecersiz kullanici adi veya sifre")
    token, expires = create_admin_token()
    return AdminLoginResponse(token=token, expires_at=expires)


# ---------- Photo: cinematic transform ----------
ERA_PROMPTS: dict[str, str] = {
    "1950s": (
        "Restore and dramatize this old portrait. "
        "Fix quality: denoise heavily, sharpen face, fix blur, recover skin texture, clear sharp eyes. "
        "Reframe so subject faces camera directly, frontal headshot, eyes meeting lens, shoulders centered. "
        "Preserve identity, age, hair and clothing exactly. "
        "Replace background with pure solid black. "
        "Apply 1950s Hollywood black and white noir lighting: strong key light from one side, "
        "soft warm rim light, deep chiaroscuro shadows, silver-warm grayscale tones, fine film grain. "
        "Output only the final sharp image ready for animation."
    ),
    "80s": (
        "Restore and dramatize this old portrait. "
        "Fix quality: denoise heavily, sharpen face, fix blur, restore skin texture, clear sharp eyes. "
        "Reframe so subject faces camera directly, frontal headshot, eyes meeting lens. "
        "Preserve identity, age, hair and clothing exactly. "
        "Replace background with pure solid black. "
        "Apply warm 80s studio portrait lighting: strong golden rim light, soft fill light, "
        "warm color grade, faded pastel tones, soft halation, gentle film grain. Keep full color. "
        "Output only the final sharp image ready for animation."
    ),
    "modern": (
        "Restore and dramatize this portrait. "
        "Fix quality: denoise heavily, sharpen face, fix blur, restore skin texture, clear sharp eyes. "
        "Reframe so subject faces camera directly, frontal headshot, head straight, shoulders centered. "
        "Preserve identity, age, hair and clothing exactly. "
        "Replace background with pure solid black. "
        "Apply modern cinematic studio lighting: dramatic side rim light with warm golden highlights, "
        "deep true blacks, rich shadows, clean neutral midtones, delicate film grain. "
        "Output only the final sharp image ready for animation."
    ),
}

# ---------- Veo video prompts (English only, no Turkish text, no special chars) ----------
VEO_OPENING_PROMPTS = [
    (
        "Cinematic studio portrait. Pure black background only, no room, no environment. "
        "Dramatic warm golden rim light from the right, soft fill light from the left. "
        "Person faces camera directly with a calm, warm, loving expression. "
        "Gentle natural blinks, subtle mouth movement as if beginning to speak. "
        "Medium close-up shot. Fine film grain. Static camera, no zoom, no pan. "
        "Preserve exact face, age, hair, skin and clothing from the reference image."
    ),
    (
        "Cinematic portrait on pure black background. "
        "Strong side key light with warm amber tone, deep shadow on opposite side. "
        "Subject looks softly into the camera with tender, nostalgic eyes. "
        "Slow natural blinks, slight lip movement as if about to speak. "
        "Intimate medium close-up. Cinematic film texture. Static locked camera. "
        "Preserve exact identity and appearance from reference."
    ),
    (
        "Studio cinematic portrait, pure solid black background. "
        "Gentle warm backlight creating a soft halo, delicate frontal fill light. "
        "Person gazes at camera with quiet, affectionate, deeply emotional expression. "
        "Natural slow blinks, subtle jaw and lip movement. "
        "Close-up framing, vintage film grain. No camera movement whatsoever. "
        "Identical face, age and clothing to the reference image."
    ),
]

VEO_CLOSING_PROMPTS = [
    (
        "Cinematic studio portrait. Pure black background, no environment. "
        "Warm golden three-point studio lighting with rich deep shadows. "
        "Person holds a long gaze at the camera, expression soft and emotionally resolved. "
        "Eyes slowly become glassy, a slight melancholic smile forms. Very slow blink. "
        "Medium close-up fading gently. Cinematic grain. Static camera. "
        "Preserve exact face and appearance from reference."
    ),
    (
        "Cinematic portrait on pure black background. "
        "Dramatic warm rim light, gentle fill, deep blacks. "
        "Subject finishes speaking, eyes lower briefly then return to camera with warmth. "
        "Slow deliberate blink, peaceful expression. "
        "Intimate close-up, film grain texture. Locked static camera. "
        "Exact age, face, hair and clothing from reference preserved."
    ),
]


def _make_watermarked_preview_b64(clean_b64: str) -> str:
    try:
        img = _PILImage.open(BytesIO(_b64.b64decode(clean_b64))).convert("RGBA")
        W, H = img.size

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

        layer = _PILImage.new("RGBA", img.size, (0, 0, 0, 0))
        draw = _PILDraw.Draw(layer)
        text = "HatirAI - ONIZLEME"
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
        layer = layer.rotate(-22, resample=_PILImage.BICUBIC, expand=False)
        out = _PILImage.alpha_composite(img, layer)

        draw2 = _PILDraw.Draw(out)
        corner = "HatirAI - ONIZLEME - ODEME GEREKLI"
        try:
            bb = draw2.textbbox((0, 0), corner, font=font_corner)
            cw, ch = bb[2] - bb[0], bb[3] - bb[1]
        except Exception:
            cw, ch = (W // 3, H // 30)
        pad = 16
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

        image_bytes = _base64.b64decode(image_b64)
        tmp_path = f"/tmp/upload_{photo_id}.jpg"
        with open(tmp_path, "wb") as f:
            f.write(image_bytes)
        image_url = await fal_client.upload_file_async(tmp_path)

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
        logger.info(f"[cinema] Photo {photo_id} ready (era={era})")
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
        raise HTTPException(status_code=404, detail="Fotograf bulunamadi")
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
        return f"{name}, seni cok ozledim."
    try:
        ac = _anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        response = await asyncio.to_thread(
            ac.messages.create,
            model="claude-sonnet-4-5-20250929",
            max_tokens=100,
            system=(
                "Sen Turkce konusan, sicak, sinematik, hafif duygusal bir senaryo yazarisin. "
                "Sana bir iliski, bir isim ve bir ani verilecek. "
                "Fotodaki kisi (ornegin dede), video mesajinda karsi taraftaki kisiye (isim) hitap ediyor. "
                "Fotodaki kisinin agzindan, karsi tarafin adini kullanarak, aniya atifta bulunan TEK bir cumle yaz. "
                "En fazla 18 kelime. Kliseden kacin. Sadece cumleyi dondur."
            ),
            messages=[{
                "role": "user",
                "content": (
                    f"Fotodaki kisinin kim oldugu: {relationship}\n"
                    f"Konusulan kisinin adi (hitap edilecek): {name}\n"
                    f"Paylasilan ani: {last_memory}\n\n"
                    f"Fotodaki {relationship}, {name}'e hitap ederek bu aniya dair 1 cumle soylüyor."
                )
            }]
        )
        line = (response.content[0].text or "").strip().strip('"').strip("'").splitlines()[0]
        return line[:240] if line else f"{name}, o gunleri hic unutmadim."
    except Exception as e:
        logger.warning(f"[ai-sentence] fallback ({e})")
        return f"{name}, seni cok ozledim."


async def _build_full_script(name: str, relationship: str, last_memory: Optional[str], ai_sentence: str) -> str:
    """
    Fotodaki kisi (relationship), kullaniciya (name) hitap ederek konusuyor.
    Claude aniya sadik, isim gecen dogal Turkce konusma metni uretir.
    """
    try:
        ac = _anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        memory_hint = last_memory if last_memory and last_memory.strip() else "genel ozlem ve sevgi"
        response = await asyncio.to_thread(
            ac.messages.create,
            model="claude-sonnet-4-5-20250929",
            max_tokens=300,
            system=(
                "Sen Turkce konusan duygusal bir senaryo yazarisin. "
                "Fotodaki kisi (ornegin dede), sevdigi birine kisa bir video mesaji birakiyor. "
                "Mesaj fotodaki kisinin agzindan, karsi taraftaki kisinin adini kullanarak yazilmali. "
                "Karsi tarafin adini en az 2 kez kullan. "
                "Verilen aniya mutlaka atifta bulun. Kliseden kacin. "
                "Toplam 50-70 kelime. Tek akis, paragraf yok. "
                "Sadece konusma metnini yaz, hicbir aciklama ekleme."
            ),
            messages=[{
                "role": "user",
                "content": (
                    f"Konusan kisinin kim oldugu: {relationship} (fotodaki kisi)\n"
                    f"Konusulan kisinin adi: {name}\n"
                    f"Paylasilan ani: {memory_hint}\n"
                    f"Ilk cumle su olmali: {ai_sentence}\n\n"
                    f"{relationship} olarak, {name}'e hitap eden dogal bir video mesaji yaz."
                )
            }]
        )
        script = (response.content[0].text or "").strip()
        return script if len(script) > 20 else _fallback_script(name, ai_sentence)
    except Exception as e:
        logger.warning(f"[full-script] Claude fallback ({e})")
        return _fallback_script(name, ai_sentence)


def _fallback_script(name: str, ai_sentence: str) -> str:
    import random
    MIDDLES = [
        f"Her gun aklimdasin {name}, bunu bil. Seninle gecirdigimiz her ani icimde yasatiyorum.",
        f"Seni dusunmeden tek bir gun gecmiyor {name}. Sesin hala kulaklarimda.",
        f"Seni her zaman yanimda hissediyorum {name}. Biraktigin izler hic silinmedi.",
    ]
    CLOSINGS = [
        f"Seni cok seviyorum {name}. Kendine iyi bak.",
        f"Her zaman kalbimdesin {name}. Guclu ol.",
        f"Seni seviyorum {name}.",
    ]
    return f"{ai_sentence}. {random.choice(MIDDLES)} {random.choice(CLOSINGS)}"


@api_router.post("/memory/form")
async def submit_memory_form(body: MemoryFormBody, user: Optional[dict] = Depends(current_user)):
    photo = await db.photos.find_one({"id": body.photo_id}, {"_id": 0})
    if not photo:
        raise HTTPException(status_code=404, detail="Fotograf bulunamadi")
    if photo.get("status") != "ready":
        raise HTTPException(status_code=400, detail="Fotograf henuz hazir degil")

    name = body.name.strip()[:40] or "Sevgilim"
    rel = body.relationship.strip()[:30] or "Sevdigim"
    memory = (body.last_memory or "").strip()[:400]
    ai_sentence = await _generate_ai_sentence(name, rel, memory)
    full = await _build_full_script(name, rel, memory, ai_sentence)
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
    logger.info(f"[memory/form] photo={body.photo_id} job={job.id} user={user.get('email') if user else 'anon'}")

    return {"form": rec.model_dump(), "job_id": job.id}


@api_router.get("/memory/form/{photo_id}", response_model=Optional[MemoryForm])
async def get_memory_form(photo_id: str):
    doc = await db.memory_forms.find_one({"photo_id": photo_id}, {"_id": 0}, sort=[("created_at", -1)])
    return MemoryForm(**doc) if doc else None


# ---------- ElevenLabs TTS ----------
async def _generate_elevenlabs_audio(script: str, voice_id: str, workdir: str, filename: str) -> Optional[str]:
    """
    ElevenLabs ile Turkce sesi uret, MP3 olarak kaydet, yolunu dondur.
    Basarisiz olursa None doner (sessiz video uretilir).
    """
    if not ELEVENLABS_API_KEY:
        logger.warning("[tts] ELEVENLABS_API_KEY eksik, ses atlanıyor")
        return None
    try:
        import httpx
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
        headers = {
            "xi-api-key": ELEVENLABS_API_KEY,
            "Content-Type": "application/json",
        }
        payload = {
            "text": script,
            "model_id": "eleven_multilingual_v2",
            "voice_settings": {"stability": 0.5, "similarity_boost": 0.75},
        }
        async with httpx.AsyncClient(timeout=60) as hc:
            resp = await hc.post(url, headers=headers, json=payload)
        if resp.status_code != 200:
            logger.warning(f"[tts] ElevenLabs {resp.status_code}: {resp.text[:200]}")
            return None
        audio_path = os.path.join(workdir, filename)
        with open(audio_path, "wb") as f:
            f.write(resp.content)
        logger.info(f"[tts] Ses uretildi: {audio_path} ({len(resp.content)} bytes)")
        return audio_path
    except Exception as e:
        logger.warning(f"[tts] Ses uretilemedi: {e}")
        return None


def _ffmpeg_merge_audio(video_path: str, audio_path: str, out_path: str) -> None:
    """
    Video uzerine sesi ekle. Video suresi baz alinir (audio kesilir/uzatilmaz).
    """
    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-i", audio_path,
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-c:v", "copy",
        "-c:a", "aac", "-b:a", "128k",
        "-shortest",
        out_path,
    ]
    subprocess.run(cmd, check=True, capture_output=True, timeout=120)
    logger.info(f"[ffmpeg] Ses birlestirildi: {out_path}")


# ---------- Veo pipeline ----------
NUM_CLIPS = 2  # 2 x 8s = 16s


def _ffmpeg_last_frame(video_path: str, frame_path: str) -> None:
    cmd = ["ffmpeg", "-y", "-sseof", "-1", "-i", video_path, "-vframes", "1", "-q:v", "2", frame_path]
    subprocess.run(cmd, check=True, capture_output=True, timeout=60)


def _ffmpeg_concat(video_paths: List[str], out_path: str) -> None:
    """Clipları birlestir ve kose filigran ekle."""
    tmp = tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False)
    try:
        for p in video_paths:
            tmp.write(f"file '{os.path.abspath(p)}'\n")
        tmp.flush()
        tmp.close()

        font_candidates = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        ]
        font_path = next((f for f in font_candidates if os.path.exists(f)), None)

        if font_path:
            draw = (
                f"drawtext=fontfile='{font_path}':text='HatirAI':"
                f"fontcolor=0xE6C36A@0.85:fontsize=h/26:"
                f"x=w-tw-24:y=h-th-24:"
                f"box=1:boxcolor=black@0.35:boxborderw=10"
            )
            vf_args = ["-vf", draw]
        else:
            vf_args = []
            logger.warning("[watermark] Font bulunamadi, filigransiz video")

        cmd = [
            "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", tmp.name,
            *vf_args,
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "22",
            "-c:a", "aac", "-b:a", "128k",
            out_path,
        ]
        subprocess.run(cmd, check=True, capture_output=True, timeout=300)
    finally:
        try:
            os.unlink(tmp.name)
        except Exception:
            pass


def _http_download(url: str, dest: str) -> None:
    r = _requests.get(url, timeout=120)
    r.raise_for_status()
    with open(dest, "wb") as f:
        f.write(r.content)


async def _veo_clip(image_url: str, prompt: str) -> str:
    """
    Veo 3.1 Lite ile 8 saniyelik video uret.
    - generate_audio=True: Veo kendi sinematik sesini uretsin
    - safety_tolerance integer olmali (string degil)
    - Prompt tamamen Ingilizce, Turkce metin ve ozel karakter yok
    """
    handle = await fal_client.submit_async(
        "fal-ai/veo3.1/lite/image-to-video",
        arguments={
            "prompt": prompt,
            "image_url": image_url,
            "duration": "8s",
            "resolution": "720p",
            "aspect_ratio": "9:16",
            "generate_audio": True,
            "safety_tolerance": 4,   # integer, string degil
        },
    )
    result = await handle.get()
    return result["video"]["url"]


# Ses secimi: iliskiye gore ElevenLabs voice ID
ELEVENLABS_VOICES = {
    "dede":    "FYPltOzsM2n1UbqzX19d",
    "baba":    "FYPltOzsM2n1UbqzX19d",
    "nine":    "SAz9YHcvj6GT2YYXdXww",
    "anne":    "SAz9YHcvj6GT2YYXdXww",
    "teyze":   "SAz9YHcvj6GT2YYXdXww",
    "abla":    "SAz9YHcvj6GT2YYXdXww",
    "kiz":     "SAz9YHcvj6GT2YYXdXww",
    "agabey":  "WRjHw9UKGmcRAoOgyIzT",
    "abi":     "WRjHw9UKGmcRAoOgyIzT",
    "erkek":   "WRjHw9UKGmcRAoOgyIzT",
    "default": "FYPltOzsM2n1UbqzX19d",
}

# HeyGen fallback icin ses
HEYGEN_VOICES = {
    "dede":    "baeba2c18fea4438a4d83a54a462498a",
    "baba":    "baeba2c18fea4438a4d83a54a462498a",
    "anne":    "664b73058b784aa89ddb2924c141d441",
    "nine":    "664b73058b784aa89ddb2924c141d441",
    "teyze":   "664b73058b784aa89ddb2924c141d441",
    "abla":    "664b73058b784aa89ddb2924c141d441",
    "kiz":     "664b73058b784aa89ddb2924c141d441",
    "agabey":  "836aa05e398543d08231f68bffdfc025",
    "abi":     "836aa05e398543d08231f68bffdfc025",
    "erkek":   "836aa05e398543d08231f68bffdfc025",
    "default": "baeba2c18fea4438a4d83a54a462498a",
}


def _pick_voice(relationship: str, voice_map: dict) -> str:
    rel = (relationship or "").lower().strip()
    for key, voice_id in voice_map.items():
        if key != "default" and key in rel:
            return voice_id
    return voice_map["default"]


async def _heygen_clip(image_url: str, script: str, relationship: str = "") -> str:
    """HeyGen Avatar4 fallback."""
    voice_id = _pick_voice(relationship, HEYGEN_VOICES)
    handle = await fal_client.submit_async(
        "fal-ai/heygen/avatar4/image-to-video",
        arguments={
            "image_url": image_url,
            "prompt": script,
            "voice_id": voice_id,
            "talking_style": "expressive",
            "aspect_ratio": "9:16",
            "resolution": "720p",
            "background": {"type": "color", "value": "#000000"},
        },
    )
    result = await handle.get()
    return result["video"]["url"]


async def _run_veo_pipeline(job_id: str):
    """
    2 x 8s Veo klibi uret, son kare gecisl ile birlestir → 16s final video.
    Veo generate_audio=True: kendi sinematik sesini uretir.
    Prompt tamamen Ingilizce, Turkce metin veya ozel karakter YOK.
    """
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
        full_script = (form or {}).get("full_script") or "Merhaba, seni cok ozledim."
        relationship = (form or {}).get("relationship", "")

        # Baslangic fotografını kaydet ve yukle
        start_path = os.path.join(workdir, "start.png")
        with open(start_path, "wb") as f:
            f.write(_b64.b64decode(photo["noir_b64"]))
        ref_url = await fal_client.upload_file_async(start_path)
        logger.info(f"[veo {job_id}] Baslangic karesi yuklendi: {ref_url}")

        import random

        LIGHTINGS = [
            "dramatic warm golden rim light from the right, soft fill from the left, ",
            "strong side key light with warm amber tone, deep shadow on opposite side, ",
            "gentle warm backlight creating a halo effect, soft frontal fill light, ",
            "cinematic three-point studio lighting, warm golden highlights, ",
        ]
        EXPRESSIONS = [
            "The person looks straight at the camera with a gentle, loving, warm expression. ",
            "The person gazes softly at the camera with a tender, nostalgic expression. ",
            "The person looks into the camera with calm, warm, deeply affectionate eyes. ",
            "The person faces the camera with a quiet, loving, emotional expression. ",
        ]
        MOVEMENTS = [
            "Subtle natural lip sync as if speaking quietly, slow blinks, intimate medium close-up, soft film grain. ",
            "Gentle lip movement, soft natural blinks, warm close-up, cinematic film grain. ",
            "Natural speech movement, slow blinks, medium close-up, vintage film texture. ",
            "Subtle mouth movement, intimate close-up, slow blinks. ",
        ]

        # Script'i klip sayisina gore esit parcalara bol
        sentences = [s.strip() for s in full_script.replace("...", ".").split(".") if s.strip()]
        per = max(1, len(sentences) // NUM_CLIPS)
        chunks = [". ".join(sentences[i * per:(i + 1) * per]) for i in range(NUM_CLIPS - 1)]
        chunks.append(". ".join(sentences[(NUM_CLIPS - 1) * per:]))

        # Anı bilgileri — isim oldugu gibi geciyor, Turkce karakter sorun degil
        name_for_prompt = (form or {}).get("name") or "sevdigi biri"

        clip_paths: List[str] = []
        current_ref = ref_url

        for i in range(NUM_CLIPS):
            lighting = random.choice(LIGHTINGS)
            expression = random.choice(EXPRESSIONS)
            movement = random.choice(MOVEMENTS)

            # Bu klip icin soylenecek cumleleri al
            chunk_text = chunks[i] if chunks[i] else full_script
            is_last = (i == NUM_CLIPS - 1)

            prompt = (
                "Cinematic studio portrait on a pure black background, no environment, no room, no objects. "
                + lighting
                + "Preserve the exact age, face, identity, hair, skin and clothing of the person "
                "in the reference image. Do not alter their appearance in any way. "
                + expression
                + f"The person speaks the following lines out loud in Turkish, directly addressing {name_for_prompt}, "
                f"with deep emotion as a {relationship} would: "
                f'"{chunk_text}" '
                + ("Their expression softens with quiet emotional resolve as they finish. " if is_last else "")
                + movement
                + "Static camera, no zoom, no pan."
            )

            logger.info(f"[veo {job_id}] Klip {i+1}/{NUM_CLIPS} basliyor...")
            try:
                video_url = await _veo_clip(current_ref, prompt)
                logger.info(f"[veo {job_id}] Klip {i+1} tamamlandi: {video_url}")
            except Exception as veo_err:
                logger.warning(f"[veo {job_id}] Veo basarisiz, HeyGen fallback: {veo_err}")
                try:
                    video_url = await _heygen_clip(current_ref, full_script, relationship)
                    logger.info(f"[veo {job_id}] HeyGen fallback basarili")
                except Exception as hg_err:
                    logger.error(f"[veo {job_id}] HeyGen de basarisiz: {hg_err}")
                    await db.video_jobs.update_one(
                        {"id": job_id},
                        {"$set": {"status": "failed", "error": f"Veo: {veo_err} | HeyGen: {hg_err}"}}
                    )
                    return

            clip_local = os.path.join(workdir, f"c{i}.mp4")
            await asyncio.to_thread(_http_download, video_url, clip_local)
            clip_paths.append(clip_local)

            await db.video_jobs.update_one(
                {"id": job_id},
                {"$set": {"progress": int((i + 1) / NUM_CLIPS * 90)}},
            )

            # Sonraki klip icin son kareyi gecis noktasi olarak kullan
            if i < NUM_CLIPS - 1:
                next_frame = os.path.join(workdir, f"f{i}.png")
                await asyncio.to_thread(_ffmpeg_last_frame, clip_local, next_frame)
                current_ref = await fal_client.upload_file_async(next_frame)
                logger.info(f"[veo {job_id}] Gecis karesi yuklendi: {current_ref}")

        # Klipleri birlestir (ses dahil, Veo tarafindan uretildi)
        final_local = os.path.join(workdir, "final.mp4")
        await asyncio.to_thread(_ffmpeg_concat, clip_paths, final_local)
        logger.info(f"[veo {job_id}] Klipler birlestirildi")

        final_url = await fal_client.upload_file_async(final_local)
        logger.info(f"[veo {job_id}] Final yuklendi: {final_url}")

        await db.video_jobs.update_one(
            {"id": job_id},
            {"$set": {"status": "ready", "media_url": final_url, "kind": "video", "progress": 100}},
        )
    except Exception as e:
        logger.exception(f"[veo {job_id}] Pipeline hatasi")
        await db.video_jobs.update_one(
            {"id": job_id}, {"$set": {"status": "failed", "error": str(e)[:500]}}
        )
    finally:
        try:
            shutil.rmtree(workdir, ignore_errors=True)
        except Exception:
            pass


# ---------- Video job lifecycle ----------
@api_router.post("/video/request", response_model=VideoJob)
async def request_video(body: VideoRequestBody):
    photo = await db.photos.find_one({"id": body.photo_id}, {"_id": 0})
    if not photo:
        raise HTTPException(status_code=404, detail="Fotograf bulunamadi")
    if photo.get("status") != "ready":
        raise HTTPException(status_code=400, detail="Fotograf henuz hazir degil")
    job = VideoJob(photo_id=body.photo_id, user_email=body.user_email)
    await db.video_jobs.insert_one(job.model_dump())
    return job


@api_router.post("/payment/dev-skip/{job_id}", response_model=VideoJobPublic)
async def payment_dev_skip(job_id: str):
    if IYZICO_MODE == "production":
        raise HTTPException(status_code=403, detail="Dev-skip production'da devre disi")
    job = await db.video_jobs.find_one({"id": job_id}, {"_id": 0})
    if not job:
        raise HTTPException(status_code=404, detail="Is bulunamadi")

    if job.get("payment_status") == "paid":
        return VideoJobPublic(
            id=job_id, status=job.get("status", "generating"),
            payment_status="paid", media_url=job.get("media_url"),
            kind=job.get("kind", "video"), progress=int(job.get("progress") or 0),
        )

    photo = await db.photos.find_one({"id": job["photo_id"]}, {"_id": 0})
    if not photo or not photo.get("noir_b64"):
        raise HTTPException(status_code=400, detail="Fotograf hazir degil")

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
        logger.info(f"[dev-skip] {job_id} acildi, Veo pipeline basladi")
        return VideoJobPublic(
            id=job_id, status="generating",
            payment_status="paid", media_url=None,
            kind="video", progress=0,
        )
    logger.info(f"[dev-skip] {job_id} zaten hazir, yeniden kullaniliyor")
    return VideoJobPublic(
        id=job_id, status=job.get("status", "ready"),
        payment_status="paid", media_url=job.get("media_url"),
        kind=job.get("kind", "video"), progress=int(job.get("progress") or 100),
    )


@api_router.post("/payment/initiate")
async def payment_initiate(body: PaymentInitiateBody, request: Request):
    job = await db.video_jobs.find_one({"id": body.job_id}, {"_id": 0})
    if not job:
        raise HTTPException(status_code=404, detail="Is bulunamadi")
    if job.get("payment_status") == "paid":
        raise HTTPException(status_code=400, detail="Zaten odenmis")

    photo = await db.photos.find_one({"id": job["photo_id"]}, {"_id": 0})
    if not photo or not photo.get("noir_b64"):
        raise HTTPException(status_code=400, detail="Fotograf hazir degil")

    conversation_id = str(uuid.uuid4())
    buyer_ip = (request.client.host if request.client else None) or "85.34.78.112"

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
        "price": "99.00",
        "paidPrice": "99.00",
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
                "name": "HatirAI Sinematik Canlandirma",
                "category1": "Dijital Icerik",
                "itemType": "VIRTUAL",
                "price": "99.00",
            }
        ],
    }

    try:
        resp = iyzipay.CheckoutFormInitialize().create(checkout_request, IYZICO_OPTIONS)
        data = _json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        logger.exception("[iyzico] initialize failed")
        raise HTTPException(status_code=502, detail=f"Iyzico hatasi: {e}")

    if data.get("status") != "success":
        err = data.get("errorMessage") or "Bilinmeyen Iyzico hatasi"
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
        logger.warning(f"[iyzico] odeme basarisiz: status={data.get('status')} paymentStatus={data.get('paymentStatus')}")
        job = await db.video_jobs.find_one({"iyzico_token": token}, {"_id": 0})
        if job:
            await db.video_jobs.update_one(
                {"id": job["id"]},
                {"$set": {"status": "failed", "iyzico_error": data.get("errorMessage")}},
            )
        return data

    job = await db.video_jobs.find_one({"iyzico_token": token}, {"_id": 0})
    if not job:
        logger.error("[iyzico] token icin job bulunamadi")
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
    already_ready = job.get("status") == "ready" and job.get("media_url")
    if not already_ready:
        asyncio.create_task(_run_veo_pipeline(job["id"]))
        logger.info(f"[iyzico] job {job['id']} odendi, Veo pipeline basladi")
    else:
        logger.info(f"[iyzico] job {job['id']} odendi, video zaten hazir")
    return data


@api_router.post("/payment/callback")
async def payment_callback_post(request: Request):
    form = await request.form()
    token = form.get("token")
    result = await _finalize_job_by_token(token) if token else None

    ok = bool(result and result.get("paymentStatus") == "SUCCESS")
    title = "Odeme Basarili" if ok else "Odeme Basarisiz"
    color = "#C9A961" if ok else "#E59A9A"
    body_msg = (
        "Uygulamaya geri donebilirsiniz. Icerik hazir."
        if ok else
        "Odeme tamamlanamadi. Uygulamaya donun ve tekrar deneyin."
    )
    html = f"""<!doctype html><html lang="tr"><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>HatirAI - {title}</title>
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
 <p class="mono">HATIRAI - SINEMATIK GALERI</p>
</div>
<script>setTimeout(()=>{{try{{window.close()}}catch(e){{}}}},4000)</script>
</body></html>"""
    return HTMLResponse(content=html, status_code=200)


@api_router.get("/payment/callback")
async def payment_callback_get(token: Optional[str] = None):
    result = await _finalize_job_by_token(token) if token else None
    ok = bool(result and result.get("paymentStatus") == "SUCCESS")
    color = "#C9A961" if ok else "#E59A9A"
    title = "Odeme Basarili" if ok else "Odeme Basarisiz"
    html = (
        "<html><body style='background:#000;color:#fff;font-family:serif;text-align:center;padding:80px'>"
        f"<h1 style='color:{color}'>{title}</h1>"
        "<p>Uygulamaya donebilirsiniz.</p></body></html>"
    )
    return HTMLResponse(content=html)


@api_router.post("/payment/webhook")
async def payment_webhook(request: Request):
    body = await request.body()
    try:
        payload = _json.loads(body.decode("utf-8")) if body else {}
    except Exception:
        raise HTTPException(status_code=400, detail="Bad JSON")

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
            logger.warning("[iyzico-webhook] imza uyusmadi")
            raise HTTPException(status_code=401, detail="Invalid signature")
    except HTTPException:
        raise
    except Exception:
        logger.exception("[iyzico-webhook] imza kontrolu hatasi")

    job = await db.video_jobs.find_one({"iyzico_conversation_id": cid}, {"_id": 0})
    if job and st == "SUCCESS":
        token = job.get("iyzico_token")
        await _finalize_job_by_token(token)
    return {"ok": True}


@api_router.get("/video/{job_id}", response_model=VideoJobPublic)
async def get_video(job_id: str):
    job = await db.video_jobs.find_one({"id": job_id}, {"_id": 0})
    if not job:
        raise HTTPException(status_code=404, detail="Is bulunamadi")
    return VideoJobPublic(
        id=job["id"],
        status=job.get("status", "pending_payment"),
        payment_status=job.get("payment_status", "unpaid"),
        media_url=job.get("media_url"),
        kind=job.get("kind", "image"),
        progress=int(job.get("progress") or 0),
        retry_count=int(job.get("retry_count") or 0),
    )


# ---------- User Auth ----------
EMERGENT_AUTH_SESSION_URL = "https://demobackend.emergentagent.com/auth/v1/env/oauth/session-data"


class EmailRegisterBody(BaseModel):
    email: str
    password: str
    name: str


class EmailLoginBody(BaseModel):
    email: str
    password: str


RESEND_API_KEY = os.environ.get("RESEND_API_KEY")

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

_login_attempts: dict = defaultdict(list)
_RATE_LIMIT_MAX = 10
_RATE_LIMIT_WINDOW = 300


def _check_rate_limit(ip: str):
    now = time.time()
    attempts = [t for t in _login_attempts[ip] if now - t < _RATE_LIMIT_WINDOW]
    _login_attempts[ip] = attempts
    if len(attempts) >= _RATE_LIMIT_MAX:
        raise HTTPException(status_code=429, detail="Cok fazla deneme. 5 dakika bekleyin.")
    _login_attempts[ip].append(now)


class ForgotPasswordBody(BaseModel):
    email: str


class ResetPasswordBody(BaseModel):
    token: str
    password: str


@api_router.post("/auth/forgot-password")
async def forgot_password(body: ForgotPasswordBody):
    import secrets
    email = body.email.lower().strip()
    user = await db.users.find_one({"email": email}, {"_id": 0})
    if not user:
        return {"message": "Eger bu email kayitliysa sifirlama linki gonderildi."}
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
            async with httpx.AsyncClient(timeout=10) as hc:
                resp = await hc.post(
                    "https://api.resend.com/emails",
                    headers={"Authorization": f"Bearer {RESEND_API_KEY}", "Content-Type": "application/json"},
                    json={
                        "from": "HatirAI <noreply@hatirai.com>",
                        "to": [email],
                        "subject": "Sifre Sifirlama - HatirAI",
                        "html": f"""
                        <div style="background:#080808;color:#E8E0D0;font-family:Georgia,serif;padding:48px;max-width:480px;margin:0 auto;">
                          <h1 style="color:#C9A961;font-size:32px;letter-spacing:4px;margin-bottom:8px;">HATIR<span style="color:#E8E0D0;">AI</span></h1>
                          <hr style="border-color:#1E1C18;margin:24px 0;">
                          <p style="font-size:16px;line-height:1.8;color:#9C9A93;">Sifrenizi sifirlamak icin asagidaki butona tiklayin. Link 1 saat gecerlidir.</p>
                          <a href="{reset_url}" style="display:inline-block;background:#C9A961;color:#080808;font-family:monospace;font-size:12px;letter-spacing:3px;padding:16px 32px;text-decoration:none;margin:32px 0;">SIFREYI SIFIRLA</a>
                          <p style="font-size:11px;color:#4A4540;">Bu istegi siz yapmadıysaniz bu emaili gormezden gelin.</p>
                        </div>
                        """
                    }
                )
                logger.info(f"[resend] status={resp.status_code}")
        except Exception as e:
            logger.error(f"[resend] Email gonderilemedi: {e}")
    return {"message": "Eger bu email kayitliysa sifirlama linki gonderildi."}


@api_router.post("/auth/reset-password")
async def reset_password(body: ResetPasswordBody):
    record = await db.password_resets.find_one({"token": body.token, "used": False})
    if not record:
        raise HTTPException(status_code=400, detail="Gecersiz veya suresi dolmus link.")
    if record["expires_at"].replace(tzinfo=timezone.utc) < datetime.now(timezone.utc):
        raise HTTPException(status_code=400, detail="Link suresi dolmus.")
    pw_hash = pwd_context.hash(body.password)
    await db.users.update_one({"email": record["email"]}, {"$set": {"pw_hash": pw_hash}})
    await db.password_resets.update_one({"token": body.token}, {"$set": {"used": True}})
    return {"message": "Sifreniz basariyla guncellendi."}


@api_router.post("/auth/register")
async def auth_register(body: EmailRegisterBody, request: Request):
    _check_rate_limit(request.client.host)
    email = body.email.lower().strip()
    if not email or not body.password:
        raise HTTPException(status_code=400, detail="Email ve sifre gerekli")
    if len(body.password) < 6:
        raise HTTPException(status_code=400, detail="Sifre en az 6 karakter olmali")
    existing = await db.users.find_one({"email": email}, {"_id": 0})
    if existing:
        raise HTTPException(status_code=400, detail="Bu email zaten kayitli")
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
    if RESEND_API_KEY:
        try:
            import httpx
            async with httpx.AsyncClient(timeout=10) as hc:
                await hc.post(
                    "https://api.resend.com/emails",
                    headers={"Authorization": f"Bearer {RESEND_API_KEY}", "Content-Type": "application/json"},
                    json={
                        "from": "HatirAI <noreply@hatirai.com>",
                        "to": [email],
                        "subject": "HatirAI'ya Hos Geldiniz",
                        "html": f"""
                        <div style="background:#080808;color:#E8E0D0;font-family:Georgia,serif;padding:48px;max-width:480px;margin:0 auto;">
                          <h1 style="color:#C9A961;font-size:32px;letter-spacing:4px;margin-bottom:8px;">HATIR<span style="color:#E8E0D0;">AI</span></h1>
                          <hr style="border-color:#1E1C18;margin:24px 0;">
                          <p style="font-size:16px;line-height:1.8;color:#9C9A93;">Merhaba {body.name.strip()},</p>
                          <p style="font-size:16px;line-height:1.8;color:#9C9A93;">HatirAI'ya hos geldiniz. Artik yakinlarinizin anilari ni sinematik videolara donusturebilirsiniz.</p>
                          <a href="https://hatirai.com" style="display:inline-block;background:#C9A961;color:#080808;font-family:monospace;font-size:12px;letter-spacing:3px;padding:16px 32px;text-decoration:none;margin:32px 0;">UYGULAMAYA GIT</a>
                        </div>
                        """
                    }
                )
        except Exception as e:
            logger.warning(f"[register] Hosgeldin maili gonderilemedi: {e}")
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
        raise HTTPException(status_code=401, detail="Email veya sifre hatali")
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


@api_router.post("/auth/session")
async def auth_session(body: AuthSessionBody):
    try:
        r = _requests.get(
            EMERGENT_AUTH_SESSION_URL,
            headers={"X-Session-ID": body.session_id},
            timeout=10,
        )
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Oturum dogrulanamadi: {e}")

    email = (data.get("email") or "").lower().strip()
    name = data.get("name") or "Anonim"
    picture = data.get("picture")
    session_token = data.get("session_token") or str(uuid.uuid4())
    if not email:
        raise HTTPException(status_code=401, detail="Email alinamadi")

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


ADMIN_EMAIL = os.environ.get("ADMIN_EMAIL", "administrator@hatirai.com")


def _is_admin_user(user: dict) -> bool:
    """Kullanicinin administrator olup olmadigini kontrol et."""
    return (
        user.get("is_admin") is True
        or user.get("email", "").lower().strip() == ADMIN_EMAIL.lower().strip()
    )


@api_router.get("/auth/me")
async def auth_me(user: dict = Depends(require_user)):
    return {
        "user_id": user["user_id"],
        "email": user["email"],
        "name": user["name"],
        "picture": user.get("picture"),
        "is_admin": _is_admin_user(user),  # frontend admin butonunu bununla gosterir
    }


@api_router.post("/payment/admin-free/{job_id}")
async def admin_free_video(job_id: str, user: dict = Depends(require_user)):
    """
    Sadece administrator hesabina acik: odeme olmadan direkt Veo pipeline baslatir.
    Frontend'de is_admin=true olan kullaniciya ozel buton ile cagirilir.
    """
    if not _is_admin_user(user):
        raise HTTPException(status_code=403, detail="Bu islem sadece administrator hesabina aciktir.")

    job = await db.video_jobs.find_one({"id": job_id}, {"_id": 0})
    if not job:
        raise HTTPException(status_code=404, detail="Is bulunamadi")

    photo = await db.photos.find_one({"id": job["photo_id"]}, {"_id": 0})
    if not photo or not photo.get("noir_b64"):
        raise HTTPException(status_code=400, detail="Fotograf hazir degil")

    # Zaten uretiliyorsa tekrar baslatma
    if job.get("status") == "generating":
        return VideoJobPublic(
            id=job_id, status="generating",
            payment_status=job.get("payment_status", "unpaid"),
            media_url=None, kind="video", progress=int(job.get("progress") or 0),
        )

    # Zaten hazirsa direkt dondur
    if job.get("status") == "ready" and job.get("media_url"):
        return VideoJobPublic(
            id=job_id, status="ready",
            payment_status=job.get("payment_status", "unpaid"),
            media_url=job.get("media_url"), kind="video", progress=100,
        )

    # Odeme flagini set et ve pipeline'i baslat
    await db.video_jobs.update_one(
        {"id": job_id},
        {"$set": {
            "payment_status": "paid",
            "iyzico_payment_id": f"ADMIN_FREE_{user['user_id']}",
            "iyzico_paid_price": "0.00",
        }},
    )
    asyncio.create_task(_run_veo_pipeline(job_id))
    logger.info(f"[admin-free] job={job_id} admin={user['email']} → Veo pipeline basladi")

    return VideoJobPublic(
        id=job_id, status="generating",
        payment_status="paid", media_url=None,
        kind="video", progress=0,
    )


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
        thumb_b64 = None
        if noir_b64:
            try:
                from PIL import Image as PilImage
                import io
                img_bytes = _b64.b64decode(noir_b64)
                img = PilImage.open(io.BytesIO(img_bytes))
                img.thumbnail((200, 300))
                buf = io.BytesIO()
                img.save(buf, format="JPEG", quality=60)
                thumb_b64 = _b64.b64encode(buf.getvalue()).decode()
            except Exception:
                thumb_b64 = None
        result.append({
            "job_id": i.get("id"),
            "photo_id": i.get("photo_id"),
            "status": i.get("status"),
            "payment_status": i.get("payment_status"),
            "kind": i.get("kind"),
            "media_url": i.get("media_url"),
            "created_at": i.get("created_at"),
            "noir_b64": thumb_b64,
        })
    return result


@api_router.post("/payment/lemonsqueezy-init")
async def lemonsqueezy_init(request: Request):
    import httpx
    try:
        body = await request.json()
        job_id = body.get("job_id")
        if not job_id:
            raise HTTPException(status_code=400, detail="job_id gerekli")

        job = await db.video_jobs.find_one({"id": job_id}, {"_id": 0})
        if not job:
            raise HTTPException(status_code=404, detail="Job bulunamadi")

        if job.get("payment_status") == "paid":
            raise HTTPException(status_code=400, detail="Zaten odendi")

        api_key = os.environ.get("LEMONSQUEEZY_API_KEY", "")
        variant_id = os.environ.get("LEMONSQUEEZY_VARIANT_ID", "")
        store_id = os.environ.get("LEMONSQUEEZY_STORE_ID", "")

        async with httpx.AsyncClient() as hc:
            resp = await hc.post(
                "https://api.lemonsqueezy.com/v1/checkouts",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/vnd.api+json",
                    "Accept": "application/vnd.api+json",
                },
                json={
                    "data": {
                        "type": "checkouts",
                        "attributes": {
                            "checkout_data": {
                                "custom": {"job_id": job_id}
                            },
                            "product_options": {
                                "redirect_url": f"{PUBLIC_BACKEND_URL}/result?id={job.get('photo_id', '')}&job={job_id}",
                            }
                        },
                        "relationships": {
                            "store": {"data": {"type": "stores", "id": store_id}},
                            "variant": {"data": {"type": "variants", "id": variant_id}}
                        }
                    }
                }
            )
            resp.raise_for_status()
            checkout = resp.json()
            checkout_url = checkout["data"]["attributes"]["url"]
            logger.info(f"[lemonsqueezy-init] checkout_url={checkout_url} job_id={job_id}")
            return {"checkout_url": checkout_url}

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"[lemonsqueezy-init] Hata: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@api_router.post("/payment/lemonsqueezy-webhook")
async def lemonsqueezy_webhook(request: Request):
    import hmac, hashlib
    try:
        body = await request.body()
        secret = os.environ.get("LEMONSQUEEZY_WEBHOOK_SECRET", "")
        signature = request.headers.get("X-Signature", "")
        if secret:
            expected = hmac.new(
                secret.encode("utf-8"),
                body,
                hashlib.sha256
            ).hexdigest()
            if not hmac.compare_digest(expected, signature):
                logger.warning("[lemonsqueezy] Imza dogrulamasi basarisiz")
                raise HTTPException(status_code=401, detail="Invalid signature")

        data = await request.json()
        event = data.get("meta", {}).get("event_name", "")
        if event != "order_created":
            return {"status": "ignored"}

        order = data.get("data", {})
        attrs = order.get("attributes", {})
        status = attrs.get("status", "")

        if status != "paid":
            return {"status": "ignored"}

        meta = data.get("meta", {})
        custom_data = meta.get("custom_data", {})
        job_id = custom_data.get("job_id", "") or attrs.get("notes", "")

        if not job_id:
            logger.error("[lemonsqueezy] job_id bulunamadi")
            return {"status": "error", "detail": "job_id missing"}

        job = await db.video_jobs.find_one({"id": job_id}, {"_id": 0})
        if not job:
            return {"status": "error", "detail": "job not found"}
        if job.get("payment_status") == "paid":
            return {"status": "already_paid"}

        await db.video_jobs.update_one(
            {"id": job_id},
            {"$set": {
                "payment_status": "paid",
                "lemonsqueezy_order_id": order.get("id"),
                "paid_at": datetime.now(timezone.utc),
            }}
        )
        asyncio.create_task(_run_veo_pipeline(job_id))
        logger.info(f"[lemonsqueezy] Odeme islendi, pipeline basladi: {job_id}")
        return {"status": "ok"}

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"[lemonsqueezy] Hata: {e}")
        return {"status": "error", "detail": str(e)}


@api_router.post("/payment/shopier-init")
async def shopier_init(request: Request):
    import hashlib, hmac, time as _time
    try:
        body = await request.json()
        job_id = body.get("job_id")
        if not job_id:
            raise HTTPException(status_code=400, detail="job_id gerekli")

        job = await db.video_jobs.find_one({"id": job_id}, {"_id": 0})
        if not job:
            raise HTTPException(status_code=404, detail="Job bulunamadi")
        if job.get("payment_status") == "paid":
            raise HTTPException(status_code=400, detail="Zaten odendi")

        api_key = os.environ.get("SHOPIER_API_KEY", "")
        api_secret = os.environ.get("SHOPIER_API_SECRET", "")
        random_nr = str(int(_time.time()))
        data = random_nr + job_id + "99.00" + "0"
        signature = hmac.new(
            api_secret.encode('utf-8'),
            data.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        callback_url = f"{PUBLIC_BACKEND_URL}/api/payment/shopier-callback?job_id={job_id}"

        params = {
            "API_key": api_key,
            "website_index": "1",
            "platform_order_id": job_id,
            "product_name": "HatirAI Sinematik Video",
            "product_type": "2",
            "buyer_name": "Musteri",
            "buyer_surname": "",
            "buyer_email": "musteri@hatirai.com",
            "buyer_phone": "5000000000",
            "buyer_address": "Istanbul",
            "buyer_city": "Istanbul",
            "buyer_country": "Turkey",
            "buyer_postcode": "34000",
            "shipping_address": "Istanbul",
            "shipping_city": "Istanbul",
            "shipping_country": "Turkey",
            "shipping_postcode": "34000",
            "total_order_value": "99.00",
            "currency": "0",
            "random_nr": random_nr,
            "signature": signature,
            "callback": callback_url,
        }
        return {"params": params, "action": "https://www.shopier.com/ShowProduct/api_pay4.php"}

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"[shopier-init] Hata: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@api_router.get("/payment/shopier-callback")
async def shopier_callback(job_id: str, request: Request):
    params = dict(request.query_params)
    status = params.get("status", "")
    logger.info(f"[shopier-callback] job_id={job_id} status={status}")

    if status == "success":
        job = await db.video_jobs.find_one({"id": job_id}, {"_id": 0})
        if job and job.get("payment_status") != "paid":
            await db.video_jobs.update_one(
                {"id": job_id},
                {"$set": {"payment_status": "paid", "paid_at": datetime.now(timezone.utc)}}
            )
            asyncio.create_task(_run_veo_pipeline(job_id))
            logger.info(f"[shopier-callback] Video basladi: {job_id}")

    photo_id = ""
    if job_id:
        job = await db.video_jobs.find_one({"id": job_id}, {"_id": 0})
        if job:
            photo_id = job.get("photo_id", "")

    return RedirectResponse(
        url=f"{PUBLIC_BACKEND_URL}/result?id={photo_id}&job={job_id}",
        status_code=302
    )


@api_router.post("/payment/shopier-osb")
async def shopier_osb(request: Request):
    try:
        form = await request.form()
        data = {key: value for key, value in form.items()}
        logger.info(f"[shopier-osb] data={data}")

        payment_status = data.get("STATUS", "")
        order_id = data.get("BILL_ORDER_ID", "")
        platform_order_id = data.get("PLATFORM_ORDER_ID", "")

        if payment_status != "success":
            return {"status": "ignored"}

        job_id = order_id or platform_order_id
        if not job_id:
            return {"status": "error", "detail": "job_id missing"}

        job = await db.video_jobs.find_one({"id": job_id}, {"_id": 0})
        if not job:
            job = await db.video_jobs.find_one({"shopier_order_id": job_id}, {"_id": 0})
        if not job:
            return {"status": "error", "detail": "job not found"}
        if job.get("payment_status") == "paid":
            return {"status": "already_paid"}

        await db.video_jobs.update_one(
            {"id": job["id"]},
            {"$set": {
                "payment_status": "paid",
                "shopier_order_id": order_id,
                "paid_at": datetime.now(timezone.utc),
            }}
        )
        asyncio.create_task(_run_veo_pipeline(job["id"]))
        logger.info(f"[shopier-osb] Odeme islendi: {job['id']}")
        return {"status": "ok"}

    except Exception as e:
        logger.exception(f"[shopier-osb] Hata: {e}")
        return {"status": "error", "detail": str(e)}


@api_router.post("/admin/test-payment/{job_id}")
async def admin_test_payment(job_id: str, _: str = Depends(require_admin)):
    job = await db.video_jobs.find_one({"id": job_id}, {"_id": 0})
    if not job:
        raise HTTPException(status_code=404, detail="Job bulunamadi")
    await db.video_jobs.update_one(
        {"id": job_id},
        {"$set": {"payment_status": "paid", "iyzico_payment_id": "test_payment"}}
    )
    asyncio.create_task(_run_veo_pipeline(job_id))
    return {"message": f"Job {job_id} odendi, video uretimi basladi"}


@api_router.post("/video/retry/{job_id}")
async def retry_video(job_id: str):
    job = await db.video_jobs.find_one({"id": job_id}, {"_id": 0})
    if not job:
        raise HTTPException(status_code=404, detail="Is bulunamadi")
    if job.get("payment_status") != "paid":
        raise HTTPException(status_code=403, detail="Odeme yapilmamis")
    if job.get("status") == "generating":
        return {"message": "Zaten uretiliyor"}
    # Maksimum 1 tekrar deneme hakki
    if job.get("retry_count", 0) >= 1:
        raise HTTPException(status_code=403, detail="Tekrar deneme hakkiniz doldu. Odemeniz iade edilecektir.")
    await db.video_jobs.update_one(
        {"id": job_id},
        {"$set": {"status": "generating", "progress": 0, "error": None, "media_url": None},
         "$inc": {"retry_count": 1}}
    )
    asyncio.create_task(_run_veo_pipeline(job_id))
    logger.info(f"[retry] {job_id} yeniden basladi (retry_count={job.get('retry_count', 0) + 1})")
    return {"message": "Yeniden baslatildi"}


@api_router.get("/admin/jobs", response_model=List[VideoJob])
async def admin_all(_: str = Depends(require_admin)):
    cursor = db.video_jobs.find({}, {"_id": 0}).sort("created_at", -1)
    items = await cursor.to_list(length=500)
    return [VideoJob(**i) for i in items]


# ===================== SEEDANCE ANI VİDEOSU =====================

# Konsept promptları — Türkçe kullanıcıya gösterilir, İngilizce Seedance'a gider
SEEDANCE_CONCEPTS = {

    # ── EĞLENCE ──────────────────────────────────────────────────────────
    "aquapark": {
        "label": "Aquapark",
        "emoji": "💦",
        "prompt": (
            "Cinematic multishot memory video. No dialogue, only music and ambient sound. "
            "Upbeat summer pop music throughout. "
            "[Shot 1 - Wide] {people} arrive at a massive water park entrance, laughing and pointing excitedly at the slides. Bright summer noon light, vibrant colors. "
            "[Shot 2 - Slow motion] They slide down a giant water slide together, arms raised, water exploding around them, huge grins frozen in joy. "
            "[Shot 3 - Close-up] Faces mid-splash, eyes wide with delight, water droplets catching sunlight like diamonds. "
            "[Shot 4 - Wide] They float together on inflatable rings in a lazy river, golden afternoon light, relaxed and happy. "
            "Warm color grade, handheld energy, feel-good summer film aesthetic."
        ),
    },
    "lunapark": {
        "label": "Lunapark",
        "emoji": "🎡",
        "prompt": (
            "Cinematic multishot memory video. No dialogue, only music and ambient sound. "
            "Dreamy indie pop music throughout. "
            "[Shot 1 - Wide] {people} walking hand in hand through a glowing fairground at night, colorful lights everywhere, ferris wheel spinning behind them. "
            "[Shot 2 - Close-up] Sharing cotton candy, laughing, neon reflections in their eyes. "
            "[Shot 3 - Slow motion] On the ferris wheel at the top, city lights below, looking at each other and smiling. "
            "[Shot 4 - Wide] Walking away through the crowd, lights bokeh blurring beautifully behind them. "
            "Warm golden and neon color grade, magical night atmosphere."
        ),
    },
    "dans": {
        "label": "Dans Sahnesi",
        "emoji": "🕺",
        "prompt": (
            "Cinematic multishot music video. No dialogue, only energetic music. "
            "Upbeat cinematic music throughout. "
            "[Shot 1 - Wide] {people} in a stunning dance hall with dramatic lighting, facing each other ready to dance. "
            "[Shot 2 - Dynamic] They begin dancing together, camera spinning around them, colorful stage lights sweeping. "
            "[Shot 3 - Slow motion] A dramatic dip move, one holding the other, confetti falling around them. "
            "[Shot 4 - Close-up] Their faces laughing joyfully mid-dance, pure happiness. "
            "High contrast dramatic lighting, cinematic color grade, euphoric energy."
        ),
    },
    "surf": {
        "label": "Sörf",
        "emoji": "🏄",
        "prompt": (
            "Cinematic multishot action memory video. No dialogue, only music and ocean sounds. "
            "Epic cinematic surf music throughout. "
            "[Shot 1 - Wide] {people} standing on a tropical beach at golden hour, surfboards in hand, looking at massive waves. "
            "[Shot 2 - Slow motion] Paddling into a huge turquoise wave, water spraying, sun behind them creating a halo. "
            "[Shot 3 - Dynamic] Riding the wave together, exhilarated expressions, ocean stretching to the horizon. "
            "[Shot 4 - Close-up] High-five mid-wave, water flying everywhere in slow motion. "
            "Epic wide cinematography, warm tropical color grade, adrenaline and joy."
        ),
    },
    "kar": {
        "label": "Kar Topu Savaşı",
        "emoji": "❄️",
        "prompt": (
            "Cinematic multishot winter memory video. No dialogue, only playful music and snow sounds. "
            "Warm cheerful acoustic music throughout. "
            "[Shot 1 - Wide] {people} in a snow-covered forest clearing, breath visible in cold air, building up snowballs with mischievous grins. "
            "[Shot 2 - Slow motion] Snowball thrown and hitting with a perfect powder explosion, both erupting in laughter. "
            "[Shot 3 - Close-up] One catching snowflakes on their tongue, the other watching warmly. "
            "[Shot 4 - Wide] Both collapsing into the snow making snow angels, laughing up at the grey winter sky. "
            "Soft cold light, cozy winter color grade, pure joy and warmth."
        ),
    },
    "kamp": {
        "label": "Kamp Ateşi",
        "emoji": "🏕️",
        "prompt": (
            "Cinematic multishot night memory video. No dialogue, only soft music and nature sounds. "
            "Warm acoustic guitar music throughout. "
            "[Shot 1 - Wide] {people} arriving at a forest campsite at dusk, setting up a tent, golden light through the trees. "
            "[Shot 2 - Close-up] Both roasting marshmallows over a glowing campfire, faces lit warmly by the flames, smiling at each other. "
            "[Shot 3 - Wide] Lying on a blanket staring up at a stunning star-filled sky, the milky way visible above. "
            "[Shot 4 - Close-up] One resting their head on the other's shoulder, firelight flickering, peaceful and content. "
            "Deep warm tones, cinematic night photography, intimate and nostalgic."
        ),
    },
    "western": {
        "label": "Western",
        "emoji": "🤠",
        "prompt": (
            "Cinematic multishot Western film. No dialogue, only dramatic Western score. "
            "Epic Ennio Morricone-style music throughout. "
            "[Shot 1 - Wide] {people} riding horses across a vast desert landscape at golden hour, dust rising behind them, dramatic sky. "
            "[Shot 2 - Close-up] Both with cowboy hats, exchanging a meaningful glance, wind blowing. "
            "[Shot 3 - Slow motion] Galloping side by side into the sunset, silhouettes against a burning orange sky. "
            "[Shot 4 - Wide] Stopping on a cliff edge overlooking a canyon, looking out at the epic landscape together. "
            "Classic Western color grade, dust and golden light, epic and cinematic."
        ),
    },

    # ── ROMANTİK ─────────────────────────────────────────────────────────
    "kir": {
        "label": "Kırda Buluşma",
        "emoji": "🌸",
        "prompt": (
            "Cinematic multishot romantic memory video. No dialogue, only soft romantic music. "
            "Beautiful emotional strings music throughout. "
            "[Shot 1 - Wide] {people} running toward each other through a vast field of wildflowers at golden hour, warm backlight creating a halo around them. "
            "[Shot 2 - Slow motion] They meet and embrace, wildflowers swaying around them, petals carried by the wind. "
            "[Shot 3 - Close-up] Looking into each other's eyes tenderly, one holding a single red rose, soft smile. "
            "[Shot 4 - Wide] Walking together through the flower field hand in hand as the sun sets, silhouettes glowing. "
            "Warm golden hour cinematography, soft bokeh, deeply romantic and emotional."
        ),
    },
    "paris": {
        "label": "Paris Sokakları",
        "emoji": "🗼",
        "prompt": (
            "Cinematic multishot romantic Paris film. No dialogue, only romantic French music. "
            "Soft accordion and strings music throughout. "
            "[Shot 1 - Wide] {people} walking through a beautiful Parisian street at dusk, warm cafe lights glowing, Eiffel Tower visible in the distance. "
            "[Shot 2 - Close-up] Sharing a coffee at a sidewalk cafe, looking at each other over the rim of their cups, smiling. "
            "[Shot 3 - Slow motion] Walking in the rain under a shared umbrella, Paris lights reflecting on wet cobblestones, leaning close. "
            "[Shot 4 - Wide] Standing in front of the illuminated Eiffel Tower at night, looking up together in wonder. "
            "Vintage cinematic color grade, romantic Paris atmosphere, timeless and elegant."
        ),
    },
    "bogaz": {
        "label": "Boğaz'da Gün Batımı",
        "emoji": "🌉",
        "prompt": (
            "Cinematic multishot romantic Istanbul film. No dialogue, only beautiful ambient music. "
            "Emotional cinematic music throughout. "
            "[Shot 1 - Wide] {people} on the deck of a wooden boat on the Bosphorus, Istanbul skyline behind them, mosques and minarets glowing in orange light. "
            "[Shot 2 - Close-up] Both leaning on the railing, wind in their hair, gazing at the stunning sunset over the water. "
            "[Shot 3 - Slow motion] One handing the other a single red rose, their fingers touching, warm smiles. "
            "[Shot 4 - Wide] The boat sailing into the golden sunset, their silhouettes close together, city lights beginning to flicker on. "
            "Rich warm tones, cinematic Istanbul atmosphere, deeply romantic and iconic."
        ),
    },
    "mum": {
        "label": "Mum Işığında Akşam Yemeği",
        "emoji": "🕯️",
        "prompt": (
            "Cinematic multishot intimate romantic dinner film. No dialogue, only soft jazz music. "
            "Warm intimate jazz music throughout. "
            "[Shot 1 - Wide] {people} at an elegantly set table in a rustic stone room, dozens of candles glowing around them, rose petals on the table. "
            "[Shot 2 - Close-up] Clinking wine glasses gently, eyes locked, candlelight dancing in their eyes. "
            "[Shot 3 - Close-up] One reaching across and holding the other's hand on the table, a tender knowing smile. "
            "[Shot 4 - Slow motion] Rose petals falling softly onto the table, both laughing as one catches a petal. "
            "Deep warm candlelight tones, intimate and luxurious, deeply romantic."
        ),
    },
    "kapadokya": {
        "label": "Kapadokya Balonları",
        "emoji": "🎈",
        "prompt": (
            "Cinematic multishot magical Cappadocia film. No dialogue, only dreamy music. "
            "Ethereal cinematic music throughout. "
            "[Shot 1 - Wide] {people} standing on a cliff at dawn in Cappadocia, hundreds of colorful hot air balloons rising into a pink and orange sky behind them. "
            "[Shot 2 - Close-up] Inside a balloon basket, both leaning over the edge in wonder, fairy chimneys stretching below them. "
            "[Shot 3 - Slow motion] A single balloon drifting close by, the landscape stretching magnificently, their hair blowing in the gentle breeze. "
            "[Shot 4 - Wide] Watching the balloons drift away as the sun fully rises, arms around each other, breathtaking landscape all around. "
            "Magical golden dawn light, dreamy color grade, wonder and romance."
        ),
    },

    # ── FANTASTİK ────────────────────────────────────────────────────────
    "uzay": {
        "label": "Uzayda Yürüyüş",
        "emoji": "🌌",
        "prompt": (
            "Cinematic multishot epic space adventure film. No dialogue, only epic orchestral music. "
            "Hans Zimmer-style cinematic music throughout. "
            "[Shot 1 - Wide] {people} in sleek astronaut suits floating in the void of space, Earth glowing blue below them, stars infinite around them. "
            "[Shot 2 - Close-up] Visors touching in space, their faces visible through the helmets, smiling at each other against a backdrop of galaxies. "
            "[Shot 3 - Wide] Floating hand in hand past a spectacular ringed planet, aurora-like nebula colors surrounding them. "
            "[Shot 4 - Slow motion] Both reaching out to touch a passing comet's tail, light streaming past them in slow motion. "
            "Breathtaking space cinematography, deep blacks and vibrant cosmos colors, epic and awe-inspiring."
        ),
    },
    "ejderha": {
        "label": "Ejderha Üzerinde",
        "emoji": "🐉",
        "prompt": (
            "Cinematic multishot epic fantasy film. No dialogue, only epic fantasy music. "
            "Sweeping orchestral fantasy music throughout. "
            "[Shot 1 - Wide] {people} standing on a mountain peak at sunset, a magnificent dragon landing behind them, wings spreading dramatically. "
            "[Shot 2 - Dynamic] Both riding on the dragon's back, soaring above clouds, landscape far below, expressions of pure exhilaration. "
            "[Shot 3 - Slow motion] The dragon swooping through golden clouds, their hair and cloaks streaming, sun breaking through dramatically. "
            "[Shot 4 - Wide] Landing on a cliff at golden hour, dismounting the dragon, looking out at an epic fantasy landscape together. "
            "Rich fantasy cinematography, epic scale, golden and dramatic color grade."
        ),
    },
    "okyanus_alti": {
        "label": "Okyanus Altı",
        "emoji": "🌊",
        "prompt": (
            "Cinematic multishot magical underwater film. No dialogue, only magical ambient music. "
            "Ethereal underwater ambient music throughout. "
            "[Shot 1 - Wide] {people} in elegant diving suits sinking slowly into a crystal-clear tropical ocean, rays of sunlight piercing the water above them. "
            "[Shot 2 - Wide] Swimming side by side through a vast colorful coral reef, schools of tropical fish parting around them like a living curtain. "
            "[Shot 3 - Close-up] Both stopping to watch a majestic sea turtle gliding past, their faces lit by the blue underwater glow, eyes wide with wonder. "
            "[Shot 4 - Wide] Swimming upward toward the shimmering surface, sunlight above them, silhouettes rising together through the blue. "
            "Magical teal and blue tones, magical realism, wonder and serenity."
        ),
    },
    "buyulu_orman": {
        "label": "Büyülü Orman",
        "emoji": "🧚",
        "prompt": (
            "Cinematic multishot magical forest fantasy film. No dialogue, only enchanted music. "
            "Magical whimsical music with soft bells throughout. "
            "[Shot 1 - Wide] {people} entering an ancient glowing forest at dusk, fireflies beginning to light up all around them, bioluminescent mushrooms at their feet. "
            "[Shot 2 - Close-up] Fireflies landing on their outstretched hands, their faces glowing with soft golden light, expressions of pure wonder. "
            "[Shot 3 - Wide] Dancing slowly among the fireflies, the whole forest glowing around them, magical light particles floating everywhere. "
            "[Shot 4 - Slow motion] They look up as thousands of glowing fireflies rise around them into the dark sky like living stars. "
            "Magical bioluminescent palette, dark enchanted forest, pure wonder and magic."
        ),
    },
    "lale": {
        "label": "Lale Bahçesi",
        "emoji": "🌷",
        "prompt": (
            "Cinematic multishot romantic spring film. No dialogue, only beautiful spring music. "
            "Light and joyful classical music throughout. "
            "[Shot 1 - Wide] {people} entering a stunning tulip garden in full bloom, thousands of red and pink tulips stretching to the horizon, Istanbul visible in the distance. "
            "[Shot 2 - Slow motion] Walking slowly between tall rows of tulips, petals gently swaying, warm spring sunlight filtering through. "
            "[Shot 3 - Close-up] One handing the other a single perfect red tulip, their fingers touching, warm smiles exchanged. "
            "[Shot 4 - Wide] Both sitting among the tulips as petals blow in the breeze around them, a perfect spring moment. "
            "Rich vibrant spring colors, warm golden light, deeply romantic and joyful."
        ),
    },
    "sahil": {
        "label": "Sahilde Gün Batımı",
        "emoji": "🌅",
        "prompt": (
            "Cinematic multishot emotional beach film. No dialogue, only beautiful ambient music. "
            "Emotional piano and strings music throughout. "
            "[Shot 1 - Wide] {people} walking barefoot on a deserted beach at golden hour, waves gently washing over their feet, pink and gold sky stretching endlessly. "
            "[Shot 2 - Slow motion] Spinning together in the shallow water, laughter silent but visible, water splashing around them, backlit by the setting sun. "
            "[Shot 3 - Close-up] Sitting together on the sand, watching the sun touch the horizon, one resting their head on the other's shoulder. "
            "[Shot 4 - Wide] The sun disappears below the horizon, first stars appearing, both silhouetted against the glowing sky. "
            "Warm golden hour cinematography, emotional and serene, deeply beautiful."
        ),
    },
    "japon_kiraz": {
        "label": "Japon Kiraz Bahçesi",
        "emoji": "🌸",
        "prompt": (
            "Cinematic multishot romantic Japanese spring film. No dialogue, only delicate music. "
            "Soft koto and strings music throughout. "
            "[Shot 1 - Wide] {people} walking along a stunning avenue of cherry blossom trees in full bloom, petals raining down gently around them like pink snow. "
            "[Shot 2 - Slow motion] Cherry blossom petals falling in slow motion around their faces, both looking up in wonder, pink light everywhere. "
            "[Shot 3 - Close-up] Sitting beneath a massive cherry blossom tree, petals covering their shoulders, looking at each other tenderly. "
            "[Shot 4 - Wide] Walking away down the petal-covered path into a soft pink haze of blossoms, a perfect farewell frame. "
            "Soft pink and white tones, delicate and romantic, timelessly beautiful."
        ),
    },
    "park": {
        "label": "Parkta Gün",
        "emoji": "🌳",
        "prompt": (
            "Cinematic multishot warm memory film. No dialogue, only cheerful music. "
            "Light acoustic and strings music throughout. "
            "[Shot 1 - Wide] {people} arriving at a beautiful park on a perfect sunny day, trees in full leaf, dappled sunlight on the path ahead. "
            "[Shot 2 - Close-up] Sharing a picnic on a blanket under a tree, laughing over food, golden afternoon light filtering through the leaves. "
            "[Shot 3 - Wide] Running across an open green field together, one chasing the other playfully, pure joy. "
            "[Shot 4 - Slow motion] Lying on the grass side by side, looking up at the sky through the tree canopy, completely at peace. "
            "Warm natural tones, nostalgic home-video feel, deeply heartwarming."
        ),
    },
}



class SeedanceJobModel(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    concept: str
    status: str = "awaiting_payment"
    payment_status: str = "unpaid"
    video_url: Optional[str] = None
    error: Optional[str] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


async def _add_seedance_watermark(job_id: str, video_url: str) -> Optional[str]:
    """
    9:16 videoya HATIR ◆ AI watermark ekle — ffmpeg, kredi harcamaz.
    Ust orta: HATIR (beyaz) ◆ (gold) AI (gold)
    Alt orta: hatirai.com (soluk)
    Basarisiz olursa None doner, orijinal video kullanilir.
    """
    import tempfile as _tf
    workdir = _tf.mkdtemp(prefix=f"wm-{job_id}-")
    try:
        video_local = os.path.join(workdir, "input.mp4")
        await asyncio.to_thread(_http_download, video_url, video_local)
        output_local = os.path.join(workdir, "watermarked.mp4")

        font_candidates = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSerif-Bold.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        ]
        font_path = next((f for f in font_candidates if os.path.exists(f)), None)
        if not font_path:
            logger.warning(f"[watermark] Font bulunamadi, atlanıyor")
            return None

        # Üst ortaya: HATIR beyaz + boşluk + AI gold
        # ◆ sembolü fontlarda olmayabilir, basit | veya · kullanalım
        # İki ayrı drawtext katmanı: önce HATIR sonra AI, ortaya hizala
        # Basit yaklaşım: tek metin "HATIR · AI" beyaz + gold efekti yok ama güvenilir

        top_y = "h/20"
        bottom_y = "h-h/14"
        font_size_main = "h/14"
        font_size_small = "h/48"

        # 3 katman: HATIR (beyaz) | · (gold) | AI (gold) + alt site
        # x koordinatları: toplam genişliği hesaplayamayız kolayca,
        # bu yüzden "HATIR  AI" tek metin beyaz yazıp üstüne AI kısmını gold yazıyoruz
        # Güvenilir yöntem: her şeyi ayrı drawtext, x offset ile hizala

        # HATIRAI birlesik yaz — HATIR beyaz, AI gold
        # Yontem: once HATIRAI tamamen beyaz yaz, sonra AI kismini gold ile ustune yaz
        # AI metni, HATIRAI'nin son 2 karakteri — x offseti: (w+tw_full)/2 - tw_ai
        # tw ffmpeg'de o anki metnin genisligini verir
        # HATIR kismi icin x: (w-tw_full)/2 — tam ortada baslangic
        # AI kismi x: (w-tw_full)/2 + tw_hatir — HATIR'dan sonra baslar

        vf = ",".join([
            # Gölge — okunabilirlik
            f"drawtext=fontfile='{font_path}':text='HATIRAI':"
            f"fontcolor=black@0.6:fontsize={font_size_main}:"
            f"x=(w-tw)/2+2:y={top_y}+2",

            # HATIRAI tamamen beyaz (altta kalacak)
            f"drawtext=fontfile='{font_path}':text='HATIRAI':"
            f"fontcolor=0xF4F1EA@0.93:fontsize={font_size_main}:"
            f"x=(w-tw)/2:y={top_y}",

            # AI — gold, HATIR'ın hemen sağına konumla
            # x = (w/2) + (tw_HATIRAI/2) - (tw_AI) — sağdan iki karakter
            # ffmpeg'de dinamik tw kullanamayız, sabit ratio kullanıyoruz:
            # HATIRAI = 7 karakter, AI = 2 karakter → AI başlangıcı = %71 oranında
            f"drawtext=fontfile='{font_path}':text='AI':"
            f"fontcolor=0xC9A961@1.0:fontsize={font_size_main}:"
            f"x=(w-tw)/2+(tw)*5/7:y={top_y}",

            # Alt — hatirai.com soluk
            f"drawtext=fontfile='{font_path}':text='hatirai.com':"
            f"fontcolor=0xF4F1EA@0.28:fontsize={font_size_small}:"
            f"x=(w-tw)/2:y={bottom_y}",
        ])

        cmd = [
            "ffmpeg", "-y", "-i", video_local,
            "-vf", vf,
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
            "-c:a", "copy",
            output_local,
        ]
        result = await asyncio.to_thread(
            lambda: subprocess.run(cmd, capture_output=True, timeout=120)
        )

        if result.returncode != 0:
            logger.warning(f"[watermark] ffmpeg hatasi: {result.stderr.decode()[:300]}")
            return None

        # Watermarklı videoyu fal.ai'ye yükle
        wm_url = await fal_client.upload_file_async(output_local)
        logger.info(f"[watermark {job_id}] Watermark eklendi: {wm_url}")
        return wm_url

    except Exception as e:
        logger.warning(f"[watermark {job_id}] Hata, orijinal kullanilacak: {e}")
        return None
    finally:
        try:
            shutil.rmtree(workdir, ignore_errors=True)
        except Exception:
            pass


async def _run_seedance(job_id: str, photos: list, concept: str):
    """
    Seedance 2.0 reference-to-video ile anı videosu üret.
    photos: [{"b64": "...", "role": "Me", "relation": "father"}]
    """
    import tempfile as _tf
    try:
        concept_data = SEEDANCE_CONCEPTS.get(concept, SEEDANCE_CONCEPTS["sahil"])
        base_prompt = concept_data["prompt"]

        # Kişi tanımlaması — {people} placeholder'ını doldur
        people_parts = []
        for i, p in enumerate(photos):
            rel = p.get("relation", "person")
            people_parts.append(f"@Image{i+1} ({rel})")
        people_str = " and ".join(people_parts)
        prompt = base_prompt.replace("{people}", people_str)

        # Her fotoğrafı fal.ai'ye yükle
        image_urls = []
        for i, p in enumerate(photos):
            tmp = _tf.NamedTemporaryFile(suffix=".jpg", delete=False)
            tmp.write(_b64.b64decode(p["b64"]))
            tmp.close()
            url = await fal_client.upload_file_async(tmp.name)
            os.unlink(tmp.name)
            image_urls.append(url)
            logger.info(f"[seedance {job_id}] @Image{i+1} yuklendi: {url}")

        # Kimlik koruma talimatını başa ekle
        full_prompt = (
            f"IMPORTANT: The people in this video are {people_str}. "
            f"Preserve the exact face, appearance and identity of each person "
            f"from their reference image throughout ALL shots. "
            f"{prompt}"
        )
        logger.info(f"[seedance {job_id}] Prompt ({len(full_prompt)} chars): {full_prompt[:300]}...")

        handle = await fal_client.submit_async(
            "bytedance/seedance-2.0/fast/reference-to-video",
            arguments={
                "prompt": full_prompt,
                "image_urls": image_urls,
                "resolution": "480p",
                "duration": "15",
                "aspect_ratio": "9:16",
                "generate_audio": True,
            },
        )
        result = await handle.get()
        video_url = result["video"]["url"]
        logger.info(f"[seedance {job_id}] Ham video hazir: {video_url}")

        # Watermark ekle
        watermarked_url = await _add_seedance_watermark(job_id, video_url)
        final_video_url = watermarked_url or video_url

        await db.seedance_jobs.update_one(
            {"id": job_id},
            {"$set": {"status": "ready", "video_url": final_video_url}},
        )
    except Exception as e:
        logger.exception(f"[seedance {job_id}] Hata")
        await db.seedance_jobs.update_one(
            {"id": job_id},
            {"$set": {"status": "failed", "error": str(e)[:500]}},
        )


@api_router.get("/seedance/concepts")
async def seedance_concepts():
    return [
        {"id": k, "label": v["label"], "emoji": v["emoji"]}
        for k, v in SEEDANCE_CONCEPTS.items()
    ]


@api_router.post("/seedance/create")
async def seedance_create(request: Request):
    """Çoklu fotoğraf + ilişki + konsept ile Seedance job oluştur — ödeme bekleniyor."""
    if not FAL_KEY:
        raise HTTPException(status_code=500, detail="FAL_KEY eksik")
    try:
        import base64 as _base64
        form = await request.form()
        concept = form.get("concept", "park")

        if concept not in SEEDANCE_CONCEPTS:
            raise HTTPException(status_code=400, detail=f"Gecersiz konsept: {concept}")

        photos = []
        for i in range(4):
            photo_file = form.get(f"photo_{i}")
            if not photo_file:
                continue
            photo_bytes = await photo_file.read()
            if not photo_bytes:
                continue
            b64 = _base64.b64encode(photo_bytes).decode()
            role = form.get(f"role_{i}", f"Kisi {i+1}")
            relation = form.get(f"relation_{i}", "")
            photos.append({"b64": b64, "role": role, "relation": relation})

        if not photos:
            raise HTTPException(status_code=400, detail="En az 1 fotograf gerekli")

        job = SeedanceJobModel(concept=concept, status="awaiting_payment", payment_status="unpaid")
        job_doc = job.model_dump()
        job_doc["photos"] = photos  # fotoğrafları DB'de sakla
        await db.seedance_jobs.insert_one(job_doc)
        logger.info(f"[seedance] job={job.id} concept={concept} {len(photos)} fotograf — odeme bekleniyor")

        return {"job_id": job.id, "status": "awaiting_payment"}
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("[seedance/create] Hata")
        raise HTTPException(status_code=500, detail=str(e))


@api_router.post("/seedance/payment/init")
async def seedance_payment_init(request: Request):
    """Seedance job için Lemonsqueezy ödeme başlat."""
    import httpx
    try:
        body = await request.json()
        job_id = body.get("job_id")
        if not job_id:
            raise HTTPException(status_code=400, detail="job_id gerekli")

        job = await db.seedance_jobs.find_one({"id": job_id}, {"_id": 0})
        if not job:
            raise HTTPException(status_code=404, detail="Job bulunamadi")
        if job.get("payment_status") == "paid":
            raise HTTPException(status_code=400, detail="Zaten odendi")

        api_key = os.environ.get("LEMONSQUEEZY_API_KEY", "")
        variant_id = "1643185"
        store_id = "1047868"

        async with httpx.AsyncClient() as hc:
            resp = await hc.post(
                "https://api.lemonsqueezy.com/v1/checkouts",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/vnd.api+json",
                    "Accept": "application/vnd.api+json",
                },
                json={
                    "data": {
                        "type": "checkouts",
                        "attributes": {
                            "checkout_data": {
                                "custom": {"seedance_job_id": job_id}
                            },
                            "product_options": {
                                "redirect_url": f"{PUBLIC_BACKEND_URL}/api/ani?job={job_id}",
                            }
                        },
                        "relationships": {
                            "store": {"data": {"type": "stores", "id": store_id}},
                            "variant": {"data": {"type": "variants", "id": variant_id}}
                        }
                    }
                }
            )
            resp.raise_for_status()
            checkout = resp.json()
            checkout_url = checkout["data"]["attributes"]["url"]
            logger.info(f"[seedance-payment] checkout_url={checkout_url} job_id={job_id}")
            return {"checkout_url": checkout_url}

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"[seedance-payment] Hata: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@api_router.post("/seedance/payment/webhook")
async def seedance_payment_webhook(request: Request):
    """Lemonsqueezy webhook — ödeme onaylanınca Seedance pipeline başlat."""
    import hmac, hashlib
    try:
        body = await request.body()
        secret = os.environ.get("LEMONSQUEEZY_WEBHOOK_SECRET", "")
        signature = request.headers.get("X-Signature", "")
        if secret:
            expected = hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
            if not hmac.compare_digest(expected, signature):
                raise HTTPException(status_code=401, detail="Invalid signature")

        data = await request.json()
        event = data.get("meta", {}).get("event_name", "")
        if event != "order_created":
            return {"status": "ignored"}

        attrs = data.get("data", {}).get("attributes", {})
        if attrs.get("status") != "paid":
            return {"status": "ignored"}

        custom = data.get("meta", {}).get("custom_data", {})
        job_id = custom.get("seedance_job_id", "")
        if not job_id:
            return {"status": "error", "detail": "seedance_job_id missing"}

        job = await db.seedance_jobs.find_one({"id": job_id}, {"_id": 0})
        if not job:
            return {"status": "error", "detail": "job not found"}
        if job.get("payment_status") == "paid":
            return {"status": "already_paid"}

        await db.seedance_jobs.update_one(
            {"id": job_id},
            {"$set": {"payment_status": "paid", "status": "processing", "paid_at": datetime.now(timezone.utc)}}
        )
        photos = job.get("photos", [])
        asyncio.create_task(_run_seedance(job_id, photos, job["concept"]))
        logger.info(f"[seedance-webhook] Odeme onaylandi, pipeline basladi: {job_id}")
        return {"status": "ok"}

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"[seedance-webhook] Hata: {e}")
        return {"status": "error", "detail": str(e)}


@api_router.post("/seedance/admin-free/{job_id}")
async def seedance_admin_free(job_id: str, user: dict = Depends(require_user)):
    """Sadece administrator: ödeme olmadan Seedance pipeline başlat."""
    if not _is_admin_user(user):
        raise HTTPException(status_code=403, detail="Sadece administrator erisebilir")

    job = await db.seedance_jobs.find_one({"id": job_id}, {"_id": 0})
    if not job:
        raise HTTPException(status_code=404, detail="Job bulunamadi")
    if job.get("status") == "processing":
        return {"status": "processing", "message": "Zaten uretiliyor"}
    if job.get("status") == "ready":
        return {"status": "ready", "video_url": job.get("video_url")}

    await db.seedance_jobs.update_one(
        {"id": job_id},
        {"$set": {"payment_status": "paid", "status": "processing", "paid_at": datetime.now(timezone.utc)}}
    )
    photos = job.get("photos", [])
    asyncio.create_task(_run_seedance(job_id, photos, job["concept"]))
    logger.info(f"[seedance-admin-free] job={job_id} admin={user['email']} pipeline basladi")
    return {"status": "processing", "message": "Pipeline basladi"}


@api_router.get("/seedance/job/{job_id}")
async def seedance_job_status(job_id: str):
    """Seedance iş durumunu döndür."""
    doc = await db.seedance_jobs.find_one({"id": job_id}, {"_id": 0})
    if not doc:
        raise HTTPException(status_code=404, detail="Is bulunamadi")
    return {
        "id": doc["id"],
        "concept": doc.get("concept"),
        "status": doc.get("status", "awaiting_payment"),
        "payment_status": doc.get("payment_status", "unpaid"),
        "video_url": doc.get("video_url"),
        "error": doc.get("error"),
    }


@api_router.get("/ani")
async def ani_page():
    """Anı videosu sayfası."""
    ani_path = Path("/app/ani.html")
    if ani_path.exists():
        return FileResponse(ani_path)
    raise HTTPException(status_code=404, detail="Ani sayfasi bulunamadi")


# ===================== CHAT / LIVEAVATAR =====================

@api_router.post("/chat/heygen-token")
async def heygen_token():
    import httpx
    api_key = os.environ.get("LIVEAVATAR_API_KEY", "")
    avatar_id = os.environ.get("LIVEAVATAR_AVATAR_ID", "fc4125a5-83fa-45e2-8574-bf657ac19998")
    if not api_key:
        raise HTTPException(status_code=500, detail="LIVEAVATAR_API_KEY eksik")
    try:
        async with httpx.AsyncClient() as hc:
            resp = await hc.post(
                "https://api.liveavatar.com/v1/sessions/token",
                headers={"X-API-KEY": api_key, "Content-Type": "application/json"},
                json={
                    "mode": "FULL",
                    "avatar_id": avatar_id,
                    "avatar_persona": {
                        "prompt": (
                            "You are a warm, loving family member speaking Turkish. "
                            "Respond with short, emotional, sincere sentences in Turkish. "
                            "You deeply miss the person you are speaking to. "
                            "Keep responses under 2 sentences."
                        )
                    },
                },
                timeout=15,
            )
            logger.info(f"[heygen-token] {resp.status_code} {resp.text[:300]}")
            if resp.status_code == 200:
                data = resp.json()
                token = data.get("data", {}).get("session_token", "") or data.get("token", "")
                if not token:
                    logger.error(f"[heygen-token] Token bos geldi: {data}")
                    raise HTTPException(status_code=500, detail="Token bos geldi")
                return {"token": token}
            raise HTTPException(status_code=resp.status_code, detail=resp.text)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"[heygen-token] Hata: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@api_router.post("/chat/prepare")
async def chat_prepare(request: Request):
    import base64, httpx
    try:
        form = await request.form()
        photo_file = form.get("photo")
        era = form.get("era", "modern")

        if not photo_file:
            raise HTTPException(status_code=400, detail="Fotograf gerekli")

        photo_bytes = await photo_file.read()
        photo_b64 = base64.b64encode(photo_bytes).decode()
        should_restore = form.get("restore", "1") == "1"
        photo_id = f"chat_{uuid.uuid4().hex[:8]}"

        if should_restore:
            prompt = ERA_PROMPTS.get(era, ERA_PROMPTS["modern"])
            image_bytes = base64.b64decode(photo_b64)
            tmp_path = f"/tmp/chat_{photo_id}.jpg"
            with open(tmp_path, "wb") as f:
                f.write(image_bytes)

            image_url = await fal_client.upload_file_async(tmp_path)
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

            restored_b64 = None
            if result.get("images"):
                img_url = result["images"][0].get("url", "")
                if img_url:
                    async with httpx.AsyncClient() as hc:
                        r = await hc.get(img_url, timeout=60)
                        restored_b64 = base64.b64encode(r.content).decode()

            if not restored_b64:
                restored_b64 = photo_b64
        else:
            restored_b64 = photo_b64
            logger.info("[chat/prepare] Restore atlanidi")

        did_key = os.environ.get("DID_API_KEY", "")
        stream_data = None
        if did_key:
            img_bytes = base64.b64decode(restored_b64)
            tmp_did = f"/tmp/did_{photo_id}.jpg"
            with open(tmp_did, "wb") as f:
                f.write(img_bytes)
            photo_url = await fal_client.upload_file_async(tmp_did)

            async with httpx.AsyncClient() as hc:
                resp = await hc.post(
                    "https://api.d-id.com/talks/streams",
                    headers={"Content-Type": "application/json", "Authorization": f"Basic {did_key}"},
                    json={"source_url": photo_url},
                    timeout=30,
                )
                if resp.status_code == 200:
                    stream_data = resp.json()

        return {"restored_b64": restored_b64, "stream_data": stream_data}

    except Exception as e:
        logger.exception(f"[chat/prepare] Hata: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@api_router.post("/chat/did-token")
async def did_get_token():
    api_key = os.environ.get("DID_API_KEY", "")
    if not api_key:
        raise HTTPException(status_code=500, detail="DID_API_KEY eksik")
    return {"token": api_key}


@api_router.post("/chat/respond")
async def chat_respond(request: Request):
    import httpx, base64
    try:
        form = await request.form()
        person_name = form.get("person_name", "Sevgili")
        relation = form.get("relation", "yakin")
        context = form.get("context", "")
        user_text = form.get("user_text", "") or "Seni cok ozledim"
        voice_id = form.get("voice_id", "FYPltOzsM2n1UbqzX19d")

        import anthropic as ant
        ant_client = ant.AsyncAnthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))
        system_prompt = (
            f"Sen {person_name} adli kisisin. Kullanicinin {relation}. "
            f"Hakkinda bilgi: {context} "
            "Kullanici sana konusuyor. Sen o kisi gibi, Turkce, kisa (1-2 cumle), samimi cevap ver. "
            "Asla yapay zeka oldugunu belirtme. Maksimum 2 cumle."
        )

        message = await ant_client.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=150,
            system=system_prompt,
            messages=[{"role": "user", "content": user_text}]
        )
        response_text = message.content[0].text

        audio_b64 = None
        if ELEVENLABS_API_KEY:
            async with httpx.AsyncClient() as hc:
                resp = await hc.post(
                    f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}",
                    headers={"xi-api-key": ELEVENLABS_API_KEY, "Content-Type": "application/json"},
                    json={
                        "text": response_text,
                        "model_id": "eleven_multilingual_v2",
                        "voice_settings": {"stability": 0.5, "similarity_boost": 0.75}
                    },
                    timeout=30,
                )
                if resp.status_code == 200:
                    audio_b64 = base64.b64encode(resp.content).decode()

        return {"user_text": user_text, "response": response_text, "audio_b64": audio_b64}

    except Exception as e:
        logger.exception(f"[chat/respond] Hata: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@api_router.get("/sohbet")
async def sohbet_page():
    sohbet_path = Path("/app/sohbet.html")
    if sohbet_path.exists():
        return FileResponse(sohbet_path)
    raise HTTPException(status_code=404, detail="Sohbet sayfasi bulunamadi")


# ===================== APP MOUNT =====================
app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

DIST_DIR = Path("/app/frontend/dist")

if DIST_DIR.is_dir():
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
        if full_path.startswith("api/") or full_path.startswith("api"):
            raise HTTPException(status_code=404)
        direct = DIST_DIR / full_path
        if direct.is_file():
            return FileResponse(direct)
        if full_path:
            html = DIST_DIR / f"{full_path}.html"
            if html.is_file():
                return FileResponse(html)
        index = DIST_DIR / "index.html"
        if index.is_file():
            return FileResponse(index)
        not_found = DIST_DIR / "+not-found.html"
        if not_found.is_file():
            return FileResponse(not_found, status_code=404)
        raise HTTPException(status_code=404)

    logger.info(f"[static] Expo web bundle: {DIST_DIR}")
else:
    logger.info(f"[static] {DIST_DIR} yok, frontend dis sunucudan servis ediliyor")


@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()
