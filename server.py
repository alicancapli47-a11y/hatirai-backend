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
    """Clipları birlestir ve logo watermark ekle."""
    tmp = tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False)
    try:
        for p in video_paths:
            tmp.write(f"file '{os.path.abspath(p)}'\n")
        tmp.flush()
        tmp.close()

        logo_candidates = [
            "/app/hatirai_logo_watermark.png",
            os.path.join(str(ROOT_DIR), "hatirai_logo_watermark.png"),
        ]
        logo_path = next((p for p in logo_candidates if os.path.exists(p)), None)

        if logo_path:
            vf = (
                f"[1:v]scale=iw*0.35:-1,format=rgba,colorchannelmixer=aa=0.75[logo];"
                f"[0:v][logo]overlay=(W-w)/2:H-h-50"
            )
            cmd = [
                "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", tmp.name,
                "-i", logo_path,
                "-filter_complex", vf,
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "22",
                "-c:a", "aac", "-b:a", "128k",
                out_path,
            ]
        else:
            logger.warning("[watermark] Logo bulunamadi, filigransiz video")
            cmd = [
                "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", tmp.name,
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
            "Fast-cut cinematic action video, no dialogue, upbeat summer music. "
            "[Shot 1 - Low angle wide, 2s] {people} sprint toward a massive water slide entrance, laughing, pointing up at the towering slide. Bright noon sun, saturated colors. "
            "[Shot 2 - POV tracking, 2s] Rushing down a steep twisting water slide at full speed, water spraying violently left and right, screaming with excitement. "
            "[Shot 3 - Slow motion 0.5s] The exact moment of splashdown — a massive wall of white water exploding outward, faces frozen mid-scream in joy. "
            "[Shot 4 - Wide action, 2s] Both racing across a shallow splash pad, soaking each other, sprinting and laughing hysterically. "
            "[Shot 5 - Close-up, 1s] High five underwater in a pool, bubbles everywhere, huge grins. "
            "Handheld camera energy, quick cuts every 1-2 seconds, vibrant summer color grade."
        ),
    },
    "lunapark": {
        "label": "Lunapark",
        "emoji": "🎡",
        "prompt": (
            "Fast-cut cinematic night video, no dialogue, upbeat pop music. "
            "[Shot 1 - Wide tracking, 2s] {people} running through a glowing fairground at night, colorful lights everywhere, ferris wheel spinning behind them. "
            "[Shot 2 - Close action, 1.5s] Screaming on a fast spinning ride, hands up, hair flying, neon lights blurring around them. "
            "[Shot 3 - Slow motion, 1s] Cotton candy pulled apart and shoved in each other's faces, erupting in laughter. "
            "[Shot 4 - POV, 1.5s] Racing through a bumper car arena, crashing head-on, whiplash reaction and laughing. "
            "[Shot 5 - Wide, 2s] At the top of the ferris wheel, city lights below, both arms raised triumphantly. "
            "Quick cuts, handheld shaky energy, warm neon color grade."
        ),
    },
    "dans": {
        "label": "Dans Sahnesi",
        "emoji": "🕺",
        "prompt": (
            "Fast-cut energetic dance video, no dialogue, pumping beat music. "
            "[Shot 1 - Wide, 1.5s] {people} burst onto a dance floor with dramatic spotlight, striking an opening pose. "
            "[Shot 2 - Close tracking, 2s] Synchronized dance moves, camera spinning rapidly around them, strobe lights firing. "
            "[Shot 3 - Low angle, 1.5s] Feet moving fast on the floor, rhythm tight, energy explosive. "
            "[Shot 4 - Slow motion, 1s] One lifts the other dramatically, confetti exploding around them mid-air. "
            "[Shot 5 - Wide pull-back, 2s] Full crowd around them, everyone cheering as they hit a final pose together. "
            "Rapid cuts every second, dynamic camera angles, high contrast club lighting."
        ),
    },
    "surf": {
        "label": "Sörf",
        "emoji": "🏄",
        "prompt": (
            "Fast-cut extreme action surf video, no dialogue, adrenaline surf music. "
            "[Shot 1 - Wide drone, 2s] {people} paddling hard into a massive turquoise wave, spray flying. "
            "[Shot 2 - Side tracking, 2s] Both up on boards riding the face of a huge wave, carving hard, water sheeting off the rails. "
            "[Shot 3 - Slow motion, 1s] One person launches off the wave lip into the air, board spinning, sun behind them. "
            "[Shot 4 - Underwater, 1.5s] Wiping out, tumbling underwater in a chaos of bubbles and white foam. "
            "[Shot 5 - Beach close-up, 1s] Emerging from the water laughing, boards under arms, fist pump to each other. "
            "Extreme action cuts, GoPro-style angles, tropical color grade."
        ),
    },
    "kar": {
        "label": "Kar Topu Savaşı",
        "emoji": "❄️",
        "prompt": (
            "Fast-cut winter action video, no dialogue, playful energetic music. "
            "[Shot 1 - Wide, 1.5s] {people} sprint toward each other across a snowy field, both loading up massive snowballs. "
            "[Shot 2 - Slow motion, 1s] Snowball hits square in the face — powder explosion in perfect detail, shocked expression. "
            "[Shot 3 - POV chase, 2s] Running full speed through the snow, ducking behind trees, ambushing from behind a snowbank. "
            "[Shot 4 - Close action, 1.5s] Building and launching snowballs rapid-fire, breath steaming, laughing uncontrollably. "
            "[Shot 5 - Wide, 1.5s] Both collapse backward into the snow making snow angels, completely exhausted and laughing. "
            "Quick cuts, handheld action cam style, cold crisp winter colors."
        ),
    },
    "kamp": {
        "label": "Kamp Ateşi",
        "emoji": "🏕️",
        "prompt": (
            "Cinematic night camp video, no dialogue, warm acoustic music. "
            "[Shot 1 - Wide, 2s] {people} arrive at campsite at dusk, dropping gear excitedly, starting to set up. "
            "[Shot 2 - Close action, 1.5s] Rapidly chopping wood, striking flint, a fire roaring to life — fast cut sequence. "
            "[Shot 3 - Close, 2s] Both roasting marshmallows, one catching fire, panicked blowing it out, laughing. "
            "[Shot 4 - Wide, 2s] Lying back looking at a stunning star-filled sky, milky way blazing above. "
            "[Shot 5 - Close warmth, 1.5s] Leaning on each other by the fire, faces glowing orange, smiling peacefully. "
            "Mix of fast action cuts and slow warm moments, firelight color grade."
        ),
    },
    # ── ROMANTİK ─────────────────────────────────────────────────────────
    "kir": {
        "label": "Kırda Buluşma",
        "emoji": "🌸",
        "prompt": (
            "Romantic cinematic video, no dialogue, emotional strings music. "
            "[Shot 1 - Wide slow-mo, 2s] {people} running toward each other through a flower field at golden hour, backlit, hair flowing. "
            "[Shot 2 - Close, 1.5s] They collide in an embrace, spinning, wildflowers scattering around them. "
            "[Shot 3 - Tight close-up, 1.5s] Foreheads touching, eyes locked, soft smile, single red rose between them. "
            "[Shot 4 - Wide, 2s] Walking hand in hand through the field, sun setting behind, long golden shadows. "
            "[Shot 5 - Slow motion, 1.5s] Petals blowing across their faces in the breeze, laughing softly. "
            "Gentle pacing with emotional beats, golden hour cinematography, warm color grade."
        ),
    },
    "paris": {
        "label": "Paris Sokakları",
        "emoji": "🗼",
        "prompt": (
            "Romantic cinematic Paris video, no dialogue, soft French accordion music. "
            "[Shot 1 - Wide tracking, 2s] {people} walking fast through busy Parisian streets, dodging crowds, laughing. "
            "[Shot 2 - Close, 1.5s] Sharing a croissant at a sidewalk café, stealing bites from each other, playful. "
            "[Shot 3 - Action, 2s] Running across a bridge over the Seine as light rain begins, sharing one umbrella. "
            "[Shot 4 - Close slow-mo, 1s] Rain drops on the umbrella, their faces close together, smiling. "
            "[Shot 5 - Wide night, 2s] Eiffel Tower sparkling behind them as they look up in wonder together. "
            "Mix of playful action and romantic moments, Parisian color grade."
        ),
    },
    "bogaz": {
        "label": "Boğaz'da Gün Batımı",
        "emoji": "🌉",
        "prompt": (
            "Romantic Istanbul cinematic video, no dialogue, emotional music. "
            "[Shot 1 - Wide drone-style, 2s] {people} on a wooden boat deck cutting through Bosphorus waters, Istanbul skyline blazing orange behind them. "
            "[Shot 2 - Close action, 1.5s] Wind hitting hard, hair flying wildly, both bracing and laughing at the bow. "
            "[Shot 3 - Slow motion, 1.5s] One hands the other a red rose, fingers touching, warm smile against the golden backdrop. "
            "[Shot 4 - Wide, 2s] Both leaning on the railing as the sun touches the water, silhouettes glowing. "
            "[Shot 5 - Close, 1.5s] Looking at each other as the city lights begin flickering on around them. "
            "Romantic pacing, rich golden Istanbul tones."
        ),
    },
    "mum": {
        "label": "Mum Işığında Akşam",
        "emoji": "🕯️",
        "prompt": (
            "Intimate romantic dinner video, no dialogue, soft jazz music. "
            "[Shot 1 - Wide, 2s] {people} sit down at an elegantly set candlelit table, dozens of candles and rose petals everywhere. "
            "[Shot 2 - Close, 1.5s] Clinking wine glasses, eyes meeting over the rims, candlelight dancing in their pupils. "
            "[Shot 3 - Action close, 1.5s] One reaches across and steals food from the other's plate, playful protest, laughing. "
            "[Shot 4 - Slow motion, 1s] Rose petals drifting down onto the table, one catching a petal in their palm. "
            "[Shot 5 - Close warmth, 2s] Hands intertwined on the table, leaning toward each other across the candles. "
            "Intimate warm cuts, deep candlelight gold tones."
        ),
    },
    "kapadokya": {
        "label": "Kapadokya Balonları",
        "emoji": "🎈",
        "prompt": (
            "Epic cinematic Cappadocia video, no dialogue, sweeping orchestral music. "
            "[Shot 1 - Wide drone-style, 2s] {people} standing at cliff edge at dawn, hundreds of hot air balloons rising explosively all around them, sky on fire with color. "
            "[Shot 2 - Action, 1.5s] Racing to the balloon basket, climbing in excitedly, grabbing the rim as it lurches upward. "
            "[Shot 3 - Wide, 2s] Inside balloon looking straight down — fairy chimneys and valleys far below, breathtaking drop. "
            "[Shot 4 - Close slow-mo, 1.5s] Wind hitting their faces at altitude, hair and scarves blasting backward, exhilarated expressions. "
            "[Shot 5 - Wide epic, 2s] Hundreds of balloons surrounding them in every direction as the sun fully rises, overwhelming beauty. "
            "Epic wide shots with fast action cuts, dawn golden color grade."
        ),
    },

    # ── FANTASTİK ────────────────────────────────────────────────────────
    "uzay": {
        "label": "Uzayda Yürüyüş",
        "emoji": "🌌",
        "prompt": (
            "Epic space adventure video, no dialogue, Hans Zimmer-style orchestral music. "
            "[Shot 1 - Wide, 2s] {people} in astronaut suits burst through an airlock into open space, Earth glowing below. "
            "[Shot 2 - Action, 1.5s] Both firing jetpacks, accelerating through the void, stars streaking past. "
            "[Shot 3 - Close slow-mo, 1.5s] Visors touching in space, smiling faces visible through helmets, galaxy reflected in the visors. "
            "[Shot 4 - Wide, 2s] Flying side by side past a massive ringed planet, its rings filling the entire frame. "
            "[Shot 5 - Epic close, 1.5s] Reaching out together to touch a comet's glowing tail, light exploding around their gloves. "
            "Epic fast cuts, breathtaking scale, deep space color palette."
        ),
    },
    "ejderha": {
        "label": "Ejderha Üzerinde",
        "emoji": "🐉",
        "prompt": (
            "Epic fantasy action video, no dialogue, sweeping orchestral fantasy score. "
            "[Shot 1 - Wide, 2s] {people} sprint across a mountain peak, a massive dragon landing thunderously behind them, wings slamming down. "
            "[Shot 2 - Action close, 1.5s] Climbing fast onto the dragon's back, grabbing its scales, it lurches upward violently. "
            "[Shot 3 - Wide POV, 2s] Dragon banking hard through golden clouds at speed, landscape tilting wildly below them, holding on for life. "
            "[Shot 4 - Slow motion, 1.5s] Dragon swoops directly at camera, wings spread massive, roaring — cut to their exhilarated screaming faces. "
            "[Shot 5 - Wide epic, 2s] Banking over a vast fantasy landscape at sunset, silhouettes against a blood-orange sky. "
            "High-energy action cuts, epic fantasy color grade."
        ),
    },
    "okyanus_alti": {
        "label": "Okyanus Altı",
        "emoji": "🌊",
        "prompt": (
            "Magical underwater action video, no dialogue, ethereal ambient music. "
            "[Shot 1 - Wide, 2s] {people} dive off a boat and plunge deep underwater, bubbles exploding around them as they sink fast. "
            "[Shot 2 - Action tracking, 2s] Swimming fast through a coral reef, fish scattering explosively in every direction around them. "
            "[Shot 3 - Close slow-mo, 1.5s] A massive sea turtle glides inches past their faces — wide-eyed shock then delight. "
            "[Shot 4 - Action, 1.5s] Racing each other through an underwater arch, bubbles trailing, kicking hard. "
            "[Shot 5 - Wide, 2s] Both shoot upward toward the shimmering surface, reaching up, light breaking through from above. "
            "Fluid fast cuts, magical teal and blue tones."
        ),
    },
    "buyulu_orman": {
        "label": "Büyülü Orman",
        "emoji": "🧚",
        "prompt": (
            "Magical fantasy forest video, no dialogue, enchanted whimsical music. "
            "[Shot 1 - Wide, 2s] {people} run into a glowing magical forest at dusk, fireflies exploding to life around them as they enter. "
            "[Shot 2 - Close action, 1.5s] Fireflies landing on outstretched hands — one suddenly illuminates brightly, startling them both. "
            "[Shot 3 - Wide action, 2s] Running and spinning through the glowing forest, light particles scattering with every step. "
            "[Shot 4 - Slow motion, 1.5s] Thousands of fireflies rising around them simultaneously in a massive glowing tornado. "
            "[Shot 5 - Close, 1.5s] Looking at each other's faces lit by firefly glow, pure wonder and joy. "
            "Mix of fast magical action and wonder moments, bioluminescent palette."
        ),
    },
    "lale": {
        "label": "Lale Bahçesi",
        "emoji": "🌷",
        "prompt": (
            "Vibrant romantic spring video, no dialogue, joyful music. "
            "[Shot 1 - Wide tracking, 2s] {people} run into a stunning tulip garden in full bloom, thousands of red tulips stretching to the horizon, Istanbul visible behind. "
            "[Shot 2 - Action close, 1.5s] Spinning each other between the tall tulip rows, petals flying everywhere around them. "
            "[Shot 3 - Slow motion, 1.5s] One dramatically presents a single red tulip, petals blowing in slow motion. "
            "[Shot 4 - Close playful, 1.5s] Chasing each other through the tulip rows, ducking, laughing, tulips swaying. "
            "[Shot 5 - Wide, 2s] Collapsing together sitting among the tulips, catching breath, smiling at each other. "
            "Vibrant spring colors, energetic then tender pacing."
        ),
    },
    "sahil": {
        "label": "Sahilde Gün Batımı",
        "emoji": "🌅",
        "prompt": (
            "Dynamic beach sunset video, no dialogue, emotional cinematic music. "
            "[Shot 1 - Wide, 2s] {people} sprint into the ocean at golden hour, clothes on, crashing through shallow waves at full speed. "
            "[Shot 2 - Slow motion, 1.5s] Wave hits them both simultaneously — wall of water, arms out, screaming with joy. "
            "[Shot 3 - Action, 1.5s] Splashing each other furiously in waist-deep water, laughing uncontrollably. "
            "[Shot 4 - Close slow-mo, 1s] Water droplets suspended in golden backlight around their faces. "
            "[Shot 5 - Wide, 2s] Walking out of the water hand in hand, soaking wet, sun setting perfectly behind them. "
            "Fast action cuts mixed with slow-motion beauty, golden hour color grade."
        ),
    },
    "japon_kiraz": {
        "label": "Japon Kiraz Bahçesi",
        "emoji": "🌸",
        "prompt": (
            "Beautiful romantic cherry blossom video, no dialogue, delicate koto music. "
            "[Shot 1 - Wide tracking, 2s] {people} running down a stunning cherry blossom avenue, petals raining down like pink snow. "
            "[Shot 2 - Slow motion, 1.5s] Both jump simultaneously, catching petals mid-air, laughing as they land. "
            "[Shot 3 - Close action, 1.5s] One grabs a branch and shakes it — massive petal shower erupts over them both. "
            "[Shot 4 - Close romantic, 1.5s] Sitting under the biggest tree, petals covering their shoulders, looking at each other. "
            "[Shot 5 - Wide, 2s] Walking away down the petal-covered path into a pink blossom haze, hand in hand. "
            "Playful action mixed with romantic beauty, soft pink tones."
        ),
    },
    "park": {
        "label": "Parkta Gün",
        "emoji": "🌳",
        "prompt": (
            "Fun energetic park day video, no dialogue, cheerful acoustic music. "
            "[Shot 1 - Wide action, 2s] {people} sprint across a sunny park, one chasing the other, laughing and dodging. "
            "[Shot 2 - Close, 1.5s] Collapsing on a picnic blanket, out of breath, stealing food from each other's hands. "
            "[Shot 3 - Action, 1.5s] Frisbee thrown — one leaping and catching it dramatically, landing in a roll. "
            "[Shot 4 - Close, 1.5s] Pushing each other on a swing, one flying high, screaming excitedly. "
            "[Shot 5 - Wide slow-mo, 2s] Both lying in the grass side by side looking up at trees, completely at peace, laughing about something. "
            "Energetic action cuts, warm natural tones, feel-good energy."
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
                "generate_audio": False,
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
        store_id = "370282"

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

# ===================== SEO =====================
from fastapi.responses import PlainTextResponse

@app.get("/sitemap.xml", include_in_schema=False)
async def sitemap():
    content = """<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
  <url>
    <loc>https://hatirai.com/</loc>
    <changefreq>weekly</changefreq>
    <priority>1.0</priority>
  </url>
  <url>
    <loc>https://hatirai.com/terms</loc>
    <changefreq>monthly</changefreq>
    <priority>0.5</priority>
  </url>
  <url>
    <loc>https://hatirai-backend-production.up.railway.app/api/ani</loc>
    <changefreq>weekly</changefreq>
    <priority>0.8</priority>
  </url>
</urlset>"""
    return Response(content=content, media_type="application/xml")

@app.get("/robots.txt", include_in_schema=False)
async def robots():
    content = """User-agent: *
Allow: /
Sitemap: https://hatirai.com/sitemap.xml"""
    return PlainTextResponse(content=content)

@app.get("/google5207070550b5c3d1.html", include_in_schema=False)
async def google_verify():
    return PlainTextResponse("google-site-verification: google5207070550b5c3d1.html")

@app.get("/googlesiteverification", include_in_schema=False)
async def google_verify_meta():
    html = """<!DOCTYPE html><html><head>
<meta name="google-site-verification" content="uCNBGtwP68mV4KJmMF_PvzBSGPVV7WZwHaOuyyyXP8M" />
</head><body></body></html>"""
    return HTMLResponse(content=html)

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
