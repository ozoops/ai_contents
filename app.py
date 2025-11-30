"""
Streamlit 앱에서 이미지/영상 생성 워크플로우를 제공하는 메인 스크립트.
OpenAI API와 LangChain의 런너블을 결합해 프롬프트 번역과 생성 과정을 구성한다.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import requests
import streamlit as st
from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from openai import APIStatusError, OpenAI
from openai.types.video import Video


# -------------------------------------------------------------------
# OpenAI 클라이언트 초기화
# -------------------------------------------------------------------
# -------------------------------------------------------------------
# 페이지 설정 (가장 먼저 호출되어야 함)
# -------------------------------------------------------------------
try:
    st.set_page_config(page_title="AI 비주얼 생성 스튜디오", layout="wide")
except Exception:
    pass

# -------------------------------------------------------------------
# OpenAI 클라이언트 초기화
# -------------------------------------------------------------------
if "openai_api_key" not in st.session_state:
    st.session_state.openai_api_key = os.environ.get("OPENAI_API_KEY", "")

if not st.session_state.openai_api_key:
    with st.sidebar:
        st.session_state.openai_api_key = st.text_input(
            "OpenAI API Key",
            type="password",
            help="환경 변수에 키가 설정되지 않았습니다. API 키를 직접 입력하세요."
        )

try:
    if st.session_state.openai_api_key:
        client = OpenAI(api_key=st.session_state.openai_api_key)
    else:
        client = None
except Exception as exc:
    st.error(f"OpenAI 클라이언트를 초기화하는 중 오류가 발생했습니다: {exc}")
    client = None


# -------------------------------------------------------------------
# 이미지/영상 생성 관련 상수
# -------------------------------------------------------------------
IMAGE_TOOLS: Dict[str, str] = {
    "DALL·E 3 (OpenAI)": "dall-e-3",
    "GPT-Image-1 (OpenAI)": "gpt-image-1",
}

VIDEO_TOOLS: Dict[str, str] = {
    "SORA-2 (OpenAI)": "sora-2",
    "Stable Diffusion Video (커스텀)": "stable-diffusion-video",
}

VIDEO_SIZES: Dict[str, str] = {
    "1280 x 720 (Landscape)": "1280x720",
    "720 x 1280 (Portrait)": "720x1280",
    "1024 x 1792 (Vertical)": "1024x1792",
    "1080 x 1920 (Vertical)": "1080x1920",
    "1792 x 1024 (Horizontal)": "1792x1024",
    "1920 x 1080 (Horizontal)": "1920x1080",
}

VIDEO_TOOL_SIZES: Dict[str, Tuple[str, ...]] = {
    "sora-2": ("1280x720", "720x1280", "1024x1792", "1792x1024"),
    "stable-diffusion-video": tuple(VIDEO_SIZES.values()),
}

VALID_VIDEO_SIZES = {size for sizes in VIDEO_TOOL_SIZES.values() for size in sizes}
VIDEO_SECONDS: Tuple[int, int, int] = (4, 8, 12)

DEFAULT_IMAGE_PROMPTS: List[str] = [
    "안개 낀 산 정상에서 떠오르는 태양과 빛나는 구름을 담은 장면",
    "차분한 바닷가에서 파도가 부드럽게 부서지는 석양 풍경",
]

DEFAULT_VIDEO_PROMPTS: List[str] = [
    "미래 도시의 네온사인 거리를 걷는 사람들의 느린 장면",
    "자연 속에서 흐르는 폭포를 다양한 각도로 담아낸 몽환적인 영상",
]

BACKGROUND_IMAGE_URL = (
    "https://images.unsplash.com/photo-1500530855697-b586d89ba3ee"
    "?auto=format&fit=crop&w=1920&q=80"
)


# -------------------------------------------------------------------
# 공용 유틸리티
# -------------------------------------------------------------------
def ensure_client(feature: str) -> OpenAI:
    """해당 기능을 호출하기 전에 클라이언트가 준비되어 있는지 확인."""
    if client is None:
        raise RuntimeError(f"{feature} 기능을 사용하려면 사이드바에서 OpenAI API Key를 입력하거나 환경 변수를 설정해야 합니다.")
    return client


TRANSLATION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a professional translator. Translate each line of the user input into "
            "{target_language} while keeping the line count identical. If a line is already "
            "written in the requested language, return it unchanged.",
        ),
        ("user", "{prompts}"),
    ]
)


@lru_cache(maxsize=1)
def get_translation_chain() -> RunnableLambda:
    """프롬프트 번역을 수행하는 LangChain 러너블."""
    parser = StrOutputParser()

    def _invoke(inputs: Dict[str, str]) -> str:
        ensure_client("번역")
        prompt_value = TRANSLATION_PROMPT.invoke(inputs)
        if hasattr(prompt_value, "to_messages"):
            messages = prompt_value.to_messages()
        else:  # pragma: no cover
            messages = prompt_value

        role_map = {"system": "system", "human": "user", "ai": "assistant"}
        formatted: List[Dict[str, str]] = []
        for msg in messages:
            if isinstance(msg, BaseMessage):
                role = role_map.get(msg.type, "user")
                content = msg.content
            else:
                role = msg.get("role", "user")
                content = msg.get("content", "")
            if not isinstance(content, str):
                content = str(content)
            formatted.append({"role": role, "content": content})

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=formatted,
            temperature=0.2,
        )
        return parser.invoke(response.choices[0].message.content)

    return RunnableLambda(_invoke)


def translate_prompts(prompts: Iterable[str], target_language: str) -> List[str]:
    """프롬프트 목록을 원하는 언어로 번역."""
    prompt_list = [p.strip() for p in prompts if p.strip()]
    if not prompt_list:
        return []

    combined = "\n".join(prompt_list)
    chain = get_translation_chain()
    translated = chain.invoke({"target_language": target_language, "prompts": combined})
    return [line.strip() for line in translated.splitlines() if line.strip()]


def ensure_prompt_state(prompts_key: str, textarea_key: str) -> None:
    """세션 상태에 저장된 프롬프트를 텍스트 영역과 동기화."""
    sync_flag = f"{textarea_key}__needs_sync"
    prompts = st.session_state.get(prompts_key, [])
    needs_sync = st.session_state.pop(sync_flag, False)
    if needs_sync or textarea_key not in st.session_state:
        st.session_state[textarea_key] = "\n".join(prompts)


def rerun_app() -> None:
    """Streamlit 버전에 따라 rerun 호출."""
    if hasattr(st, "rerun"):
        st.rerun()
    else:  # pragma: no cover
        st.experimental_rerun()


# -------------------------------------------------------------------
# 이미지 생성
# -------------------------------------------------------------------
def generate_image(tool_id: str, prompt: str, size: str) -> str:
    ensure_client("이미지 생성")
    if tool_id not in IMAGE_TOOLS.values():
        raise NotImplementedError(f"{tool_id} 모델은 지원하지 않습니다.")

    response = client.images.generate(
        model=tool_id,
        prompt=prompt,
        n=1,
        size=size,
        quality="standard",
    )
    return response.data[0].url


@lru_cache(maxsize=12)
def get_image_generation_chain(tool_id: str, size: str) -> RunnableLambda:
    """이미지 생성을 래핑한 Runnable."""

    def _invoke(prompt: str) -> Dict[str, str]:
        image_url = generate_image(tool_id, prompt, size)
        return {"prompt": prompt, "url": image_url}

    return RunnableLambda(_invoke)


def download_asset(url: str, path: Path) -> Path:
    """URL 자산을 디스크에 다운로드."""
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        for chunk in response.iter_content(1024 * 128):
            handle.write(chunk)
    return path


# -------------------------------------------------------------------
# 영상 생성
# -------------------------------------------------------------------
def generate_video_clip(tool_id: str, prompt: str, seconds: int, size: str) -> Video:
    api = ensure_client("영상 생성")
    allowed = VIDEO_TOOL_SIZES.get(tool_id)
    if allowed and size not in allowed:
        allowed_text = ", ".join(allowed)
        raise ValueError(f"선택한 모델은 {size} 해상도를 지원하지 않습니다. 사용 가능: {allowed_text}")

    if tool_id == "sora-2":
        request_kwargs = {
            "model": tool_id,
            "prompt": prompt,
            "seconds": str(seconds),
            "size": size,
        }
        try:
            video_job = api.videos.create_and_poll(**request_kwargs)
        except APIStatusError as exc:
            if getattr(exc, "status_code", None) == 403:
                raise RuntimeError("SORA-2 접근 권한이 필요합니다. OpenAI 지원을 통해 권한을 활성화하세요.") from exc
            raise RuntimeError(f"SORA-2 API 오류: {getattr(exc, 'message', exc)}") from exc

        if video_job.status != "completed":
            error_msg = getattr(getattr(video_job, "error", None), "message", "원인 미확인")
            raise RuntimeError(f"SORA-2 영상 생성이 완료되지 않았습니다. 상태: {video_job.status}, 사유: {error_msg}")
        return video_job

    raise NotImplementedError(f"{tool_id} 모델은 아직 영상 생성을 지원하지 않습니다.")


@lru_cache(maxsize=12)
def get_video_generation_chain(tool_id: str, seconds: int, size: str) -> RunnableLambda:
    """영상 생성을 래핑한 Runnable."""

    def _invoke(prompt: str) -> Dict[str, Video]:
        video_obj = generate_video_clip(tool_id, prompt, seconds, size)
        return {"prompt": prompt, "video": video_obj}

    return RunnableLambda(_invoke)


def download_video(video: Video, output_path: Path) -> Path:
    """영상 결과를 파일로 저장."""
    ensure_client("영상 다운로드")
    content = client.videos.download_content(video.id)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    content.write_to_file(str(output_path))
    return output_path


# -------------------------------------------------------------------
# UI 렌더링
# -------------------------------------------------------------------
def render_image_page() -> None:
    st.title("🌌🪄 AI 이미지 생성 스튜디오")
    st.info("장면별 프롬프트를 입력하고 원하는 모델과 해상도로 이미지를 생성해보세요.")

    if "image_prompts" not in st.session_state:
        st.session_state.image_prompts = DEFAULT_IMAGE_PROMPTS.copy()
    if "image_outputs" not in st.session_state:
        st.session_state.image_outputs = []

    with st.sidebar:
        st.header("🖌️🌠 이미지 옵션")
        tool_name = st.selectbox("이미지 생성 도구", list(IMAGE_TOOLS.keys()), index=0)
        tool_id = IMAGE_TOOLS[tool_name]
        image_size = st.selectbox("이미지 해상도", ["1024x1024", "1792x1024", "1024x1792"], index=0)
        st.caption(f"모델: **{tool_name}** · 해상도: {image_size}")

    ensure_prompt_state("image_prompts", "image_prompt_input")
    prompt_input = st.text_area(
        "장면(Scene) 프롬프트 입력",
        height=200,
        help="행 단위로 입력하면 각 행마다 개별 이미지가 생성됩니다.",
        key="image_prompt_input",
    )
    prompt_value = st.session_state.get("image_prompt_input", "")
    st.session_state.image_prompts = [line.strip() for line in prompt_value.split("\n") if line.strip()]

    col_en, col_ko, col_generate, _ = st.columns([1, 1, 1, 5])
    if col_en.button("🇺🇸 영어로 번역", help="현재 프롬프트를 영어로 번역합니다."):
        if st.session_state.image_prompts:
            with st.spinner("번역 중..."):
                st.session_state.image_prompts = translate_prompts(st.session_state.image_prompts, "English")
                st.session_state["image_prompt_input__needs_sync"] = True
            st.success("영어 번역 완료")
            rerun_app()
        else:
            st.toast("번역할 프롬프트가 없습니다.")

    if col_ko.button("🇰🇷 한국어로 번역", help="현재 프롬프트를 한국어로 번역합니다."):
        if st.session_state.image_prompts:
            with st.spinner("번역 중..."):
                st.session_state.image_prompts = translate_prompts(st.session_state.image_prompts, "Korean")
                st.session_state["image_prompt_input__needs_sync"] = True
            st.success("한국어 번역 완료")
            rerun_app()
        else:
            st.toast("번역할 프롬프트가 없습니다.")

    if col_generate.button("🚀 이미지 생성", type="primary"):
        if not st.session_state.image_prompts:
            st.warning("생성할 프롬프트를 입력해주세요.")
            st.session_state.image_outputs = []
        else:
            st.session_state.image_outputs = []
            temp_dir = Path("temp_images")
            temp_dir.mkdir(exist_ok=True)
            image_chain = get_image_generation_chain(tool_id, image_size)

            try:
                status_placeholder = st.empty()
                progress = st.progress(0)
                total = len(st.session_state.image_prompts)

                for idx, scene_prompt in enumerate(st.session_state.image_prompts, start=1):
                    progress.progress(int(idx / total * 100))
                    status_placeholder.text(f"[{idx}/{total}] 이미지 생성 중: '{scene_prompt}'")
                    result = image_chain.invoke(scene_prompt)
                    image_url = result["url"]
                    output_path = temp_dir / f"scene_{idx}.png"
                    download_asset(image_url, output_path)
                    st.session_state.image_outputs.append({"path": output_path, "prompt": scene_prompt})

                status_placeholder.success("🎉 모든 이미지가 생성되었습니다!")
            except (RuntimeError, ValueError, NotImplementedError, APIStatusError) as exc:
                st.error(f"⚠️ 오류: {exc}")
            except Exception as exc:  # pragma: no cover
                st.error(f"⚠️ 예기치 못한 오류가 발생했습니다: {exc}")

    if st.session_state.image_outputs:
        st.markdown("---")
        st.subheader("📂 생성된 이미지")
        for idx, info in enumerate(list(st.session_state.image_outputs), start=1):
            img_path: Path = info["path"]
            cols = st.columns([3, 1])
            with cols[0]:
                st.image(str(img_path), caption=f"장면 #{idx}: {info['prompt']}")
            with cols[1]:
                if st.button(f"🗑️ 삭제 {idx}", key=f"delete_image_{idx}"):
                    if img_path.exists():
                        img_path.unlink()
                    st.session_state.image_outputs = [
                        item for item in st.session_state.image_outputs if item["path"] != img_path
                    ]
                    rerun_app()
                with img_path.open("rb") as file:
                    st.download_button(
                        label=f"⬇️ 다운로드 #{idx}",
                        data=file,
                        file_name=img_path.name,
                        mime="image/png",
                    )
            st.markdown("&nbsp;")


def render_video_page() -> None:
    st.title("🎞️⚡ AI 영상 생성 스튜디오")
    st.info("장면 프롬프트와 옵션을 지정해 짧은 클립을 생성합니다. SORA-2 사용에는 권한이 필요합니다.")

    if "video_prompts" not in st.session_state:
        st.session_state.video_prompts = DEFAULT_VIDEO_PROMPTS.copy()
    if "video_outputs" not in st.session_state:
        st.session_state.video_outputs = []

    with st.sidebar:
        st.header("🛰️🎛️ 영상 옵션")
        tool_name = st.selectbox("영상 생성 도구", list(VIDEO_TOOLS.keys()), index=0)
        tool_id = VIDEO_TOOLS[tool_name]
        clip_seconds = st.select_slider("영상 길이(초)", options=list(VIDEO_SECONDS), value=VIDEO_SECONDS[0])
        allowed_sizes = VIDEO_TOOL_SIZES.get(tool_id, tuple(VIDEO_SIZES.values()))
        size_labels = [label for label, code in VIDEO_SIZES.items() if code in allowed_sizes]
        size_label = st.selectbox("영상 해상도", size_labels, index=0)
        video_size = VIDEO_SIZES[size_label]
        st.caption(f"모델: **{tool_name}** · 길이: {clip_seconds}s · 해상도: {video_size}")
        if len(size_labels) < len(VIDEO_SIZES):
            st.caption(f"{tool_name} 모델이 지원하는 해상도: {', '.join(allowed_sizes)}")

    ensure_prompt_state("video_prompts", "video_prompt_input")
    prompt_input = st.text_area(
        "장면(Scene) 프롬프트 입력",
        height=200,
        help="행 단위로 입력하면 각 행마다 개별 영상이 생성됩니다.",
        key="video_prompt_input",
    )
    prompt_value = st.session_state.get("video_prompt_input", "")
    st.session_state.video_prompts = [line.strip() for line in prompt_value.split("\n") if line.strip()]

    col_en, col_ko, col_generate, _ = st.columns([1, 1, 1, 5])
    if col_en.button("🇺🇸 영어로 번역", key="video_translate_en"):
        if st.session_state.video_prompts:
            with st.spinner("번역 중..."):
                st.session_state.video_prompts = translate_prompts(st.session_state.video_prompts, "English")
                st.session_state["video_prompt_input__needs_sync"] = True
            st.success("영어 번역 완료")
            rerun_app()
        else:
            st.toast("번역할 프롬프트가 없습니다.")

    if col_ko.button("🇰🇷 한국어로 번역", key="video_translate_ko"):
        if st.session_state.video_prompts:
            with st.spinner("번역 중..."):
                st.session_state.video_prompts = translate_prompts(st.session_state.video_prompts, "Korean")
                st.session_state["video_prompt_input__needs_sync"] = True
            st.success("한국어 번역 완료")
            rerun_app()
        else:
            st.toast("번역할 프롬프트가 없습니다.")

    if col_generate.button("🎬 영상 생성", type="primary", key="video_generate"):
        if not st.session_state.video_prompts:
            st.warning("생성할 프롬프트를 입력해주세요.")
            st.session_state.video_outputs = []
        else:
            st.session_state.video_outputs = []
            temp_dir = Path("temp_videos")
            temp_dir.mkdir(exist_ok=True)
            video_chain = get_video_generation_chain(tool_id, clip_seconds, video_size)

            try:
                status_placeholder = st.empty()
                progress = st.progress(0)
                total = len(st.session_state.video_prompts)

                for idx, scene_prompt in enumerate(st.session_state.video_prompts, start=1):
                    progress.progress(int(idx / total * 100))
                    status_placeholder.text(f"[{idx}/{total}] 영상 생성 중: '{scene_prompt}'")
                    result = video_chain.invoke(scene_prompt)
                    video_obj = result["video"]
                    output_path = temp_dir / f"scene_{idx}.mp4"
                    download_video(video_obj, output_path)
                    st.session_state.video_outputs.append({"path": output_path, "prompt": scene_prompt})

                status_placeholder.success("🎉 모든 영상이 생성되었습니다!")
            except (RuntimeError, ValueError, NotImplementedError, APIStatusError) as exc:
                st.error(f"⚠️ 오류: {exc}")
            except Exception as exc:  # pragma: no cover
                st.error(f"⚠️ 예기치 못한 오류가 발생했습니다: {exc}")

    if st.session_state.video_outputs:
        st.markdown("---")
        st.subheader("📂 생성된 영상")
        for idx, info in enumerate(list(st.session_state.video_outputs), start=1):
            video_path: Path = info["path"]
            cols = st.columns([3, 1])
            with cols[0]:
                st.video(str(video_path))
                st.caption(f"장면 #{idx}: {info['prompt']}")
            with cols[1]:
                if st.button(f"🗑️ 삭제 {idx}", key=f"delete_video_{idx}"):
                    if video_path.exists():
                        video_path.unlink()
                    st.session_state.video_outputs = [
                        item for item in st.session_state.video_outputs if item["path"] != video_path
                    ]
                    rerun_app()
                with video_path.open("rb") as file:
                    st.download_button(
                        label=f"⬇️ 다운로드 #{idx}",
                        data=file,
                        file_name=video_path.name,
                        mime="video/mp4",
                    )
            st.markdown("&nbsp;")


# -------------------------------------------------------------------
# 페이지 구성 및 스타일
# -------------------------------------------------------------------
# st.set_page_config moved to top

st.markdown(
    f"""
    <style>
    .stApp {{
        background: linear-gradient(135deg, rgba(8, 11, 24, 0.92), rgba(24, 18, 36, 0.85)),
                    url('{BACKGROUND_IMAGE_URL}');
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
        background-position: center;
    }}

    .stMainBlockContainer, .block-container {{
        background-color: rgba(8, 10, 20, 0.68);
        padding: 2.2rem 2.5rem;
        border-radius: 20px;
        backdrop-filter: blur(14px);
        box-shadow: 0 0 35px rgba(0, 0, 0, 0.45);
    }}

    section[data-testid="stSidebar"] > div {{
        background-color: rgba(9, 12, 24, 0.75);
        border-radius: 18px;
        padding-top: 1rem;
        backdrop-filter: blur(10px);
        box-shadow: 0 0 20px rgba(0, 0, 0, 0.4);
    }}

    .stButton button {{
        white-space: nowrap;
        border-radius: 999px;
        padding: 0.45rem 1.2rem;
        font-weight: 600;
    }}

    .stTextInput textarea, .stTextArea textarea {{
        background-color: rgba(18, 22, 38, 0.65);
        color: #f5f5f7;
        border-radius: 14px;
    }}

    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4 {{
        color: #f8f9fd;
    }}

    .stMarkdown p, .stMarkdown li, .stMarkdown span {{
        color: #e6e6f0;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.title("✨🎨 AI 비주얼 제작실")

    page_choice = st.radio("작업 모드", options=["이미지 생성", "영상 생성"], index=0)

if page_choice == "이미지 생성":
    render_image_page()
else:
    render_video_page()
