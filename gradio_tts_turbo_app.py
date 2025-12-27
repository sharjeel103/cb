import random
import numpy as np
import torch
import gradio as gr
import re
import io
import os
import time
import logging
import uuid
import tempfile
import gc
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Set, Any

# --- Dependencies Check ---
try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    print("Warning: Librosa not found. Advanced audio features will be limited.")

try:
    import soundfile as sf
except ImportError:
    print("Warning: soundfile not found. Audio operations may fail.")

try:
    from scipy.signal import butter, filtfilt
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("Warning: scipy not found. DC offset removal will be disabled.")

try:
    import parselmouth
    PARSELMOUTH_AVAILABLE = True
except ImportError:
    PARSELMOUTH_AVAILABLE = False

from chatterbox.tts_turbo import ChatterboxTurboTTS

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==========================================
#        CONSTANTS & CONFIGURATION
# ==========================================
CROSSFADE_MS = 20           # Crossfade duration
SAFETY_FADE_MS = 3          # Minimal edge fade
ENABLE_DC_REMOVAL = True    # Remove DC offset
DC_HIGHPASS_HZ = 15         # Cutoff for DC removal
PEAK_NORMALIZE_TARGET = 0.95
MAX_REF_DURATION_SEC = 30   # Server limit for reference audio

# --- OPTIMIZED BATCH SETTINGS ---
CHUNK_SIZE_DEFAULT = 275    # Optimized Chunk Size
DEFAULT_BATCH_SIZE = 18     # Best speed
SAFE_BATCH_SIZE = 14        # Fallback for OOM

EVENT_TAGS = [
  "[advertisement]", "[angry]", "[chuckle]", "[clear throat]", "[cough]", "[crying]",
  "[dramatic]", "[fear]", "[gasp]", "[groan]", "[happy]", "[laugh]", "[narration]",
  "[sarcastic]", "[shush]", "[sigh]", "[sniff]", "[surprised]", "[whispering]"
]

# ==========================================
#      UTILS: TEXT PROCESSING
# ==========================================

ABBREVIATIONS: Set[str] = {
    "mr.", "mrs.", "ms.", "dr.", "prof.", "rev.", "hon.", "st.", "etc.", "e.g.", "i.e.",
    "vs.", "approx.", "apt.", "dept.", "fig.", "gen.", "gov.", "inc.", "jr.", "sr.",
    "ltd.", "no.", "p.", "pp.", "vol.", "op.", "cit.", "ca.", "cf.", "ed.", "esp.",
    "et.", "al.", "ibid.", "id.", "inf.", "sup.", "viz.", "sc.", "fl.", "d.", "b.",
    "r.", "c.", "v.", "u.s.", "u.k.", "a.m.", "p.m.", "a.d.", "b.c.",
}

NUMBER_DOT_NUMBER_PATTERN = re.compile(r"(?<!\d\.)\d*\.\d+")
VERSION_PATTERN = re.compile(r"[vV]?\d+(\.\d+)+")
POTENTIAL_END_PATTERN = re.compile(r'([.!?])(["\']?)([\s,\n]+|$)')
BULLET_POINT_PATTERN = re.compile(r"(?:^|\n)\s*([-•*]|\d+\.)\s+")
NON_VERBAL_CUE_PATTERN = re.compile(r"(\([\w\s'-]+\))")

def _is_valid_sentence_end(text: str, period_index: int) -> bool:
    word_start_before_period = period_index - 1
    scan_limit = max(0, period_index - 10)
    while word_start_before_period >= scan_limit and not text[word_start_before_period].isspace():
        word_start_before_period -= 1
    word_before_period = text[word_start_before_period + 1 : period_index + 1].lower()
    if word_before_period in ABBREVIATIONS:
        return False
    context_start = max(0, period_index - 10)
    context_end = min(len(text), period_index + 10)
    context_segment = text[context_start:context_end]
    relative_period_index_in_context = period_index - context_start
    for pattern in [NUMBER_DOT_NUMBER_PATTERN, VERSION_PATTERN]:
        for match in pattern.finditer(context_segment):
            if match.start() <= relative_period_index_in_context < match.end():
                if not (relative_period_index_in_context == match.end() - 1 and (period_index + 1 == len(text) or text[period_index + 1].isspace())):
                    return False
    return True

def _split_text_by_punctuation(text: str) -> List[str]:
    sentences: List[str] = []
    last_split_index = 0
    text_length = len(text)
    for match in POTENTIAL_END_PATTERN.finditer(text):
        punctuation_char_index = match.start(1)
        punctuation_char = text[punctuation_char_index]
        slice_end_after_punctuation = match.start(1) + 1 + len(match.group(2) or "")
        
        if punctuation_char in ["!", "?"]:
            s = text[last_split_index:slice_end_after_punctuation].strip()
            if s: sentences.append(s)
            last_split_index = match.end()
            continue

        if punctuation_char == ".":
            if (punctuation_char_index > 0 and text[punctuation_char_index - 1] == ".") or \
               (punctuation_char_index < text_length - 1 and text[punctuation_char_index + 1] == "."):
                continue
            if _is_valid_sentence_end(text, punctuation_char_index):
                s = text[last_split_index:slice_end_after_punctuation].strip()
                if s: sentences.append(s)
                last_split_index = match.end()

    remaining = text[last_split_index:].strip()
    if remaining: sentences.append(remaining)
    return sentences if sentences else [text.strip()] if text.strip() else []

def split_into_sentences(text: str) -> List[str]:
    if not text or text.isspace(): return []
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    bullet_matches = list(BULLET_POINT_PATTERN.finditer(text))
    if bullet_matches:
        sentences = []
        curr = 0
        for i, match in enumerate(bullet_matches):
            start = match.start()
            if i == 0 and start > curr:
                seg = text[curr:start].strip()
                if seg: sentences.extend(_split_text_by_punctuation(seg))
            next_start = bullet_matches[i+1].start() if i + 1 < len(bullet_matches) else len(text)
            seg = text[start:next_start].strip()
            if seg: sentences.append(seg)
            curr = next_start
        if curr < len(text):
            seg = text[curr:].strip()
            if seg: sentences.extend(_split_text_by_punctuation(seg))
        return [s for s in sentences if s]
    else:
        return _split_text_by_punctuation(text)

def _preprocess_and_segment_text(full_text: str) -> List[Tuple[Optional[str], str]]:
    if not full_text or full_text.isspace(): return []
    segmented = []
    parts = NON_VERBAL_CUE_PATTERN.split(full_text)
    for part in parts:
        if not part or part.isspace(): continue
        if NON_VERBAL_CUE_PATTERN.fullmatch(part):
            segmented.append((None, part.strip()))
        else:
            for s in split_into_sentences(part.strip()):
                if s: segmented.append((None, s))
    if not segmented and full_text.strip():
        segmented.append((None, full_text.strip()))
    return segmented

def chunk_text_by_sentences(full_text: str, chunk_size: int) -> List[str]:
    if not full_text or full_text.isspace(): return []
    if chunk_size <= 0: chunk_size = float("inf")
    
    # [THE AXE] Adjusted for new smaller chunk size
    HARD_LIMIT = chunk_size + 50
    
    segments = _preprocess_and_segment_text(full_text)
    if not segments: return []
    
    chunks = []
    curr_sents = []
    curr_len = 0
    
    for _, seg_text in segments:
        slen = len(seg_text)
        
        # --- THE AXE LOGIC ---
        if slen > HARD_LIMIT:
            if curr_sents:
                chunks.append(" ".join(curr_sents))
                curr_sents = []
                curr_len = 0
            
            print(f"Warning: Forced chop applied to long sentence ({slen} chars)")
            for i in range(0, slen, chunk_size):
                chunks.append(seg_text[i : i + chunk_size])
            continue
        # ---------------------

        if not curr_sents:
            curr_sents.append(seg_text)
            curr_len = slen
        elif curr_len + 1 + slen <= chunk_size:
            curr_sents.append(seg_text)
            curr_len += 1 + slen
        else:
            if curr_sents: chunks.append(" ".join(curr_sents))
            curr_sents = [seg_text]
            curr_len = slen
        
        if curr_len > chunk_size and len(curr_sents) == 1:
            chunks.append(" ".join(curr_sents))
            curr_sents = []
            curr_len = 0
            
    if curr_sents: chunks.append(" ".join(curr_sents))
    return [c for c in chunks if c.strip()]

# ==========================================
#      UTILS: AUDIO PROCESSING
# ==========================================

def trim_lead_trail_silence(audio_array: np.ndarray, sample_rate: int, silence_thresh_db: float = -40.0, padding_ms: int = 50) -> np.ndarray:
    if not LIBROSA_AVAILABLE or audio_array is None or audio_array.size == 0: return audio_array
    try:
        _, index = librosa.effects.trim(y=audio_array, top_db=abs(silence_thresh_db))
        pad_samples = int((padding_ms / 1000.0) * sample_rate)
        start = max(0, index[0] - pad_samples)
        end = min(len(audio_array), index[1] + pad_samples)
        if end > start: return audio_array[start:end]
        return audio_array
    except Exception:
        return audio_array

def fix_internal_silence(audio_array: np.ndarray, sample_rate: int, min_silence_ms: int = 700, max_allowed_ms: int = 300) -> np.ndarray:
    if not LIBROSA_AVAILABLE or audio_array is None or audio_array.size == 0: return audio_array
    try:
        intervals = librosa.effects.split(y=audio_array, top_db=40)
        if len(intervals) <= 1: return audio_array
        parts = []
        last_end = 0
        min_samples = int((min_silence_ms/1000)*sample_rate)
        max_keep = int((max_allowed_ms/1000)*sample_rate)
        for start, end in intervals:
            silence_dur = start - last_end
            if silence_dur > 0:
                if silence_dur >= min_samples:
                    parts.append(audio_array[last_end : last_end + max_keep])
                else:
                    parts.append(audio_array[last_end : start])
            parts.append(audio_array[start:end])
            last_end = end
        return np.concatenate(parts)
    except Exception:
        return audio_array

def remove_long_unvoiced(audio_array: np.ndarray, sample_rate: int, min_dur_ms: int = 300) -> np.ndarray:
    if not PARSELMOUTH_AVAILABLE or audio_array is None or audio_array.size == 0: return audio_array
    try:
        sound = parselmouth.Sound(audio_array.astype(np.float64), sampling_frequency=sample_rate)
        pitch = sound.to_pitch(pitch_floor=75, pitch_ceiling=600)
        vu = pitch.get_VoicedVoicelessUnvoiced()
        keep = []
        curr = 0
        min_samples = int((min_dur_ms/1000)*sample_rate)
        for i in range(len(vu.time_intervals)):
            start_t, end_t, label = vu.time_intervals[i]
            start_s = int(start_t * sample_rate)
            end_s = int(end_t * sample_rate)
            dur = end_s - start_s
            if label == "voiced":
                keep.append(audio_array[curr:end_s])
                curr = end_s
            else:
                if dur < min_samples:
                    keep.append(audio_array[curr:end_s])
                    curr = end_s
                else:
                    if start_s > curr:
                        keep.append(audio_array[curr:start_s])
                    curr = end_s
        if curr < len(audio_array): keep.append(audio_array[curr:])
        return np.concatenate(keep) if keep else audio_array
    except Exception:
        return audio_array

# ==========================================
#     STITCHING HELPERS
# ==========================================

def _generate_equal_power_curves(n_samples: int):
    t = np.linspace(0, np.pi / 2, n_samples, dtype=np.float32)
    return np.cos(t) ** 2, np.sin(t) ** 2

def _crossfade_with_overlap(chunk_a: np.ndarray, chunk_b: np.ndarray, fade_samples: int) -> np.ndarray:
    fade_samples = min(fade_samples, len(chunk_a), len(chunk_b))
    if fade_samples <= 0: return np.concatenate([chunk_a, chunk_b])
    fade_out, fade_in = _generate_equal_power_curves(fade_samples)
    crossfaded = (chunk_a[-fade_samples:] * fade_out) + (chunk_b[:fade_samples] * fade_in)
    return np.concatenate([chunk_a[:-fade_samples], crossfaded, chunk_b[fade_samples:]])

def _remove_dc_offset(audio: np.ndarray, sample_rate: int, cutoff_hz: float = 15.0) -> np.ndarray:
    if not SCIPY_AVAILABLE: return audio
    try:
        nyquist = sample_rate / 2
        b, a = butter(2, cutoff_hz / nyquist, btype="high")
        return filtfilt(b, a, audio).astype(np.float32)
    except Exception:
        return audio

def stitch_audio_segments(audio_segments: List[np.ndarray], engine_sr: int, sentence_pause_s: float) -> Optional[np.ndarray]:
    """Reusable stitching logic for Batches and Chunks."""
    if not audio_segments:
        return None
    if len(audio_segments) == 1:
        return audio_segments[0]
        
    fade_samples = int(CROSSFADE_MS / 1000 * engine_sr)
    desired_silence_samples = int(sentence_pause_s * engine_sr)
    silence_buffer_samples = desired_silence_samples + (fade_samples * 2)
    
    chunks = []
    for c in audio_segments:
        processed = c.astype(np.float32, copy=True)
        if ENABLE_DC_REMOVAL:
            processed = _remove_dc_offset(processed, engine_sr, DC_HIGHPASS_HZ)
        chunks.append(processed)
        
    result = chunks[0]
    for i in range(1, len(chunks)):
        silence = np.zeros(silence_buffer_samples, dtype=np.float32)
        # Fade out previous -> silence
        result = _crossfade_with_overlap(result, silence, fade_samples)
        # Fade silence -> next chunk
        result = _crossfade_with_overlap(result, chunks[i], fade_samples)
        
    return result

# ==========================================
#              GRADIO APP
# ==========================================

CUSTOM_CSS = """
.tag-container { display: flex !important; flex-wrap: wrap !important; gap: 8px !important; margin-top: 5px !important; margin-bottom: 10px !important; border: none !important; background: transparent !important; }
.tag-btn { min-width: fit-content !important; width: auto !important; height: 32px !important; font-size: 13px !important; background: #eef2ff !important; border: 1px solid #c7d2fe !important; color: #3730a3 !important; border-radius: 6px !important; padding: 0 10px !important; margin: 0 !important; box-shadow: none !important; }
.tag-btn:hover { background: #c7d2fe !important; transform: translateY(-1px); }
"""

INSERT_TAG_JS = """
(tag_val, current_text) => {
    const textarea = document.querySelector('#main_textbox textarea');
    if (!textarea) return current_text + " " + tag_val; 
    const start = textarea.selectionStart;
    const end = textarea.selectionEnd;
    let prefix = (start === 0 || current_text[start - 1] === ' ') ? "" : " ";
    let suffix = (end < current_text.length && current_text[end] === ' ') ? "" : " ";
    return current_text.slice(0, start) + prefix + tag_val + suffix + current_text.slice(end);
}
"""

def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)

def load_model():
    print(f"Loading Chatterbox-Turbo on {DEVICE}...")
    model = ChatterboxTurboTTS.from_pretrained(DEVICE)
    return model

def prepare_reference_audio(audio_path):
    if not audio_path or not os.path.exists(audio_path): return audio_path
    try:
        info = sf.info(audio_path)
        if info.duration <= MAX_REF_DURATION_SEC: return audio_path
        print(f"Reference audio {info.duration:.1f}s > {MAX_REF_DURATION_SEC}s. Cropping...")
        y, sr = librosa.load(audio_path, sr=None)
        max_samples = int(MAX_REF_DURATION_SEC * sr)
        y_cropped = y[:max_samples]
        temp_dir = tempfile.gettempdir()
        temp_path = os.path.join(temp_dir, f"cropped_{uuid.uuid4().hex}.wav")
        sf.write(temp_path, y_cropped, sr)
        return temp_path
    except Exception as e:
        print(f"Error checking reference: {e}")
        return audio_path

def generate(
        model,
        text,
        audio_prompt_path,
        temperature,
        seed_num,
        top_p,
        top_k,
        repetition_penalty,
        norm_loudness,
        sentence_pause_s,
        trim_silence,
        fix_int_silence,
        remove_unvoiced
):
    if model is None:
        model = ChatterboxTurboTTS.from_pretrained(DEVICE)

    if seed_num != 0: set_seed(int(seed_num))

    # [TIMER START]
    start_time = time.time()

    # [CLEANING]
    text = text.replace('",', '" ').replace('",', '" ') 

    # --- 0. Reference Handling ---
    final_ref_path = prepare_reference_audio(audio_prompt_path)

    # --- 1. Text Chunking ---
    print("Preprocessing text...")
    text_chunks = chunk_text_by_sentences(text, chunk_size=CHUNK_SIZE_DEFAULT)
    if not text_chunks:
        print("Warning: No usable text chunks.")
        return None, "Error: No text chunks"

    all_batch_audios_np = []
    engine_sr = None

    print(f"Generating audio for {len(text_chunks)} chunks...")
    
    # --- 2. ROBUST BATCHED GENERATION LOOP ---
    current_batch_size = DEFAULT_BATCH_SIZE
    chunk_idx = 0
    total_chunks = len(text_chunks)

    with torch.inference_mode():
        while chunk_idx < total_chunks:
            # Determine end index for this batch
            end_idx = min(chunk_idx + current_batch_size, total_chunks)
            batch_texts = text_chunks[chunk_idx : end_idx]
            
            print(f"Processing batch {chunk_idx}-{end_idx} (Size: {len(batch_texts)})...")
            
            try:
                # [GENERATION ATTEMPT]
                wavs = model.generate_batch(
                    batch_texts,
                    audio_prompt_paths=final_ref_path,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=int(top_k),
                    repetition_penalty=repetition_penalty,
                    norm_loudness=norm_loudness,
                )
                
                if engine_sr is None: engine_sr = model.sr
                
                # Convert Batch Tensors -> Numpy
                batch_segments_np = []
                for wav_tensor in wavs:
                    audio_np = wav_tensor.cpu().numpy().squeeze()
                    batch_segments_np.append(audio_np)
                    del wav_tensor 
                
                # Stitch WITHIN batch
                batch_audio_merged = stitch_audio_segments(batch_segments_np, engine_sr, sentence_pause_s)
                if batch_audio_merged is not None:
                    all_batch_audios_np.append(batch_audio_merged)

                # [SUCCESS]
                chunk_idx += len(batch_texts)          # Advance index
                current_batch_size = DEFAULT_BATCH_SIZE # Reset to speed mode
                
                # Cleanup
                del wavs
                torch.cuda.empty_cache()

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    # [OOM HANDLER]
                    print(f"⚠️ OOM DETECTED at batch size {current_batch_size}!")
                    
                    # 1. Clear Memory Immediately
                    if 'wavs' in locals(): del wavs
                    gc.collect()
                    torch.cuda.empty_cache()
                    
                    # 2. Strategy: Reduce Batch Size
                    if current_batch_size > SAFE_BATCH_SIZE:
                        print(f"♻️ Retrying with Safe Batch Size ({SAFE_BATCH_SIZE})...")
                        current_batch_size = SAFE_BATCH_SIZE
                        # Loop continues, retrying same chunk_idx
                    elif current_batch_size > 1:
                        print("♻️ Safe Batch Failed. Retrying with Single Processing (Size 1)...")
                        current_batch_size = 1
                    else:
                        print("❌ OOM even at batch size 1. Skipping this chunk.")
                        chunk_idx += 1 # Skip problematic chunk to keep server alive
                else:
                    # Non-OOM Error
                    print(f"Error during generation: {e}")
                    chunk_idx += len(batch_texts) # Skip to avoid loop death
                    
            except Exception as e:
                print(f"General Error: {e}")
                chunk_idx += len(batch_texts)

    # --- 3. LEVEL 2 MERGE ---
    print("Stitching all batches...")
    final_audio_np = stitch_audio_segments(all_batch_audios_np, engine_sr, sentence_pause_s)
    
    if final_audio_np is None:
        return None, "Error: Generation Failed"

    # --- 4. Post-Processing ---
    print("Post-processing...")
    final_audio_np = final_audio_np.astype(np.float32)

    if trim_silence:
        final_audio_np = trim_lead_trail_silence(final_audio_np, engine_sr)
    if fix_int_silence:
        final_audio_np = fix_internal_silence(final_audio_np, engine_sr)
    if remove_unvoiced:
        final_audio_np = remove_long_unvoiced(final_audio_np, engine_sr)

    peak = np.abs(final_audio_np).max()
    if peak > 0:
        final_audio_np = final_audio_np / peak * PEAK_NORMALIZE_TARGET

    # [TIMER END]
    elapsed_time = time.time() - start_time
    time_str = f"Done in {elapsed_time:.2f} seconds ({len(text_chunks)} chunks)"

    return (engine_sr, final_audio_np), time_str


with gr.Blocks(title="Chatterbox Turbo", css=CUSTOM_CSS) as demo:
    gr.Markdown("# ⚡ Chatterbox Turbo (Robust Batched)")
    model_state = gr.State(None)
    with gr.Row():
        with gr.Column():
            text = gr.Textbox(
                value="Oh, that's hilarious! [chuckle] Um anyway, we do have a new model in store. It's the SkyNet T-800 series and it's got basically everything. Including AI integration with ChatGPT and um all that jazz. Would you like me to get some prices for you?",
                label="Text to synthesize",
                max_lines=5,
                elem_id="main_textbox"
            )
            with gr.Row(elem_classes=["tag-container"]):
                for tag in EVENT_TAGS:
                    btn = gr.Button(tag, elem_classes=["tag-btn"])
                    btn.click(fn=None, inputs=[btn, text], outputs=text, js=INSERT_TAG_JS)
            ref_wav = gr.Audio(
                sources=["upload", "microphone"],
                type="filepath",
                label="Reference Audio File",
                value="https://storage.googleapis.com/chatterbox-demo-samples/prompts/female_random_podcast.wav"
            )
            run_btn = gr.Button("Generate ⚡", variant="primary")
        with gr.Column():
            audio_output = gr.Audio(label="Output Audio")
            
            time_output = gr.Textbox(label="Generation Status", value="Ready", interactive=False)
            
            with gr.Accordion("Advanced Options", open=False):
                seed_num = gr.Number(value=0, label="Random seed (0 for random)")
                temp = gr.Slider(0.05, 2.0, step=.05, label="Temperature", value=0.8)
                
                sentence_pause_s = gr.Slider(0.0, 2.0, step=0.1, label="Sentence Pause (seconds)", value=0.2)
                
                top_p = gr.Slider(0.00, 1.00, step=0.01, label="Top P", value=0.95)
                top_k = gr.Slider(0, 1000, step=10, label="Top K", value=1000)
                repetition_penalty = gr.Slider(1.00, 2.00, step=0.05, label="Repetition Penalty", value=1.2)
                
                norm_loudness = gr.Checkbox(value=True, label="Normalize Loudness (-27 LUFS) [During Gen]")
                gr.Markdown("### Post-Processing")
                trim_silence_chk = gr.Checkbox(value=False, label="Trim Leading/Trailing Silence")
                fix_internal_silence_chk = gr.Checkbox(value=False, label="Fix Long Internal Silences")
                remove_unvoiced_chk = gr.Checkbox(value=False, label="Remove Long Unvoiced Segments (Parselmouth)")

    demo.load(fn=load_model, inputs=[], outputs=model_state)
    run_btn.click(
        fn=generate,
        inputs=[
            model_state, text, ref_wav, temp, seed_num, 
            top_p, top_k, repetition_penalty, norm_loudness, 
            sentence_pause_s, trim_silence_chk, fix_internal_silence_chk, remove_unvoiced_chk
        ],
        outputs=[audio_output, time_output],
    )

if __name__ == "__main__":
    demo.queue(max_size=50, default_concurrency_limit=1).launch(share=True)
