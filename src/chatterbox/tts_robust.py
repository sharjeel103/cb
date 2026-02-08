import re
import os
import random
import logging
import tempfile
import gc
import uuid
from collections import defaultdict
from typing import List, Optional, Tuple, Union, Dict

import numpy as np
import torch
import torch.nn.functional as F
import librosa
import soundfile as sf
import parselmouth
from scipy.signal import butter, filtfilt

from .tts import ChatterboxTTS

# ==========================================
#        CONSTANTS & CONFIGURATION
# ==========================================
CROSSFADE_MS = 20           
ENABLE_DC_REMOVAL = True    
DC_HIGHPASS_HZ = 15         
PEAK_NORMALIZE_TARGET = 0.95
MAX_REF_DURATION_SEC = 30   
CHUNK_SIZE_DEFAULT = 275    
SAFE_BATCH_SIZE = 4         

# ==========================================
#      TEXT PROCESSING UTILS
# ==========================================
ABBREVIATIONS = {
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
    sentences = []
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
    
    HARD_LIMIT = chunk_size + 50
    segments = _preprocess_and_segment_text(full_text)
    if not segments: return []
    
    chunks = []
    curr_sents = []
    curr_len = 0
    
    for _, seg_text in segments:
        slen = len(seg_text)
        
        # Hard chop for extremely long sentences
        if slen > HARD_LIMIT:
            if curr_sents:
                chunks.append(" ".join(curr_sents))
                curr_sents = []
                curr_len = 0
            
            for i in range(0, slen, chunk_size):
                chunks.append(seg_text[i : i + chunk_size])
            continue

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
#      AUDIO PROCESSING UTILS
# ==========================================

def trim_lead_trail_silence(audio_array: np.ndarray, sample_rate: int, silence_thresh_db: float = -40.0, padding_ms: int = 50) -> np.ndarray:
    if audio_array is None or audio_array.size == 0: return audio_array
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
    if audio_array is None or audio_array.size == 0: return audio_array
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
    if audio_array is None or audio_array.size == 0: return audio_array
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
    try:
        nyquist = sample_rate / 2
        b, a = butter(2, cutoff_hz / nyquist, btype="high")
        return filtfilt(b, a, audio).astype(np.float32)
    except Exception:
        return audio

def stitch_audio_segments(audio_segments: List[np.ndarray], engine_sr: int, sentence_pause_s: float) -> Optional[np.ndarray]:
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

def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)


class RobustChatterboxTTS(ChatterboxTTS):
    """
    Enhanced TTS wrapper with:
    1. Outer batching (processing multiple unrelated requests).
    2. Internal batching (filling VRAM by flattening all text chunks).
    3. Robust OOM recovery.
    4. Post-processing tools.
    """

    def prepare_reference_robust(self, audio_path: str) -> str:
        """Checks duration of reference audio and crops if necessary."""
        if not audio_path or not os.path.exists(audio_path):
            return audio_path
        
        try:
            info = sf.info(audio_path)
            duration = info.duration
            sr = info.samplerate

            if duration <= MAX_REF_DURATION_SEC:
                return audio_path

            # Crop if too long
            print(f"Reference audio {duration:.1f}s > {MAX_REF_DURATION_SEC}s. Cropping...")
            data, sr = librosa.load(audio_path, sr=None)

            max_samples = int(MAX_REF_DURATION_SEC * sr)
            data_cropped = data[:max_samples]
            
            # Save to temp file
            temp_dir = tempfile.gettempdir()
            temp_path = os.path.join(temp_dir, f"cropped_{uuid.uuid4().hex}.wav")
            sf.write(temp_path, data_cropped, sr)
            return temp_path
            
        except Exception as e:
            print(f"Error checking reference: {e}")
            return audio_path

    def generate_batch_robust(
        self,
        texts: List[str],
        audio_prompts: Union[str, List[str]] = None,
        # Generation Params
        exaggeration: float = 0.5,
        cfg_weight: float = 0.0,
        temperature: float = 0.8,
        seed_num: int = 0,
        top_p: float = 1.0,
        top_k: int = 1000,
        min_p: float = 0.05,
        repetition_penalty: float = 1.2,
        # Robust/Post-proc params
        sentence_pause_s: float = 0.2,
        norm_loudness: bool = True,
        trim_silence: bool = False,
        fix_int_silence: bool = False,
        remove_unvoiced: bool = False,
        target_batch_size: int = 12
    ) -> List[np.ndarray]:
        """
        Process multiple input texts, breaking them into chunks, and filling GPU batches efficiently.
        
        Args:
            texts: List of full text strings to synthesize.
            audio_prompts: Single path (str) applied to all, or List[str] matching len(texts).
            
        Returns:
            List[np.ndarray]: One audio array per input text (at model.sr).
        """
        if seed_num != 0:
            set_seed(int(seed_num))

        num_requests = len(texts)
        if num_requests == 0:
            return []

        # 1. Handle Reference Audio Paths
        if isinstance(audio_prompts, str) or audio_prompts is None:
            ref_paths = [audio_prompts] * num_requests
        else:
            if len(audio_prompts) != num_requests:
                raise ValueError(f"Mismatch: {len(texts)} texts provided but {len(audio_prompts)} reference paths.")
            ref_paths = audio_prompts
        
        # Pre-process references (cropping)
        final_ref_paths = [self.prepare_reference_robust(p) for p in ref_paths]

        # 2. Phase 1: Flatten everything into chunks (Internal Batching Prep)
        # We need to map: Flat Batch Index -> (Original Request Index, Stitch Order Index)
        flat_batch_items = []  # List of dicts: {'text': chunk_str, 'ref': path, 'req_idx': int, 'chunk_idx': int}
        
        for req_idx, (text, ref_path) in enumerate(zip(texts, final_ref_paths)):
            cleaned_text = text.replace('",', '" ').replace('",', '" ')
            chunks = chunk_text_by_sentences(cleaned_text, chunk_size=CHUNK_SIZE_DEFAULT)
            
            if not chunks:
                # Handle empty text case (store a placeholder/None or skip)
                continue
                
            for chunk_idx, chunk_text in enumerate(chunks):
                flat_batch_items.append({
                    'text': chunk_text,
                    'ref': ref_path,
                    'req_idx': req_idx,
                    'chunk_idx': chunk_idx
                })

        total_flat_items = len(flat_batch_items)
        if total_flat_items == 0:
            return [np.zeros(16000, dtype=np.float32) for _ in range(num_requests)]

        # 3. Phase 2: Process Flat List in Optimal GPU Batches
        # Adjust batch size based on CFG
        if cfg_weight > 0.0:
            current_target_batch = target_batch_size 
        else:
            current_target_batch = int(target_batch_size * 1.5)

        current_batch_size = current_target_batch
        flat_idx = 0
        
        # Store results: map[req_idx] -> list of (chunk_idx, audio_np)
        generated_chunks_map = defaultdict(list)
        engine_sr = self.sr # Default if not set yet

        with torch.inference_mode():
            while flat_idx < total_flat_items:
                end_idx = min(flat_idx + current_batch_size, total_flat_items)
                batch_slice = flat_batch_items[flat_idx : end_idx]
                
                # Extract lists for the model
                batch_texts_in = [item['text'] for item in batch_slice]
                batch_refs_in = [item['ref'] for item in batch_slice]
                
                try:
                    # CALL BASE MODEL
                    wav_tensors = self.generate_batch(
                        batch_texts_in,
                        audio_prompt_paths=batch_refs_in,
                        exaggeration=exaggeration,
                        cfg_weight=cfg_weight,
                        temperature=temperature,
                        top_p=top_p,
                        min_p=min_p,
                        repetition_penalty=repetition_penalty,
                    )
                    
                    if hasattr(self, 'sr'): engine_sr = self.sr
                    
                    # Distribute results back to map
                    for i, wav_tensor in enumerate(wav_tensors):
                        audio_np = wav_tensor.cpu().numpy().squeeze()
                        meta = batch_slice[i]
                        generated_chunks_map[meta['req_idx']].append(
                            (meta['chunk_idx'], audio_np)
                        )
                        del wav_tensor
                    
                    # Advance
                    flat_idx += len(batch_slice)
                    
                    # Restore batch size if it was lowered by OOM
                    if current_batch_size < current_target_batch:
                         current_batch_size = current_target_batch
                    
                    del wav_tensors
                    torch.cuda.empty_cache()

                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        print(f"⚠️ OOM DETECTED at batch size {current_batch_size}!")
                        gc.collect()
                        torch.cuda.empty_cache()
                        
                        if current_batch_size > SAFE_BATCH_SIZE:
                            print(f"♻️ Retrying with Safe Batch Size ({SAFE_BATCH_SIZE})...")
                            current_batch_size = SAFE_BATCH_SIZE
                        elif current_batch_size > 1:
                            print("♻️ Safe Batch Failed. Retrying with Single Processing (Size 1)...")
                            current_batch_size = 1
                        else:
                            print("❌ OOM even at batch size 1. Skipping this chunk.")
                            flat_idx += 1 
                    else:
                        raise e

        # 4. Phase 3: Stitching & Post-Processing per Request
        final_outputs = []
        
        for i in range(num_requests):
            # Get chunks for this request
            chunk_list = generated_chunks_map.get(i, [])
            
            if not chunk_list:
                # Fallback silence
                final_outputs.append(np.zeros(int(engine_sr), dtype=np.float32))
                continue
                
            # Sort by chunk_idx to ensure correct order
            chunk_list.sort(key=lambda x: x[0])
            audio_segments = [x[1] for x in chunk_list]
            
            # Stitch
            full_audio = stitch_audio_segments(audio_segments, engine_sr, sentence_pause_s)
            
            if full_audio is None:
                full_audio = np.zeros(int(engine_sr), dtype=np.float32)
                
            full_audio = full_audio.astype(np.float32)

            # Post-Process
            if trim_silence:
                full_audio = trim_lead_trail_silence(full_audio, engine_sr)
            if fix_int_silence:
                full_audio = fix_internal_silence(full_audio, engine_sr)
            if remove_unvoiced:
                full_audio = remove_long_unvoiced(full_audio, engine_sr)
            
            if norm_loudness:
                peak = np.abs(full_audio).max()
                if peak > 0:
                    full_audio = full_audio / peak * PEAK_NORMALIZE_TARGET
            
            final_outputs.append(full_audio)

        return final_outputs

    def generate_robust(self, *args, **kwargs):
        """Wrapper for single-text backward compatibility."""
        # This just calls generate_batch_robust with a list of 1
        if 'text' in kwargs:
            text = kwargs.pop('text')
        else:
            text = args[0]
            args = args[1:]
        
        audio_prompt_path = kwargs.get('audio_prompt_path', None)
        # Handle positional arg for prompt if present
        if len(args) > 0:
            audio_prompt_path = args[0]

        res = self.generate_batch_robust(
            texts=[text], 
            audio_prompts=audio_prompt_path, 
            **kwargs
        )
        return self.sr, res[0]
