import os
import math
from dataclasses import dataclass
from pathlib import Path

import librosa
import torch
import perth
import pyloudnorm as ln

from safetensors.torch import load_file
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

from .models.t3 import T3
from .models.s3tokenizer import S3_SR
from .models.s3gen import S3GEN_SR, S3Gen
from .models.tokenizers import EnTokenizer
from .models.voice_encoder import VoiceEncoder
from .models.t3.modules.cond_enc import T3Cond
from .models.t3.modules.t3_config import T3Config
from .models.s3gen.const import S3GEN_SIL
import logging
logger = logging.getLogger(__name__)

REPO_ID = "ResembleAI/chatterbox-turbo"


def punc_norm(text: str) -> str:
    """
        Quick cleanup func for punctuation from LLMs or
        containing chars not seen often in the dataset
    """
    if len(text) == 0:
        return "You need to add some text for me to talk."

    # Capitalise first letter
    if text[0].islower():
        text = text[0].upper() + text[1:]

    # Remove multiple space chars
    text = " ".join(text.split())

    # Replace uncommon/llm punc
    punc_to_replace = [
        ("…", ", "),
        (":", ","),
        ("—", "-"),
        ("–", "-"),
        (" ,", ","),
        ("“", "\""),
        ("”", "\""),
        ("‘", "'"),
        ("’", "'"),
    ]
    for old_char_sequence, new_char in punc_to_replace:
        text = text.replace(old_char_sequence, new_char)

    # Add full stop if no ending punc
    text = text.rstrip(" ")
    sentence_enders = {".", "!", "?", "-", ","}
    if not any(text.endswith(p) for p in sentence_enders):
        text += "."

    return text


@dataclass
class Conditionals:
    """
    Conditionals for T3 and S3Gen
    - T3 conditionals:
        - speaker_emb
        - clap_emb
        - cond_prompt_speech_tokens
        - cond_prompt_speech_emb
        - emotion_adv
    - S3Gen conditionals:
        - prompt_token
        - prompt_token_len
        - prompt_feat
        - prompt_feat_len
        - embedding
    """
    t3: T3Cond
    gen: dict

    def to(self, device):
        self.t3 = self.t3.to(device=device)
        for k, v in self.gen.items():
            if torch.is_tensor(v):
                self.gen[k] = v.to(device=device)
        return self

    def save(self, fpath: Path):
        arg_dict = dict(
            t3=self.t3.__dict__,
            gen=self.gen
        )
        torch.save(arg_dict, fpath)

    @classmethod
    def load(cls, fpath, map_location="cpu"):
        if isinstance(map_location, str):
            map_location = torch.device(map_location)
        kwargs = torch.load(fpath, map_location=map_location, weights_only=True)
        return cls(T3Cond(**kwargs['t3']), kwargs['gen'])


class ChatterboxTurboTTS:
    ENC_COND_LEN = 15 * S3_SR
    DEC_COND_LEN = 10 * S3GEN_SR

    def __init__(
        self,
        t3: T3,
        s3gen: S3Gen,
        ve: VoiceEncoder,
        tokenizer: EnTokenizer,
        device: str,
        conds: Conditionals = None,
    ):
        self.sr = S3GEN_SR  # sample rate of synthesized audio
        self.t3 = t3
        self.s3gen = s3gen
        self.ve = ve
        self.tokenizer = tokenizer
        self.device = device
        self.conds = conds
        self.watermarker = perth.PerthImplicitWatermarker()

    @classmethod
    def from_local(cls, ckpt_dir, device) -> 'ChatterboxTurboTTS':
        ckpt_dir = Path(ckpt_dir)

        # Always load to CPU first for non-CUDA devices to handle CUDA-saved models
        if device in ["cpu", "mps"]:
            map_location = torch.device('cpu')
        else:
            map_location = None

        ve = VoiceEncoder()
        ve.load_state_dict(
            load_file(ckpt_dir / "ve.safetensors")
        )
        ve.to(device).eval()

        # Turbo specific hp
        hp = T3Config(text_tokens_dict_size=50276)
        hp.llama_config_name = "GPT2_medium"
        hp.speech_tokens_dict_size = 6563
        hp.input_pos_emb = None
        hp.speech_cond_prompt_len = 375
        hp.use_perceiver_resampler = False
        hp.emotion_adv = False

        t3 = T3(hp)
        t3_state = load_file(ckpt_dir / "t3_turbo_v1.safetensors")
        if "model" in t3_state.keys():
            t3_state = t3_state["model"][0]
        t3.load_state_dict(t3_state)
        del t3.tfmr.wte
        t3.to(device).eval()

        s3gen = S3Gen(meanflow=True)
        weights = load_file(ckpt_dir / "s3gen_meanflow.safetensors")
        s3gen.load_state_dict(
            weights, strict=True
        )
        s3gen.to(device).eval()

        tokenizer = AutoTokenizer.from_pretrained(ckpt_dir)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        if len(tokenizer) != 50276:
            print(f"WARNING: Tokenizer len {len(tokenizer)} != 50276")

        conds = None
        builtin_voice = ckpt_dir / "conds.pt"
        if builtin_voice.exists():
            conds = Conditionals.load(builtin_voice, map_location=map_location).to(device)

        return cls(t3, s3gen, ve, tokenizer, device, conds=conds)

    @classmethod
    def from_pretrained(cls, device) -> 'ChatterboxTurboTTS':
        # Check if MPS is available on macOS
        if device == "mps" and not torch.backends.mps.is_available():
            if not torch.backends.mps.is_built():
                print("MPS not available because the current PyTorch install was not built with MPS enabled.")
            else:
                print("MPS not available because the current MacOS version is not 12.3+ and/or you do not have an MPS-enabled device on this machine.")
            device = "cpu"

        local_path = snapshot_download(
            repo_id=REPO_ID,
            token=os.getenv("HF_TOKEN") or True,
            # Optional: Filter to download only what you need
            allow_patterns=["*.safetensors", "*.json", "*.txt", "*.pt", "*.model"]
        )

        return cls.from_local(local_path, device)

    def norm_loudness(self, wav, sr, target_lufs=-27):
        try:
            meter = ln.Meter(sr)
            loudness = meter.integrated_loudness(wav)
            gain_db = target_lufs - loudness
            gain_linear = 10.0 ** (gain_db / 20.0)
            if math.isfinite(gain_linear) and gain_linear > 0.0:
                wav = wav * gain_linear
        except Exception as e:
            print(f"Warning: Error in norm_loudness, skipping: {e}")

        return wav

    def prepare_conditionals(self, wav_fpath, exaggeration=0.5, norm_loudness=True):
        ## Load and norm reference wav
        s3gen_ref_wav, _sr = librosa.load(wav_fpath, sr=S3GEN_SR)

        assert len(s3gen_ref_wav) / _sr > 5.0, "Audio prompt must be longer than 5 seconds!"

        if norm_loudness:
            s3gen_ref_wav = self.norm_loudness(s3gen_ref_wav, _sr)

        ref_16k_wav = librosa.resample(s3gen_ref_wav, orig_sr=S3GEN_SR, target_sr=S3_SR)

        s3gen_ref_wav = s3gen_ref_wav[:self.DEC_COND_LEN]
        s3gen_ref_dict = self.s3gen.embed_ref(s3gen_ref_wav, S3GEN_SR, device=self.device)

        # Speech cond prompt tokens
        if plen := self.t3.hp.speech_cond_prompt_len:
            s3_tokzr = self.s3gen.tokenizer
            t3_cond_prompt_tokens, _ = s3_tokzr.forward([ref_16k_wav[:self.ENC_COND_LEN]], max_len=plen)
            t3_cond_prompt_tokens = torch.atleast_2d(t3_cond_prompt_tokens).to(self.device)

        # Voice-encoder speaker embedding
        ve_embed = torch.from_numpy(self.ve.embeds_from_wavs([ref_16k_wav], sample_rate=S3_SR))
        ve_embed = ve_embed.mean(axis=0, keepdim=True).to(self.device)

        t3_cond = T3Cond(
            speaker_emb=ve_embed,
            cond_prompt_speech_tokens=t3_cond_prompt_tokens,
            emotion_adv=exaggeration * torch.ones(1, 1, 1),
        ).to(device=self.device)
        self.conds = Conditionals(t3_cond, s3gen_ref_dict)
        
    def prepare_conditionals_batch(self, wav_fpaths: list[str], exaggeration=0.5, norm_loudness=True):
        """
        Prepares a batch of conditionals. 
        wav_fpaths: List of file paths (equal to batch size) OR single file path (broadcasted).
        """
        if isinstance(wav_fpaths, str):
            wav_fpaths = [wav_fpaths]

        t3_conds_list = []
        gen_refs_list = []

        for wav_fpath in wav_fpaths:
            # Re-using the single logic to safely process each audio file
            
            ## Load and norm reference wav
            s3gen_ref_wav, _sr = librosa.load(wav_fpath, sr=S3GEN_SR)
            assert len(s3gen_ref_wav) / _sr > 5.0, f"Audio prompt {wav_fpath} must be > 5s!"

            if norm_loudness:
                s3gen_ref_wav = self.norm_loudness(s3gen_ref_wav, _sr)

            ref_16k_wav = librosa.resample(s3gen_ref_wav, orig_sr=S3GEN_SR, target_sr=S3_SR)

            # S3Gen Conditioning (Prompt Feats)
            s3gen_ref_wav_trunc = s3gen_ref_wav[:self.DEC_COND_LEN]
            s3gen_ref_dict = self.s3gen.embed_ref(s3gen_ref_wav_trunc, S3GEN_SR, device=self.device)
            gen_refs_list.append(s3gen_ref_dict)

            # T3 Conditioning
            t3_cond_prompt_tokens = None
            if plen := self.t3.hp.speech_cond_prompt_len:
                s3_tokzr = self.s3gen.tokenizer
                # Note: s3_tokzr.forward usually expects batch, so wrapping in list
                t3_cond_prompt_tokens, _ = s3_tokzr.forward([ref_16k_wav[:self.ENC_COND_LEN]], max_len=plen)
                t3_cond_prompt_tokens = torch.atleast_2d(t3_cond_prompt_tokens).to(self.device)

            ve_embed = torch.from_numpy(self.ve.embeds_from_wavs([ref_16k_wav], sample_rate=S3_SR))
            ve_embed = ve_embed.mean(axis=0, keepdim=True).to(self.device)

            t3_cond_item = T3Cond(
                speaker_emb=ve_embed,
                cond_prompt_speech_tokens=t3_cond_prompt_tokens,
                emotion_adv=exaggeration * torch.ones(1, 1, 1).to(self.device),
            )
            t3_conds_list.append(t3_cond_item)

        # --- Collate Batch ---
        
        # 1. Collate T3 Conds (Stack tensors)
        batched_speaker_emb = torch.cat([c.speaker_emb for c in t3_conds_list], dim=0)
        # Note: Handling None for prompt tokens if disabled in config, but usually enabled for Turbo
        batched_speech_tokens = torch.cat([c.cond_prompt_speech_tokens for c in t3_conds_list], dim=0)
        batched_emotion = torch.cat([c.emotion_adv for c in t3_conds_list], dim=0)
        
        batched_t3_cond = T3Cond(
            speaker_emb=batched_speaker_emb,
            cond_prompt_speech_tokens=batched_speech_tokens,
            emotion_adv=batched_emotion,
        ).to(self.device)
        
        # 2. S3Gen refs remain a list of dicts because S3Gen will run sequentially
        self.conds_batch = {
            "t3": batched_t3_cond,
            "gen_list": gen_refs_list
        }
   def generate_batch(
        self,
        texts: list[str],
        audio_prompt_paths: list[str] = None,
        repetition_penalty=1.2,
        top_p=0.95,
        temperature=0.8,
        top_k=1000,
        exaggeration=0.0,
        norm_loudness=True,
    ):
        batch_size = len(texts)
        
        # ... (Audio Prompt setup remains the same) ...
        if audio_prompt_paths:
            if isinstance(audio_prompt_paths, str):
                audio_prompt_paths = [audio_prompt_paths] * batch_size
            elif len(audio_prompt_paths) == 1:
                audio_prompt_paths = audio_prompt_paths * batch_size
            self.prepare_conditionals_batch(audio_prompt_paths, exaggeration=exaggeration, norm_loudness=norm_loudness)
        else:
             assert hasattr(self, 'conds_batch'), "Please call prepare_conditionals_batch first"

        t3_conds = self.conds_batch["t3"]
        s3gen_refs = self.conds_batch["gen_list"]

        # --- Tokenize with Padding & Attention Mask ---
        cleaned_texts = [punc_norm(t) for t in texts]
        
        # returns 'input_ids' AND 'attention_mask'
        tokenized_output = self.tokenizer(cleaned_texts, return_tensors="pt", padding=True, truncation=True)
        
        text_tokens = tokenized_output.input_ids.to(self.device)
        attention_mask = tokenized_output.attention_mask.to(self.device)

        # --- Batch Inference (Pass Mask) ---
        speech_tokens_batch = self.t3.inference_turbo(
            t3_cond=t3_conds,
            text_tokens=text_tokens,
            attention_mask=attention_mask,  # <--- NEW
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
        )

        # --- Sequential S3Gen & Cleanup ---
        generated_wavs = []
        silence = torch.tensor([S3GEN_SIL, S3GEN_SIL, S3GEN_SIL]).long().to(self.device)
        stop_token = self.t3.hp.stop_speech_token

        for i in range(batch_size):
            sequence = speech_tokens_batch[i]
            
            # --- Cleanup Logic (Solves Garbage) ---
            eot_indices = (sequence == stop_token).nonzero(as_tuple=True)[0]
            if len(eot_indices) > 0:
                cutoff = eot_indices[0]
                sequence = sequence[:cutoff]
            
            sequence = sequence[sequence < 6561] # Remove OOVs
            sequence = torch.cat([sequence, silence])
            
            if len(sequence) <= 3:
                generated_wavs.append(torch.zeros(1, 16000))
                continue

            # Sequential Vocoder (Safe)
            sequence = sequence.unsqueeze(0)
            wav, _ = self.s3gen.inference(
                speech_tokens=sequence,
                ref_dict=s3gen_refs[i],
                n_cfm_timesteps=2,
            )
            wav = wav.squeeze(0).detach().cpu().numpy()
            watermarked_wav = self.watermarker.apply_watermark(wav, sample_rate=self.sr)
            generated_wavs.append(torch.from_numpy(watermarked_wav).unsqueeze(0))

        return generated_wavs
