"""Reusable GPU (NVDEC) video frame reader for the VLINCS workspace.

Cross-domain utility: embed / detect / track / viz all decode video. The batch CPU re-embed
stalled on cv2 CPU decode with no thread caps (174-core thrash, GPU idle). This module decodes on
the GPU via NVIDIA NVDEC and returns frames row-aligned to requested frame indices, so a re-embed
(or any frame-indexed crop job) is GPU-bound rather than CPU-bound.

Primary backend: PyNvVideoCodec.SimpleDecoder (NVIDIA official, NVDEC -> CUDA frames -> RGB).
Designed for the embed_t02_boxes.py access pattern: "give me frames at this set of indices, in
order, decoded once." Returns HxWx3 uint8 RGB numpy arrays (RGB, NOT BGR — callers that cropped
from cv2 BGR should account for this; embed_t02 fed PIL RGB anyway via frame[...,::-1]).

Guarded against the known NVDEC-segfault history: a one-call self-test (`gpu_decode_available`)
decodes a couple of frames from a probe video and verifies non-degenerate output BEFORE any real
job commits to the GPU path. If it raises/segfaults at import or self-test, callers fall back to
thread-capped CPU cv2.

API
---
    reader = NvdecFrameReader(video_path, gpu_id=0)        # raises if backend unavailable
    n      = reader.num_frames                              # total decodable frames (best-effort)
    for frame_idx, rgb in reader.iter_indices(sorted_idx):  # rgb: HxWx3 uint8 RGB
        ...
    reader.close()

`iter_indices(indices, batch=64)` yields (frame_idx, rgb) for each requested index, decoding in
ascending-index batches via SimpleDecoder.get_batch_frames_by_index. Indices need not be
contiguous; they are sorted internally and yielded in ascending order.
"""
from __future__ import annotations
from typing import Iterable, Iterator, Tuple
import numpy as np


class NvdecFrameReader:
    """NVDEC-backed frame reader over PyNvVideoCodec.SimpleDecoder, returning RGB uint8 numpy frames."""

    def __init__(self, video_path: str, gpu_id: int = 0, decoder_cache_size: int = 4):
        import PyNvVideoCodec as nvc  # raises ImportError if unavailable
        self._nvc = nvc
        # NATIVE output (NV12 on device); we convert per-frame with nv12_to_rgb -> host numpy.
        self._dec = nvc.SimpleDecoder(
            video_path, gpu_id=gpu_id, use_device_memory=True,
            decoder_cache_size=decoder_cache_size,
            output_color_type=nvc.OutputColorType.RGB,
        )
        try:
            self._n = len(self._dec)
        except Exception:
            md = self._dec.get_stream_metadata()
            self._n = int(getattr(md, "num_frames", 0) or 0)

    @property
    def num_frames(self) -> int:
        return self._n

    @staticmethod
    def _to_rgb_numpy(frame) -> np.ndarray:
        """Convert a DecodedFrame (RGB output_color_type, device memory) to HxWx3 uint8 RGB host array."""
        # Frame exposes CUDA-array-interface via .cuda(); move to torch then host. RGB interleaved.
        import torch
        t = torch.from_dlpack(frame)  # zero-copy device tensor (PyNvVideoCodec frames are DLPack-capable)
        arr = t.detach().to("cpu").numpy()
        # RGB output is typically (H, W, 3) interleaved already; squeeze any leading singleton.
        arr = np.asarray(arr)
        if arr.ndim == 3 and arr.shape[0] in (1,) and arr.shape[-1] != 3:
            arr = arr[0]
        return np.ascontiguousarray(arr.astype(np.uint8))

    def iter_indices(self, indices: Iterable[int], batch: int = 64) -> Iterator[Tuple[int, np.ndarray]]:
        idx = sorted(int(i) for i in indices)
        if not idx:
            return
        for s in range(0, len(idx), batch):
            chunk = idx[s:s + batch]
            frames = self._dec.get_batch_frames_by_index(chunk)
            for fi, fr in zip(chunk, frames):
                yield fi, self._to_rgb_numpy(fr)

    def close(self):
        try:
            self._dec.stop()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()


def gpu_decode_available(probe_video: str, gpu_id: int = 0) -> tuple[bool, str]:
    """Self-test the NVDEC path on a real video; guards the NVDEC-segfault history.

    Decodes the first few frames + a mid frame and verifies non-degenerate (nonzero, correct ndim)
    output. Returns (ok, detail). Never raises — any failure (ImportError, CUDA error, bad frame)
    returns (False, reason) so the caller can fall back to CPU.
    """
    try:
        r = NvdecFrameReader(probe_video, gpu_id=gpu_id)
        n = r.num_frames
        probe_idx = [0, 1, max(0, n // 2)] if n > 2 else [0]
        got = {}
        for fi, rgb in r.iter_indices(probe_idx, batch=8):
            got[fi] = rgb
        r.close()
        if not got:
            return False, "no frames returned"
        for fi, rgb in got.items():
            if rgb.ndim != 3 or rgb.shape[-1] != 3:
                return False, f"frame {fi} bad shape {rgb.shape}"
            if int(rgb.max()) == 0:
                return False, f"frame {fi} all-zero (degenerate decode)"
        h, w = next(iter(got.values())).shape[:2]
        return True, f"ok: {len(got)} probe frames, {w}x{h}, num_frames~{n}"
    except BaseException as e:  # BaseException to also trap unusual native errors short of segfault
        return False, f"{type(e).__name__}: {e}"
