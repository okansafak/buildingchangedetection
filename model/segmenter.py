"""
Building footprint segmentation with the ChangeStar ViT-B ONNX export
(geobase/changestar-building-segmentation-vitb). Large images are processed
as overlapping 1024 px tiles blended with a feathered weight.

On an NVIDIA GPU (onnxruntime-gpu with CUDA) the fp32 model runs ~15x faster
than the int8 model on the CPU with the same accuracy; anything else, or a
CUDA failure while running (4 GB laptop GPUs can run out of memory), falls
back to the CPU int8 model. ATLAS_DEVICE=cpu forces the CPU.
"""
import logging
import os

import numpy as np

MODEL_REPO = "geobase/changestar-building-segmentation-vitb"
MODEL_FILE = "onnx/model_quantized.onnx"   # int8, for the CPU
MODEL_FILE_GPU = "onnx/model.onnx"         # fp32, for CUDA (int8 ops are slow or unsupported on GPUs)
MODEL_REVISION = "8a6d7676d2fc9ea787b8786f2f05a65989c2606c"
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")

TILE = 1024     # the exported ViT has fixed positional embeddings: tiles must be exactly 1024 px
OVERLAP = 64
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class ModelUnavailable(RuntimeError):
    """The ONNX model could not be downloaded or loaded."""


def tile_origins(length, tile=TILE, overlap=OVERLAP):
    """Tile start offsets covering [0, length); the last tile is aligned to the end."""
    if length <= tile:
        return [0]
    starts = list(range(0, length - tile, tile - overlap))
    starts.append(length - tile)
    return starts


def _feather(size, overlap):
    ramp = np.ones(size, dtype=np.float32)
    edge = np.linspace(0.05, 1.0, overlap, dtype=np.float32)
    ramp[:overlap] = edge
    ramp[-overlap:] = edge[::-1]
    return np.outer(ramp, ramp)


log = logging.getLogger(__name__)


def _model_path(fname, model_dir):
    from huggingface_hub import hf_hub_download
    return hf_hub_download(MODEL_REPO, fname, revision=MODEL_REVISION, cache_dir=model_dir)


class BuildingSegmenter:
    def __init__(self, session=None, model_dir=MODEL_DIR):
        self._session = session
        self.model_dir = model_dir
        self.device = "cpu" if session is not None else None  # "cuda" | "cpu" once a session is open

    def _open_cuda_session(self):
        """fp32 model on CUDA, or None when onnxruntime-gpu / an NVIDIA GPU is not available."""
        try:
            import onnxruntime as ort
            if "CUDAExecutionProvider" not in ort.get_available_providers():
                return None
            if hasattr(ort, "preload_dlls"):
                ort.preload_dlls()  # CUDA / cuDNN DLLs installed by pip (onnxruntime-gpu[cuda,cudnn])
            options = ort.SessionOptions()
            options.enable_mem_pattern = False  # with kSameAsRequested: steady 0.6 s/tile on a 4 GB GPU
            session = ort.InferenceSession(
                _model_path(MODEL_FILE_GPU, self.model_dir), sess_options=options,
                providers=[("CUDAExecutionProvider", {"arena_extend_strategy": "kSameAsRequested"}), "CPUExecutionProvider"],
            )
            return session if session.get_providers()[0] == "CUDAExecutionProvider" else None
        except Exception as e:
            log.warning("CUDA kullanılamıyor, CPU'ya geçiliyor: %s", e)
            return None

    def _open_cpu_session(self):
        try:
            import onnxruntime as ort
            return ort.InferenceSession(_model_path(MODEL_FILE, self.model_dir), providers=["CPUExecutionProvider"])
        except Exception as e:
            raise ModelUnavailable(str(e)) from e

    def _get_session(self):
        if self._session is None:
            session = None
            if os.environ.get("ATLAS_DEVICE", "").lower() != "cpu":
                session = self._open_cuda_session()
            self.device = "cuda" if session is not None else "cpu"
            self._session = session if session is not None else self._open_cpu_session()
        return self._session

    def count_tiles(self, h, w):
        return len(tile_origins(h)) * len(tile_origins(w))

    def predict(self, rgb, progress=None):
        """Building probability (H x W float32 in [0, 1]) for an H x W x 3 uint8 RGB image.
        progress(done_tiles, total_tiles) is called after every tile."""
        session = self._get_session()
        try:
            return self._predict(session, rgb, progress)
        except Exception as e:
            if self.device != "cuda":
                raise
            log.warning("CUDA çalışırken hata verdi, görüntü CPU'da yeniden işleniyor: %s", e)
            self._session = self._open_cpu_session()
            self.device = "cpu"
            return self._predict(self._session, rgb, progress)

    def _predict(self, session, rgb, progress):
        input_name = session.get_inputs()[0].name
        h, w = rgb.shape[:2]
        accum = np.zeros((h, w), dtype=np.float32)
        weight = np.zeros((h, w), dtype=np.float32)
        feather = _feather(TILE, OVERLAP)
        ys, xs = tile_origins(h), tile_origins(w)
        total, done = len(ys) * len(xs), 0
        for y in ys:
            for x in xs:
                tile = rgb[y:y + TILE, x:x + TILE].astype(np.float32) / 255.0
                th, tw = tile.shape[:2]
                if th < TILE or tw < TILE:
                    tile = np.pad(tile, ((0, TILE - th), (0, TILE - tw), (0, 0)), mode="reflect")
                batch = ((tile - _MEAN) / _STD).transpose(2, 0, 1)[None].astype(np.float32)
                prob = np.asarray(session.run(None, {input_name: batch})[0]).reshape(TILE, TILE)[:th, :tw]
                accum[y:y + th, x:x + tw] += prob * feather[:th, :tw]
                weight[y:y + th, x:x + tw] += feather[:th, :tw]
                done += 1
                if progress:
                    progress(done, total)
        return accum / np.maximum(weight, 1e-6)
