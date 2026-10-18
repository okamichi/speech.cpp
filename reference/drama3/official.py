"""The official Drama 3 as fish-speech's server runs it, on the CPU in float32: the model, the codec, and the engine
that turns a request into audio."""

import queue
import threading
import traceback
from unittest import mock

import torch

from fish_speech.inference_engine import TTSInferenceEngine
from fish_speech.models.text2semantic.fish_omni import model as fish_omni_model
from fish_speech.models.text2semantic.fish_omni.model import FishOmniModel
from fish_speech.models.text2semantic.inference import (
    GenerateRequest,
    WrappedGenerateResponse,
    generate_long,
    load_codec_model,
)

import pins

DTYPE = torch.float32


def load_model(folder):
    """The model as FishOmniModel.from_pretrained loads it, on the CPU in float32 with its experts kept in FP8.

    from_pretrained keeps the checkpoint's FP8 experts only where it finds SM89+ CUDA devices, and otherwise
    dequantizes all 256 experts of every layer as it loads them, 128 GB in float32. Told that the CPU qualifies, it
    keeps them in FP8 (32.2 GB), and MoEExperts, which runs its Triton kernel only on CUDA tensors, dequantizes the
    experts each step selects to float32 with the same dequantize_fp8_blocks() the load would have used, so the
    values are the same.
    """
    with mock.patch.object(fish_omni_model, "fp8_gemm_supported", lambda devices: True):
        return FishOmniModel.from_pretrained(folder, devices=["cpu"], tower_device="cpu", dtype=DTYPE)


def load_codec(folder):
    """The codec as fish-speech's server loads it for Drama 3, on the CPU in float32."""
    return load_codec_model(f"{folder}/codec.pth", "cpu", DTYPE)


def serve(model):
    """The queue fish-speech's launch_thread_safe_queue() serves, with `model` in place of the one init_model() loads
    on CUDA: each request runs generate_long() and puts every response it yields, or its error, on its own queue."""
    requests = queue.Queue()

    def worker():
        while (item := requests.get()) is not None:
            try:
                for response in generate_long(model=model, decode_one_token=None, **item.request):
                    item.response_queue.put(WrappedGenerateResponse(status="success", response=response))
            except Exception as error:
                traceback.print_exc()
                item.response_queue.put(WrappedGenerateResponse(status="error", response=error))

    threading.Thread(target=worker, daemon=True).start()
    return requests


def engine():
    """fish-speech's TTSInferenceEngine over the pinned checkpoint, as tools/api_server.py builds it for Drama 3."""
    folder = pins.snapshot()
    torch.set_grad_enabled(False)
    model = load_model(folder)
    return model, TTSInferenceEngine(
        llama_queue=serve(model), decoder_model=load_codec(folder), precision=DTYPE, compile=False, model_family="drama3")
