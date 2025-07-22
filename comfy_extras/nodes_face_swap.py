import os
import torch
import numpy as np
import cv2
import insightface
from insightface.app import FaceAnalysis

class FaceSwap:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source_image": ("IMAGE",),
                "target_image": ("IMAGE",),
                "model_path": ("STRING", {"default": "models/faceswap/inswapper_128.onnx"})
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "swap"
    CATEGORY = "image/face"

    def swap(self, source_image: torch.Tensor, target_image: torch.Tensor, model_path: str):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Face swap model not found at {model_path}. Download 'inswapper_128.onnx' and place it there.")

        ctx_id = -1
        app = FaceAnalysis(name="buffalo_l")
        app.prepare(ctx_id=ctx_id, det_size=(640, 640))
        swapper = insightface.model_zoo.get_model(model_path, download=False)

        batch_size = min(source_image.shape[0], target_image.shape[0])
        outputs = []
        for i in range(batch_size):
            src = (source_image[i].cpu().numpy() * 255).astype(np.uint8)
            tgt = (target_image[i].cpu().numpy() * 255).astype(np.uint8)
            src_bgr = cv2.cvtColor(src, cv2.COLOR_RGB2BGR)
            tgt_bgr = cv2.cvtColor(tgt, cv2.COLOR_RGB2BGR)
            src_faces = app.get(src_bgr)
            tgt_faces = app.get(tgt_bgr)
            if len(src_faces) == 0:
                raise RuntimeError("No face detected in source image.")
            if len(tgt_faces) == 0:
                raise RuntimeError("No face detected in target image.")
            src_face = src_faces[0]
            result = tgt_bgr.copy()
            for face in tgt_faces:
                result = swapper.get(result, face, src_face, paste_back=True)
            result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
            outputs.append(torch.from_numpy(result.astype(np.float32) / 255.0))
        return (torch.stack(outputs, dim=0),)


NODE_CLASS_MAPPINGS = {
    "FaceSwap": FaceSwap,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FaceSwap": "Face Swap",
}
