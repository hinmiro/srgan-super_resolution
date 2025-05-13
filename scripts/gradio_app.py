import gradio as gr
import numpy as np
from models.model import build_generator


def super_resolve_large_image(model, input_img, patch_size=256, overlap=32):
    """
    Super-resolve a large image by tiling it into patches.
    Args:
        model: Trained super-resolution model.
        input_img: NumPy array, shape (H, W, 3), float32 in [0,1].
        patch_size: Size of each patch to process.
        overlap: Overlap between patches to avoid seams.
    Returns:
        Super-resolved image as NumPy array.
    """
    h, w, c = input_img.shape
    scale = 4  # Change if your model uses a different scale

    out_h, out_w = h * scale, w * scale
    output_img = np.zeros((out_h, out_w, c), dtype=np.float32)
    weight_mask = np.zeros((out_h, out_w, c), dtype=np.float32)

    stride = patch_size - overlap
    for y in range(0, h, stride):
        for x in range(0, w, stride):
            patch = input_img[y : y + patch_size, x : x + patch_size, :]
            patch_h, patch_w = patch.shape[:2]
            patch_input = np.expand_dims(patch, axis=0)
            sr_patch = model.predict(patch_input)[0]
            y1, x1 = y * scale, x * scale
            y2, x2 = y1 + patch_h * scale, x1 + patch_w * scale
            output_img[y1:y2, x1:x2, :] += sr_patch[
                : patch_h * scale, : patch_w * scale, :
            ]
            weight_mask[y1:y2, x1:x2, :] += 1.0

    # Avoid division by zero
    output_img /= np.maximum(weight_mask, 1e-8)
    return np.clip(output_img, 0, 1)


def superRes(input_img, generator):
    img_array = input_img.astype(np.float32) / 255.0
    sr_img = super_resolve_large_image(generator, img_array, patch_size=256, overlap=32)
    return (sr_img * 255).astype(np.uint8)


def launch_gradio(generator):
    demo = gr.Interface(
        fn=lambda img: superRes(img, generator),
        inputs=gr.Image(type="numpy"),
        outputs="image",
    )
    demo.launch(share=True)
