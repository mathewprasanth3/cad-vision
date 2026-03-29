"""
app/app.py

CADVision Gradio demo.
Accepts an STL file, renders 4 views, predicts machining feature,
generates Grad-CAM heatmap overlays.

Mirrors the role of app.py (FastAPI) in CNC Fault Detector
but uses Gradio web UI instead of REST API.

Usage:
    uv run python app/app.py
"""

import sys
import torch
sys.path.append(".")

import gradio as gr
from src.utils.infer import CADVisionPredictor, render_stl, TRANSFORM
from src.utils.gradcam import GradCAM


# load model once at startup — same pattern as CNC Fault Detector
predictor = CADVisionPredictor()
gcam = GradCAM(predictor.model)


def predict(stl_file):
    """
    Main prediction function called by Gradio on every upload.
    Takes STL file path, returns prediction text + 4 heatmap images.
    """
    if stl_file is None:
        return "please upload an STL file", None, None, None, None

    try:
        # render 4 views from STL
        pil_images = render_stl(stl_file)

        # prepare tensor for model
        tensors = [TRANSFORM(img) for img in pil_images]
        views = torch.stack(tensors).unsqueeze(0).to(predictor.device)

        # run inference
        result = predictor.predict(stl_file)
        predicted_class = result["predicted_class"]
        confidence = result["confidence"]

        # generate grad-cam heatmaps
        predicted_idx = list(range(24))[
            [r["class"] for r in result["top3"]].index(predicted_class)
            if predicted_class in [r["class"] for r in result["top3"]]
            else 0
        ]
        heatmaps = gcam.generate(views, predicted_idx, pil_images)

        # format output text
        output_text = f"Predicted Feature:  {predicted_class.replace('_', ' ').title()}\n"
        output_text += f"Confidence:         {confidence * 100:.2f}%\n\n"
        output_text += "Top 3 Predictions:\n"
        for i, pred in enumerate(result["top3"], 1):
            output_text += f"  {i}. {pred['class'].replace('_', ' ').title():<30} {pred['confidence']*100:.2f}%\n"

        return (
            output_text,
            heatmaps[0],  # front
            heatmaps[1],  # side
            heatmaps[2],  # top
            heatmaps[3],  # isometric
        )

    except Exception as e:
        return f"error: {str(e)}", None, None, None, None


# build Gradio interface
with gr.Blocks(title="CADVision") as demo:

    # header row — title left, name right
    with gr.Row():
        with gr.Column(scale=3):
            gr.Markdown("# CADVision")
            gr.Markdown(
                "Multi-view CNN for automated machining feature recognition. "
                "Upload an STL file of a mechanical part to identify its machining feature."
            )
        with gr.Column(scale=1):
            gr.Markdown(
                "<div style='text-align: right;'>"
                "<strong>Mathew Prasanth, PE</strong><br>AI/ML Engineer"
                "</div>"
            )

    gr.Markdown(
        "> **Note:** Model trained on single-feature parts. "
        "For best results upload a part with one dominant machining feature."
    )

    with gr.Row():
        with gr.Column(scale=1):
            stl_input = gr.File(
                label="Upload STL file",
                file_types=[".stl", ".STL"],
            )
            predict_btn = gr.Button("Predict", variant="primary")

        with gr.Column(scale=2):
            output_text = gr.Textbox(
                label="Prediction",
                lines=6,
            )

    gr.Markdown("### Grad-CAM Heatmaps — where the model focused")
    gr.Markdown("Red/yellow = model focused here. Blue = model ignored.")

    with gr.Row():
        front_img = gr.Image(label="Front view")
        side_img = gr.Image(label="Side view")
        top_img = gr.Image(label="Top view")
        iso_img = gr.Image(label="Isometric view")

    predict_btn.click(
        fn=predict,
        inputs=[stl_input],
        outputs=[output_text, front_img, side_img, top_img, iso_img],
    )

    gr.Markdown("---")
    gr.Markdown(
        "**Supported features:** O-ring, Through Hole, Blind Hole, "
        "Triangular/Rectangular Passage, Circular/Triangular/Rectangular Through Slot, "
        "Rectangular Blind Slot, Triangular/Rectangular/Circular End Pocket, "
        "Triangular/Circular/Rectangular Blind Step, Rectangular Through Step, "
        "Two Sides Through Step, Slanted Through Step, Chamfer, Round, "
        "V/H Circular End Blind Slot, Six Sides Passage, Six Sides Pocket"
    )


if __name__ == "__main__":
    demo.launch()