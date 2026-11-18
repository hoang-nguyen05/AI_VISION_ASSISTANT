import os
import sys
from pathlib import Path
import gradio as gr

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.components import create_model_selector, get_available_adapters
from app.callbacks import handle_vqa, handle_caption, handle_compare

CUSTOM_CSS = """
.main-title {
    text-align: center;
    color: #1e3a8a;
    font-weight: 800;
    margin-bottom: 0.2rem;
}
.sub-title {
    text-align: center;
    color: #4b5563;
    margin-bottom: 1.5rem;
    font-size: 1.05rem;
}
.metric-box {
    background-color: #f3f4f6;
    padding: 0.5rem;
    border-radius: 0.5rem;
    border: 1px solid #e5e7eb;
}
"""

def create_app():
    with gr.Blocks(title="AI Vision Assistant - VQA & Captioning Tiếng Việt") as demo:
        gr.Markdown(
            """
            # 👁️ AI VISION ASSISTANT
            ### Hệ thống Mô tả Hình ảnh & Hỏi đáp Thị giác Tiếng Việt Đa Phương thức (Multimodal VQA)
            """
        )

        with gr.Tabs() as tabs:
            # ================= TAB 1: VQA =================
            with gr.TabItem("❓ Hỏi đáp Thị giác (VQA)"):
                with gr.Row():
                    with gr.Column(scale=5):
                        vqa_image = gr.Image(type="pil", label="Tải ảnh lên (JPG / PNG / WebP)")
                        vqa_question = gr.Textbox(
                            label="Đặt câu hỏi về bức ảnh",
                            placeholder="Ví dụ: Trên bàn có bao nhiêu quả táo và chúng màu gì?",
                            lines=2
                        )
                        vqa_submit_btn = gr.Button("🚀 Phân tích & Trả lời", variant="primary")

                    with gr.Column(scale=5):
                        vqa_model, vqa_use_adapter, vqa_adapter = create_model_selector()
                        vqa_output = gr.Textbox(label="Câu trả lời từ AI", lines=4, interactive=False)
                        with gr.Row():
                            vqa_latency = gr.Textbox(label="⏱️ Độ trễ (Latency)", value="0.0s", interactive=False)
                            vqa_vram = gr.Textbox(label="💾 Peak VRAM", value="0.0 MB", interactive=False)

                vqa_submit_btn.click(
                    fn=handle_vqa,
                    inputs=[vqa_image, vqa_question, vqa_model, vqa_use_adapter, vqa_adapter],
                    outputs=[vqa_output, vqa_latency, vqa_vram]
                )

            # ================= TAB 2: CAPTIONING =================
            with gr.TabItem("📝 Mô tả Hình ảnh (Captioning)"):
                with gr.Row():
                    with gr.Column(scale=5):
                        cap_image = gr.Image(type="pil", label="Tải ảnh lên")
                        cap_mode = gr.Radio(
                            choices=["Ngắn gọn (1-2 câu)", "Chi tiết"],
                            value="Ngắn gọn (1-2 câu)",
                            label="Độ chi tiết mô tả"
                        )
                        cap_submit_btn = gr.Button("✨ Sinh mô tả tiếng Việt", variant="primary")

                    with gr.Column(scale=5):
                        cap_model, cap_use_adapter, cap_adapter = create_model_selector()
                        cap_output = gr.Textbox(label="Đoạn mô tả sinh tự động", lines=4, interactive=False)
                        with gr.Row():
                            cap_latency = gr.Textbox(label="⏱️ Độ trễ", value="0.0s", interactive=False)
                            cap_vram = gr.Textbox(label="💾 Peak VRAM", value="0.0 MB", interactive=False)

                cap_submit_btn.click(
                    fn=handle_caption,
                    inputs=[cap_image, cap_mode, cap_model, cap_use_adapter, cap_adapter],
                    outputs=[cap_output, cap_latency, cap_vram]
                )

            # ================= TAB 3: COMPARE MODE =================
            with gr.TabItem("⚖️ So sánh Trực tiếp (Base vs LoRA)"):
                gr.Markdown("Đưa cùng một ảnh và câu hỏi vào **Base Pretrained Model** và **Fine-tuned LoRA Adapter** để đối chứng sự cải thiện về độ tự nhiên tiếng Việt và độ chính xác.")
                with gr.Row():
                    with gr.Column(scale=4):
                        cmp_image = gr.Image(type="pil", label="Ảnh so sánh")
                        cmp_question = gr.Textbox(
                            label="Câu hỏi / Yêu cầu mô tả",
                            placeholder="Ví dụ: Người trong ảnh đang làm gì?",
                            lines=2
                        )
                        cmp_model = gr.Dropdown(choices=["smolvlm", "qwen"], value="smolvlm", label="Mô hình nền")
                        cmp_adapter = gr.Dropdown(choices=get_available_adapters(), value=get_available_adapters()[0] if get_available_adapters() else "", label="Adapter đối chứng")
                        cmp_btn = gr.Button("🔍 Chạy So sánh Đối chứng", variant="primary")

                    with gr.Column(scale=3):
                        gr.Markdown("### 🏛️ Base Pretrained Model")
                        cmp_base_out = gr.Textbox(label="Kết quả Base Model", lines=4, interactive=False)
                        cmp_base_meta = gr.Textbox(label="Hiệu năng Base", interactive=False)

                    with gr.Column(scale=3):
                        gr.Markdown("### 🚀 Fine-tuned (Base + LoRA)")
                        cmp_ft_out = gr.Textbox(label="Kết quả LoRA Adapter", lines=4, interactive=False)
                        cmp_ft_meta = gr.Textbox(label="Hiệu năng LoRA", interactive=False)

                cmp_btn.click(
                    fn=handle_compare,
                    inputs=[cmp_image, cmp_question, cmp_model, cmp_adapter],
                    outputs=[cmp_base_out, cmp_base_meta, cmp_ft_out, cmp_ft_meta]
                )

            # ================= TAB 4: METRICS & REPORTS =================
            with gr.TabItem("📊 Báo cáo Thực nghiệm & Đánh giá"):
                gr.Markdown("### 📈 Biểu đồ Loss & Chỉ số Đánh giá Thực tế")
                with gr.Row():
                    loss_img_path = PROJECT_ROOT / "results" / "plots" / "loss_curve.png"
                    comp_img_path = PROJECT_ROOT / "results" / "plots" / "model_comparison.png"
                    
                    gr.Image(
                        value=str(loss_img_path) if loss_img_path.exists() else None,
                        label="Tiến trình Giảm Loss Huấn luyện SFT"
                    )
                    gr.Image(
                        value=str(comp_img_path) if comp_img_path.exists() else None,
                        label="Biểu đồ So sánh Định lượng (EM / F1 / BLEU / ROUGE)"
                    )

        gr.Markdown(
            """
            ---
            *AI Vision Assistant - Dự án nghiên cứu Vision-Language Model cho tiếng Việt | Powered by SmolVLM, Qwen2-VL & PEFT LoRA.*
            """
        )

    return demo

if __name__ == "__main__":
    app = create_app()
    app.launch(server_name="127.0.0.1", server_port=7860, share=False, css=CUSTOM_CSS)
