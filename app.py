import gradio as gr
import cv2
import numpy as np
import os

def process_image(
    image,
    brightness, contrast,
    r, g, b,
    y, u, v,
    gauss_k, median_k, bilateral_d, sharpen_val,
    effect
):
    if image is None:
        return None

    img = image.copy()

    # 防呆處理：統一轉為 3 通道 RGB（避免上傳透明 PNG 或灰階圖報錯）
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)

    # 1. 亮度與對比、RGB 通道調整
    img = img.astype(np.float32)
    img = img * contrast + brightness
    img[:, :, 0] += r
    img[:, :, 1] += g
    img[:, :, 2] += b
    img = np.clip(img, 0, 255).astype(np.uint8)

    # 2. YUV 通道調整
    if y != 0 or u != 0 or v != 0:
        yuv_img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV).astype(np.float32)
        yuv_img[:, :, 0] += y
        yuv_img[:, :, 1] += u
        yuv_img[:, :, 2] += v
        yuv_img = np.clip(yuv_img, 0, 255).astype(np.uint8)
        img = cv2.cvtColor(yuv_img, cv2.COLOR_YUV2RGB)

    # 3. 模糊與降噪濾波器
    if gauss_k > 0:
        k = int(gauss_k) * 2 + 1  # 確保為奇數
        img = cv2.GaussianBlur(img, (k, k), 0)

    if median_k > 0:
        k = int(median_k) * 2 + 1
        img = cv2.medianBlur(img, k)

    if bilateral_d > 0:
        img = cv2.bilateralFilter(img, d=int(bilateral_d), sigmaColor=75, sigmaSpace=75)

    # 4. 銳化 (Unsharp Masking)
    if sharpen_val > 0:
        blurred = cv2.GaussianBlur(img, (0, 0), sigmaX=2.0)
        img = cv2.addWeighted(img, 1.0 + sharpen_val, blurred, -sharpen_val, 0)

    # 5. 風格與邊緣偵測濾鏡
    if effect == "灰階 (Grayscale)":
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

    elif effect == "負片反轉 (Invert)":
        img = cv2.bitwise_not(img)

    elif effect == "復古懷舊 (Sepia)":
        kernel = np.array([
            [0.393, 0.769, 0.189],
            [0.349, 0.686, 0.168],
            [0.272, 0.534, 0.131]
        ])
        img = cv2.transform(img, kernel)
        img = np.clip(img, 0, 255).astype(np.uint8)

    elif effect == "自動對比增強 (CLAHE)":
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        l, a, b_ch = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        img = cv2.cvtColor(cv2.merge((cl, a, b_ch)), cv2.COLOR_LAB2RGB)

    elif effect == "Canny 邊緣偵測":
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        img = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)

    elif effect == "Sobel 邊緣偵測":
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        sx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        mag = cv2.magnitude(sx, sy)
        mag = np.clip(mag, 0, 255).astype(np.uint8)
        img = cv2.cvtColor(mag, cv2.COLOR_GRAY2RGB)

    elif effect == "浮雕效果 (Emboss)":
        kernel = np.array([
            [-2, -1, 0],
            [-1,  1, 1],
            [ 0,  1, 2]
        ])
        img = cv2.filter2D(img, -1, kernel)

    elif effect == "鉛筆素描 (Sketch)":
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        inv = 255 - gray
        blur = cv2.GaussianBlur(inv, (21, 21), 0)
        sketch = cv2.divide(gray, 255 - blur, scale=256)
        img = cv2.cvtColor(sketch, cv2.COLOR_GRAY2RGB)

    return img

def reset_all(image):
    # 回傳所有控制項的預設值 + 原圖
    return 0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0, "無", image

with gr.Blocks() as demo:
    gr.Markdown("### 影像處理與濾波器控制台")

    with gr.Row():
        # 手機直立瀏覽時，先顯示圖片區塊方便對照
        img_input = gr.Image(label="原圖 (支援手機相簿上傳或拍照)", type="numpy")
        img_output = gr.Image(label="處理結果 (右上角可下載)", interactive=False)

    with gr.Tabs():
        with gr.Tab("濾波與銳化"):
            gauss_slider = gr.Slider(0, 15, 0, step=1, label="高斯模糊 (Gaussian Blur)")
            median_slider = gr.Slider(0, 10, 0, step=1, label="中值濾波 (去雜訊)")
            bilateral_slider = gr.Slider(0, 15, 0, step=1, label="雙邊濾波 (保邊平滑／磨皮)")
            sharpen_slider = gr.Slider(0.0, 3.0, 0.0, step=0.1, label="銳化強度 (Unsharp Mask)")

        with gr.Tab("特效與邊緣"):
            effect_dropdown = gr.Dropdown(
                choices=[
                    "無",
                    "灰階 (Grayscale)",
                    "負片反轉 (Invert)",
                    "復古懷舊 (Sepia)",
                    "自動對比增強 (CLAHE)",
                    "Canny 邊緣偵測",
                    "Sobel 邊緣偵測",
                    "浮雕效果 (Emboss)",
                    "鉛筆素描 (Sketch)"
                ],
                value="無",
                label="選擇特效或邊緣濾鏡"
            )

        with gr.Tab("亮度與色彩"):
            with gr.Row():
                brightness = gr.Slider(-100, 100, 0, step=1, label="亮度")
                contrast = gr.Slider(0.1, 3.0, 1.0, step=0.1, label="對比")
            with gr.Row():
                r_slider = gr.Slider(-100, 100, 0, step=1, label="R (紅)")
                g_slider = gr.Slider(-100, 100, 0, step=1, label="G (綠)")
                b_slider = gr.Slider(-100, 100, 0, step=1, label="B (藍)")
            with gr.Row():
                y_slider = gr.Slider(-100, 100, 0, step=1, label="Y (明度)")
                u_slider = gr.Slider(-100, 100, 0, step=1, label="U (色度)")
                v_slider = gr.Slider(-100, 100, 0, step=1, label="V (色度)")

    reset_btn = gr.Button("重置所有參數", variant="secondary")

    sliders = [
        brightness, contrast,
        r_slider, g_slider, b_slider,
        y_slider, u_slider, v_slider,
        gauss_slider, median_slider, bilateral_slider, sharpen_slider
    ]
    all_inputs = [img_input] + sliders + [effect_dropdown]

    # 上傳圖片與切換下拉選單時立即觸發
    img_input.change(fn=process_image, inputs=all_inputs, outputs=img_output)
    effect_dropdown.change(fn=process_image, inputs=all_inputs, outputs=img_output)

    # 滑桿改用 release（放開手指／滑鼠時才計算），大幅提升手機網路操作流暢度
    for s in sliders:
        s.release(fn=process_image, inputs=all_inputs, outputs=img_output)

    reset_btn.click(
        fn=reset_all,
        inputs=[img_input],
        outputs=sliders + [effect_dropdown, img_output]
    )

if __name__ == "__main__":
    # 讀取 Render 給的 PORT，如果沒有則預設 7860
    port = int(os.environ.get("PORT", 7860))
    demo.launch(theme=gr.themes.Soft(), server_name="0.0.0.0", server_port=port)