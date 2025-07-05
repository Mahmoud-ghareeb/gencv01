import gradio as gr
import os
import tempfile
import threading
import time
import sys
import io
from contextlib import redirect_stdout, redirect_stderr
from PIL import Image
from runners.simple_runner import SimpleRunner


print("Loading SFE model...")
MODEL_PATH = 'pretrained_models/sfe_editor_light.pt'
runner = SimpleRunner(editor_ckpt_pth=MODEL_PATH)
print("Model loaded successfully!")


EDITING_TYPES = ['age', 'fs_glasses', 'fs_smiling', 'angry', 'curly_hair', 'head_angle_up', 'rotation', 'grey hair', 'sideburns', 'sslipstick']


current_result = None
current_status = "Ready"
processing_lock = threading.Lock()
console_output = ""
is_processing = False

class ConsoleCapture:
    def __init__(self):
        self.buffer = []
        self.lock = threading.Lock()
    
    def write(self, text):
        with self.lock:
            self.buffer.append(text)
            
            if len(self.buffer) > 100:
                self.buffer = self.buffer[-100:]
        
        if hasattr(sys, '__stdout__') and sys.__stdout__ is not None:
            sys.__stdout__.write(text)
            sys.__stdout__.flush()
        elif hasattr(sys, 'stdout') and sys.stdout is not None:
            sys.stdout.write(text)
            sys.stdout.flush()
    
    def flush(self):
        pass
    
    def get_output(self):
        with self.lock:
            return ''.join(self.buffer)


console_capture = ConsoleCapture()

def process_image_background(image, editing_type, power, align_face, use_mask, mask_threshold, is_styleclip=False, neutral_prompt="", target_prompt="", disentanglement=0.14):
    """Background processing function"""
    global current_result, current_status, console_output, is_processing
    
    try:
        with processing_lock:
            current_status = "🚀 Starting processing..."
            current_result = None
            is_processing = True
            console_output = "🚀 Processing started...\n"
        
        
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = console_capture
        sys.stderr = console_capture
        
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_input:
            image.save(temp_input.name)
            temp_input_path = temp_input.name
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_output:
            temp_output_path = temp_output.name
        
        print("📁 Image files created, starting processing...")
        
        with processing_lock:
            current_status = "🔄 Processing image..."
        
        
        # Determine editing name based on type
        if is_styleclip:
            editing_name = f"styleclip_global_{neutral_prompt}_{target_prompt}_{disentanglement}"
            print(f"🎨 StyleCLIP editing: {neutral_prompt} -> {target_prompt} (disentanglement: {disentanglement})")
        else:
            editing_name = editing_type
            print(f"✨ Standard editing: {editing_type}")
        
        runner.edit(
            orig_img_pth=temp_input_path,
            editing_name=editing_name,
            edited_power=power,
            save_pth=temp_output_path,
            align=align_face,
            save_inversion=False,
            use_mask=use_mask,
            mask_trashold=mask_threshold
        )
        
        print("🎯 Processing completed, loading result...")
        
        
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        
        with processing_lock:
            current_status = "📥 Finalizing..."
        
        
        if os.path.exists(temp_output_path):
            edited_image = Image.open(temp_output_path)
            
            
            os.unlink(temp_input_path)
            os.unlink(temp_output_path)
            
            with processing_lock:
                current_result = edited_image
                current_status = "✅ COMPLETE! Click 'Get Results' to see the image."
                is_processing = False
                
            print("✅ Image processing finished successfully!")
            print("📥 Click 'Get Results' button to see your edited image!")
        else:
            with processing_lock:
                current_result = None
                current_status = "❌ Failed to generate edited image"
                is_processing = False
            print("❌ Failed to generate edited image")
            
    except Exception as e:
        
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        
        with processing_lock:
            current_result = None
            current_status = f"❌ Error: {str(e)}"
            is_processing = False
        print(f"❌ Error occurred: {str(e)}")

def start_processing_standard(image, editing_type, power, align_face, use_mask, mask_threshold):
    """Start standard processing"""
    global current_result, current_status, console_output, is_processing
    
    if image is None:
        return None, "Please upload an image first"
    
    
    console_capture.buffer.clear()
    
    with processing_lock:
        current_status = "🚀 Starting..."
        current_result = None
        is_processing = True
        console_output = "🚀 Processing started...\n"
    
    
    thread = threading.Thread(
        target=process_image_background,
        args=(image, editing_type, power, align_face, use_mask, mask_threshold, False, "", "", 0.14)
    )
    thread.start()
    
    return None, "🚀 Processing started... Click 'Refresh Status' to see live output!"

def start_processing_styleclip(image, neutral_prompt, target_prompt, disentanglement, power, align_face, use_mask, mask_threshold):
    """Start StyleCLIP processing"""
    global current_result, current_status, console_output, is_processing
    
    if image is None:
        return None, "Please upload an image first"
    
    if not neutral_prompt.strip() or not target_prompt.strip():
        return None, "Please provide both neutral and target prompts"
    
    console_capture.buffer.clear()
    
    with processing_lock:
        current_status = "🚀 Starting StyleCLIP..."
        current_result = None
        is_processing = True
        console_output = "🚀 StyleCLIP processing started...\n"
    
    thread = threading.Thread(
        target=process_image_background,
        args=(image, "", power, align_face, use_mask, mask_threshold, True, neutral_prompt.strip(), target_prompt.strip(), disentanglement)
    )
    thread.start()
    
    return None, f"🚀 StyleCLIP started: {neutral_prompt} -> {target_prompt}"

def get_results():
    """Get current results"""
    global current_result, current_status
    
    with processing_lock:
        if current_result is not None:
            
            result_image = current_result
            current_result = None
            current_status = "Ready for next image"
            return result_image, "✅ Success! Image loaded."
        else:
            return None, current_status

def get_live_status():
    """Get live status with streaming console output"""
    global current_status, is_processing
    
    live_output = console_capture.get_output()
    
    with processing_lock:
        status_line = f"\n{'='*50}\nStatus: {current_status}\n{'='*50}"
        
        if is_processing:
            status_line += f"\n🔄 Processing... (Click 'Refresh Status' for updates)"
        elif current_result is not None:
            status_line += f"\n✅ Ready! Click 'Get Results' to see image"
        
        return live_output + status_line


with gr.Blocks(title="StyleFeatureEditor") as demo:
    gr.Markdown("# StyleFeatureEditor - Live Console Output")
    gr.Markdown("Choose between **Standard Editing** or **Custom StyleCLIP** editing")
    
    with gr.Tabs():
        with gr.TabItem("Standard Editing"):
            gr.Markdown("**Instructions:**")
            gr.Markdown("1. Upload image and select editing type")
            gr.Markdown("2. Click 'Start Processing' and watch console output")
            gr.Markdown("3. Click 'Refresh Status' to see updates")
            gr.Markdown("4. When you see '✅ COMPLETE!' click 'Get Results'")
            
            with gr.Row():
                with gr.Column():
                    image_input1 = gr.Image(type="pil", label="Input Image")
                    editing_dropdown = gr.Dropdown(EDITING_TYPES, label="Edit Type", value="age")
                    power_slider1 = gr.Slider(-15, 15, value=0, step=0.5, label="Power")
                    align_face1 = gr.Checkbox(label="Align Face", value=True)
                    use_mask1 = gr.Checkbox(label="Use Mask", value=True)
                    mask_threshold1 = gr.Slider(0.01, 0.3, value=0.095, step=0.005, label="Mask Threshold")
                    
                    with gr.Row():
                        start_btn1 = gr.Button("🚀 Start Processing", variant="primary")
                        get_results_btn1 = gr.Button("📥 Get Results", variant="secondary")
                        refresh_btn1 = gr.Button("🔄 Refresh Status", variant="secondary")
                
                with gr.Column():
                    image_output1 = gr.Image(type="pil", label="Output Image")
                    status_text1 = gr.Textbox(
                        label="🔴 LIVE Console Output & Status", 
                        value="Ready - Upload an image to start", 
                        lines=15,
                        max_lines=20,
                        interactive=False
                    )
        
        with gr.TabItem("Custom StyleCLIP"):
            gr.Markdown("**StyleCLIP Text-Based Editing:**")
            gr.Markdown("1. Upload image and enter text prompts")
            gr.Markdown("2. Neutral prompt: describe current state (e.g., 'face')")
            gr.Markdown("3. Target prompt: describe desired result (e.g., 'face with curly afro')")
            gr.Markdown("4. Adjust disentanglement (0.1-0.3 recommended)")
            
            with gr.Row():
                with gr.Column():
                    image_input2 = gr.Image(type="pil", label="Input Image")
                    
                    with gr.Row():
                        neutral_prompt = gr.Textbox(label="Neutral Prompt", placeholder="e.g., face", value="face")
                        target_prompt = gr.Textbox(label="Target Prompt", placeholder="e.g., face with curly afro", value="face with curly afro")
                    
                    disentanglement_slider = gr.Slider(0.05, 0.5, value=0.14, step=0.01, label="Disentanglement")
                    power_slider2 = gr.Slider(-15, 15, value=5, step=0.5, label="Power")
                    align_face2 = gr.Checkbox(label="Align Face", value=True)
                    use_mask2 = gr.Checkbox(label="Use Mask", value=True)
                    mask_threshold2 = gr.Slider(0.01, 0.3, value=0.095, step=0.005, label="Mask Threshold")
                    
                    with gr.Row():
                        start_btn2 = gr.Button("🎨 Start StyleCLIP", variant="primary")
                        get_results_btn2 = gr.Button("📥 Get Results", variant="secondary")
                        refresh_btn2 = gr.Button("🔄 Refresh Status", variant="secondary")
                
                with gr.Column():
                    image_output2 = gr.Image(type="pil", label="Output Image")
                    status_text2 = gr.Textbox(
                        label="🔴 LIVE Console Output & Status", 
                        value="Ready - Upload an image to start", 
                        lines=15,
                        max_lines=20,
                        interactive=False
                    )
    
    # Event handlers for Standard Editing
    start_btn1.click(
        start_processing_standard,
        inputs=[image_input1, editing_dropdown, power_slider1, align_face1, use_mask1, mask_threshold1],
        outputs=[image_output1, status_text1]
    )
    
    get_results_btn1.click(
        get_results,
        outputs=[image_output1, status_text1]
    )
    
    refresh_btn1.click(
        get_live_status,
        outputs=[status_text1]
    )
    
    # Event handlers for StyleCLIP
    start_btn2.click(
        start_processing_styleclip,
        inputs=[image_input2, neutral_prompt, target_prompt, disentanglement_slider, power_slider2, align_face2, use_mask2, mask_threshold2],
        outputs=[image_output2, status_text2]
    )
    
    get_results_btn2.click(
        get_results,
        outputs=[image_output2, status_text2]
    )
    
    refresh_btn2.click(
        get_live_status,
        outputs=[status_text2]
    )

if __name__ == "__main__":
    demo.launch(
        debug=True, 
        share=True,
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True
    )