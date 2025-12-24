# app.py
import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image, ImageDraw
import cv2
import matplotlib.pyplot as plt
import io
import time
from datetime import datetime
import base64

# Load the trained model
print("🔄 Loading the trained model...")
try:
    model = tf.keras.models.load_model('model.h5')
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    raise

class DigitRecognizer:
    def __init__(self, model):
        self.model = model
        self.preprocessing_time = 0
        self.prediction_time = 0
    
    def enhance_image(self, image):
        """Advanced image preprocessing"""
        # Convert to numpy array
        img_array = np.array(image)
        
        # Apply Gaussian blur to reduce noise
        img_array = cv2.GaussianBlur(img_array, (3, 3), 0)
        
        return Image.fromarray(img_array)
    
    def preprocess_image(self, image):
        """Professional preprocessing pipeline"""
        start_time = time.time()
        
        try:
            # Convert to grayscale if needed
            if image.mode != 'L':
                image = image.convert('L')
            
            # Enhance image quality
            image = self.enhance_image(image)
            
            # Resize to 28x28 pixels
            image = image.resize((28, 28), Image.LANCZOS)
            
            # Convert to numpy array
            img_array = np.array(image)
            
            # Advanced normalization
            img_array = img_array.astype('float32')
            
            # Auto-contrast enhancement
            p2, p98 = np.percentile(img_array, (2, 98))
            if p98 > p2:
                img_array = np.clip((img_array - p2) * 255.0 / (p98 - p2), 0, 255)
            
            # Normalize to 0-1
            img_array = img_array / 255.0
            
            # Smart background detection and inversion
            if np.mean(img_array) > 0.5:
                img_array = 1.0 - img_array
            
            # Reshape for model
            img_array = img_array.reshape(1, 28, 28, 1)
            
            self.preprocessing_time = time.time() - start_time
            return img_array
            
        except Exception as e:
            print(f"Preprocessing error: {e}")
            return None
    
    def predict(self, image):
        """Advanced prediction with confidence analysis"""
        start_time = time.time()
        
        try:
            if image is None:
                return None, None, None, None
            
            # Preprocess image
            processed_image = self.preprocess_image(image)
            if processed_image is None:
                return None, None, None, None
            
            # Make prediction
            predictions = self.model.predict(processed_image, verbose=0)[0]
            
            # Get top prediction
            predicted_digit = np.argmax(predictions)
            confidence = np.max(predictions)
            
            # Calculate prediction certainty metrics
            sorted_probs = np.sort(predictions)[::-1]
            certainty_score = sorted_probs[0] - sorted_probs[1]
            
            # Create detailed confidence analysis
            confidence_data = []
            for i, prob in enumerate(predictions):
                confidence_data.append({
                    'digit': i,
                    'confidence': float(prob),
                    'percentage': f"{prob:.2%}",
                    'is_top': i == predicted_digit
                })
            
            # Sort by confidence
            confidence_data.sort(key=lambda x: x['confidence'], reverse=True)
            
            self.prediction_time = time.time() - start_time
            
            return predicted_digit, confidence, certainty_score, confidence_data
            
        except Exception as e:
            print(f"Prediction error: {e}")
            return None, None, None, None

# Initialize recognizer
recognizer = DigitRecognizer(model)

def create_confidence_chart(confidence_data):
    """Create a beautiful confidence visualization"""
    if not confidence_data:
        return None
    
    digits = [str(item['digit']) for item in confidence_data]
    confidences = [item['confidence'] for item in confidence_data]
    colors = ['#FF6B6B' if item['is_top'] else '#4ECDC4' for item in confidence_data]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(digits, confidences, color=colors, alpha=0.8, edgecolor='white', linewidth=2)
    
    # Add value labels on bars
    for bar, confidence in zip(bars, confidences):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{confidence:.1%}', ha='center', va='bottom', fontweight='bold')
    
    ax.set_ylabel('Confidence Score', fontweight='bold')
    ax.set_xlabel('Digits', fontweight='bold')
    ax.set_title('Digit Recognition Confidence Distribution', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    # Convert to image properly for Gradio
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    plt.close()
    
    # Return as PIL Image
    return Image.open(buf)

def create_confidence_table(confidence_data):
    """Create a formatted confidence table"""
    if not confidence_data:
        return ""
    
    table_html = """
    <div style="background: linear-gradient(135deg, #667eea, #764ba2); padding: 20px; border-radius: 10px; color: white;">
        <h3 style="margin-top: 0; text-align: center;">📊 Confidence Analysis</h3>
        <table style="width: 100%; border-collapse: collapse; color: white;">
            <thead>
                <tr style="border-bottom: 2px solid white;">
                    <th style="padding: 10px; text-align: left;">Digit</th>
                    <th style="padding: 10px; text-align: center;">Confidence</th>
                    <th style="padding: 10px; text-align: right;">Percentage</th>
                </tr>
            </thead>
            <tbody>
    """
    
    for i, item in enumerate(confidence_data[:5]):  # Show top 5
        bar_width = int(item['confidence'] * 100)
        bar_color = "#FF6B6B" if item['is_top'] else "#4ECDC4"
        
        table_html += f"""
                <tr style="border-bottom: 1px solid rgba(255,255,255,0.1);">
                    <td style="padding: 8px; font-weight: {'bold' if item['is_top'] else 'normal'};">
                        {'🎯' if item['is_top'] else ''} {item['digit']}
                    </td>
                    <td style="padding: 8px; text-align: center;">
                        <div style="background: #e0e0e0; border-radius: 10px; height: 20px; width: 100%; overflow: hidden;">
                            <div style="background: {bar_color}; height: 100%; width: {bar_width}%; border-radius: 10px; transition: width 0.3s ease;"></div>
                        </div>
                    </td>
                    <td style="padding: 8px; text-align: right; font-weight: {'bold' if item['is_top'] else 'normal'};">
                        {item['percentage']}
                    </td>
                </tr>
        """
    
    table_html += """
            </tbody>
        </table>
    </div>
    """
    
    return table_html

def recognize_digit(image):
    """Main recognition function with professional output"""
    if image is None:
        return "🖊️ Please draw a digit first", None, "0 ms", "0 ms", ""
    
    digit, confidence, certainty, confidence_data = recognizer.predict(image)
    
    if digit is None:
        return "❌ Could not process image", None, "0 ms", "0 ms", ""
    
    # Create result message based on confidence
    if confidence > 0.95:
        emoji = "🎯"
        message = "Excellent match!"
        color = "#10B981"
    elif confidence > 0.8:
        emoji = "✅"
        message = "Good recognition!"
        color = "#3B82F6"
    elif confidence > 0.6:
        emoji = "⚠️"
        message = "Moderate confidence"
        color = "#F59E0B"
    else:
        emoji = "❓"
        message = "Uncertain prediction"
        color = "#EF4444"
    
    # Format timing
    preprocess_time = f"{recognizer.preprocessing_time*1000:.1f} ms"
    predict_time = f"{recognizer.prediction_time*1000:.1f} ms"
    
    # Create detailed output
    output = f"""
<div style="background: {color}10; padding: 20px; border-radius: 15px; border-left: 5px solid {color};">
    <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 15px;">
        <span style="font-size: 2em;">{emoji}</span>
        <h2 style="margin: 0; color: {color};">Prediction Result</h2>
    </div>
    
    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px;">
        <div style="background: white; padding: 15px; border-radius: 10px;">
            <strong>Digit:</strong><br>
            <span style="font-size: 2em; font-weight: bold; color: {color};">{digit}</span>
        </div>
        <div style="background: white; padding: 15px; border-radius: 10px;">
            <strong>Confidence:</strong><br>
            <span style="font-size: 1.5em; font-weight: bold; color: {color};">{confidence:.2%}</span>
        </div>
    </div>
    
    <div style="margin-top: 15px; background: white; padding: 15px; border-radius: 10px;">
        <strong>Status:</strong> {message}<br>
        <strong>Certainty Score:</strong> {certainty:.3f}<br>
        <strong>Timestamp:</strong> {datetime.now().strftime('%H:%M:%S')}
    </div>
</div>
"""
    
    # Create confidence chart and table
    chart_image = create_confidence_chart(confidence_data)
    confidence_table = create_confidence_table(confidence_data)
    
    return output, chart_image, preprocess_time, predict_time, confidence_table

def clear_all():
    """Clear all inputs and outputs"""
    return None, None, "0 ms", "0 ms", ""

# Professional CSS styling
custom_css = """
:root {
    --primary: #2563eb;
    --secondary: #7c3aed;
    --success: #059669;
    --warning: #d97706;
    --error: #dc2626;
}

.gradio-container {
    font-family: 'Segoe UI', system-ui, sans-serif !important;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    min-height: 100vh;
}

.main-container {
    max-width: 1400px !important;
    margin: 0 auto !important;
    padding: 20px;
}

.header {
    text-align: center;
    padding: 2rem 0;
    background: rgba(255, 255, 255, 0.95);
    border-radius: 20px;
    margin-bottom: 2rem;
    box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
}

.drawing-area {
    border: 3px dashed #cbd5e1 !important;
    border-radius: 15px !important;
    background: #f8fafc !important;
    min-height: 300px !important;
}

.control-btn {
    border-radius: 25px !important;
    font-weight: 600 !important;
    padding: 1rem 2rem !important;
    border: none !important;
    transition: all 0.3s ease !important;
}

.control-btn:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 10px 20px rgba(0, 0, 0, 0.2) !important;
}

.stats-card {
    background: rgba(255, 255, 255, 0.95) !important;
    padding: 1.5rem !important;
    border-radius: 15px !important;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1) !important;
    border: none !important;
}

.gr-box {
    border: none !important;
    border-radius: 15px !important;
    background: rgba(255, 255, 255, 0.95) !important;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1) !important;
}
"""

# Create expert-level interface
with gr.Blocks(css=custom_css, theme=gr.themes.Soft(), title="Expert Digit Recognition") as demo:
    
    # Header Section
    with gr.Column(elem_classes=["main-container"]):
        with gr.Column(elem_classes=["header"]):
            gr.Markdown(
                """
                # 🚀 Advanced Digit Recognition System
                ### Enterprise-Grade CNN Powered Classification
                *Real-time AI inference with professional analytics*
                """
            )
    
    # Main Content
    with gr.Row():
        # Left Column - Input and Controls
        with gr.Column(scale=1):
            with gr.Group():
                gr.Markdown("### 🎨 Digital Canvas")
                input_image = gr.Image(
                    label="Draw your digit (0-9) here",
                    image_mode="L",
                    type="pil",
                    height=400,
                    elem_classes=["drawing-area"]
                )
                
                with gr.Row():
                    clear_btn = gr.Button(
                        "🗑️ Clear Canvas", 
                        variant="secondary",
                        elem_classes=["control-btn"],
                        scale=1
                    )
                    predict_btn = gr.Button(
                        "🚀 Analyze Digit", 
                        variant="primary",
                        elem_classes=["control-btn"],
                        scale=1
                    )
            
            # Performance Metrics
            with gr.Group():
                gr.Markdown("### ⚡ Performance Metrics")
                with gr.Row():
                    preprocess_time = gr.Textbox(
                        label="🛠️ Preprocessing Time",
                        value="0 ms",
                        interactive=False,
                        elem_classes=["stats-card"]
                    )
                    inference_time = gr.Textbox(
                        label="🤖 Inference Time", 
                        value="0 ms",
                        interactive=False,
                        elem_classes=["stats-card"]
                    )
        
        # Right Column - Results
        with gr.Column(scale=1):
            # Prediction Results
            with gr.Group():
                gr.Markdown("### 📊 Analysis Report")
                output_text = gr.HTML(
                    label="Prediction Results",
                    value="<div style='text-align: center; padding: 40px; color: #6B7280;'>🖊️ Please draw a digit to begin analysis</div>"
                )
            
            # Confidence Chart
            with gr.Group():
                gr.Markdown("### 📈 Confidence Visualization")
                confidence_chart = gr.Image(
                    label="Confidence Distribution",
                    interactive=False,
                    height=300,
                    elem_classes=["stats-card"]
                )
            
            # Confidence Table
            with gr.Group():
                gr.Markdown("### 🎯 Top Predictions")
                confidence_table = gr.HTML(
                    label="Confidence Rankings",
                    value=""
                )
    
    # Footer with technical details
    with gr.Row():
        with gr.Column():
            gr.Markdown(
                """
                ---
                ### 🔧 System Architecture
                
                | Component | Specification |
                |-----------|---------------|
                | **Model** | Convolutional Neural Network (CNN) |
                | **Framework** | TensorFlow 2.x + Keras |
                | **Training Data** | MNIST Dataset (70,000 samples) |
                | **Accuracy** | >99% validation accuracy |
                | **Preprocessing** | Advanced image enhancement pipeline |
                | **Deployment** | Hugging Face Spaces + Gradio |
                
                *Optimized for high-accuracy real-time digit classification*
                """
            )
    
    # Event handlers
    predict_btn.click(
        fn=recognize_digit,
        inputs=[input_image],
        outputs=[output_text, confidence_chart, preprocess_time, inference_time, confidence_table]
    )
    
    clear_btn.click(
        fn=clear_all,
        inputs=[],
        outputs=[input_image, output_text, preprocess_time, inference_time, confidence_table]
    )

# Launch with professional settings
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        show_error=True
    )