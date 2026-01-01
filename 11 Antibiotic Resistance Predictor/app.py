# app.py - FINAL WORKING VERSION
import gradio as gr
import pandas as pd
import numpy as np
import joblib
import os
import sys

print("🚀 INITIALIZING ANTIBIOTIC RESISTANCE PREDICTOR...")

# Check if models folder exists and has files
models_dir = 'models'
if os.path.exists(models_dir) and os.path.isdir(models_dir):
    print(f"📁 Found models directory: {models_dir}")
    pkl_files = [f for f in os.listdir(models_dir) if f.endswith('.pkl')]
    print(f"🔍 Found {len(pkl_files)} .pkl files in models folder:")
    for pkl in pkl_files:
        size = os.path.getsize(os.path.join(models_dir, pkl))
        print(f"   ✅ {pkl} ({size} bytes)")
else:
    print("❌ models directory not found!")
    pkl_files = []

# Load models from models/ folder
def load_models_from_folder():
    """Load all models from the models/ folder"""
    try:
        model_paths = {
            'model': 'models/antibiotic_resistance_model.pkl',
            'label_encoders': 'models/label_encoders.pkl',
            'target_encoder': 'models/target_encoder.pkl',
            'feature_names': 'models/feature_names.pkl'
        }
        
        # Check if all files exist
        for name, path in model_paths.items():
            if not os.path.exists(path):
                print(f"❌ Missing: {path}")
                return None, None, None, None
        
        print("📥 Loading models from models/ folder...")
        model = joblib.load(model_paths['model'])
        label_encoders = joblib.load(model_paths['label_encoders'])
        target_encoder = joblib.load(model_paths['target_encoder'])
        feature_names = joblib.load(model_paths['feature_names'])
        
        print("✅ ALL MODELS LOADED SUCCESSFULLY!")
        print(f"📊 Model type: {type(model).__name__}")
        print(f"🎯 Target classes: {target_encoder.classes_}")
        print(f"📈 Features: {len(feature_names)}")
        
        return model, label_encoders, target_encoder, feature_names
        
    except Exception as e:
        print(f"💥 Error loading models: {e}")
        return None, None, None, None

# Load the models
model, label_encoders, target_encoder, feature_names = load_models_from_folder()

# Prediction function
def predict_resistance(antibiotic, lab_method, testing_standard, measurement_value, measurement_unit):
    """Make resistance prediction"""
    if model is None:
        return "❌ Models not loaded", "Please check that all .pkl files are in the 'models' folder"
    
    try:
        # Create input data
        input_data = {
            'Antibiotic': antibiotic,
            'Laboratory Typing Method': lab_method,
            'Testing Standard': testing_standard,
            'Measurement Value': float(measurement_value),
            'Measurement Unit': measurement_unit
        }
        
        # Convert to DataFrame
        input_df = pd.DataFrame([input_data])
        
        # Encode categorical variables
        for col in input_df.columns:
            if col in label_encoders:
                try:
                    input_df[col] = label_encoders[col].transform([input_df[col].iloc[0]])[0]
                except:
                    input_df[col] = 0  # Default for unknown categories
        
        # Ensure all features are present
        for feature in feature_names:
            if feature not in input_df.columns:
                input_df[feature] = 0
        
        input_df = input_df[feature_names]
        
        # Make prediction
        probabilities = model.predict_proba(input_df)[0]
        prediction = model.predict(input_df)[0]
        
        # Format results
        results = {}
        for i, class_name in enumerate(target_encoder.classes_):
            results[class_name] = float(probabilities[i] * 100)
        
        # Sort by probability
        sorted_results = dict(sorted(results.items(), key=lambda x: x[1], reverse=True))
        top_class = list(sorted_results.keys())[0]
        top_prob = sorted_results[top_class]
        
        # Determine confidence level
        if top_prob > 80:
            confidence = "HIGH"
            emoji = "🟢"
            advice = "Strong clinical confidence"
        elif top_prob > 60:
            confidence = "MEDIUM"
            emoji = "🟡"
            advice = "Moderate clinical confidence"
        else:
            confidence = "LOW"
            emoji = "🔴"
            advice = "Consider additional testing"
        
        # Create summary
        summary = f"{emoji} **PREDICTION: {top_class}**\n\n"
        summary += f"**Confidence:** {confidence} ({top_prob:.1f}%)\n"
        summary += f"**Clinical Advice:** {advice}\n\n"
        summary += f"**Test Parameters:**\n"
        summary += f"- **Antibiotic:** {antibiotic}\n"
        summary += f"- **Method:** {lab_method}\n"
        summary += f"- **Standard:** {testing_standard}\n"
        summary += f"- **Value:** {measurement_value} {measurement_unit}"
        
        # Create details
        details = "**Probability Distribution:**\n"
        for class_name, prob in list(sorted_results.items())[:5]:
            bar = "█" * int(prob / 5)
            details += f"- {class_name}: {prob:.1f}% {bar}\n"
        
        return summary, details
        
    except Exception as e:
        return f"❌ Prediction error: {str(e)}", "Please check the input values"

# Create the interface
with gr.Blocks(
    title="Antibiotic Resistance Predictor",
    theme=gr.themes.Soft(primary_hue="blue"),
    css="""
    .success-box { 
        background: linear-gradient(135deg, #d4edda, #c3e6cb); 
        padding: 20px; 
        border-radius: 10px; 
        border-left: 4px solid #28a745;
        margin-bottom: 20px;
    }
    .error-box { 
        background: linear-gradient(135deg, #f8d7da, #f5c6cb); 
        padding: 20px; 
        border-radius: 10px; 
        border-left: 4px solid #dc3545;
        margin-bottom: 20px;
    }
    """
) as demo:
    
    # Status display
    if model is not None:
        status_html = """
        <div class="success-box">
            <h3>✅ SYSTEM READY</h3>
            <p>Antibiotic Resistance Predictor is loaded and ready for analysis.</p>
        </div>
        """
    else:
        status_html = """
        <div class="error-box">
            <h3>❌ SYSTEM OFFLINE</h3>
            <p>Please ensure all .pkl files are in the 'models' folder:</p>
            <ul>
                <li>antibiotic_resistance_model.pkl</li>
                <li>label_encoders.pkl</li>
                <li>target_encoder.pkl</li>
                <li>feature_names.pkl</li>
            </ul>
        </div>
        """
    
    gr.Markdown("""
    # 🦠 Antibiotic Resistance Predictor
    ### *AI-Powered Clinical Decision Support*
    """)
    
    gr.HTML(status_html)
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### 🔬 Laboratory Parameters")
            
            antibiotic = gr.Dropdown(
                choices=["gentamicin", "ciprofloxacin", "ceftazidime", "cefotaxime", 
                        "meropenem", "ampicillin", "piperacillin/tazobactam"],
                label="Antibiotic Agent",
                value="gentamicin",
                interactive=True
            )
            
            lab_method = gr.Dropdown(
                choices=["MIC", "Broth dilution", "Disk diffusion", "Agar dilution", "E-test"],
                label="Testing Methodology",
                value="MIC"
            )
            
            testing_standard = gr.Dropdown(
                choices=["CLSI", "EUCAST", "FDA", "ISO"],
                label="Quality Standard",
                value="CLSI"
            )
            
            with gr.Row():
                measurement_value = gr.Number(
                    label="Measurement Value",
                    value=2.0,
                    minimum=0.0,
                    maximum=1000.0
                )
                
                measurement_unit = gr.Dropdown(
                    choices=["mg/L", "mm", "μg/mL"],
                    label="Units",
                    value="mg/L"
                )
            
            predict_btn = gr.Button("🔍 Analyze Resistance", variant="primary", size="lg")
        
        with gr.Column():
            gr.Markdown("### 📊 AI Analysis")
            
            with gr.Group():
                summary_output = gr.Markdown(
                    "### 🎯 Prediction Results\n\n*Enter test parameters and click 'Analyze Resistance'*"
                )
            
            with gr.Accordion("📈 Detailed Probability Analysis", open=False):
                details_output = gr.Markdown("")
    
    # Examples section
    gr.Markdown("### 💡 Common Test Scenarios")
    examples = gr.Examples(
        examples=[
            ["gentamicin", "MIC", "CLSI", 4.0, "mg/L"],
            ["ciprofloxacin", "Disk diffusion", "EUCAST", 25.0, "mm"],
            ["meropenem", "Broth dilution", "CLSI", 0.25, "mg/L"],
            ["ampicillin", "MIC", "CLSI", 32.0, "mg/L"]
        ],
        inputs=[antibiotic, lab_method, testing_standard, measurement_value, measurement_unit],
        outputs=[summary_output, details_output],
        fn=predict_resistance,
        run_on_click=True,
        label="Click any example to test:"
    )
    
    # Footer
    gr.Markdown("---")
    gr.Markdown("""
    <div style="text-align: center; color: #666;">
        <p><strong>⚠️ Medical Disclaimer:</strong> This tool is for research and clinical decision support only. 
        Always verify predictions with qualified healthcare professionals.</p>
        <p>Antibiotic Resistance Predictor v1.0 | Built with Machine Learning</p>
    </div>
    """)
    
    # Connect the prediction button
    if model is not None:
        predict_btn.click(
            predict_resistance,
            inputs=[antibiotic, lab_method, testing_standard, measurement_value, measurement_unit],
            outputs=[summary_output, details_output]
        )

if __name__ == "__main__":
    print("\n🌐 Starting web interface...")
    print("📱 Open http://localhost:7860 in your browser")
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)