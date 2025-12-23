# app.py - Expert-Level Sentiment Analysis App
import gradio as gr
import joblib
import pandas as pd
import numpy as np
import re
import time
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer

# Configuration
CONFIG = {
    "app_title": "🚀 Sentiment Intelligence Pro",
    "model_paths": {
        "model": "sentiment_model.pkl",
        "vectorizer": "tfidf_vectorizer.pkl"
    },
    "theme": "soft"
}

class SentimentAnalyzer:
    """Expert-level sentiment analysis engine"""
    
    def __init__(self):
        self.model = None
        self.vectorizer = None
        self.load_models()
        self.analysis_count = 0
        
    def load_models(self):
        """Load trained models with error handling"""
        try:
            self.model = joblib.load(CONFIG["model_paths"]["model"])
            self.vectorizer = joblib.load(CONFIG["model_paths"]["vectorizer"])
            print("✅ AI Models loaded successfully!")
        except Exception as e:
            print(f"⚠️ Model version warning (safe to continue): {e}")
            # Try to load anyway
            try:
                self.model = joblib.load(CONFIG["model_paths"]["model"])
                self.vectorizer = joblib.load(CONFIG["model_paths"]["vectorizer"])
                print("✅ Models loaded despite version differences!")
            except Exception as e2:
                raise Exception(f"❌ Model loading failed: {str(e2)}")
    
    def advanced_text_clean(self, text):
        """Advanced text preprocessing pipeline"""
        if not isinstance(text, str) or not text.strip():
            return ""
        
        # Multi-stage cleaning pipeline
        cleaning_steps = [
            lambda x: re.sub(r'http\S+', '', x),      # URL removal
            lambda x: re.sub(r'@\w+', '', x),         # User mention removal
            lambda x: re.sub(r'#(\w+)', r'\1', x),    # Hashtag normalization
            lambda x: re.sub(r'[^\w\s]', ' ', x),     # Punctuation removal
            lambda x: re.sub(r'\d+', '', x),          # Number removal
            lambda x: ' '.join(x.split()),            # Whitespace normalization
            lambda x: x.lower().strip()               # Case normalization
        ]
        
        cleaned_text = text
        for step in cleaning_steps:
            cleaned_text = step(cleaned_text)
            
        return cleaned_text
    
    def analyze_sentiment(self, text, detailed=False):
        """Advanced sentiment analysis with confidence scoring"""
        start_time = time.time()
        
        if not text or len(text.strip()) < 3:
            return self._format_error("Text too short for analysis")
        
        try:
            # Advanced text cleaning
            cleaned_text = self.advanced_text_clean(text)
            
            if len(cleaned_text.split()) < 1:
                return self._format_error("No meaningful text detected")
            
            # Feature transformation
            text_features = self.vectorizer.transform([cleaned_text])
            
            # Prediction with confidence
            prediction = self.model.predict(text_features)[0]
            probabilities = self.model.predict_proba(text_features)[0]
            confidence = max(probabilities)
            
            # Sentiment interpretation
            sentiment_label = "POSITIVE 🎉" if prediction == 1 else "NEGATIVE ⚠️"
            sentiment_emoji = "😊" if prediction == 1 else "😞"
            
            # Confidence level categorization
            confidence_level = self._get_confidence_level(confidence)
            
            # Response time
            response_time = round((time.time() - start_time) * 1000, 2)
            
            # Update analytics
            self.analysis_count += 1
            
            return self._format_response(
                sentiment_label, sentiment_emoji, confidence, 
                confidence_level, response_time, cleaned_text, 
                probabilities, detailed
            )
                
        except Exception as e:
            return self._format_error(f"Analysis error: {str(e)}")
    
    def _get_confidence_level(self, confidence):
        """Categorize confidence levels"""
        if confidence >= 0.9: return "VERY HIGH"
        elif confidence >= 0.8: return "HIGH"
        elif confidence >= 0.7: return "MODERATE"
        elif confidence >= 0.6: return "LOW"
        else: return "VERY LOW"
    
    def _format_response(self, label, emoji, confidence, level, response_time, cleaned_text, probabilities, detailed):
        """Format response based on detail level"""
        base_response = {
            "sentiment": f"{label} {emoji}",
            "confidence": float(confidence),
            "confidence_level": level,
            "timestamp": datetime.now().strftime("%H:%M:%S")
        }
        
        if detailed:
            base_response.update({
                "response_time": f"{response_time}ms",
                "positive_prob": float(probabilities[1]),
                "negative_prob": float(probabilities[0]),
                "cleaned_text": cleaned_text[:100] + "..." if len(cleaned_text) > 100 else cleaned_text,
                "analysis_id": f"SA{self.analysis_count:04d}"
            })
        
        return base_response
    
    def _format_error(self, message):
        """Format error response"""
        return {
            "sentiment": "ERROR ❌",
            "confidence": 0.0,
            "confidence_level": "N/A",
            "error": message,
            "timestamp": datetime.now().strftime("%H:%M:%S")
        }

# Initialize analyzer
try:
    analyzer = SentimentAnalyzer()
    print("🚀 Sentiment Intelligence Pro initialized successfully!")
except Exception as e:
    print(f"❌ Initialization failed: {e}")
    exit(1)

# Expert-level examples
EXPERT_EXAMPLES = [
    "This product absolutely exceeded my expectations! The quality is outstanding and delivery was lightning fast. Highly recommended! 🌟",
    "Terrible customer service and poor product quality. Will never purchase again. Complete waste of money and time. 👎",
    "The movie had some good moments but overall felt predictable and lacked originality. Not bad, but not great either.",
    "Absolutely phenomenal experience from start to finish! Every detail was perfect and the team went above and beyond. 💯",
    "Extremely disappointed with the service. Multiple issues were ignored and the support team was unhelpful. Would not recommend.",
]

# Create expert-level Gradio interface
def create_expert_interface():
    with gr.Blocks(
        title="Sentiment Intelligence Pro",
        theme=gr.themes.Soft(
            primary_hue="blue",
            secondary_hue="slate",
        ),
        css="""
        .gradio-container {
            max-width: 1200px !important;
            margin: 0 auto;
        }
        .positive-card { 
            background: linear-gradient(135deg, #d4edda, #c3e6cb);
            border: 2px solid #28a745;
            color: #155724;
            padding: 15px;
            border-radius: 10px;
            margin: 10px 0;
        }
        .negative-card { 
            background: linear-gradient(135deg, #f8d7da, #f5c6cb);
            border: 2px solid #dc3545;
            color: #721c24;
            padding: 15px;
            border-radius: 10px;
            margin: 10px 0;
        }
        .confidence-high { color: #28a745; font-weight: 800; }
        .confidence-medium { color: #ffc107; font-weight: 700; }
        .confidence-low { color: #dc3545; font-weight: 600; }
        .analysis-card {
            background: white;
            border-radius: 15px;
            padding: 20px;
            margin: 10px 0;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            border: 1px solid #e9ecef;
        }
        """
    ) as demo:
        
        # Header Section
        gr.Markdown(
            """
            <div style='text-align: center; padding: 20px; background: linear-gradient(135deg, #667eea, #764ba2); border-radius: 15px; color: white;'>
                <h1 style='margin: 0; font-size: 2.5em;'>🚀 Sentiment Intelligence Pro</h1>
                <p style='margin: 10px 0 0 0; font-size: 1.2em;'>Enterprise-Grade Text Sentiment Analysis Platform</p>
                <p style='margin: 5px 0 0 0; opacity: 0.9;'>Powered by Advanced Machine Learning • Real-time Analytics • Production Ready</p>
            </div>
            """
        )
        
        with gr.Row(equal_height=True):
            # Input Column
            with gr.Column(scale=1):
                with gr.Group():
                    gr.Markdown("### 📝 Input Text Analysis")
                    
                    text_input = gr.Textbox(
                        label="Enter Text for Sentiment Analysis",
                        placeholder="Paste your text, tweet, review, or comment here...",
                        lines=4,
                        max_lines=8,
                        show_copy_button=True
                    )
                    
                    with gr.Row():
                        analyze_btn = gr.Button(
                            "🚀 Analyze Sentiment", 
                            variant="primary",
                            scale=2
                        )
                        clear_btn = gr.Button(
                            "🗑️ Clear", 
                            variant="secondary",
                            scale=1
                        )
                    
                    detailed_toggle = gr.Checkbox(
                        label="📊 Show Detailed Analysis",
                        value=True,
                        info="Display comprehensive analysis results"
                    )
                
                # Quick Examples
                gr.Markdown("### 💡 Quick Analysis Examples")
                gr.Examples(
                    examples=EXPERT_EXAMPLES,
                    inputs=text_input,
                    label="Click any example for instant analysis:",
                    examples_per_page=3
                )
            
            # Output Column
            with gr.Column(scale=1):
                gr.Markdown("### 📊 Analysis Results")
                
                # Results Display
                with gr.Group():
                    # Sentiment Result with custom HTML for better display
                    sentiment_display = gr.HTML(
                        value="<div class='analysis-card'><h3>🎯 Sentiment Classification</h3><p>Awaiting analysis...</p></div>",
                        label="Sentiment Result"
                    )
                    
                    with gr.Row():
                        confidence_output = gr.Number(
                            label="Confidence Score",
                            value=0.0,
                            minimum=0.0,
                            maximum=1.0,
                            precision=3
                        )
                        
                        confidence_level = gr.Textbox(
                            label="Confidence Level",
                            value="N/A",
                            interactive=False
                        )
                    
                    # Detailed Results Section
                    with gr.Column(visible=True) as detailed_col:
                        with gr.Row():
                            response_time = gr.Textbox(
                                label="⚡ Response Time",
                                value="N/A",
                                interactive=False
                            )
                            
                            analysis_id = gr.Textbox(
                                label="📋 Analysis ID",
                                value="N/A",
                                interactive=False
                            )
                        
                        with gr.Row():
                            positive_prob = gr.Number(
                                label="Positive Probability",
                                value=0.0,
                                minimum=0.0,
                                maximum=1.0,
                                precision=3
                            )
                            negative_prob = gr.Number(
                                label="Negative Probability",
                                value=0.0,
                                minimum=0.0,
                                maximum=1.0,
                                precision=3
                            )
                        
                        cleaned_preview = gr.Textbox(
                            label="🔍 Processed Text Preview",
                            value="N/A",
                            lines=2,
                            interactive=False
                        )
        
        # Analytics Section
        with gr.Row():
            with gr.Column():
                gr.Markdown("### 📈 System Information")
                with gr.Row():
                    model_info = gr.Textbox(
                        label="🤖 Model Type",
                        value="Logistic Regression + TF-IDF",
                        interactive=False
                    )
                    total_analyses = gr.Number(
                        label="📊 Total Analyses",
                        value=0,
                        interactive=False
                    )
        
        # Footer
        gr.Markdown("---")
        gr.Markdown(
            """
            <div style='text-align: center; padding: 20px; background: #f8f9fa; border-radius: 10px;'>
                <p style='margin: 0; font-weight: bold;'>Sentiment Intelligence Pro</p>
                <p style='margin: 5px 0 0 0; color: #666;'>Built with Enterprise-Grade Machine Learning • Real-time Analysis • High Accuracy</p>
            </div>
            """
        )
        
        # Interactive Functions
        def analyze_sentiment_wrapper(text, detailed):
            if not text or len(text.strip()) < 3:
                return [
                    "<div class='analysis-card'><h3>🎯 Sentiment Classification</h3><p>Please enter text to analyze...</p></div>",
                    0.0, "N/A", 0, "N/A", "N/A", 0.0, 0.0, "N/A"
                ]
            
            result = analyzer.analyze_sentiment(text, detailed)
            total_count = analyzer.analysis_count
            
            # Create beautiful sentiment display
            sentiment_class = "positive-card" if "POSITIVE" in result["sentiment"] else "negative-card" if "NEGATIVE" in result["sentiment"] else "analysis-card"
            sentiment_html = f"""
            <div class='{sentiment_class}'>
                <h3>🎯 Sentiment Classification</h3>
                <h2 style='margin: 10px 0;'>{result['sentiment']}</h2>
                <p><strong>Confidence:</strong> {result['confidence']:.3f} ({result['confidence_level']})</p>
                <p><strong>Time:</strong> {result['timestamp']}</p>
            </div>
            """
            
            outputs = [
                sentiment_html, 
                float(result.get("confidence", 0.0)), 
                result.get("confidence_level", "N/A"), 
                total_count
            ]
            
            # Add detailed outputs if available
            if detailed and "response_time" in result:
                outputs.extend([
                    result.get("response_time", "N/A"),
                    result.get("analysis_id", "N/A"),
                    float(result.get("positive_prob", 0.0)),
                    float(result.get("negative_prob", 0.0)),
                    result.get("cleaned_text", "N/A")
                ])
            else:
                outputs.extend(["N/A", "N/A", 0.0, 0.0, "N/A"])
            
            return outputs
        
        def toggle_detailed_view(detailed):
            return gr.update(visible=detailed)
        
        def clear_all():
            return [
                "",
                "<div class='analysis-card'><h3>🎯 Sentiment Classification</h3><p>Awaiting analysis...</p></div>",
                0.0, "N/A", 0, "N/A", "N/A", 0.0, 0.0, "N/A"
            ]
        
        # Event Handlers
        analyze_btn.click(
            fn=analyze_sentiment_wrapper,
            inputs=[text_input, detailed_toggle],
            outputs=[
                sentiment_display, confidence_output, confidence_level, total_analyses,
                response_time, analysis_id, positive_prob, negative_prob, cleaned_preview
            ]
        )
        
        text_input.submit(
            fn=analyze_sentiment_wrapper,
            inputs=[text_input, detailed_toggle],
            outputs=[
                sentiment_display, confidence_output, confidence_level, total_analyses,
                response_time, analysis_id, positive_prob, negative_prob, cleaned_preview
            ]
        )
        
        detailed_toggle.change(
            fn=toggle_detailed_view,
            inputs=detailed_toggle,
            outputs=detailed_col
        )
        
        clear_btn.click(
            fn=clear_all,
            inputs=[],
            outputs=[
                text_input, sentiment_display, confidence_output, confidence_level, total_analyses,
                response_time, analysis_id, positive_prob, negative_prob, cleaned_preview
            ]
        )
    
    return demo

# Create and launch the expert interface
if __name__ == "__main__":
    demo = create_expert_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        favicon_path=None,
        inbrowser=True
    )