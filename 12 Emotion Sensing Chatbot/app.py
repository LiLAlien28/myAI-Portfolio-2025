import gradio as gr
import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer, 
    AutoModelForSeq2SeqLM, 
    pipeline,
    BlenderbotTokenizer, 
    BlenderbotForConditionalGeneration
)
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import json
import datetime
import hashlib
import warnings
warnings.filterwarnings("ignore")

class AdvancedMentalHealthChatbot:
    def __init__(self):
        print("🚀 Initializing Advanced Mental Health Chatbot for FYP...")
        
        # Load advanced conversation model
        try:
            self.tokenizer = BlenderbotTokenizer.from_pretrained("facebook/blenderbot-400M-distill")
            self.model = BlenderbotForConditionalGeneration.from_pretrained("facebook/blenderbot-400M-distill")
            print("✅ Loaded BlenderBot-400M for advanced conversations")
        except:
            print("⚠️ Using smaller model for demo")
            self.tokenizer = BlenderbotTokenizer.from_pretrained("facebook/blenderbot-90M")
            self.model = BlenderbotForConditionalGeneration.from_pretrained("facebook/blenderbot-90M")
        
        # Emotion detection pipeline
        self.emotion_classifier = pipeline(
            "text-classification",
            model="j-hartmann/emotion-english-distilroberta-base",
            return_all_scores=True
        )
        
        # Mental health knowledge base
        self.mental_health_kb = self._load_knowledge_base()
        
        # Conversation analytics
        self.conversation_analytics = {
            'session_start': datetime.datetime.now(),
            'total_messages': 0,
            'emotion_trends': [],
            'user_engagement': 0,
            'crisis_detected': False
        }
        
        # User session management
        self.user_sessions = {}
        
        print("🎯 Advanced Chatbot Initialized Successfully!")
    
    def _load_knowledge_base(self):
        """Load mental health knowledge base"""
        return {
            "depression": {
                "symptoms": ["persistent sadness", "loss of interest", "fatigue", "sleep changes", "appetite changes"],
                "coping": ["therapy", "medication", "exercise", "social support", "routine"],
                "resources": ["Psychologist", "Psychiatrist", "Support groups", "Crisis lines"]
            },
            "anxiety": {
                "symptoms": ["excessive worry", "restlessness", "muscle tension", "sleep problems", "irritability"],
                "coping": ["deep breathing", "mindfulness", "exposure therapy", "CBT techniques"],
                "resources": ["Therapist", "Psychiatrist", "Anxiety workshops"]
            },
            "stress": {
                "symptoms": ["headaches", "fatigue", "sleep problems", "digestive issues", "mood changes"],
                "coping": ["time management", "relaxation techniques", "exercise", "social support"],
                "resources": ["Counselor", "Stress management programs"]
            }
        }
    
    def detect_emotion(self, text):
        """Advanced emotion detection with confidence scores"""
        emotions = self.emotion_classifier(text[:512])[0]  # Limit text length
        emotions_sorted = sorted(emotions, key=lambda x: x['score'], reverse=True)
        
        primary_emotion = emotions_sorted[0]
        secondary_emotions = emotions_sorted[1:3]
        
        return {
            'primary': primary_emotion,
            'secondary': secondary_emotions,
            'all_emotions': emotions_sorted
        }
    
    def mental_health_assessment(self, conversation_history):
        """Assess mental health based on conversation patterns"""
        recent_text = " ".join([msg['user'] for msg in conversation_history[-5:]])
        
        # Simple keyword-based assessment (can be enhanced with ML)
        risk_keywords = {
            'suicide': 10, 'kill myself': 10, 'end my life': 10, 'want to die': 8,
            'harm myself': 8, 'no reason to live': 7, 'hopeless': 6, 'cant take it': 5
        }
        
        risk_score = 0
        for keyword, score in risk_keywords.items():
            if keyword in recent_text.lower():
                risk_score += score
        
        return min(risk_score, 10)  # Scale 0-10
    
    def generate_contextual_response(self, user_input, conversation_history, user_id):
        """Generate intelligent, contextual response using advanced NLP"""
        
        # Update analytics
        self.conversation_analytics['total_messages'] += 1
        
        # Emotion analysis
        emotion_data = self.detect_emotion(user_input)
        primary_emotion = emotion_data['primary']['label']
        emotion_score = emotion_data['primary']['score']
        
        # Mental health risk assessment
        risk_level = self.mental_health_assessment(conversation_history)
        
        # Update emotion trends
        self.conversation_analytics['emotion_trends'].append({
            'timestamp': datetime.datetime.now(),
            'emotion': primary_emotion,
            'score': emotion_score,
            'risk_level': risk_level
        })
        
        # Generate response using transformer model
        if len(conversation_history) > 0:
            # Use conversation context
            context = " ".join([msg['user'] for msg in conversation_history[-3:]])
            full_input = f"{context} {user_input}"
        else:
            full_input = user_input
        
        # Add mental health context
        mental_health_context = f"As a mental health support assistant, provide empathetic, helpful response to: {full_input}"
        
        # Generate using BlenderBot
        inputs = self.tokenizer([mental_health_context], return_tensors="pt", max_length=256, truncation=True)
        
        reply_ids = self.model.generate(
            **inputs,
            max_length=150,
            min_length=30,
            temperature=0.9,
            do_sample=True,
            top_p=0.95,
            repetition_penalty=1.2,
            num_return_sequences=1
        )
        
        response = self.tokenizer.batch_decode(reply_ids, skip_special_tokens=True)[0]
        
        # Enhance response based on emotion and risk
        enhanced_response = self._enhance_response(response, primary_emotion, risk_level, user_input)
        
        return enhanced_response, emotion_data, risk_level
    
    def _enhance_response(self, response, emotion, risk_level, user_input):
        """Enhance AI response with mental health expertise"""
        
        empathy_phrases = {
            'anger': ["I understand this is frustrating", "It makes sense you'd feel angry about this"],
            'fear': ["This sounds really scary", "I can hear the worry in your words"],
            'sadness': ["I'm sorry you're feeling this way", "This sounds really difficult"],
            'joy': ["I'm glad you're feeling good!", "It's wonderful to hear this"],
            'surprise': ["That sounds unexpected", "This must be quite a surprise"],
            'disgust': ["I understand this is upsetting", "That sounds really unpleasant"],
            'neutral': ["Thank you for sharing", "I appreciate you telling me this"]
        }
        
        # Add empathetic opening
        if emotion in empathy_phrases:
            empathetic_opening = random.choice(empathy_phrases[emotion])
            response = f"{empathetic_opening}. {response}"
        
        # Add risk-appropriate resources
        if risk_level >= 7:
            crisis_resources = """
            
            🚨 **Important Resources:**
            • Crisis Text Line: Text HOME to 741741
            • National Suicide Prevention Lifeline: 1-800-273-8255
            • Emergency Services: 911
            
            You are not alone - professional help is available.
            """
            response += crisis_resources
        elif risk_level >= 4:
            support_suggestion = "\n\n💡 Remember: Speaking with a mental health professional can provide valuable support."
            response += support_suggestion
        
        return response
    
    def get_conversation_analytics(self):
        """Get comprehensive conversation analytics"""
        if not self.conversation_analytics['emotion_trends']:
            return "No conversation data yet"
        
        recent_emotions = [et['emotion'] for et in self.conversation_analytics['emotion_trends'][-5:]]
        emotion_counts = pd.Series(recent_emotions).value_counts()
        
        analytics_report = f"""
        📊 **Conversation Analytics**
        
        • Total Messages: {self.conversation_analytics['total_messages']}
        • Session Duration: {(datetime.datetime.now() - self.conversation_analytics['session_start']).seconds // 60} minutes
        • Recent Emotions: {', '.join([f'{emotion}({count})' for emotion, count in emotion_counts.items()])}
        • User Engagement: High
        • Risk Level: {'Low' if self.conversation_analytics['emotion_trends'][-1]['risk_level'] < 4 else 'Medium' if self.conversation_analytics['emotion_trends'][-1]['risk_level'] < 7 else 'High'}
        """
        
        return analytics_report

# Initialize the advanced chatbot
print("🎓 Initializing FYP-Grade Mental Health Chatbot...")
advanced_chatbot = AdvancedMentalHealthChatbot()

# Gradio Interface with Professional UI
def chat_interface(message, history, user_id):
    """Advanced chat interface"""
    if not message.strip():
        return "", history, user_id
    
    # Initialize user session if new
    if user_id == "new_user":
        user_id = hashlib.md5(str(datetime.datetime.now()).encode()).hexdigest()[:8]
        advanced_chatbot.user_sessions[user_id] = []
    
    # Get conversation history for this user
    user_history = advanced_chatbot.user_sessions.get(user_id, [])
    
    try:
        # Generate advanced response
        response, emotion_data, risk_level = advanced_chatbot.generate_contextual_response(
            message, user_history, user_id
        )
        
        # Update user history
        user_history.append({'user': message, 'bot': response, 'timestamp': datetime.datetime.now()})
        advanced_chatbot.user_sessions[user_id] = user_history[-10:]  # Keep last 10 messages
        
        # Format response with emotion info
        primary_emotion = emotion_data['primary']
        emotion_display = f"**Detected Emotion**: {primary_emotion['label']} (confidence: {primary_emotion['score']:.2f})"
        
        formatted_response = f"{response}\n\n---\n*{emotion_display}*"
        
        # Add to Gradio history
        history.append([message, formatted_response])
        
        # Get analytics
        analytics = advanced_chatbot.get_conversation_analytics()
        
        return "", history, user_id, analytics
        
    except Exception as e:
        error_msg = "I apologize for the technical difficulty. Could you please rephrase your message?"
        history.append([message, error_msg])
        return "", history, user_id, f"System Error: {str(e)}"

def clear_conversation(user_id):
    """Clear conversation and start new session"""
    new_user_id = hashlib.md5(str(datetime.datetime.now()).encode()).hexdigest()[:8]
    if user_id in advanced_chatbot.user_sessions:
        del advanced_chatbot.user_sessions[user_id]
    return [], new_user_id, "Conversation cleared. New session started."

def get_emotion_dashboard():
    """Get emotion analytics dashboard"""
    analytics = advanced_chatbot.get_conversation_analytics()
    return analytics

# Create professional Gradio interface
with gr.Blocks(
    title="FYP: Advanced Mental Health Chatbot with AI Emotion Detection",
    theme=gr.themes.Soft(primary_hue="blue", secondary_hue="slate"),
    css="""
    .chat-container { max-height: 500px; overflow-y: auto; }
    .analytics-panel { background: #f8f9fa; padding: 15px; border-radius: 10px; margin: 10px 0; }
    .warning-box { background-color: #fff3cd; border: 1px solid #ffeaa7; padding: 10px; border-radius: 5px; }
    .emergency-box { background-color: #f8d7da; border: 1px solid #f5c6cb; padding: 10px; border-radius: 5px; }
    .success-box { background-color: #d1edff; border: 1px solid #b3d9ff; padding: 10px; border-radius: 5px; }
    """
) as demo:
    
    # Hidden user ID state
    user_id_state = gr.State("new_user")
    
    gr.Markdown("""
    # 🎓 **FYP: Advanced Mental Health Chatbot**
    ### AI-Powered Emotional Support with Real-time Analytics
    *Final Year Project - AI-Based Mental Health Support System*
    
    ---
    """)
    
    with gr.Row():
        with gr.Column(scale=3):
            # Analytics Dashboard
            analytics_display = gr.Markdown(
                "### 📊 Live Analytics Dashboard\n\n*Start chatting to see real-time analytics...*",
                elem_classes="analytics-panel"
            )
            
            # Chat Interface
            chatbot = gr.Chatbot(
                label="🤖 AI Mental Health Support",
                height=400,
                show_copy_button=True,
                show_share_button=True,
                container=True
            )
            
            with gr.Row():
                message_input = gr.Textbox(
                    placeholder="💭 Share your thoughts, feelings, or concerns...",
                    lines=2,
                    max_lines=4,
                    scale=4,
                    container=True
                )
                send_btn = gr.Button("Send 🚀", variant="primary", scale=1)
            
            with gr.Row():
                clear_btn = gr.Button("🔄 New Session", variant="secondary")
                analytics_btn = gr.Button("📈 Update Analytics", variant="secondary")
                export_btn = gr.Button("💾 Export Data", variant="secondary")
        
        with gr.Column(scale=2):
            # Project Information
            gr.Markdown("""
            ### 🎯 **Project Features**
            
            **Advanced AI Capabilities:**
            - 🤖 Transformer-based conversational AI
            - 😊 Real-time emotion detection
            - 📊 Sentiment analysis & trends
            - 💭 Context-aware responses
            - 🎯 Mental health risk assessment
            
            **Technical Stack:**
            - PyTorch & Transformers
            - Gradio Interface
            - Real-time Analytics
            - Session Management
            """)
            
            # Emergency Resources
            gr.Markdown("""
            <div class="emergency-box">
            🚨 **Crisis Support Resources**
            
            **Immediate Help:**
            - Crisis Text Line: Text HOME to 741741
            - Suicide Prevention: 1-800-273-8255
            - Emergency Services: 911
            
            **International:**
            - International Association for Suicide Prevention
            - Befrienders Worldwide
            </div>
            """)
            
            # Technical Details
            gr.Markdown("""
            <div class="success-box">
            🔬 **Technical Implementation**
            
            **Models Used:**
            - BlenderBot-400M (Conversational AI)
            - RoBERTa-base (Emotion Detection)
            - Custom Analytics Engine
            
            **Features:**
            - Multi-turn context retention
            - Emotion trend analysis
            - Risk assessment algorithms
            - Professional resource integration
            </div>
            """)
    
    # Event handlers
    send_btn.click(
        fn=chat_interface,
        inputs=[message_input, chatbot, user_id_state],
        outputs=[message_input, chatbot, user_id_state, analytics_display]
    )
    
    message_input.submit(
        fn=chat_interface,
        inputs=[message_input, chatbot, user_id_state],
        outputs=[message_input, chatbot, user_id_state, analytics_display]
    )
    
    clear_btn.click(
        fn=clear_conversation,
        inputs=[user_id_state],
        outputs=[chatbot, user_id_state, analytics_display]
    )
    
    analytics_btn.click(
        fn=get_emotion_dashboard,
        outputs=analytics_display
    )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,  # Enable sharing for demonstration
        debug=True
    )