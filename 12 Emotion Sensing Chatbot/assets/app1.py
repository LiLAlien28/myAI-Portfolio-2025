import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import random
import warnings
warnings.filterwarnings("ignore")

class LLMMentalHealthChatbot:
    def __init__(self):
        print("Loading Mental Health LLM...")
        
        # Try to use a smaller, faster model that's good for conversation
        try:
            # Option 1: Use a smaller conversational model
            self.chat_pipeline = pipeline(
                "text-generation",
                model="microsoft/DialoGPT-medium",
                tokenizer="microsoft/DialoGPT-medium",
                torch_dtype=torch.float16,
                device_map="auto" if torch.cuda.is_available() else None,
            )
            self.model_type = "dialogpt"
            print("✅ Loaded DialoGPT-medium for conversations")
            
        except Exception as e:
            print(f"❌ Error loading DialoGPT: {e}")
            try:
                # Fallback: Use a smaller model
                self.chat_pipeline = pipeline(
                    "text-generation",
                    model="distilgpt2",
                    torch_dtype=torch.float16
                )
                self.model_type = "distilgpt2"
                print("✅ Loaded DistilGPT2 as fallback")
            except Exception as e2:
                print(f"❌ Error loading fallback model: {e2}")
                self.chat_pipeline = None
                self.model_type = "none"
        
        # Mental health context and guidelines
        self.mental_health_guidelines = """
        You are a compassionate, empathetic mental health support assistant. Your role is to:
        - Provide emotional support and active listening
        - Offer evidence-based mental health information
        - Suggest coping strategies and resources
        - Maintain professional boundaries
        - Encourage seeking professional help when needed
        - Be non-judgmental and validating
        - Use empathetic language and show understanding
        """
        
        self.conversation_history = []
        self.safety_keywords = ["suicide", "kill myself", "end my life", "want to die", "harm myself"]
        
    def contains_safety_concern(self, text):
        """Check for urgent safety concerns"""
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in self.safety_keywords)
    
    def get_safety_response(self):
        """Get emergency response for safety concerns"""
        emergency_resources = """
        **🚨 Immediate Support Resources:**
        
        **Crisis Text Line**: Text HOME to 741741
        **National Suicide Prevention Lifeline**: 1-800-273-8255
        **Emergency Services**: 911
        
        You are not alone, and there are people who want to help. Please reach out to these resources immediately.
        """
        return emergency_resources
    
    def generate_llm_response(self, user_input):
        """Generate response using LLM"""
        
        # First check for safety concerns
        if self.contains_safety_concern(user_input):
            return self.get_safety_response(), "safety_concern"
        
        if self.chat_pipeline is None:
            # Fallback to rule-based responses if no model
            return self.fallback_response(user_input), "fallback"
        
        try:
            # Build conversation context
            context = self.mental_health_guidelines + "\n\nConversation:\n"
            
            # Add recent conversation history for context
            if self.conversation_history:
                for i, (user, bot) in enumerate(self.conversation_history[-4:]):  # Last 4 exchanges
                    context += f"User: {user}\nAssistant: {bot}\n"
            
            context += f"User: {user_input}\nAssistant:"
            
            # Generate response
            if self.model_type == "dialogpt":
                response = self.chat_pipeline(
                    context,
                    max_new_tokens=150,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=self.chat_pipeline.tokenizer.eos_token_id,
                    repetition_penalty=1.1
                )[0]['generated_text']
                
                # Extract only the new response
                if "Assistant:" in response:
                    response = response.split("Assistant:")[-1].strip()
                
            else:  # distilgpt2 or other
                response = self.chat_pipeline(
                    context,
                    max_new_tokens=100,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=50256
                )[0]['generated_text']
                
                if "Assistant:" in response:
                    response = response.split("Assistant:")[-1].strip()
            
            # Clean up response
            response = response.split('\n')[0].split('User:')[0].strip()
            
            # Ensure response is appropriate length
            if len(response.split()) < 3:
                response = self.fallback_response(user_input)
            
            return response, "llm_generated"
            
        except Exception as e:
            print(f"Error in LLM generation: {e}")
            return self.fallback_response(user_input), "error"
    
    def fallback_response(self, user_input):
        """Intelligent fallback responses when LLM fails"""
        
        user_lower = user_input.lower()
        
        # Emotional support responses
        if any(word in user_lower for word in ['sad', 'depressed', 'unhappy', 'miserable']):
            responses = [
                "I hear that you're feeling down, and I want you to know that your feelings are completely valid. It takes courage to acknowledge when we're struggling. Would you like to talk more about what's been weighing on you?",
                "I'm really sorry you're going through this. Feeling sad can be incredibly heavy. Remember that you don't have to face this alone - I'm here to listen whenever you need to talk.",
                "Thank you for sharing that with me. It sounds like you're carrying a lot right now. What's one small thing that might bring you a moment of comfort today?"
            ]
        elif any(word in user_lower for word in ['anxious', 'worried', 'nervous', 'stress', 'overwhelmed']):
            responses = [
                "Anxiety can feel absolutely overwhelming. Let's take a moment together - try taking three deep breaths with me. What specifically has been causing these anxious feelings?",
                "I understand how consuming anxiety can be. It might help to break things down - what's one small step you could take to feel slightly more grounded right now?",
                "Thank you for trusting me with these feelings. Anxiety can make everything feel urgent. Remember that you can handle this moment by moment."
            ]
        elif any(word in user_lower for word in ['angry', 'frustrated', 'mad', 'pissed']):
            responses = [
                "Anger is such a powerful emotion, and it often points to something that really matters to us. What do you think might be underneath this anger?",
                "I hear the frustration in your words. It's completely valid to feel angry when things aren't fair or when boundaries are crossed. Would taking a moment to step away help?",
                "Anger can be protective - it often shows us where our values have been challenged. What would feel like a constructive way to express this right now?"
            ]
        elif any(word in user_lower for word in ['happy', 'good', 'great', 'excited', 'joy']):
            responses = [
                "That's wonderful to hear! I'm genuinely happy for you. What's contributing to these positive feelings?",
                "It's beautiful that you're experiencing joy right now. Savor these moments - they're precious. Want to share what's bringing you happiness?",
                "I'm smiling hearing this! Positive emotions are so important for our wellbeing. How can you nurture this good feeling?"
            ]
        elif any(word in user_lower for word in ['lonely', 'alone', 'isolated']):
            responses = [
                "Loneliness can feel incredibly heavy, even when we're surrounded by people. You're reaching out, and that's a brave step. What does connection mean to you right now?",
                "I hear how isolated you're feeling. That pain is real and valid. Would you like to talk about what kind of connection you're craving?",
                "Loneliness can be so profound. Remember that your worth isn't defined by how connected you feel in this moment. I'm here with you right now."
            ]
        else:
            # General empathetic responses
            responses = [
                "Thank you for sharing that with me. I'm here to listen and support you. Could you tell me more about what you're experiencing?",
                "I appreciate you opening up about this. Let's explore this together - how has this been affecting you?",
                "I'm listening carefully, and I want to understand your perspective fully. What else would you like me to know about this situation?",
                "That sounds really significant. I'm here with you as you process this. What's coming up for you as you share this?",
                "Thank you for trusting me with this. I'm fully present and want to support you through this. How can I best be here for you right now?"
            ]
        
        return random.choice(responses)
    
    def chat(self, user_input):
        """Main chat function"""
        
        # Add to conversation history
        self.conversation_history.append((user_input, ""))
        
        # Generate response
        response, response_type = self.generate_llm_response(user_input)
        
        # Update conversation history with response
        self.conversation_history[-1] = (user_input, response)
        
        # Keep history manageable
        if len(self.conversation_history) > 10:
            self.conversation_history.pop(0)
        
        return response, response_type

# Initialize the chatbot
print("Initializing Mental Health LLM Chatbot...")
chatbot = LLMMentalHealthChatbot()

# Gradio Interface
def chat_interface(message, history):
    """Gradio chat function"""
    if not message.strip():
        return "", history
    
    try:
        response, response_type = chatbot.chat(message)
        
        # Add to Gradio history
        history.append([message, response])
        
        return "", history
        
    except Exception as e:
        error_msg = "I apologize, but I'm having trouble responding right now. Could you try again?"
        history.append([message, error_msg])
        return "", history

def clear_chat():
    """Clear conversation history"""
    chatbot.conversation_history = []
    return []

with gr.Blocks(
    title="LLM Mental Health Chatbot",
    theme=gr.themes.Soft(),
    css="""
    .chat-container { max-height: 500px; overflow-y: auto; }
    .warning-box { background-color: #fff3cd; border: 1px solid #ffeaa7; padding: 10px; border-radius: 5px; margin: 10px 0; }
    .emergency-box { background-color: #f8d7da; border: 1px solid #f5c6cb; padding: 10px; border-radius: 5px; margin: 10px 0; }
    """
) as demo:
    
    gr.Markdown("""
    # 🧠 LLM Mental Health Support Chatbot
    ### Intelligent, Empathetic Conversations Powered by Advanced AI
    
    This chatbot uses large language model technology to provide thoughtful, contextual mental health support.
    """)
    
    with gr.Row():
        with gr.Column(scale=2):
            # Chat interface
            chatbot_interface = gr.Chatbot(
                label="Mental Health Conversation",
                height=500,
                show_copy_button=True,
                container=True
            )
            
            with gr.Row():
                message_input = gr.Textbox(
                    placeholder="Share what's on your mind... (I'll provide thoughtful, contextual responses)",
                    lines=2,
                    max_lines=4,
                    scale=4,
                    container=True
                )
                submit_btn = gr.Button("Send 💬", variant="primary", scale=1)
            
            with gr.Row():
                clear_btn = gr.Button("🔄 New Conversation", variant="secondary")
        
        with gr.Column(scale=1):
            gr.Markdown("""
            ## 💡 How to Use This Chatbot
            
            **Share openly about:**
            - Your feelings and emotions
            - Life challenges and stressors
            - Relationships and social situations
            - Work or academic pressure
            - Personal growth and self-reflection
            
            **You'll receive:**
            - Empathetic, thoughtful responses
            - Evidence-based mental health information
            - Coping strategy suggestions
            - Continuous conversation context
            - Professional resource guidance
            """)
            
            gr.Markdown("""
            ## 🎯 Example Conversations
            
            **Try saying:**
            - "I've been feeling really overwhelmed with work lately"
            - "I'm struggling with my relationship with my family"
            - "I want to improve my mental health but don't know where to start"
            - "I've been having trouble sleeping because of anxiety"
            - "I feel like I've lost motivation for things I used to enjoy"
            """)
            
            # Emergency resources (always visible)
            gr.Markdown("""
            <div class="emergency-box">
            🚨 **Immediate Crisis Support**
            
            If you're in crisis or having thoughts of harm:
            - **Crisis Text Line**: Text HOME to 741741
            - **Suicide Prevention Lifeline**: 1-800-273-8255
            - **Emergency Services**: 911
            - **The Trevor Project** (LGBTQ+): 1-866-488-7386
            
            You are not alone - help is available 24/7.
            </div>
            """)
    
    # Event handlers
    submit_btn.click(
        fn=chat_interface,
        inputs=[message_input, chatbot_interface],
        outputs=[message_input, chatbot_interface]
    )
    
    message_input.submit(
        fn=chat_interface,
        inputs=[message_input, chatbot_interface],
        outputs=[message_input, chatbot_interface]
    )
    
    clear_btn.click(
        fn=clear_chat,
        outputs=[chatbot_interface]
    )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )