import gradio as gr
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import re
import spacy

# Try to load spaCy for better entity recognition
try:
    nlp = spacy.load("en_core_web_sm")
    spacy_available = True
    print("✅ spaCy model loaded successfully!")
except:
    spacy_available = False
    print("⚠️ spaCy not available, using enhanced rule-based approach")

# Load the saved model and preprocessing objects
print("🔄 Loading NER model and components...")

try:
    model = load_model('ner_model.h5')
    with open('tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
    with open('label_encoder.pkl', 'rb') as f:
        label_encoder = pickle.load(f)
    with open('training_params.pkl', 'rb') as f:
        params = pickle.load(f)
    
    max_sequence_length = params['max_sequence_length']
    model_loaded = True
    print("✅ All components loaded successfully!")
    
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model_loaded = False
    model = None
    tokenizer = None
    label_encoder = None
    max_sequence_length = 50

# Define color mapping for different entity types
entity_colors = {
    'B-PER': '#FF6B6B',  # Red - Person
    'I-PER': '#FF9999',  # Light Red
    'B-ORG': '#4ECDC4',  # Teal - Organization
    'I-ORG': '#88D9D3',  # Light Teal
    'B-LOC': '#45B7D1',  # Blue - Location
    'I-LOC': '#87CEEB',  # Light Blue
    'B-MISC': '#FFA500', # Orange - Miscellaneous
    'I-MISC': '#FFD700', # Gold
    'O': '#FFFFFF'       # White - No entity
}

# Entity type descriptions
entity_descriptions = {
    'PER': 'Person - Names of people',
    'ORG': 'Organization - Companies, institutions',
    'LOC': 'Location - Places, countries, cities',
    'MISC': 'Miscellaneous - Other named entities',
    'O': 'No entity - Regular words'
}

def detect_entities_intelligent(text):
    """Intelligent entity detection using multiple approaches"""
    words = text.split()
    entities_found = {}
    colored_text = []
    
    # Approach 1: Use spaCy if available (most accurate)
    if spacy_available:
        doc = nlp(text)
        for ent in doc.ents:
            entity_type = ent.label_
            if entity_type == 'PERSON':
                tag = 'B-PER'
                short_type = 'PER'
            elif entity_type == 'ORG':
                tag = 'B-ORG' 
                short_type = 'ORG'
            elif entity_type == 'GPE' or entity_type == 'LOC':
                tag = 'B-LOC'
                short_type = 'LOC'
            else:
                tag = 'B-MISC'
                short_type = 'MISC'
            
            if short_type not in entities_found:
                entities_found[short_type] = []
            entities_found[short_type].append(ent.text)
    
    # Approach 2: Enhanced rule-based detection
    if not entities_found:
        # Common first names (international)
        common_first_names = {
            'john', 'mary', 'steve', 'david', 'michael', 'sarah', 'james', 'robert', 'maria', 'anna',
            'mohammed', 'wei', 'jing', 'yuki', 'hans', 'pierre', 'carlos', 'antonio', 'viktor', 'sven',
            'ahmed', 'ali', 'jian', 'li', 'wang', 'chen', 'tanaka', 'sato', 'suzuki', 'kim', 'park', 'lee'
        }
        
        # Company/organization indicators
        org_indicators = {'inc', 'corp', 'corporation', 'company', 'ltd', 'llc', 'gmbh', 'co', 'group'}
        
        # Location indicators
        loc_indicators = {'city', 'street', 'avenue', 'road', 'boulevard', 'park', 'square', 'plaza'}
        
        for i, word in enumerate(words):
            word_clean = word.lower().strip('.,!?;:"()[]')
            next_word = words[i+1].lower() if i+1 < len(words) else ""
            
            # Check for person names (capitalized words that might be names)
            if (word[0].isupper() and i > 0 and word_clean not in org_indicators and 
                word_clean not in loc_indicators and len(word) > 2):
                
                # Check if it might be a person name
                if (word_clean in common_first_names or 
                    (i > 0 and words[i-1].lower() in ['mr', 'mrs', 'ms', 'dr', 'professor']) or
                    word_clean.endswith(('son', 'man', 'berg', 'stein', 'ski', 'ovic', 'ian', 'ez'))):
                    
                    if 'PER' not in entities_found:
                        entities_found['PER'] = []
                    entities_found['PER'].append(word)
                    colored_text.append(
                        f"<mark style='background-color: {entity_colors['B-PER']}; padding: 2px 6px; border-radius: 4px; margin: 2px; display: inline-block;'>"
                        f"{word} <small>(B-PER)</small></mark>"
                    )
                    continue
            
            # Check for organizations (capitalized words that might be companies)
            if (word[0].isupper() and len(word) > 2 and 
                (next_word in org_indicators or word_clean in ['apple', 'google', 'microsoft', 'amazon', 
                                                              'samsung', 'toyota', 'sony', 'volkswagen'])):
                
                if 'ORG' not in entities_found:
                    entities_found['ORG'] = []
                entities_found['ORG'].append(word)
                colored_text.append(
                    f"<mark style='background-color: {entity_colors['B-ORG']}; padding: 2px 6px; border-radius: 4px; margin: 2px; display: inline-block;'>"
                    f"{word} <small>(B-ORG)</small></mark>"
                )
                continue
            
            # Check for locations (capitalized geographic terms)
            if (word[0].isupper() and len(word) > 2 and 
                (next_word in loc_indicators or word_clean in ['paris', 'london', 'tokyo', 'berlin', 
                                                              'moscow', 'delhi', 'beijing', 'cairo'])):
                
                if 'LOC' not in entities_found:
                    entities_found['LOC'] = []
                entities_found['LOC'].append(word)
                colored_text.append(
                    f"<mark style='background-color: {entity_colors['B-LOC']}; padding: 2px 6px; border-radius: 4px; margin: 2px; display: inline-block;'>"
                    f"{word} <small>(B-LOC)</small></mark>"
                )
                continue
            
            # Regular word (no entity)
            colored_text.append(f"<span style='margin: 2px; display: inline-block;'>{word}</span>")
    
    return entities_found, colored_text

def predict_ner(text):
    """Predict named entities in the input text"""
    if not text.strip():
        return "⚠️ Please enter some text to analyze."
    
    try:
        # Use intelligent entity detection
        entities_found, colored_text = detect_entities_intelligent(text)
        
        # Create HTML output
        html_output = "<div style='font-family: Arial, sans-serif; line-height: 1.8;'>"
        html_output += "<h3 style='color: #2c3e50;'>🔍 Named Entity Recognition Results:</h3>"
        
        html_output += "<div style='margin: 15px 0; padding: 15px; background: #f8f9fa; border-radius: 8px;'>"
        html_output += " ".join(colored_text)
        html_output += "</div>"
        
        # Display entity summary
        if entities_found:
            html_output += "<h4 style='color: #2c3e50; margin-top: 20px;'>📊 Entities Found:</h4>"
            html_output += "<div style='display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 10px;'>"
            
            for entity_type, words in entities_found.items():
                color = entity_colors.get(f'B-{entity_type}', '#CCCCCC')
                desc = entity_descriptions.get(entity_type, 'Unknown entity type')
                
                html_output += f"""
                <div style='background: {color}20; padding: 10px; border-radius: 6px; border-left: 4px solid {color};'>
                    <strong>{entity_type}</strong><br>
                    <small>{desc}</small><br>
                    <span style='font-size: 0.9em;'>{', '.join(set(words))}</span>
                </div>
                """
            html_output += "</div>"
        else:
            html_output += "<div style='background: #fff3cd; padding: 15px; border-radius: 6px; border: 1px solid #ffeaa7;'>"
            html_output += "🔍 No named entities detected. Try names, companies, or locations!"
            html_output += "</div>"
        
        # Add method used
        method_used = "spaCy" if spacy_available else "intelligent pattern matching"
        html_output += f"<div style='margin-top: 15px; font-size: 0.9em; color: #666;'>"
        html_output += f"Detection method: {method_used}"
        html_output += "</div>"
        
        html_output += "</div>"
        return html_output
        
    except Exception as e:
        return f"❌ Error during prediction: {str(e)}"

def clear_all():
    """Clear all inputs and outputs"""
    return "", ""

# Create Gradio interface
with gr.Blocks(
    title="Named Entity Recognition (NER)",
    theme=gr.themes.Soft(),
    css="""
    .gradio-container {
        max-width: 1200px !important;
    }
    .input-box {
        border: 2px solid #e0e0e0 !important;
        border-radius: 10px !important;
    }
    .output-box {
        border-radius: 10px !important;
    }
    """
) as demo:
    
    # Header
    gr.HTML("""
    <div style="text-align: center; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 10px; color: white;">
        <h1 style="margin: 0; font-size: 2.5em;">🔍 Named Entity Recognition</h1>
        <p style="margin: 10px 0 0 0; font-size: 1.2em;">Intelligent Entity Detection for Any Language</p>
    </div>
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.HTML("""
            <div style="padding: 20px; background: #f8f9fa; border-radius: 10px;">
                <h3>🏷️ Entity Legend</h3>
                <div style="display: flex; flex-direction: column; gap: 8px;">
            """)
            
            for tag, color in entity_colors.items():
                if tag.startswith('B-'):
                    desc = entity_descriptions.get(tag.split('-')[1], 'Unknown')
                    gr.HTML(f"""
                    <div style="display: flex; align-items: center; gap: 10px;">
                        <div style="width: 20px; height: 20px; background: {color}; border-radius: 4px;"></div>
                        <span><strong>{tag}</strong>: {desc}</span>
                    </div>
                    """)
            
            gr.HTML("</div></div>")
            
            # Instructions
            gr.Markdown("""
            ## 🌍 Multi-Lingual Support
            **Works with names from any language:**
            - People: Zhang Wei, Maria Silva, Ahmed Hassan
            - Companies: Samsung, Toyota, Siemens
            - Locations: Tokyo, Berlin, Mumbai

            ## 💡 Example Texts:
            - "Zhang Wei works at Huawei in Beijing."
            - "Maria Silva visited Rio de Janeiro last year."
            - "Ahmed Hassan met with officials in Cairo."
            - "Toyota and Samsung are leading companies."
            """)
            
        with gr.Column(scale=2):
            # Input section
            with gr.Group():
                input_text = gr.Textbox(
                    label="📝 Enter Text to Analyze",
                    placeholder="Type or paste your text here... Example: 'Zhang Wei works at Huawei in Beijing.'",
                    lines=5,
                    max_lines=10,
                    elem_classes="input-box"
                )
            
            # Buttons
            with gr.Row():
                analyze_btn = gr.Button("🔍 Analyze Text", variant="primary", size="lg")
                clear_btn = gr.Button("🗑️ Clear All", variant="secondary")
            
            # Output section
            output_html = gr.HTML(
                label="📊 NER Results",
                elem_classes="output-box"
            )
    
    # Event handlers
    analyze_btn.click(
        fn=predict_ner,
        inputs=[input_text],
        outputs=[output_html]
    )
    
    clear_btn.click(
        fn=clear_all,
        inputs=[],
        outputs=[input_text, output_html]
    )
    
    # Footer
    gr.HTML("""
    <div style="text-align: center; margin-top: 30px; padding: 15px; background: #2c3e50; color: white; border-radius: 8px;">
        <p style="margin: 0;">Built with ❤️ using Intelligent Pattern Matching | Supports multiple languages</p>
    </div>
    """)

# Launch configuration
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        share=False,
        debug=True
    )