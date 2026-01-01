import gradio as gr
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder

# Load the saved models and preprocessing objects
try:
    linear_model = joblib.load('linear_model.pkl')
    polynomial_model = joblib.load('polynomial_model.pkl')
    label_encoders = joblib.load('label_encoders.pkl')
    poly_features = joblib.load('polynomial_features.pkl')
    scaler = joblib.load('scaler.pkl')
    print("✅ All models loaded successfully!")
except Exception as e:
    print(f"❌ Error loading models: {e}")
    raise e

# Define feature names in correct order (as used during training)
feature_names = [
    'Hours_Studied', 'Attendance', 'Parental_Involvement', 'Access_to_Resources',
    'Extracurricular_Activities', 'Sleep_Hours', 'Previous_Scores', 'Motivation_Level',
    'Internet_Access', 'Tutoring_Sessions', 'Family_Income', 'Teacher_Quality',
    'School_Type', 'Peer_Influence', 'Physical_Activity', 'Learning_Disabilities',
    'Parental_Education_Level', 'Distance_from_Home', 'Gender'
]

def predict_exam_score(
    hours_studied, attendance, parental_involvement, access_to_resources,
    extracurricular_activities, sleep_hours, previous_scores, motivation_level,
    internet_access, tutoring_sessions, family_income, teacher_quality,
    school_type, peer_influence, physical_activity, learning_disabilities,
    parental_education_level, distance_from_home, gender, model_type
):
    """
    Predict exam score based on input features
    """
    try:
        # Create input dictionary
        input_dict = {
            'Hours_Studied': hours_studied,
            'Attendance': attendance,
            'Parental_Involvement': parental_involvement,
            'Access_to_Resources': access_to_resources,
            'Extracurricular_Activities': extracurricular_activities,
            'Sleep_Hours': sleep_hours,
            'Previous_Scores': previous_scores,
            'Motivation_Level': motivation_level,
            'Internet_Access': internet_access,
            'Tutoring_Sessions': tutoring_sessions,
            'Family_Income': family_income,
            'Teacher_Quality': teacher_quality,
            'School_Type': school_type,
            'Peer_Influence': peer_influence,
            'Physical_Activity': physical_activity,
            'Learning_Disabilities': learning_disabilities,
            'Parental_Education_Level': parental_education_level,
            'Distance_from_Home': distance_from_home,
            'Gender': gender
        }
        
        # Convert to DataFrame
        input_df = pd.DataFrame([input_dict])
        
        # Encode categorical variables using saved label encoders
        for column in input_df.columns:
            if column in label_encoders:
                # Handle unseen labels by mapping to most common class
                try:
                    input_df[column] = label_encoders[column].transform([input_df[column].iloc[0]])[0]
                except ValueError:
                    # If unseen label, use the first class (usually most common)
                    input_df[column] = 0
        
        # Ensure correct column order
        input_df = input_df[feature_names]
        
        # Make prediction based on model type
        if model_type == "Polynomial Regression":
            # Transform features for polynomial regression
            input_poly = poly_features.transform(input_df)
            prediction = polynomial_model.predict(input_poly)[0]
        else:
            # Use linear regression
            prediction = linear_model.predict(input_df)[0]
        
        # Ensure prediction is within reasonable bounds (0-100)
        prediction = max(0, min(100, prediction))
        
        # Performance interpretation
        if prediction >= 90:
            performance = "🎉 Excellent! Outstanding performance!"
            color = "#10B981"
        elif prediction >= 80:
            performance = "👍 Very Good! Strong performance!"
            color = "#34D399"
        elif prediction >= 70:
            performance = "✅ Good! Solid performance."
            color = "#60A5FA"
        elif prediction >= 60:
            performance = "⚠️ Average. Room for improvement."
            color = "#FBBF24"
        else:
            performance = "🚨 Needs Attention. Consider additional support."
            color = "#EF4444"
        
        return (
            f"<div style='text-align: center; padding: 20px; background-color: {color}20; border-radius: 10px; border-left: 4px solid {color};'>"
            f"<h2 style='color: {color}; margin-bottom: 10px;'>Predicted Exam Score</h2>"
            f"<h1 style='color: {color}; font-size: 48px; margin: 10px 0;'>{prediction:.1f}/100</h1>"
            f"<p style='color: #666; font-size: 16px;'>{performance}</p>"
            f"<p style='color: #888; font-size: 14px; margin-top: 10px;'>Model Used: {model_type}</p>"
            f"</div>"
        )
        
    except Exception as e:
        return f"❌ Error in prediction: {str(e)}"

# Create the Gradio interface
with gr.Blocks(theme=gr.themes.Soft(), title="Student Grade Predictor") as demo:
    gr.Markdown(
        """
        # 🎓 Student Grade Predictor
        ### Predict student exam scores using Linear & Polynomial Regression
        
        Fill in the student information below to predict their exam performance.
        """
    )
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📊 Academic Factors")
            hours_studied = gr.Slider(0, 30, value=10, label="Hours Studied per Week", info="Weekly study hours")
            attendance = gr.Slider(0, 100, value=85, label="Attendance Rate (%)", info="Class attendance percentage")
            previous_scores = gr.Slider(0, 100, value=75, label="Previous Scores", info="Average of previous exam scores")
            tutoring_sessions = gr.Slider(0, 10, value=2, label="Tutoring Sessions", info="Weekly tutoring sessions")
            
            gr.Markdown("### 🏫 School Environment")
            teacher_quality = gr.Radio(["Low", "Medium", "High"], value="Medium", label="Teacher Quality")
            school_type = gr.Radio(["Public", "Private"], value="Public", label="School Type")
            peer_influence = gr.Radio(["Positive", "Neutral", "Negative"], value="Neutral", label="Peer Influence")
        
        with gr.Column(scale=1):
            gr.Markdown("### 🏠 Home & Personal Factors")
            parental_involvement = gr.Radio(["Low", "Medium", "High"], value="Medium", label="Parental Involvement")
            access_to_resources = gr.Radio(["Low", "Medium", "High"], value="Medium", label="Access to Resources")
            family_income = gr.Radio(["Low", "Medium", "High"], value="Medium", label="Family Income")
            parental_education_level = gr.Radio(["High School", "Bachelor's", "Master's", "PhD"], value="Bachelor's", label="Parental Education Level")
            
            gr.Markdown("### 💪 Health & Activities")
            sleep_hours = gr.Slider(0, 12, value=7, label="Sleep Hours per Night", info="Average nightly sleep")
            physical_activity = gr.Slider(0, 20, value=5, label="Physical Activity (hours/week)", info="Weekly exercise hours")
            extracurricular_activities = gr.Radio(["Yes", "No"], value="Yes", label="Extracurricular Activities")
        
        with gr.Column(scale=1):
            gr.Markdown("### 🔧 Additional Factors")
            motivation_level = gr.Radio(["Low", "Medium", "High"], value="Medium", label="Motivation Level")
            internet_access = gr.Radio(["Yes", "No"], value="Yes", label="Internet Access")
            learning_disabilities = gr.Radio(["Yes", "No"], value="No", label="Learning Disabilities")
            distance_from_home = gr.Radio(["Near", "Moderate", "Far"], value="Moderate", label="Distance from Home")
            gender = gr.Radio(["Male", "Female"], value="Male", label="Gender")
            
            gr.Markdown("### 🤖 Model Selection")
            model_type = gr.Radio(
                ["Linear Regression", "Polynomial Regression"], 
                value="Polynomial Regression", 
                label="Select Prediction Model"
            )
            
            predict_btn = gr.Button("🎯 Predict Exam Score", variant="primary", size="lg")
    
    with gr.Row():
        output = gr.HTML(label="Prediction Result")
    
    # Set up prediction function
    predict_btn.click(
        fn=predict_exam_score,
        inputs=[
            hours_studied, attendance, parental_involvement, access_to_resources,
            extracurricular_activities, sleep_hours, previous_scores, motivation_level,
            internet_access, tutoring_sessions, family_income, teacher_quality,
            school_type, peer_influence, physical_activity, learning_disabilities,
            parental_education_level, distance_from_home, gender, model_type
        ],
        outputs=output
    )
    
    gr.Markdown(
        """
        ---
        ### 📈 About the Models
        - **Linear Regression**: Simple, interpretable model showing linear relationships
        - **Polynomial Regression**: Captures complex non-linear patterns in student performance
        
        *Note: Predictions are based on machine learning models and should be used as guidance rather than absolute certainty.*
        """
    )

if __name__ == "__main__":
    demo.launch(share=True, server_name="0.0.0.0", server_port=7860)