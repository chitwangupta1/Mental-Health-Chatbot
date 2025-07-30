from django.shortcuts import render
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import google.generativeai as genai
import kagglehub
import os
from peft import PeftModel

# Load model from KaggleHub
path = kagglehub.model_download("chitwangupta/mentel-harrasment/pyTorch/default/1")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Flan-T5 base and fine-tuned model
base_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large")
flan_model = PeftModel.from_pretrained(base_model, path + "/checkpoint", is_trainable=False).to(device)
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")

# Configure Gemini API
genai.configure(api_key="AIzaSyAER3pJ3aGiHoPlh4v9SuLDmCcIKqKC_UE")
gemini_model = ChatGoogleGenerativeAI(model='models/gemini-2.0-flash')


import re

def checklist(request):
    return render(request, 'Checklist.html')

def home(request):
    return render(request, 'homepage.html')

def Journal(request):
    return render(request, 'Journal.html')


def moodtracker(request):
    return render(request, 'moodtracker.html')

def motivation(request):
    return render(request, 'motivation.html')

def Resources(request):
    return render(request, 'Resources.html')

def format_gemini_response(text):
    """
    Formats Gemini's raw text to ensure clean bullet points and proper structure.
    """
    # Clean up leading/trailing whitespace
    formatted = text.strip()

    # Normalize line breaks
    formatted = re.sub(r'\r\n|\r', '\n', formatted)

    # Convert numbered lists to bullets (if any)
    formatted = re.sub(r'^\d+\.\s+', '- ', formatted, flags=re.MULTILINE)

    # Convert asterisks or dashes at start to bullets
    formatted = re.sub(r'^(\*|-)\s+', '- ', formatted, flags=re.MULTILINE)

    # Add bullet points to lines that seem like list items but lack formatting
    lines = formatted.split('\n')
    formatted_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped and not stripped.startswith(('-', '•', '*')) and not re.match(r'^\d+\.', stripped) and not stripped.endswith(':'):
            # If it's not a heading or already a list item, bullet it
            formatted_lines.append(f"- {stripped}")
        else:
            formatted_lines.append(line)
    formatted = '\n'.join(formatted_lines)

    # Convert **Bold Headers** (if Gemini gives) to just uppercase
    formatted = re.sub(r'\*\*(.*?)\*\*', lambda m: m.group(1).upper(), formatted)

    # Optional: Capitalize the first letter of each line
    formatted = '\n'.join([line.capitalize() if line else '' for line in formatted.split('\n')])

    return formatted



def chatbot_ui(request):
    return render(request, "index.html")


@csrf_exempt
def chatbot_response(request):
    if request.method == "POST":
        data = json.loads(request.body)
        user_input = data.get("message", "")

        prompt = f'Answer this question as if you are a licensed psychologist: "{user_input}"'
        inputs = tokenizer(prompt, return_tensors="pt", padding="max_length", truncation=True, max_length=216).to(device)

        output_ids = flan_model.generate(**inputs, max_new_tokens=70)
        flan_response = tokenizer.decode(output_ids[0], skip_special_tokens=True)

        if len(flan_response.strip()) < 40:
            gemini_response = gemini_model.invoke(user_input)
            gemini_text = str(gemini_response.content)
            formatted_gemini_response = format_gemini_response(gemini_text)
            return JsonResponse({
                "response": formatted_gemini_response,
                "source": "Gemini"
            })

        return JsonResponse({
            "response": flan_response,
            "source": "Flan-T5"
        })

    return JsonResponse({"error": "Invalid request"}, status=400)



@csrf_exempt
def record_feedback(request):
    if request.method == "POST":
        data = json.loads(request.body)
        response_text = data.get("response", "")
        feedback = data.get("feedback", "")
        original_question = data.get("original_question", "")
        model_used = data.get("model_used", "")

        print(f"[Feedback] Response: {response_text} | Feedback: {feedback} | Model Used: {model_used}")

        if feedback == "down":
            try:
                gemini_response = gemini_model.invoke(original_question)
                # gemini_response = gemini_model.generate_content(original_question)
                formatted_gemini_response = format_gemini_response(str(gemini_response.content))

                print(formatted_gemini_response)
                return JsonResponse({
                    "new_response": formatted_gemini_response,
                    "source": "Gemini"
                })

            except Exception as e:
                return JsonResponse({
                    "error": "Failed to regenerate using Gemini",
                    "details": str(e)
                }, status=500)

        return JsonResponse({"status": "feedback recorded"})

    return JsonResponse({"error": "Invalid request"}, status=400)
