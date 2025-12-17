import os
import re
import json
import logging

from django.http import JsonResponse
from django.views.decorators.http import require_POST
from django.conf import settings
from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.models import User

# ================= NLP / AI =================
import spacy
from langdetect import detect as lang_detect
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline as hf_pipeline

# =====================================================
# CONFIG
# =====================================================
BASE_DIR = settings.BASE_DIR
logger = logging.getLogger(__name__)

# =====================================================
# NLP INITIALIZATION
# =====================================================
nlp = spacy.load("en_core_web_sm")
embedder = SentenceTransformer("all-MiniLM-L6-v2")
# sentiment_model = hf_pipeline("sentiment-analysis")

# =====================================================
# LOAD FAQ.JSON
# =====================================================
FAQ_PATH = BASE_DIR / "faq.json"

with open(FAQ_PATH, encoding="utf-8") as f:
    FAQ = json.load(f)

FAQ_KEYS = list(FAQ.keys())
FAQ_ANSWERS = list(FAQ.values())

# Embed ONLY answers (important)
FAQ_TEXTS = [FAQ[key].lower() for key in FAQ_KEYS]
FAQ_EMB = embedder.encode(FAQ_TEXTS, convert_to_numpy=True)

# =====================================================
# INTENT MAP  ✅ (THIS IS THE FIX)
# =====================================================
INTENT_MAP = {

# ================= ACCOUNT =================
"change_password": [
    "change my password", "update password", "reset my login password"
],

"reset_password": [
    "forgot password", "reset my password", "cannot login password"
],

"two_factor_issues": [
    "2fa not working", "otp not coming", "two factor problem"
],

"account_locked": [
    "account locked", "cannot access my account", "account blocked"
],

"suspicious_activity": [
    "unknown login", "someone accessed my account", "suspicious activity"
],

# ================= PROFILE =================
"update_profile": [
    "edit profile", "update my profile details", "change personal info"
],

"change_email": [
    "change email address", "update email id", "modify email"
],

"change_phone": [
    "change phone number", "update mobile number"
],

"profile_picture": [
    "change profile picture", "upload profile photo"
],

"delete_account": [
    "delete my account", "remove my account permanently"
],

# ================= LAB TIMINGS =================
"lab_opening_time": [
    "lab opening time", "when does the lab open", "lab open time","lab start time"
],

"lab_closing_time": [
    "lab closing time", "when does the lab close", "lab close time"
],

# ================= APPOINTMENTS =================
"appointment_needed_for_test": [
    "do i need appointment", "appointment required for test","do i need any appoinment before blood testing"
],

"walk_in_test_available": [
    "can i walk in for test", "walk in test available"
],

"lab_waiting_duration": [
    "waiting time in lab", "how long will i wait in lab"
],

# ================= FASTING =================
"blood_test_fasting_time": [
    "fasting blood test time", "fasting test instructions"
],

"water_during_fasting": [
    "can i drink water fasting", "water allowed during fasting"
],

"coffee_before_blood_test": [
    "coffee before blood test", "tea before blood test"
],

"medicine_before_blood_test": [
    "medicine before blood test", "can i take tablets before test"
],

"smoking_before_blood_test": [
    "smoking before blood test"
],

# ================= SAMPLE COLLECTION =================
"blood_sample_collection_time": [
    "how long does blood collection take", "blood sample duration"
],

"blood_amount_taken": [
    "how much blood will be taken", "blood quantity taken"
],

"blood_sample_safety": [
    "is blood test safe", "blood test safety"
],

"bruise_after_blood_test": [
    "bruise after blood test", "pain after blood test"
],

"weakness_after_blood_test": [
    "feeling weak after blood test"
],

# ================= HOME COLLECTION =================
"home_blood_collection": [
    "home blood collection", "blood test at home"
],

"home_collection_charge": [
    "home collection charges", "extra cost for home sample"
],

"home_collection_reschedule": [
    "reschedule home collection", "change home test time"
],

# ================= REPORTS =================
"blood_report_availability": [
    "when will i get my blood report", "blood report time","when can i get my blood report","How long does it take to get blood test results","How fast will I get my lab report?","How much time for lab results?","Report generation time?"
],

"blood_report_download_process": [
    "download blood report", "get report online"
],

"blood_report_delay_reason": [
    "why report delayed", "blood report not ready"
],

"blood_report_abnormal": [
    "abnormal blood report", "report values abnormal"
],

# ================= TEST TYPES =================
"cbc_blood_test": [
    "what is cbc test", "cbc blood test meaning"
],

"sugar_fasting_test": [
    "fasting sugar test", "blood sugar fasting"
],

"hba1c_blood_test": [
    "hba1c test", "average sugar test"
],

"thyroid_blood_test": [
    "thyroid test", "tsh test"
],

"cholesterol_blood_test": [
    "cholesterol test", "lipid profile test"
],

# ================= ILLNESS =================
"dengue_blood_test": [
    "dengue blood test", "test for dengue"
],

"malaria_blood_test": [
    "malaria blood test"
],

"crp_blood_test": [
    "crp test", "inflammation blood test"
],

"esr_blood_test": [
    "esr test"
],

# ================= CONDITIONS =================
"blood_test_during_fever": [
    "blood test during fever"
],

"blood_test_pregnancy_safe": [
    "blood test pregnancy safe"
],

"blood_test_elderly_safe": [
    "blood test for elderly"
],

"blood_test_after_vaccination": [
    "blood test after vaccination"
],
"blood_test_cost": [
  "how much does blood test cost",
  "blood test price",
  "lab test charges",
  "cost of blood test",
  "blood test fees",
  "how expensive is blood test",
  "blood test rate"
],
"full_body_checkup_package": [
  "is there full body checkup package",
  "full body test available",
  "health checkup packages",
  "complete body checkup cost",
  "full body medical package",
  "body checkup plans",
  "annual health checkup"
],
"refund_for_cancelled_test": [
  "how do i get refund if test is cancelled",
  "refund for cancelled test",
  "blood test cancellation refund",
  "what happens if test is cancelled",
  "refund process for cancelled test",
  "cancelled test refund policy"
]
,
"refund_processing_time": [
  "when will i get refund",
  "refund processing time",
  "how long does refund take",
  "refund delay",
  "how many days for refund",
  "refund credit time",
  "money return time"
],
"menstrual_cycle_blood_test": [
  "can i do blood test during periods",
  "blood test during menstruation",
  "is it okay to do blood test while periods",
  "does periods affect blood test"
],
"fasting_mistake_before_test": [
  "i ate food before fasting test",
  "what if fasting was broken",
  "forgot to fast before blood test",
  "ate something before blood test"
],

"water_intake_limit": [
  "how much water can i drink before test",
  "is water allowed during fasting",
  "can i drink water before blood test",
  "water intake before fasting test"
],

"recent_illness_effect": [
  "will fever affect blood test",
  "recent illness blood test impact",
  "does cold affect blood test",
  "was sick recently can i do blood test"
],

"antibiotics_before_test": [
  "can antibiotics affect blood test",
  "taking antibiotics before blood test",
  "medicine effect on blood test",
  "antibiotics and lab results"
],
"dizziness_after_blood_test": [
  "feeling dizzy after blood test",
  "giddiness after blood test",
  "why i feel weak after blood test",
  "head spinning after blood test"
],

"fainting_risk_blood_test": [
  "can someone faint during blood test",
  "risk of fainting during blood test",
  "fear of fainting blood test",
  "is fainting common during blood test"
],

"post_test_food": [
  "can i eat after blood test",
  "food after blood test",
  "what to eat after blood test",
  "can i have breakfast after test"
],

"arm_pain_after_test": [
  "arm pain after blood test",
  "hand pain after blood sample",
  "soreness after blood test",
  "arm hurting after blood test"
],

"driving_after_blood_test": [
  "can i drive after blood test",
  "is it safe to drive after blood test",
  "driving post blood test",
  "can i go to work after blood test"
]
,
"slightly_abnormal_report": [
  "slightly abnormal blood report",
  "mild abnormal values in report",
  "small variation in blood report",
  "borderline blood test results"
],

"compare_old_reports": [
  "compare old and new blood report",
  "previous blood test comparison",
  "difference between reports",
  "track blood report history"
],

"lab_reference_range_difference": [
  "why normal range different in labs",
  "reference range difference",
  "normal value varies between labs",
  "lab range mismatch"
],

"report_units_confusion": [
  "dont understand blood report units",
  "blood test units confusing",
  "what do report numbers mean",
  "lab report values explanation"
],

"self_diagnosis_warning": [
  "can i diagnose myself using report",
  "self diagnosis blood report",
  "is it safe to read report myself",
  "interpret report without doctor"
]
,

"home_collection_availability_area": [
  "is home collection available in my area",
  "home sample collection near me",
  "does home collection work here",
  "home blood test service area"
],

"home_collection_time_slot": [
  "choose time for home collection",
  "home collection time slot",
  "schedule home blood test",
  "preferred time home collection"
],

"home_collection_hygiene": [
  "is home blood test hygienic",
  "home collection safety",
  "cleanliness in home blood test",
  "home blood test infection risk"
],

"home_collection_identity": [
  "how to verify technician",
  "home collection staff id",
  "technician identity verification",
  "is technician verified"
],

"home_collection_multiple_people": [
  "family blood test at home",
  "multiple people home collection",
  "blood test for family at home",
  "home collection for parents"
]
,

"repeat_test_requirement": [
  "why repeat blood test",
  "do i need test again",
  "repeat testing needed",
  "confirm blood test result"
],

"seasonal_effect_on_tests": [
  "does weather affect blood test",
  "seasonal change blood report",
  "summer winter blood test difference",
  "climate impact on blood test"
],

"pregnancy_false_results": [
  "pregnancy affect blood test",
  "false result during pregnancy",
  "blood test changes in pregnancy",
  "pregnancy lab report confusion"
],

"age_related_test_changes": [
  "blood test range by age",
  "elderly blood test values",
  "age impact on blood report",
  "normal range by age"
],

"test_accuracy_percentage": [
  "how accurate are blood tests",
  "blood test accuracy",
  "lab test reliability",
  "are blood test results correct"
]
,

"reschedule_test": [
  "reschedule blood test",
  "change test appointment",
  "modify blood test date",
  "reschedule lab test"
],

"missed_appointment": [
  "missed blood test appointment",
  "did not attend test",
  "what if appointment missed",
  "missed lab test"
],

"bulk_test_booking": [
  "book multiple tests",
  "bulk blood test booking",
  "many tests in one booking",
  "multiple lab tests together"
],

"family_test_booking": [
  "book blood test for family",
  "family lab test booking",
  "blood test for parents",
  "tests for family members"
],

"test_booking_confirmation": [
  "blood test booking confirmation",
  "how do i know test is booked",
  "appointment confirmation message",
  "lab booking confirmation"
]
,

"emergency_test_support": [
  "is blood test emergency service",
  "urgent medical emergency",
  "lab test emergency care",
  "blood test for emergency"
],

"critical_value_alert": [
  "critical blood report value",
  "dangerous blood test result",
  "critical level in report",
  "urgent abnormal result"
],

"lab_vs_hospital_test": [
  "difference between lab and hospital test",
  "lab test vs hospital test",
  "which is better lab or hospital",
  "hospital blood test vs lab"
],

"false_positive_test": [
  "false positive blood test",
  "wrong positive result",
  "incorrect blood test result",
  "false alarm in lab report"
],

"false_negative_test": [
  "false negative blood test",
  "wrong negative result",
  "test shows normal but symptoms",
  "false normal blood test"
]



}

def exact_intent_match(user_text):
    text = user_text.lower()
    for intent, phrases in INTENT_MAP.items():
        for phrase in phrases:
            if phrase in text:
                return FAQ.get(intent)
    return None

# =====================================================
# PII MASKING
# =====================================================
EMAIL_RE = re.compile(r"\S+@\S+")
PHONE_RE = re.compile(r"\b\d{10,12}\b")

def mask_pii(text):
    text = EMAIL_RE.sub("[EMAIL]", text)
    text = PHONE_RE.sub("[PHONE]", text)
    return text

# =====================================================
# LANGUAGE DETECTION
# =====================================================
def detect_language(text):
    try:
        return lang_detect(text)
    except:
        return "en"

# =====================================================
# SEMANTIC FAQ (FALLBACK ONLY)
# =====================================================
def semantic_faq(user_text):
    q_emb = embedder.encode([user_text.lower()], convert_to_numpy=True)
    sims = cosine_similarity(q_emb, FAQ_EMB)[0]

    idx = sims.argmax()
    score = sims[idx]

    if score >= 0.30:
        return FAQ_ANSWERS[idx]

    return None

# =====================================================
# NLP ENGINE
# =====================================================
def fallback_nlp_response(user_text):
    text = user_text.lower()

    # 1️⃣ EXACT INTENT MATCH (HIGHEST PRIORITY)
    intent_answer = exact_intent_match(text)
    if intent_answer:
        return intent_answer, True

    # 2️⃣ GREETINGS
    tokens = [t.text.lower() for t in nlp(text)]
    if any(t in tokens for t in ["hi","hiii","hai","hey there","hii", "hello", "hey"]):
        return "Hello! How can I help you today?", True

    # 3️⃣ SEMANTIC MATCH (SAFE FALLBACK)
    faq_answer = semantic_faq(text)
    if faq_answer:
        return faq_answer, True

    # 4️⃣ SENTIMENT FALLBACK
    # sentiment = sentiment_model(text[:512])[0]["label"]
    # if sentiment == "NEGATIVE":
    #     return (
    #         "I’m sorry for the inconvenience. Please tell me more so I can assist you."
    #     ), True

    return "I’m here to help. Please provide more details.", False

# =====================================================
# CHAT ENDPOINT
# =====================================================
@require_POST
def send_message(request):
    body = json.loads(request.body.decode())
    user_text = body.get("message", "").strip()

    if not user_text:
        return JsonResponse({"response": "Empty message"})

    masked_text = mask_pii(user_text)

    response, handled = fallback_nlp_response(masked_text)
    return JsonResponse({"response": response})

# =====================================================
# AUTH VIEWS (UNCHANGED)
# =====================================================
def login_page(request):
    if request.method == "POST":
        user = authenticate(
            request,
            username=request.POST.get("username"),
            password=request.POST.get("password")
        )
        if user:
            login(request, user)
            return redirect("chat")
        return render(request, "chat/login.html", {"error": "Invalid credentials"})
    return render(request, "chat/login.html")

def register_page(request):
    if request.method == "POST":
        username = request.POST.get("username")
        password = request.POST.get("password")
        confirm = request.POST.get("confirm_password")

        if password != confirm:
            return render(request, "chat/register.html", {"error": "Passwords do not match"})

        User.objects.create_user(username=username, password=password)
        return redirect("login")

    return render(request, "chat/register.html")

def chat_page(request):
    if not request.user.is_authenticated:
        return redirect("login")
    return render(request, "chat/chat_page.html")

def logout_view(request):
    logout(request)
    return redirect("login")
