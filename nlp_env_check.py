#!/usr/bin/env python3
"""
nlp_env_check.py

Standalone environment & functionality checker for the NLP + Gemini Django project.

Run this inside the same virtualenv used by your Django project.

It will:
 - print versions and import status for key libraries
 - test spaCy model load + NER
 - test sentence-transformers embedder
 - test HuggingFace sentiment pipeline
 - test a few helper functions (PII mask, intent detection, sentiment) from chat.views if available
 - give clear pass/fail output for each step
"""

import sys
import importlib
import traceback

CHECK_MODULES = [
    ("python", None),
    ("spacy", "spacy"),
    ("sentence_transformers", "sentence_transformers"),
    ("transformers", "transformers"),
    ("sklearn (scikit-learn)", "sklearn"),
    ("langdetect", "langdetect"),
    ("torch", "torch"),
    ("google.generativeai", "google.generativeai"),
]

def try_import(name):
    try:
        m = importlib.import_module(name)
        return True, getattr(m, "__version__", None) or "ok"
    except Exception as e:
        return False, str(e)

def print_header(s):
    print("\n" + "="*80)
    print(s)
    print("-"*80)

def run_basic_checks():
    print_header("BASIC ENVIRONMENT")
    print("Python executable:", sys.executable)
    print("Python version:", sys.version.replace("\n", " "))
    print("Current working dir:", __import__("os").getcwd())

    for pretty, mod in CHECK_MODULES[1:]:
        ok, ver = try_import(mod)
        status = "OK" if ok else "MISSING/FAIL"
        print(f"{pretty:25s} -> {status:10s}  ({ver})")

def test_spacy():
    print_header("spaCy MODEL + NER TEST")
    try:
        import spacy
        print("spaCy version:", spacy.__version__)
        try:
            nlp = spacy.load("en_core_web_sm")
            print("en_core_web_sm loaded ✅")
        except Exception as e:
            print("Failed loading en_core_web_sm:", repr(e))
            print("You can install it with: python -m spacy download en_core_web_sm")
            return False

        doc = nlp("My PAN is ABCDE1234F, email alice@example.com, phone +919876543210.")
        ents = [(ent.text, ent.label_) for ent in doc.ents]
        print("NER sample entities:", ents)
        return True
    except Exception as e:
        print("spaCy import failed:", repr(e))
        traceback.print_exc()
        return False

def test_sentence_transformers():
    print_header("sentence-transformers EMBEDDER TEST")
    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np
        print("sentence-transformers import ok")
        try:
            model = SentenceTransformer("all-MiniLM-L6-v2")
            print("Embedder loaded: all-MiniLM-L6-v2")
            emb = model.encode(["hello world", "how to reset password"], convert_to_numpy=True)
            print("Embeddings shape:", getattr(emb, "shape", str(type(emb))))
            # quick cosine similarity
            if hasattr(np, "dot"):
                v0 = emb[0]
                v1 = emb[1]
                # cosine
                cos = np.dot(v0, v1) / (np.linalg.norm(v0) * np.linalg.norm(v1))
                print("cosine(hello world, how to reset password) ~", cos)
            return True
        except Exception as e:
            print("Failed to load/run embedder:", repr(e))
            traceback.print_exc()
            return False
    except Exception as e:
        print("sentence-transformers import failed:", repr(e))
        traceback.print_exc()
        return False

def test_transformers_sentiment():
    print_header("transformers SENTIMENT PIPELINE TEST")
    try:
        from transformers import pipeline as hf_pipeline
        print("transformers import ok")
        try:
            sentiment = hf_pipeline("sentiment-analysis")
            out = sentiment("I love this app, it's great")[0]
            print("sample sentiment output:", out)
            return True
        except Exception as e:
            print("Failed to create/run HF pipeline:", repr(e))
            traceback.print_exc()
            return False
    except Exception as e:
        print("transformers import failed:", repr(e))
        traceback.print_exc()
        return False

def test_local_helpers():
    print_header("TEST HELPERS FROM chat.views (if available)")
    try:
        import chat.views as views
        print("Imported chat.views from project module.")
    except Exception as e:
        print("Could NOT import chat.views:", repr(e))
        print("Will run local fallback tests for PII/intent/sentiment.")
        views = None

    # PII mask test: use views.mask_pii if available, else local fallback
    try:
        if views and hasattr(views, "mask_pii"):
            print("Using chat.views.mask_pii")
            masked, found = views.mask_pii("Contact: alice@example.com and +919876543210 and PAN ABCDE1234F")
            print("masked:", masked)
            print("found:", found)
        else:
            print("chat.views.mask_pii not available. Running fallback mask test.")
            import re
            EMAIL_RE = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")
            PHONE_RE = re.compile(r"(?:\+\d{1,3}[- ]?)?\d{10,12}")
            s = "Contact: alice@example.com and +919876543210 and PAN ABCDE1234F"
            found_e = EMAIL_RE.findall(s)
            found_p = PHONE_RE.findall(s)
            masked = EMAIL_RE.sub("[EMAIL]", s)
            masked = PHONE_RE.sub("[PHONE]", masked)
            print("fallback masked:", masked)
            print("found email:", found_e, "found phone:", found_p)
    except Exception as e:
        print("mask_pii check failed:", repr(e))
        traceback.print_exc()

    # intent detection test
    try:
        if views and hasattr(views, "detect_intent_rule"):
            print("Using chat.views.detect_intent_rule")
            intent, conf = views.detect_intent_rule("I forgot my password please reset password")
            print("intent:", intent, "conf:", conf)
        else:
            print("chat.views.detect_intent_rule not available — running fallback rule check")
            low = "I forgot my password please reset password".lower()
            if "reset password" in low or "forgot password" in low:
                print("fallback: detected reset_password intent with conf ~ 0.95")
            else:
                print("fallback: no intent")
    except Exception as e:
        print("intent test failed:", repr(e))
        traceback.print_exc()

    # sentiment helper
    try:
        if views and hasattr(views, "analyze_sentiment"):
            print("Using chat.views.analyze_sentiment")
            print("sentiment:", views.analyze_sentiment("I love this app it's great"))
        else:
            print("chat.views.analyze_sentiment not available — trying HF pipeline if present")
            try:
                from transformers import pipeline as hf_pipeline
                s = hf_pipeline("sentiment-analysis")
                print("hf sentiment:", s("I love this app it's great")[0])
            except Exception as e:
                print("hf sentiment unavailable; running simple heuristic")
                low = set(__import__("re").findall(r"\w+", "I love this app it's great".lower()))
                pos = len(low & {"good","great","love","happy","excellent"})
                neg = len(low & {"bad","hate","terrible","worst","problem"})
                print("heuristic sentiment:", "positive" if pos>neg else "neutral")
    except Exception as e:
        print("sentiment test failed:", repr(e))
        traceback.print_exc()

    # check if send_message exists (don't call it, just confirm presence)
    if views and hasattr(views, "send_message"):
        print("chat.views.send_message exists (view handler present).")
    else:
        print("chat.views.send_message is NOT present (or import failed).")

def test_generativeai():
    print_header("google.generativeai (Gemini) QUICK CHECK (import only)")
    try:
        import google.generativeai as genai
        print("google.generativeai imported - version:", getattr(genai, "__version__", "unknown"))
        # don't call external API here — only ensure the package is installed
        try:
            # attempt to access GenerativeModel class
            cls = getattr(genai, "GenerativeModel", None)
            print("GenerativeModel available:", bool(cls))
        except Exception as e:
            print("GenerativeModel check error:", repr(e))
    except Exception as e:
        print("google.generativeai import failed:", repr(e))
        traceback.print_exc()

def main():
    run_basic_checks()
    spacy_ok = test_spacy()
    embed_ok = test_sentence_transformers()
    hf_ok = test_transformers_sentiment()
    test_local_helpers()
    test_generativeai()

    print_header("SUMMARY")
    print("spaCy ok:", spacy_ok)
    print("sentence-transformers ok:", embed_ok)
    print("transformers HF pipeline ok:", hf_ok)
    print("\nIf some steps failed, read the error above. Common fixes:")
    print(" - install scikit-learn (pip install scikit-learn)")
    print(" - install torch (pip install torch --index-url https://download.pytorch.org/whl/cpu)")
    print(" - install sentence-transformers (pip install sentence-transformers)")
    print(" - install spacy & model (pip install -U spacy; python -m spacy download en_core_web_sm)")
    print(" - install transformers (pip install transformers)")
    print(" - ensure you're running this script with the same Python interpreter used by Django (same venv)")
    print("="*80)

if __name__ == "__main__":
    main()
