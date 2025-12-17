import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()                 
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

models = genai.list_models()

print("\n=== AVAILABLE GEMINI MODELS ===\n")
for m in models:
    # Most SDK versions expose attributes like `name`, `display_name`, etc.
    print("MODEL ID:", getattr(m, "name", None))
    print("  displayName:", getattr(m, "display_name", None))
    print("  description:", getattr(m, "description", None))
    
    # Some SDKs include a list of methods in .supported_generation_methods
    methods = getattr(m, "supported_generation_methods", None)
    print("  supported methods:", methods)

    print("-" * 60)
