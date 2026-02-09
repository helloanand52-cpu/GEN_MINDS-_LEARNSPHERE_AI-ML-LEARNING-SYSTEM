import google.generativeai as genai
from google.api_core import exceptions as google_exceptions
import re
import time
import logging

logger = logging.getLogger(__name__)


def call_genai(api_key, topic, length, mode, previous_attempts=None):
    """
    Call Google Gemini AI to generate ML learning content

    Args:
        api_key: Gemini API key
        topic: ML topic to explain
        length: Length of explanation (Brief, Detailed, Comprehensive)
        mode: Output mode (Text explanation, Code with explanation, Audio, Image Explanation)
        previous_attempts: Previous attempts (for retry logic)

    Returns:
        Tuple of (briefing, code_content, audio_script, image_prompts)
    """

    genai.configure(api_key=api_key)

    # Enhanced prompt construction
    base_prompt = f"""
You are an expert Machine Learning tutor providing educational content ONLY for topics related to Artificial Intelligence (AI), Machine Learning (ML), and Deep Learning.

IMPORTANT RESTRICTIONS:
- You MUST ONLY respond to topics related to AI, ML, or Deep Learning.
- If the topic "{topic}" is NOT related to AI/ML/DL, respond with:
  "I apologize, but I can only provide information about Artificial Intelligence, Machine Learning, and Deep Learning."
- Do NOT provide information outside of AI/ML/DL domains.

Topic: "{topic}"
Required format: {mode}
Explanation depth: {length}

Teaching Guidelines:
- Start with a clear learning objective
- Provide structured explanations with examples
- Use appropriate technical depth for the topic
- Include practical applications when relevant
- Ensure accuracy and clarity
- Format output as clean text WITHOUT markdown symbols like #, *, **, etc.
- Use plain text formatting with clear paragraphs and line breaks
- For headings, use ALL CAPS or underlines instead of # symbols
- For emphasis, use Uppercase or quotation marks instead of * or **
"""

    code_instruction = ""
    audio_instruction = ""
    image_instruction = ""

    if mode == "Code with explanation":
        code_instruction = f"""
You MUST also generate a Python program that demonstrates how {topic} works.
Before the Python code block, provide a detailed but beginner-friendly explanation in plain text.
This explanation should cover the model, key functions, and evaluation.
The Python code itself should be enclosed in a single ```python``` code block.
Include helpful comments in the code explaining key steps.
Show expected outputs or results where applicable.
"""

    elif mode == "Audio":
        audio_instruction = """
Provide an AUDIO SCRIPT clearly labeled as:
Audio Script:
The script should sound natural and suitable for narration.
"""

    elif mode == "Image Explanation":
        image_instruction = """
Provide IMAGE PROMPTS clearly labeled as:
IMG-PROMPT:
Each prompt should describe a clear educational diagram or illustration.
"""

    prompt = base_prompt + code_instruction + audio_instruction + image_instruction

    safety_settings = [
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    ]

    max_retries = 3
    attempt = 0

    while attempt < max_retries:
        try:
            model = genai.GenerativeModel(
                model_name="gemini-2.0-flash-exp",
                generation_config=genai.types.GenerationConfig(),
                safety_settings=safety_settings
            )

            response = model.generate_content(prompt)
            full_response_text = response.text

            # Initialize return values
            briefing = full_response_text
            code_content = ""
            audio_script = ""
            image_prompts = []

            # Parse different modes
            if mode == "Code with explanation":
                code_match = re.search(r"```python(.*?)```", briefing, re.DOTALL)
                if code_match:
                    code_content = code_match.group(1).strip()
                    briefing = briefing.replace(code_match.group(0), "").strip()

            elif mode == "Audio":
                if "Audio Script:" in briefing:
                    parts = briefing.split("Audio Script:", 1)
                    briefing = parts[0].strip()
                    audio_script = parts[1].strip()

            elif mode == "Image Explanation":
                marker = "IMG-PROMPT:"
                if marker in briefing:
                    first_marker_pos = briefing.find(marker)
                    briefing_text = briefing[:first_marker_pos].strip()
                    prompts_text = briefing[first_marker_pos:]
                    image_prompts = [
                        p.strip() for p in prompts_text.split(marker) if p.strip()
                    ]
                    briefing = briefing_text

            return briefing, code_content, audio_script, image_prompts

        except google_exceptions.ResourceExhausted as e:
            logger.warning(
                f"Rate limit hit. Waiting 60 seconds before retrying... ({attempt + 1}/{max_retries})"
            )
            time.sleep(60)
            attempt += 1
            continue

        except Exception as e:
            logger.error(f"An unexpected error occurred during the Gemini API call: {e}")
            return None

    logger.error("API call failed after multiple retries due to rate limiting.")
    return None