import json
import subprocess
import re
import time
from typing import Optional, Dict, Any
from pydantic import BaseModel, ValidationError

# --------- Tool schemas (Pydantic) ---------
class Controller(BaseModel):
    type: str
    count: int

class Memory(BaseModel):
    type: str
    size_mb: int

class ECU(BaseModel):
    id: str
    cpu: Optional[str] = None
    memories: Optional[list[Memory]] = []
    controllers: Optional[list[Controller]] = []
    notes: Optional[str] = None
    uncertainty: Optional[list[str]] = []

class ToolCall(BaseModel):
    tool_name: str
    parameters: dict

# --------- Prompt template ---------
TOOL_DOC = """
Available tools (call exactly one):

1) add_ecu
   parameters:
     - id (string)
     - cpu (string, optional)
     - memories (list of {type:string, size_mb:int}, optional)
     - controllers (list of {type:string, count:int}, optional)
     - uncertainty (list of strings, optional)

2) add_protocol
   parameters:
     - protocol (string or list of strings)  // allowed values: CAN, Ethernet, LIN, FlexRay

3) add_hwip_block
   parameters:
     - ecu_id (string)
     - type (string)  // e.g., I2C, SPI, UART
     - count (int)

4) validate_configuration
   parameters:
     - config (object)  // pass current config object

5) create_configuration
   parameters:
     - ecu_count (int, optional)
     - ecus (list of ECU objects)
     - protocols (list of strings)
     - power_budget_w (float, optional)
     - notes (string, optional)

Rules:
- You must output ONLY a single JSON object: 
  { "tool_name": "tool_name_here", "parameters": { ... } }
- DO NOT output any extra text, explanation, or markdown.
- If uncertain, set a field to null and include its name in "uncertainty".
"""

PROMPT_TEMPLATE = """You are an assistant that converts a user instruction into a single tool call.
{tool_doc}

User instruction:
\"\"\"{user_input}\"\"\"

Return exactly one JSON object representing the tool call.
"""

# --------- JSON extractor ---------
def extract_first_json(text: str) -> Optional[str]:
    try:
        start = text.index("{")
        end = text.rfind("}")
        return text[start:end+1]
    except Exception:
        return None

def parse_tool_call(output: str) -> Optional[ToolCall]:
    j = extract_first_json(output)
    if not j:
        return None
    try:
        data = json.loads(j)
        return ToolCall(**data)
    except Exception as e:
        print("❌ JSON parse failed:", e)
        print("Raw text:", output)
        return None

# --------- Ollama runner ---------
def ollama_run(model: str, prompt: str) -> str:
    cmd = ["ollama", "run", model]
    try:
        res = subprocess.run(cmd, input=prompt, text=True, capture_output=True, check=True)
        return res.stdout.strip()
    except subprocess.CalledProcessError as e:
        print("Ollama error:", e.stderr)
        return ""

# --------- Test scenarios ---------
SCENARIOS = [
    ("S1_simple_ecu", "Create an ECU with Cortex-M4 and 2 CAN controllers."),
    ("S2_protocols", "The system should support both CAN and Ethernet communication."),
    ("S3_memory", "Add an ECU with Cortex-R5, 4MB Flash memory and 2 SPI controllers."),
    ("S4_hwip", "Attach an I2C block to ECU-1."),
    ("S5_validate", "Check if the configuration is valid."),
    ("S6_ambiguous", "Create an ECU with Cortex-M6 and 3 CAN controllers."),
]

# --------- Runner ---------
def run_tests(model: str):
    print(f" Testing model: {model}")
    for sid, user in SCENARIOS:
        print(f"\n--- {sid} ---")
        prompt = PROMPT_TEMPLATE.format(tool_doc=TOOL_DOC, user_input=user)
        start = time.time()
        out = ollama_run(model, prompt)
        elapsed = time.time() - start
        call = parse_tool_call(out)
        if call:
            print("✅ Tool:", call.tool_name)
            print("Parameters:", call.parameters)
        else:
            print("❌ Failed to parse tool call.")
        print(f" Time: {elapsed:.2f}s")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Ollama model name")
    args = parser.parse_args()
    run_tests(args.model)
