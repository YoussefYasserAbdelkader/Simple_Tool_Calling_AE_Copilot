import json
import subprocess
import time
from typing import Optional, List
from pydantic import ValidationError

from ae_xsd_schema import (
    AeCpuCluster,
    AeChipletType,
)

# --------- Normalizer for Chiplets ---------
def normalize_chiplet(data: dict) -> dict:
    """Fix common LLM mistakes in chiplet JSON before validation."""

    # ---- Fix typo key ----
    if "ucei_interface" in data:
        data["ucie_interface"] = data.pop("ucei_interface")

    # ---- Normalize short_name ----
    if isinstance(data.get("short_name"), str):
        data["short_name"] = {"name": data["short_name"]}

    # ---- Normalize axi_bus ----
    if "axi_bus" in data:
        axi = data["axi_bus"]
        if "value" in axi and "unit" in axi:
            if axi["unit"].lower().startswith("byte"):
                data["axi_bus"] = {"width": axi["value"]}
            elif axi["unit"].lower() in ["hz", "mhz"]:
                data["axi_bus"] = {"frequency": axi["value"]}

    # ---- Normalize ethernet_interface ----
    if "ethernet_interface" in data:
        if isinstance(data["ethernet_interface"], str):
            data["ethernet_interface"] = {"mode": data["ethernet_interface"]}
        elif isinstance(data["ethernet_interface"], dict):
            mode = data["ethernet_interface"].get("mode")
            if mode in ["enabled", "on", "true"]:
                data["ethernet_interface"]["mode"] = "simulated"
            elif mode in ["disabled", "off", "false"]:
                data["ethernet_interface"]["mode"] = "native"

    # ---- Normalize ucie_interface ----
    if "ucie_interface" in data:
        if isinstance(data["ucie_interface"], str):
            data["ucie_interface"] = {"mode": data["ucie_interface"]}
        elif isinstance(data["ucie_interface"], dict):
            mode = data["ucie_interface"].get("mode")
            if mode == "device":  
                data["ucie_interface"]["mode"] = "endpoint"
            elif mode not in ["host", "endpoint"]:
                data["ucie_interface"]["mode"] = "host"

    return data


# --------- Prompt Template ---------
TOOL_DOC = """
Available tools (schemas):

1) add_cpu_cluster -> conforms to AeCpuCluster schema
   Schema fields:
     - short_name { "name": string }
     - frequency { "value": number, "unit": "MHz" }
     - cores_per_cluster: integer

2) add_chiplet -> conforms to AeChipletType schema
   Schema fields:
     - short_name { "name": string }
     - axi_bus { "width": integer, "frequency": integer }
     - ethernet_interface { "mode": "simulated" | "native" }
     - ucie_interface { "mode": "host" | "endpoint" }
   Optional:
     - cpu_cluster
     - generic_hardware
    Rules:
     - Output ONLY JSON, no text or explanation.

"""

PROMPT_TEMPLATE = """You are an assistant that converts user instruction into schema JSON.
{tool_doc}

User instruction:
\"\"\"{user_input}\"\"\"

Return exactly one JSON object.
"""

# --------- JSON Parsing ---------
def extract_first_json(text: str) -> Optional[str]:
    try:
        start = text.index("{")
        end = text.rfind("}")
        return text[start:end+1]
    except Exception:
        return None
def auto_close_json(snippet: str) -> str:
    """Append missing closing brackets/braces to complete the JSON."""
    open_braces = snippet.count("{")
    close_braces = snippet.count("}")
    open_brackets = snippet.count("[")
    close_brackets = snippet.count("]")
    snippet += "}" * (open_braces - close_braces)
    snippet += "]" * (open_brackets - close_brackets)
    return snippet

def parse_json_schema(output: str, schema) -> Optional[dict]:
    j = extract_first_json(output)
    if not j:
        return None
    j = auto_close_json(j)
    try:
        data = json.loads(j)
        if isinstance(data, dict) and len(data) == 1:
            key = next(iter(data))
            if key in ("add_chiplet", "add_cpu_cluster"):
                data = data[key]
        if schema is AeChipletType:
            data = normalize_chiplet(data)  
        obj = schema(**data)
        return obj.model_dump(mode="json")
    except (ValidationError, Exception) as e:
        print("❌ JSON parse failed:", e)
        return None

# --------- Ollama Runner ---------
def ollama_run(model: str, prompt: str) -> str:
    cmd = ["ollama", "run", model]
    try:
        res = subprocess.run(
            cmd,
            input=prompt,
            text=True,
            encoding="utf-8",
            capture_output=True,
            check=True
        )
        return res.stdout.strip()
    except subprocess.CalledProcessError as e:
        print("Ollama error:", e.stderr)
        return ""

# --------- Scenarios ---------
SCENARIOS = [
    ("S1_cluster", "Create a CPU cluster named C1 with frequency 2000 MHz and 4 cores per cluster."),
    ("S2_chiplet", "Add a GPU chiplet G1 with AXI bus width 64, frequency 1000000, ethernet interface enabled, and ucie interface in host mode."),
    ("S3_cluster", "Create another CPU cluster named C2 with frequency 1500 MHz and 2 cores per cluster."),
    ("S4_chiplet_multi", "Create an NPU chiplet N1 with AXI bus width 128, frequency 2000000, ethernet disabled, and ucie in device mode."),
]

# --------- Runner ---------
def run_tests(model: str):
    print(f"🔎 Testing model: {model}")
    for sid, user in SCENARIOS:
        print(f"\n--- {sid} ---")
        prompt = PROMPT_TEMPLATE.format(tool_doc=TOOL_DOC, user_input=user)
        start = time.time()
        out = ollama_run(model, prompt)
        elapsed = time.time() - start

        schema = AeCpuCluster if "cluster" in sid else AeChipletType
        result = parse_json_schema(out, schema)

        if result:
            print("✅ Parsed →")
            print(json.dumps(result, indent=2))
        else:
            print("❌ Failed to parse JSON schema.")

        print(f" Time: {elapsed:.2f}s")


# --------- Main ---------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Ollama model name")
    args = parser.parse_args()
    run_tests(args.model)
