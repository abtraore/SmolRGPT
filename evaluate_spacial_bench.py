from pydantic import BaseModel, validator
from typing import Literal, Union

from openai import OpenAI
from statistics import mean
from pathlib import Path


from tqdm import tqdm

import pandas as pd
import json

extraction_prompt = """
Spatial Answer Extraction Task

You are extracting answers from spatial reasoning responses. Based on the question category and class, extract the appropriate answer.

QUANTITATIVE CLASSES:
Extract only the numeric value and the unit exactly as stated in the answer.
Return your output as a JSON object with two fields:
  - "value": the number only (no text)
  - "unit": the unit string (e.g., "m", "cm", "ft", "in", or "" if none given)
Examples:
- "The distance is 10 meters" → {"value": 10, "unit": "m"}
- "Width: 30 cm" → {"value": 30, "unit": "cm"}
- "5 ft" → {"value": 5, "unit": "ft"}
- "The answer is 12" → {"value": 12, "unit": ""}

QUALITATIVE CLASSES:
For PREDICATES (extract yes/no or true/false):
- front_predicate: Is object in front? (yes/no)
- behind_predicate: Is object behind? (yes/no)
- left_predicate: Is object to the left? (yes/no)
- right_predicate: Is object to the right? (yes/no)
- above_predicate: Is object above? (yes/no)
- below_predicate: Is object below? (yes/no)
- big_predicate: Is object big? (yes/no)
- small_predicate: Is object small? (yes/no)
- tall_predicate: Is object tall? (yes/no)
- short_predicate: Is object short? (yes/no)
- wide_predicate: Is object wide? (yes/no)
- thin_predicate: Is object thin? (yes/no)

For CHOICES (extract the chosen object/option):
- left_choice: Which object is on the left? Extract object name
- right_choice: Which object is on the right? Extract object name
- above_choice: Which object is above? Extract object name
- below_choice: Which object is below? Extract object name
- tall_choice: Which object is tallest? Extract object name
- short_choice: Which object is shortest? Extract object name

For DIRECTION (extract clock position):
- direction: Extract the clock hour position (e.g., "7" from "at 7 o'clock")

EXTRACTION RULES:
1. For quantitative: Extract ONLY the numerical value, no units
2. For predicates: Extract ONLY "yes" or "no" 
3. For choices: Extract ONLY the object name/identifier
4. For direction: Extract ONLY the hour number from clock position (1-12)

Examples:
- direction: "Region [1] is roughly at 7 o'clock from Region [0]" → 7
- distance_data: "The distance is 5.2 meters" → 5.2
- left_predicate: "Yes, the object is to the left" → yes
- tall_choice: "Region [3] is the tallest" → Region [3]

Input format:
Category: [category]
Class: [quantitative/qualitative]
Answer: [model's response]

Output: [extracted answer only]
"""


# Example usage with Pydantic models:
class QuantitativeAnswer(BaseModel):
    value: float
    unit: str


class QualitativeAnswer(BaseModel):
    answer: str


class PredicateAnswer(BaseModel):
    answer: Literal["yes", "no"]


# Updated Pydantic models:
class DirectionAnswer(BaseModel):
    answer: int  # Clock hour (1-12)

    @validator("answer")
    def validate_clock_hour(cls, v):
        if not 1 <= v <= 12:
            raise ValueError("Clock hour must be between 1 and 12")
        return v


# For more flexible extraction if needed:
class ClockDirectionAnswer(BaseModel):
    answer: Union[int, str]  # Can handle "7" or "7 o'clock"

    @validator("answer")
    def extract_hour(cls, v):
        if isinstance(v, str):
            # Extract number from strings like "7 o'clock" or "7"
            import re

            match = re.search(r"(\d+)", v)
            if match:
                return int(match.group(1))
        return int(v)


def convert_to_key_based_comprehension(json_array):
    """
    Convert using dictionary comprehension - more concise approach.
    """
    return {item["id"]: item for item in json_array if "id" in item}


def get_extraction_class(category: str, class_type: str):
    if class_type == "quantitative":
        return QuantitativeAnswer
    elif category == "direction":
        return DirectionAnswer
    elif "predicate" in category:
        return PredicateAnswer
    else:  # choices
        return QualitativeAnswer


def normalize_to_meters(value, unit):
    unit = unit.lower().strip()
    if unit in ("m", "meter", "meters", ""):
        return value
    elif unit in ("cm", "centimeter", "centimeters"):
        return value / 100.0
    elif unit in ("mm", "millimeter", "millimeters"):
        return value / 1000.0
    elif unit in ("km", "kilometer", "kilometers"):
        return value * 1000.0
    elif unit in ("ft", "feet", "foot"):
        return value * 0.3048
    elif unit in ("in", "inch", "inches"):
        return value * 0.0254
    else:
        return value  # fallback


def qwen_extract(
    client: OpenAI,
    answer: str,
    question: str = None,
    category: str = None,
    class_type: str = None,
):

    parsing_class = get_extraction_class(category, class_type)

    completion = client.chat.completions.create(
        model="qwen2.5:14b-instruct",
        messages=[
            {"role": "system", "content": extraction_prompt},
            {"role": "user", "content": f"question:{question}"},
            {"role": "user", "content": f"answer:{answer}"},
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "answer",
                "schema": parsing_class.model_json_schema(),
            },
        },
        temperature=0.1,
        max_completion_tokens=10,
    )

    repsonse = parsing_class.model_validate_json(completion.choices[0].message.content)

    return repsonse


client = OpenAI(api_key="ollama", base_url="http://192.168.8.202:1997/v1")


filename = "answer_spacial_bench.csv"


meta = pd.read_csv(filename, sep=";")


with open(
    "/home/atraore/embia/NanoRGPT/datasets/spacial-rgpt-bench-smolRGPT/SpatialRGPT-Bench_v1.json"
) as f:
    gd = json.load(f)


mapping = {
    "distance_data": "direct_distance",
    "front_predicate": "behind_front",
    "small_predicate": "big_small",
    "horizontal_distance_data": "hor_distance",
    "left_choice": "left_right",
    "right_predicate": "left_right",
    "thin_predicate": "wide_thin",
    "width_data": "width",
    "above_predicate": "below_above",
    "below_predicate": "below_above",
    "height_data": "height",
    "direction": "direction",
    "short_choice": "tall_short",
    "below_choice": "below_above",
    "big_predicate": "big_small",
    "tall_choice": "tall_short",
    "left_predicate": "left_right",
    "above_choice": "below_above",
    "vertical_distance_data": "ver_distance",
    "behind_predicate": "behind_front",
    "tall_predicate": "tall_short",
    "right_choice": "left_right",
    "short_predicate": "tall_short",
    "wide_predicate": "wide_thin",
}

m_keys = list(set(list(mapping.values())))

# 1) which categories are quantitative?
quantitative_cats = {
    "direct_distance",
    "hor_distance",
    "ver_distance",
    "width",
    "height",
}

# initialise the error collector
rep_relerr = {k: [] for k in m_keys}

rep = {k: [] for k in m_keys}
for item in tqdm(gd):

    class_type = item["qa_info"]["type"]

    category = item["qa_info"]["category"]

    question = item["conversations"][0]["value"]
    gd_answer = item["conversations"][1]["value"]
    ai_answer = meta[meta["id"] == item["id"]]["output"].iloc[0]

    response_ai = qwen_extract(client, ai_answer, question, category, class_type)
    response_gd = qwen_extract(client, gd_answer, question, category, class_type)

    if class_type == "qualitative":
        if response_ai.answer == response_gd.answer:
            rep[mapping[category]].append(1)
        else:
            rep[mapping[category]].append(0)
    else:  # quantitative
        try:
            ai_val = normalize_to_meters(response_ai.value, response_ai.unit)
            gd_val = normalize_to_meters(response_gd.value, response_gd.unit)
            if gd_val is not None and gd_val != 0 and ai_val is not None:
                rel_err = abs(ai_val - gd_val) / abs(gd_val)
                rep_relerr[mapping[category]].append(rel_err)  #  <-- NEW
                rep[mapping[category]].append(int(rel_err <= 0.25))
            else:
                # Could not normalize or got division by zero, count as failure
                rep[mapping[category]].append(0)
        except Exception as e:
            # Any error in the extraction or conversion: count as failure and optionally log it
            print(f"Error evaluating item {item['id']} (category {category}): {e}")
            rep[mapping[category]].append(0)


# -------------------- 2) COMPUTE THE METRICS ----------------------- #
records = []
macro_acc = []  # qualitative
macro_err = []  # quantitative

for cat in m_keys:
    if cat in quantitative_cats:
        if rep_relerr[cat]:  # avoid empty list
            mare = mean(rep_relerr[cat])  # mean abs. rel. error
            records.append(
                {
                    "category": cat,
                    "metric": "mean_abs_rel_error",
                    "value": round(mare, 4),
                }
            )
            macro_err.append(mare)
    else:
        if rep[cat]:
            acc = mean(rep[cat])  # accuracy
            records.append(
                {"category": cat, "metric": "accuracy", "value": round(acc, 4)}
            )
            macro_acc.append(acc)

# add overall (macro) aggregates
if macro_acc:
    records.append(
        {
            "category": "ALL_QUALITATIVE",
            "metric": "macro_accuracy",
            "value": round(mean(macro_acc), 4),
        }
    )
if macro_err:
    records.append(
        {
            "category": "ALL_QUANTITATIVE",
            "metric": "macro_mean_abs_rel_error",
            "value": round(mean(macro_err), 4),
        }
    )

# -------------------- 3) SAVE TO DISK ------------------------------ #
out_df = pd.DataFrame(records)
out_path = Path("metrics.csv")
out_df.to_csv(out_path, sep=";", index=False)

print(f"\nMetrics written to {out_path.resolve()}")
print(out_df)
