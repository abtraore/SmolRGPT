from pydantic import BaseModel
from typing import Literal

from openai import OpenAI

counting_prompt = """
Counting Extraction Task
From any counting observation text that follows this pattern:

Lists specific items/objects in regions
Concludes with a statement like "Therefore, [number] [items] have been positioned..."

Extract only the final count number.
Example input: "The buffer region contains items [Region 1] [Region 2] [Region 7]. Therefore, three items have been positioned in the buffer area."
Example output: 3
"""

left_right_prompt = """
Extract the direction word (left or right) from the text.

If there's a question and answer, extract from the answer only.
If there's only a statement, extract from that statement.

Example 1:
"Item A is left of Item B. Is Item C right of Item D? Yes, Item C is left of Item D."
Output: left

Example 2:
"The buffer region [Region 0] is positioned to the left of the buffer region [Region 1]."
Output: left

Output: single word only (left/right)
"""

distance_prompt = """
Distance Extraction Task

Extract the numerical distance value from text, regardless of units.

Common patterns:
- "The distance from [object] to [object] is [number] [unit]"
- "[number] [unit] apart"
- "separated by [number] [unit]"
- "[object] is [number] [unit] from [object]"

Common units:
- meters (m)
- feet (ft)
- centimeters (cm)
- kilometers (km)
- units (generic)

Extract ONLY the numerical value, not the unit.

Examples:
"The distance from pallet to buffer is 1.69 meters" → 1.69
"The objects are 5.2 feet apart" → 5.2
"Region 1 is 150 cm from Region 2" → 150
"The separation is 0.5 km" → 0.5
"They are 10 units away from each other" → 10
"The gap measures 2.35m" → 2.35

Output: numerical value only (integer or decimal)
"""

mcq_prompt = """
Multiple Choice Answer Extraction for Warehouse Scenarios

Given a question and its answer about warehouse regions, extract the region number that answers the question.

Common question types:
- "Which pallet is optimal for a transporter to pick up?" → Find the recommended pallet region
- "Which buffer zone is closest to [object]?" → Find the nearest buffer region  
- "Which [object] is leftmost/rightmost?" → Find the region in that position
- "Which region contains [specific item]?" → Find the region with that item

The answer will discuss multiple regions and conclude with one specific region as the answer.
Look for conclusion phrases like:
- "Therefore, Region [X] is the best choice"
- "Region [X] would be optimal"
- "I recommend Region [X]"
- "The answer is Region [X]"

Example:
Question: "Considering the transporters and the pallets, which pallet is the optimal choice for an empty transporter to pick up?"
Answer: "Looking at the available pallets, Region [3] is occupied, Region [5] is too far, and Region [2] is closest to the transporter and readily accessible. Therefore, Region [2] would be the optimal choice."
Output: 2

Extract only the number of the selected region.
"""


class IntAnswer(BaseModel):
    answer: int


class FloatAnswer(BaseModel):
    answer: float


class StrAnswer(BaseModel):
    answer: Literal["left", "right"]


def llm_extract(
    client: OpenAI,
    answer: str,
    task: str,
    question: str = None,
    extraction_model: str = "qwen2.5:14b-instruct",
):
    if task == "left_right":
        parsing_class = StrAnswer
        prompt = left_right_prompt
        user_content = answer
    elif task == "mcq":
        parsing_class = IntAnswer
        prompt = mcq_prompt
        # Include both question and answer for MCQ
        user_content = f"Question: {question}\nAnswer: {answer}"
    elif task == "distance":
        parsing_class = FloatAnswer
        prompt = distance_prompt
        user_content = answer
    elif task == "count":
        parsing_class = IntAnswer
        prompt = counting_prompt
        user_content = answer

    completion = client.chat.completions.create(
        model=extraction_model,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_content},
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "answer",
                "schema": parsing_class.model_json_schema(),
            },
        },
        temperature=0.1,
    )

    repsonse = parsing_class.model_validate_json(
        completion.choices[0].message.content
    ).answer

    return repsonse
