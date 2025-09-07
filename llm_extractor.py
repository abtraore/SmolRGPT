import json
import argparse
import pandas as pd
from tqdm import tqdm

import torch
from openai import OpenAI

from transformers import AutoTokenizer, AutoModelForSequenceClassification

from extractor.ollama_utils import llm_extract

from extractor.utils import (
    convert_to_key_based,
    get_class,
    clean_question_for_extraction,
)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_output", type=str)
    parser.add_argument(
        "--test_json", type=str, default="datasets/warehouse-rgbd-smolRGPT/test.json"
    )
    parser.add_argument(
        "--question_classifier",
        type=str,
        default="classifier_checkpoints/task-classifier-iccv-answer/checkpoint-500",
    )
    parser.add_argument(
        "--answer_classifier",
        type=str,
        default="classifier_checkpoints/task-classifier-iccv-question/checkpoint-500",
    )
    parser.add_argument(
        "--extraction_model",
        type=str,
        default="qwen2.5:14b-instruct",
    )
    parser.add_argument("--device", type=str, default="cuda:0")

    args = parser.parse_args()

    client = OpenAI(api_key="ollama", base_url="http://localhost:11434/v1")

    df = pd.read_csv(args.raw_output, sep=";")
    df.fillna("", inplace=True)

    with open(args.test_json) as f:
        test_data = json.load(f)

    test_data = convert_to_key_based(test_data)

    device = args.device

    model_id_answer = args.answer_classifier
    answer_tokenizer = AutoTokenizer.from_pretrained(model_id_answer)
    answer_classifier = AutoModelForSequenceClassification.from_pretrained(
        model_id_answer,
        device_map=device,
    )
    answer_classifier.eval()

    model_id_question = args.question_classifier
    question_tokenizer = AutoTokenizer.from_pretrained(model_id_question)
    question_classifier = AutoModelForSequenceClassification.from_pretrained(
        model_id_question,
        device_map=device,
    )
    question_classifier.eval()

    task_to_id = {
        "left_right": 0,
        "count": 1,
        "distance": 2,
        "mcq": 3,
    }

    id_to_task = {v: k for k, v in task_to_id.items()}

    DEBUG = False

    with torch.no_grad():
        answers = []
        for i in tqdm(range(len(df))):

            item = {}
            item["id"] = df.iloc[i]["id"]
            item["image"] = df.iloc[i]["image_id"]

            ai_answer = df.iloc[i]["output"].replace("\n", "")

            question = test_data[df.iloc[i]["id"]]["conversations"][0]["value"].replace(
                "\n", ""
            )

            answer_class = get_class(
                answer_classifier, device, ai_answer, answer_tokenizer
            )
            question_class = get_class(
                question_classifier, device, question, question_tokenizer
            )

            if ai_answer != "" and answer_class == question_class:
                extracted_answer = llm_extract(
                    client,
                    ai_answer,
                    id_to_task[answer_class],
                    clean_question_for_extraction(question),
                    args.extraction_model,
                )
                if DEBUG:
                    print(f"{ai_answer=}")
                    print(f"{id_to_task[answer_class]=}")
                    print(f"{question=}")
                    print(f"{id_to_task[question_class]=}")
                    print(f"{extracted_answer=}")
                    print("=" * 15)

                output = extracted_answer

            else:
                if question_class == 0:
                    output = "left"
                else:
                    output = 1

            item["normalized_answer"] = output

            answers.append(item)

        with open("submission.json", "w") as f:
            json.dump(answers, f)
