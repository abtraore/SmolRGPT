import torch


def convert_to_key_based(json_array):
    """
    Convert a JSON array to a key-based dictionary using the 'id' field as keys.

    Args:
        json_array (list): List of dictionaries, each containing an 'id' field

    Returns:
        dict: Dictionary with IDs as keys and original items as values
    """
    key_based_dict = {}

    # Iterate through each item in the array
    for item in json_array:
        # Check if the item has an 'id' key
        if "id" in item:
            # Use the id as the key and the entire item as the value
            key_based_dict[item["id"]] = item

    return key_based_dict


# Alternative using dictionary comprehension (more Pythonic)
def convert_to_key_based_comprehension(json_array):
    """
    Convert using dictionary comprehension - more concise approach.
    """
    return {item["id"]: item for item in json_array if "id" in item}


def get_class(model, device, text, tokenizer, return_prob=False):

    inputs = tokenizer(
        text,
        return_tensors="pt",
    )

    inputs = {k: inputs[k].to(device) for k in ("input_ids", "attention_mask")}

    logits = model(inputs["input_ids"], inputs["attention_mask"]).logits

    class_prob = torch.softmax(logits, dim=-1)[0]

    if return_prob:
        return class_prob
    else:
        return torch.argmax(class_prob).item()


def clean_question_for_extraction(question: str) -> str:
    """Remove image and mask tokens for cleaner extraction"""
    # Remove image tokens
    question = question.replace("<image>", "")

    # Remove mask tokens (both types)
    question = question.replace("<mask>", "")
    question = question.replace("<mask_rgb>", "")
    question = question.replace("<mask_depth>", "")

    # Clean up multiple spaces
    question = " ".join(question.split())

    return question.strip()
