import ujson

def filter_model_outputs():
    output_name = "omni_qa_all.json"
    with open(output_name, "r") as f:
        outputs = ujson.load(f)

    predictions, ground_truths, datasets = outputs["predictions"], outputs["ground_truths"], outputs["datasets"]
    assert len(predictions) == len(ground_truths) == len(datasets)

    filtered_outputs = {"predictions": [], "ground_truths": [], "datasets": []}
    for pred, gt, ds in zip(predictions, ground_truths, datasets):
        if ds in ["mimeqa", "intentqa", "siq2"]:
            filtered_outputs["predictions"].append(pred)
            filtered_outputs["ground_truths"].append(gt)
            filtered_outputs["datasets"].append(ds)
    print(f"Filtered {len(predictions)} -> {len(filtered_outputs['predictions'])}")
    with open("omni_qa_filtered.json", "w") as f:
        ujson.dump(filtered_outputs, f)

if __name__ == "__main__":
    filter_model_outputs()