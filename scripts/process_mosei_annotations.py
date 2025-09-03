import os

import ujson as json
import tqdm


def process_mosei_annotations(annotation_path: str) -> None:
    data = []
    with open(annotation_path, "r") as f:  # jsonl file
        for line in f:
            entry = json.loads(line.strip())
            data.append(entry)

    formatted_data = []
    for sample in tqdm.tqdm(data):
        image_path = sample["image"]
        video_id = image_path.split("images/")[1]
        # format is like images/{video_id}_{clip_id}.jpg but video_id may contain "_"
        video_id = "_".join((video_id.split("_")[:-1]))
        if image_path.count("/") > 1:
            print(image_path)
        clip_id = image_path.split("_")[-1].split(".")[0]
        raw_video_path = f"cmu_mosei/Raw/{video_id}/{clip_id}.mp4"
        assert os.path.exists(raw_video_path), f"Video path {raw_video_path} does not exist."
        problem: str = sample["conversations"][0]["value"]
        question_statement = problem.index("What is ")
        question_str = problem[question_statement:]
        answer_str = sample["conversations"][1]["value"]
        if "What is the sentiment" in question_str:
            dataset = "mosei_senti"
        elif "What is the emotion" in question_str:
            dataset = "mosei_emotion"
        else:
            raise ValueError(f"Unknown question: {question_str}")

        new_entry = {
            "videos": [raw_video_path],
            "problem": question_str,
            "answer": answer_str,
            "dataset": dataset,
        }

        # avoid adding if the video and problem already exists
        if not any(
            entry["videos"] == new_entry["videos"] and entry["problem"] == new_entry["problem"]
            for entry in formatted_data
        ):
            formatted_data.append(new_entry)

    formatted_data = sorted(formatted_data, key=lambda entry: entry["videos"])

    output_path = annotation_path.replace(".jsonl", "_formatted.jsonl")
    with open(output_path, "w") as f:
        for entry in formatted_data:
            f.write(json.dumps(entry) + "\n")

    # # Add train test split of 80-20, calling it annotations_train.jsonl and annotations_test.jsonl
    # split_index = int(0.8 * len(formatted_data))
    # train_data = formatted_data[:split_index]
    # test_data = formatted_data[split_index:]
    # folder_name = annotation_path.rsplit("/", 1)[0] if "/" in annotation_path else "."
    # train_output_path = f"{folder_name}/annotations_train.jsonl"
    # test_output_path = f"{folder_name}/annotations_test.jsonl"
    #
    # with open(train_output_path, "w") as f:
    #     for entry in train_data:
    #         f.write(json.dumps(entry) + "\n")
    #
    # with open(test_output_path, "w") as f:
    #     for entry in test_data:
    #         f.write(json.dumps(entry) + "\n")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Process MOSEI annotations")
    parser.add_argument(
        "--annotation_path",
        type=str,
        default="mosei_annotations.jsonl",
        help="Path to the MOSEI annotations file (default: mosei_annotations.jsonl)"
    )

    args = parser.parse_args()

    process_mosei_annotations(args.annotation_path)
    print(f"Processed annotations saved to {args.annotation_path.replace('.jsonl', '_formatted.jsonl')}")
