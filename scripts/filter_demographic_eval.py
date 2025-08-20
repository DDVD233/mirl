import ujson


def filter_demographic_eval():
    input_filename = "geom_valid.jsonl"
    data = []
    with open(input_filename, "r") as f:
        for line in f:
            item = ujson.loads(line)
            data.append(item)

    print(f"Loaded {len(data)} items from {input_filename}")

    filtered_data = []

    for item in data:
        if "demo" in item:
            filtered_data.append(item)

    print(f"Filtered {len(filtered_data)} items from {input_filename}")

    output_filename = "geom_valid_demo_only.jsonl"
    with open(output_filename, "w") as f:
        for item in filtered_data:
            f.write(ujson.dumps(item) + "\n")

    print(f"Saved filtered data to {output_filename}")

if __name__ == "__main__":
    filter_demographic_eval()