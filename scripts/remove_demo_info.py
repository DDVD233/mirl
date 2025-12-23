import ujson


def remove_demo_info() -> None:
    input_filename = "geom_train_demo_only.jsonl"
    output_filename = "geom_train_no_demo.jsonl"

    filtered_data = []
    with open(input_filename, "r") as infile:
        for line in infile:
            entry = ujson.loads(line.strip())
            entry.pop("demo")
            filtered_data.append(entry)

    with open(output_filename, "w") as outfile:
        for entry in filtered_data:
            outfile.write(ujson.dumps(entry) + "\n")


if __name__ == "__main__":
    remove_demo_info()