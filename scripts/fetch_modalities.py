import ujson as json
import os


def get_modalities(row_dict: dict):
    is_timeseries = False
    vision_path = row_dict['images'][0] if 'images' in row_dict and len(row_dict['images']) != 0 else None
    if vision_path is None:  # this may be video
        vision_path = row_dict['videos'][0] if 'videos' in row_dict and len(row_dict['videos']) != 0 else None
    if vision_path is None:  # this may be time series only
        vision_path = row_dict['time_series'][0] if 'time_series' in row_dict and len(
            row_dict['time_series']) != 0 else ''
        is_timeseries = True
    prompt_str = row_dict["problem"]

    if 'How long will the patient stay in the hospital?' in prompt_str:
        row_dict["data_source"] = "multimodal"
        row_dict["dataset"] = "los_prediction"
    elif 'Will the patient survive for at least 48 hours?' in prompt_str:
        row_dict["data_source"] = "multimodal"
        row_dict["dataset"] = "48_ihm"
    elif len(vision_path) != 0:
        try:
            row_dict["data_source"] = vision_path.split("/")[0]
            row_dict["dataset"] = vision_path.split("/")[1]
        except IndexError:
            row_dict["data_source"] = "unknown"
            row_dict["dataset"] = "unknown"
            print(
                f"Failed to parse vision path: {vision_path}. The annotation is {row_dict}. Using default values.")
    elif is_timeseries:
        row_dict["data_source"] = "ecg"
        # dataset already set in json
    else:
        raise ValueError("No modality found.")


def fetch_modalities(folder: str):
    files = os.listdir(folder)

    json_files = [f for f in files if f.endswith('.jsonl')]
    print(f"Found {len(json_files)} JSONL files in {folder}.")

    for json_file in json_files:
        input_path = os.path.join(folder, json_file)

        data = []
        with open(input_path, 'r') as f:
            for line in f:
                row_dict = json.loads(line)
                get_modalities(row_dict)
                data.append(row_dict)

        with open(input_path, 'w') as f:
            for row_dict in data:
                f.write(json.dumps(row_dict) + '\n')

        print(f"Processed {input_path} and updated modalities.")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fetch and update modalities in JSONL files.")
    parser.add_argument("--folder", type=str, required=True, help="Folder containing JSONL files.")

    args = parser.parse_args()

    fetch_modalities(args.folder)