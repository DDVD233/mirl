import os
from collections import defaultdict

import ujson
from tqdm import tqdm


def update_annotations(file_path: str) -> None:
    # annotation is jsonl file with each line a json object
    updated_lines = []
    with open(file_path, 'r') as f:
        for line in f:
            data = ujson.loads(line)
            dataset = data['dataset']
            original_question = data['problem']
            original_answer = data['answer']
            if dataset == 'mmsd':
                question = original_question.replace('Is the speaker being sarcastic in this utterance? Answer with one word: true or false.',
                                                     'Is the speaker being sarcastic in this utterance? Answer with one phrase: sarcasm or not sarcasm.')
                answer = 'sarcasm' if original_answer.lower() == 'true' else 'not sarcasm'
            elif dataset == 'urfunny':
                question = original_question.replace('Does this video segment contain a humorous punchline? Answer with one word: true or false.',
                                                     'Does this video segment contain a humorous punchline? Answer with one phrase: humour or not humour.')
                answer = 'humour' if original_answer.lower() == 'true' else 'not humour'
            elif dataset == 'mmpsy_depression':
                question = original_question.replace('classify the speaker’s depression severity using the PHQ-9 categories: minimal, mild, moderate, moderately severe, severe.',
                                                     'classify the speaker’s depression severity using the PHQ-9 categories: depression, no depression.')
                answer_mapping = {
                    'minimal': 'no depression',
                    'mild': 'no depression',
                    'moderate': 'depression',
                    'moderately severe': 'depression',
                    'severe': 'depression'
                }
                answer = answer_mapping.get(original_answer.lower(), original_answer)
            elif dataset == 'mmpsy_anxiety':
                question = original_question.replace('classify the speaker’s anxiety severity using the GAD-7 categories: minimal, mild, moderate, severe.',
                                                     'classify the speaker’s anxiety severity using the GAD-7 categories: anxiety, no anxiety.')
                answer_mapping = {
                    'minimal': 'no anxiety',
                    'mild': 'no anxiety',
                    'moderate': 'anxiety',
                    'severe': 'anxiety'
                }
                answer = answer_mapping.get(original_answer.lower(), original_answer)
            elif dataset == 'meld_emotion':
                index = original_question.find('Choose the most appropriate emotion from:')
                question = original_question[:index] + 'Choose the most appropriate emotion from: \nanger, \ndisgust, \nfear, \nhappy, \nsurprise, \nneutral, \nsad'
                answer_mapping = {
                    'joy': 'happy',
                    'sadness': 'sad',
                }
                answer = answer_mapping.get(original_answer.lower(), original_answer)
            elif dataset == 'mosei_senti':
                index = original_question.find("What is the sentiment of the speaker in this video?")
                question = original_question[:index] + ("What is the sentiment of the speaker in this video? "
                                                        "Answer with one phrase: highly negative, negative, weakly negative, neutral, weakly positive, positive, highly positive.")
                answer_mapping = {
                    'strongly negative': 'highly negative',
                    'strongly positive': 'highly positive',
                }
                answer = answer_mapping.get(original_answer.lower(), original_answer)
            elif dataset in ['einterface', 'expw', 'ravdess']:
                # remove this dataset
                continue
            else:
                # batch swaps: joy becomes happy, sadness -> sad, fearful -> fear, angry -> anger, surprised -> surprise,
                # strongly positive -> highly positive, strongly negative -> highly negative
                swaps = {
                    'joy': 'happy',
                    'happiness': 'happy',
                    'sadness': 'sad',
                    'fearful': 'fear',
                    'angry': 'anger',
                    'pleasant surprise': 'surprise',
                    'surprised': 'surprise',
                    'strongly positive': 'highly positive',
                    'strongly negative': 'highly negative',
                    'No PTSD': 'no ptsd',
                    'PTSD': 'ptsd',
                }
                for k, v in swaps.items():
                    if k in original_question:
                        original_question = original_question.replace(k, v)
                        original_answer = original_answer.replace(k, v)
                data['problem'] = original_question
                data['answer'] = original_answer
                updated_lines.append(data)
                continue
            assert question != original_question, f"Question not updated for dataset {dataset}"
            data['problem'] = question
            data['answer'] = answer
            updated_lines.append(data)

    # make sure annotation is ok
    mappings = {'anger': 7, 'anxiety': 20, 'calm': 12, 'depression': 18, 'disgust': 8,
                'fear': 9, 'happy': 10, 'highly negative': 0, 'highly positive': 6,
                'humour': 24, 'negative': 1, 'neutral': 11, 'no anxiety': 19, 'no depression': 17,
                'no ptsd': 15, 'not humour': 23, 'not sarcasm': 21, 'positive': 5, 'ptsd': 16,
                'sad': 13, 'sarcasm': 22, 'surprise': 14, 'weakly negative': 2, 'weakly positive': 4}
    keys = set(mappings.keys())
    for item in updated_lines:
        answer = item['answer']
        question = item['problem']
        assert answer in keys, f"Answer '{answer}' not in mappings keys for question: {question}"
        assert answer in question, f"Answer '{answer}' not found in question: {question}"

    # Add daic-woz
    if 'train' in file_path:
        with open('heldout_train_full_daicwoz.jsonl', 'r') as f:
            for line in f:
                data = ujson.loads(line)
                updated_lines.append(data)
    else:
        with open('heldout_test_daicwoz.jsonl', 'r') as f:
            for line in f:
                data = ujson.loads(line)
                updated_lines.append(data)

    new_filename = os.path.splitext(file_path)[0] + '_upd.jsonl'

    with open(new_filename, 'w') as f:
        for item in updated_lines:
            f.write(ujson.dumps(item) + '\n')

    # also write vision only (vo) and audio only (ao) versions
    data_vision_only = []
    data_audio_only = []
    for item in updated_lines:
        if len(item['audios']) == 0:
            data_vision_only.append(item)
        if len(item['audios']) > 0:
            data_audio_only.append(item)

    vo_filename = os.path.splitext(file_path)[0] + '_upd_vo.jsonl'
    with open(vo_filename, 'w') as f:
        for item in data_vision_only:
            f.write(ujson.dumps(item) + '\n')

    ao_filename = os.path.splitext(file_path)[0] + '_upd_ao.jsonl'
    with open(ao_filename, 'w') as f:
        for item in data_audio_only:
            f.write(ujson.dumps(item) + '\n')


def merge_annotations():
    # merge all together for statistics
    all_data = []
    for split in ['train', 'val', 'test']:
        file_path = f'v5_{split}_upd.jsonl'
        with open(file_path, 'r') as f:
            for line in f:
                data = ujson.loads(line)
                all_data.append(data)

    for split in ['train', 'test']:
        file_path = f'qa_{split}.jsonl'
        with open(file_path, 'r') as f:
            for line in f:
                data = ujson.loads(line)
                all_data.append(data)

    # with open('daicwoz_with_transcript.jsonl', 'r') as f:
    #     for line in f:
    #         data = ujson.loads(line)
    #         all_data.append(data)

    paths = []
    dataset_counts = defaultdict(int)

    # Remove "dataset":"ravdess"
    all_data = [item for item in all_data if item.get('dataset') not in ['ravdess', 'einterface', 'expw']]

    # # remove mosei_senti
    # all_data = [item for item in all_data if item.get('dataset') != ]
    filtered_data = []
    num_audios, num_videos, num_images = 0, 0, 0
    for item in tqdm(all_data):
        if len(item['audios']) > 0:
            path = item['audios'][0]
        elif len(item['videos']) > 0:
            path = item['videos'][0]
        elif len(item['images']) > 0:
            path = item['images'][0]
        else:
            path = item['problem']
        if path in paths:
            continue

        if len(item['videos']) > 0:
            num_videos += 1
        elif len(item['audios']) > 0:
            num_audios += 1
        elif len(item['images']) > 0:
            num_images += 1
        filtered_data.append(item)
        # paths.append(path)
        dataset_counts[item['dataset']] += 1

    print(dataset_counts)
    print(
        f"Total unique samples: {len(filtered_data)}, with {num_audios} audios, {num_videos} videos, {num_images} images.")
    with open('all_annotations.jsonl', 'w') as f:
        for item in all_data:
            f.write(ujson.dumps(item) + '\n')


if __name__ == "__main__":
    update_annotations("v5_train.jsonl")
    update_annotations("v5_val.jsonl")
    update_annotations("v5_test.jsonl")
    merge_annotations()