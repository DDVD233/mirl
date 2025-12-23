import os
import ujson as json
import tqdm


def consolidate_mimic_qa():
    base_path = '/scratch/high_modality/multimodal/mimiciv/temporal_splits/'
    out_path = '/scratch/high_modality/multimodal/mimiciv/'

    qa_types = [f'qa_type_{k}' for k in [1, 2, 3, 5, 6]]
    splits = ['train', 'val', 'test']
    no_question_count = 0

    # Put them in one annotation per split, under basepath
    for split in splits:
        out_data = []
        for qa_type in qa_types:
            annotation_file = os.path.join(base_path, qa_type, split + '.jsonl')
            data = []
            with open(annotation_file, 'r') as f:
                for line in f:
                    data.append(json.loads(line))

            for sample in tqdm.tqdm(data, desc=f'Processing {qa_type} {split}'):
                # replace ..\/ with empty exactly twice
                for index, image in enumerate(sample['images']):
                    sample['images'][index] = image.replace('../', '', 2)
                for index, time_series in enumerate(sample['time-series']):
                    sample['time-series'][index] = time_series.replace('../', '', 2)
                # add data_source, dataset
                sample['data_source'] = 'mimic_qa'
                sample['dataset'] = qa_type
                if qa_type in ['qa_type_3', 'qa_type_5', 'qa_type_6']:
                    sample['answer'] = sample['correct_choice']
                if 'question' not in sample or sample['question'] is None or not sample['question'].strip():
                    no_question_count += 1
                    continue
                if 'answer' not in sample or sample['answer'] is None or not sample['answer'].strip():
                    no_question_count += 1
                    continue
                if qa_type == 'qa_type_1':
                    sample['question'] = sample['question'] + 'The answer should start with "All diagnoses from this admission: "'
                elif qa_type == 'qa_type_2':
                    sample['question'] = sample['question'] + 'The answer should start with "The primary diagnosis of this admission is "'
                elif qa_type == 'qa_type_3':
                    sample['question'] = sample['question'] + 'Include the answer in \\boxed{} with a single letter'
                elif qa_type == 'qa_type_5':
                    sample['question'] = sample['question'] + 'Include the answer in \\boxed{}.'
                elif qa_type == 'qa_type_6':
                    sample['question'] = sample['question'] + 'Include the answer in \\boxed{} with a single letter'

                # cap at 8000 letters, clip middle part if too long
                if len(sample['question']) > 5000:
                    half_len = 5000 // 2
                    sample['question'] = sample['question'][:half_len] + ' ... ' + sample['question'][-half_len:]
                # cap number of images at 2
                if len(sample['images']) > 2:
                    sample['images'] = sample['images'][:2]

                images_count = len(sample['images'])
                image_tag_count = sample['question'].count('<image>')
                # Make sure number of <image> tags matches number of images
                if images_count > image_tag_count:
                    # append missing <image> tags at the end
                    sample['question'] = '<image>' * (images_count - image_tag_count) + ' ' + sample['question']
                elif images_count < image_tag_count:
                    # remove extra <image> tags from the end
                    for _ in range(image_tag_count - images_count):
                        last_index = sample['question'].rfind('<image>')
                        if last_index != -1:
                            sample['question'] = sample['question'][:last_index] + sample['question'][last_index + len('<image>'):]
                sample['problem'] = sample['question']

                # Make all None values to empty strings
                for key in sample:
                    if sample[key] is None and key == 'choices':
                        sample[key] = ['']
                    elif sample[key] is None:
                        sample[key] = ''
                    elif isinstance(sample[key], int) or isinstance(sample[key], float):
                        sample[key] = str(sample[key])

                out_data.append(sample)

        out_file = os.path.join(out_path, f'qa_{split}.jsonl')
        with open(out_file, 'w') as f:
            for sample in out_data:
                f.write(json.dumps(sample) + '\n')
        print(f'Wrote {len(out_data)} samples to {out_file}')
        print('Number of samples with no question:', no_question_count)

        # for test set, also make a test_mini with 1007 samples per qa_type
        if split == 'test':
            mini_data = []
            counts = {qa_type: 0 for qa_type in qa_types}
            # shuffle out_data
            import random
            random.seed(0)
            random.shuffle(out_data)
            # for sample in out_data:
            #     if counts[sample['dataset']] < 513:
            #         mini_data.append(sample)
            #         counts[sample['dataset']] += 1
            # out_file_mini = os.path.join(out_path, f'qa_{split}_mini.jsonl')
            # with open(out_file_mini, 'w') as f:
            #     for sample in mini_data:
            #         f.write(json.dumps(sample) + '\n')
            # print(f'Wrote {len(mini_data)} samples to {out_file_mini}')

            # Make another mini with only multimodal samples (samples with images)
            mini_data_mm = []
            counts_mm = {qa_type: 0 for qa_type in qa_types}
            for sample in out_data:
                if len(sample['images']) > 0 and counts_mm[sample['dataset']] < 513:
                    mini_data_mm.append(sample)
                    counts_mm[sample['dataset']] += 1
            out_file_mini_mm = os.path.join(out_path, f'qa_{split}_mini_mm.jsonl')

            with open(out_file_mini_mm, 'w') as f:
                for sample in mini_data_mm:
                    f.write(json.dumps(sample) + '\n')

            print(f'Wrote {len(mini_data_mm)} samples to {out_file_mini_mm}')

            # Make the same mini but remove images and time-series to make it unimodal
            mini_data_um = []
            for sample in mini_data_mm:
                sample_um = sample.copy()
                sample_um['images'] = []
                sample_um['time-series'] = []
                # remove any <image> tags in the question
                sample_um['question'] = sample_um['question'].replace('<image>', '')
                sample_um['problem'] = sample_um['question']
                mini_data_um.append(sample_um)
            out_file_mini_um = os.path.join(out_path, f'qa_{split}_mini_um.jsonl')
            with open(out_file_mini_um, 'w') as f:
                for sample in mini_data_um:
                    f.write(json.dumps(sample) + '\n')
            print(f'Wrote {len(mini_data_um)} samples to {out_file_mini_um}')

if __name__ == '__main__':
    consolidate_mimic_qa()