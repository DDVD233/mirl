
def __getitem__(self, item):
    """
    Note that we also return the raw_input_ids so that it can be combined with other chat template
    """
    row_dict: dict = self.dataframe[item]

    is_timeseries = False
    vision_path = row_dict['images'][0] if 'images' in row_dict and len(row_dict['images']) != 0 else None
    if vision_path is None:  # this may be video
        vision_path = row_dict['videos'][0] if 'videos' in row_dict and len(row_dict['videos']) != 0 else None
    if vision_path is None:  # this may be time series only
        vision_path = row_dict['time_series'][0] if 'time_series' in row_dict and len(
            row_dict['time_series']) != 0 else ''
        is_timeseries = True
    prompt_str = row_dict[self.prompt_key]

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

    if 'reward_model' not in row_dict:
        if 'answer' in row_dict:
            answer = row_dict['answer']
        elif 'ground_truth' in row_dict:
            answer = row_dict['ground_truth']
        else:
            raise ValueError("No answer or ground_truth found in the row_dict.")
        row_dict['reward_model'] = {'ground_truth': answer}

    for key, item in row_dict.items():
        if item is None:
            row_dict[key] = []

    # NOTE: BUILD_MESSAGES IS CALLED TWICE; 
    # NOTE: FIRST TIME IS TO GET THE LENGTH OF THE RAW PROMPT AND FILTER OUT 
    # NOTE: PROMPTS THAT DO NOT FIT THE LENGTH; 
    # NOTE: SECOND TIME IS TO BUILD THE MESSAGE TO BE PASSED INTO THE MODEL

    messages = self._build_messages(row_dict)

    if "audio" in self.modalities:
        # NOTE: Set the following prompt for qwen omni when we are training on audio
        messages.insert(0, {
            "role": "system",
            "content": [
                {"type": "text", "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, "
                                            "capable of perceiving auditory and visual inputs, as well as generating text and speech."}
            ]
        })
    model_inputs = {}
    
    # NOTE: DEBUGGING
    dbg = True
    if dbg:
        print(f"[getitem] idx=? ds={row_dict.get('dataset')} src={row_dict.get('data_source')} "
            f"modalities={self.modalities}")

    if self.processor is not None:
        # THIS CHUNK IS BASICALLY ABOUT PROCESSING ALL THE MODALITIES
        from verl.utils.dataset.vision_utils import process_image, process_video
        from verl.utils.dataset.audio_utils import process_audio

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="System prompt modified")
            raw_prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

        
        if dbg:
            print(f"[prompt] raw_prompt_chars={len(raw_prompt)}")

        multi_modal_data = {}
        processor_kwargs = {"text": [raw_prompt], "return_tensors": "pt"}

        if "images" in self.modalities and self.image_key in row_dict and row_dict.get(self.image_key, None) is not None and len(row_dict[self.image_key]) > 0:
            images = []
            for image in row_dict.get(self.image_key):
                image = os.path.join(self.base_dir, image) if isinstance(image, str) else image
                images.append(process_image(image))

            # due to the image key is "image" instead of "images" in vllm, we need to use "image" here
            # link: https://github.com/vllm-project/vllm/blob/3c545c0c3b98ee642373a308197d750d0e449403/vllm/multimodal/parse.py#L205
            multi_modal_data["image"] = images
            processor_kwargs["images"] = images

            if dbg:
                print(f"[image] n={len(images)} shapes={[tuple(x.size()) if hasattr(x,'size') else 'np' for x in images]}")


        # print(f"KEANE: Videos is next line, current processor_kwargs {processor_kwargs}")
        if "videos" in self.modalities and self.video_key in row_dict and row_dict.get(self.video_key, None) is not None and len(row_dict[self.video_key]) > 0:
            videos = []
            # print(f"KEANE: GETTING VIDEO {row_dict[self.video_key]}")

            for video in row_dict.get(self.video_key):
                video = os.path.join(self.base_dir, video) if isinstance(video, str) else video
                videos.append(process_video(video))

            # due to the video key is "video" instead of "videos" in vllm, we need to use "video" here
            # link: https://github.com/vllm-project/vllm/blob/3c545c0c3b98ee642373a308197d750d0e449403/vllm/multimodal/parse.py#L205
            multi_modal_data["video"] = [video.numpy() for video in videos]
            processor_kwargs["videos"] = videos

            if dbg:
                shapes = [tuple(v.shape) for v in videos]  # [T,3,H,W]
                toks = []
                for (T, C, H, W) in shapes:
                    toks.append(_tok_est_from_hw(H, W) * T)
                print(f"[video] n={len(videos)} shapes={shapes} est_tokens={toks} "
                    f"sum_est_tokens={sum(toks)} p99_est={_p99(toks)}")


        # NOTE: PROCESSING OF THE AUDIO TUPLES
        # if "audio" in self.modalities and self.audio_key in row_dict and row_dict.get(self.audio_key, None) is not None and len(row_dict[self.audio_key]) > 0:
        #     audios = []
        #     audio_tuples = []  # Keep tuples for multi_modal_data
        #     for audio in row_dict.get(self.audio_key):
        #         audio_path = os.path.join(self.base_dir, audio) if isinstance(audio, str) else audio
        #         audio_data, sampling_rate = process_audio(audio_path, self.processor)
        #         audio_tuples.append((audio_data, sampling_rate))
        #         # audios.append(audio_data.numpy())  # Convert to numpy array for Whisper
        #         audios.append(audio_data.detach().cpu().numpy().astype("float32"))

        #     # multi_modal_data["audio"] = audio_tuples  # Store tuples for reference
        #     multi_modal_data["audio"] = audios  # Store numpy arrays (it should not accept tuples)

        #     processor_kwargs["audio"] = audios  # Pass numpy arrays to processor

        if (
            "audio" in self.modalities
            and self.audio_key in row_dict
            and row_dict.get(self.audio_key)
            and len(row_dict[self.audio_key]) > 0
        ):
            audios_np = []
            audios_np_sr = []
            audio_tuples_debug = []  # keep tensors only for debugging
            audio_secs = []

            for audio in row_dict[self.audio_key]:
                audio_path = os.path.join(self.base_dir, audio) if isinstance(audio, str) else audio
                audio_tensor, sr = process_audio(audio_path, self.processor)

                # Debug only
                audio_tuples_debug.append((audio_tensor, sr))

                # What BOTH HF and vLLM need:
                arr = audio_tensor.detach().cpu().numpy().astype("float32")
                audios_np.append(arr)
                audios_np_sr.append((arr, int(sr)))
                audio_secs.append(_sec_from_array(arr, sr))

            # HF (Whisper / Omni processor) path
            multi_modal_data["audio"] = audios_np_sr  # Store numpy arrays (it should not accept tuples)

            processor_kwargs["audio"] = audios_np  # Pass numpy arrays to processor

            if dbg:
                print(f"[audio] n={len(audios_np)} secs_each={audio_secs} total_secs≈{round(sum([s for s in audio_secs if s!='?']),3)}")

        # NOTE: Original CODE PROCESSING    
        # TODO: Please check whether the model is processing the "audio" correctly, the processor that we are using is qwen 2.5 OMNI
        # print(f"KEANE: Processing multimodal data with processor {self.processor.__class__.__name__} ")
        # print(f"KEANE: Processor kwargs: {processor_kwargs}")
        # model_inputs = self.processor(**processor_kwargs)

        # NOTE: Replacement code
        try:
            t0 = time.time()
            model_inputs = self.processor(**processor_kwargs)
            dt = (time.time() - t0)*1000
            if dbg:
                # lengths after processor/tokenizer
                ids = model_inputs.get("input_ids")
                lens = [len(x) for x in ids] if ids is not None else []
                print(f"[processor] ok in {dt:.1f}ms; input_ids lens={lens} "
                    f"min/med/max={ (min(lens) if lens else '-')} / "
                    f"{ (sorted(lens)[len(lens)//2] if lens else '-') } / "
                    f"{ (max(lens) if lens else '-') }")
        except Exception as e:
            print(f"[processor][ERROR] {type(e).__name__}: {e}")
            # helpful context dump (small)
            print(f"[processor][ctx] has_video={videos is not None} "
                f"n_vid={len(videos) if videos is not None else 0} "
                f"n_audio={len(audio_secs) if audio_secs else 0} "
                f"raw_prompt_chars={len(raw_prompt)}")
            raise

        # NOTE: all text should be processed by self.processor()
        input_ids = model_inputs.pop("input_ids")
        attention_mask = model_inputs.pop("attention_mask")

        if "second_per_grid_ts" in model_inputs:
            model_inputs.pop("second_per_grid_ts")

        # There's a trap here, multi_modal_inputs has to be a dict, not BatchFeature
        row_dict["multi_modal_data"] = multi_modal_data

        # We will do batch.union() in the trainer,
        # so we cannot have "multi_modal_inputs" in row_dict if rollout generates new multi_modal_inputs
        if self.return_multi_modal_inputs:
            row_dict["multi_modal_inputs"] = dict(model_inputs)

            # second_per_grid_ts isn't used for training, just for mrope
            row_dict["multi_modal_inputs"].pop("second_per_grid_ts", None)

    else:
        raw_prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        model_inputs = self.tokenizer(raw_prompt, return_tensors="pt", add_special_tokens=False)
        input_ids = model_inputs.pop("input_ids")
        attention_mask = model_inputs.pop("attention_mask")

    input_ids, attention_mask = verl_F.postprocess_data(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_length=self.max_prompt_length,
        pad_token_id=self.tokenizer.pad_token_id,
        left_pad=True,
        truncation=self.truncation,
    )

    if self.processor is not None and "Qwen2VLImageProcessor" in self.processor.image_processor.__class__.__name__:
        from verl.models.transformers.qwen2_vl import get_rope_index
        
        # NOTE: printing out whether this runs
        # print("KEANE: Running getting the rope index of input ids")
        
        # NOTE: OBTAIN ROPE of rotary positional embeddings. ROPE encodes position by rotating components of query/key vectors
        # This is just for to get relative position in terms of angular differences etc.
        position_ids = [
            get_rope_index(
                self.processor,
                input_ids=input_ids[0],
                image_grid_thw=model_inputs.get("image_grid_thw"),
                video_grid_thw=model_inputs.get("video_grid_thw"),
                second_per_grid_ts=model_inputs.get("second_per_grid_ts"),
                attention_mask=attention_mask[0],
            )
        ]  # (1, 3, seq_len)

    else:
        position_ids = compute_position_id_with_mask(attention_mask)

    # Essentially training with the different input ids etc.
    row_dict["input_ids"] = input_ids[0]
    row_dict["attention_mask"] = attention_mask[0]
    row_dict["position_ids"] = position_ids[0]

    raw_prompt_ids = self.tokenizer.encode(raw_prompt, add_special_tokens=False)
    if len(raw_prompt_ids) > self.max_prompt_length:
        if self.truncation == "left":
            raw_prompt_ids = raw_prompt_ids[-self.max_prompt_length :]
        elif self.truncation == "right":
            raw_prompt_ids = raw_prompt_ids[: self.max_prompt_length]
        elif self.truncation == "middle":
            left_half = self.max_prompt_length // 2
            right_half = self.max_prompt_length - left_half
            raw_prompt_ids = raw_prompt_ids[:left_half] + raw_prompt_ids[-right_half:]
        elif self.truncation == "error":
            raise RuntimeError(f"Prompt length {len(raw_prompt_ids)} is longer than {self.max_prompt_length}.")

    row_dict["raw_prompt_ids"] = raw_prompt_ids
    # encode prompts without chat template
    if self.return_raw_chat:
        row_dict["raw_prompt"] = messages

    # get prompts with chat template
    if self.return_full_prompt:
        row_dict["full_prompts"] = raw_prompt  # array of strings

    # add index for each prompt
    index = row_dict.get("extra_info", {}).get("index", 0)
    tools_kwargs = row_dict.get("extra_info", {}).get("tools_kwargs", {})
    interaction_kwargs = row_dict.get("extra_info", {}).get("interaction_kwargs", {})
    need_tools_kwargs = row_dict.get("extra_info", {}).get("need_tools_kwargs", self.need_tools_kwargs)
    if need_tools_kwargs and not tools_kwargs:
        logger.warning("tools_kwargs is empty for index {}, data source: {}", index, row_dict["data_source"])
    row_dict["index"] = index
    row_dict["tools_kwargs"] = tools_kwargs
    row_dict["interaction_kwargs"] = interaction_kwargs
    return row_dict