"""Standalone vllm smoke test for google/gemma-3-4b-it.

Mirrors what run_gemma3-4b-fsdp_date.sh asks of vllm, minus FSDP, minus mbridge:
  - tensor_parallel_size=2
  - gpu_memory_utilization=0.6
  - default enforce_eager / load_format

If this errors with the same cublasLt failure as the full FSDP run, the
problem is vllm 0.11.2 + gemma-3 on this image (try a different image).
If this prints a coherent generation, the FSDP→vllm interface is at fault.
"""

import os

from vllm import LLM, SamplingParams


def main() -> None:
    model = os.environ.get("MODEL", "google/gemma-3-4b-it")
    tp = int(os.environ.get("TP", "2"))
    gpu_mem = float(os.environ.get("GPU_MEM", "0.6"))

    print(f">>> Building vllm engine: model={model} tp={tp} gpu_mem={gpu_mem}", flush=True)
    llm = LLM(
        model=model,
        tensor_parallel_size=tp,
        gpu_memory_utilization=gpu_mem,
        trust_remote_code=True,
    )
    print(">>> Engine ready, generating", flush=True)

    prompts = [
        "Hello, my name is",
        "The capital of France is",
        "In tactile sensing, a deformation indicates",
    ]
    out = llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=32))
    for o in out:
        print("---")
        print("PROMPT:", o.prompt)
        print("OUTPUT:", o.outputs[0].text)


if __name__ == "__main__":
    main()
