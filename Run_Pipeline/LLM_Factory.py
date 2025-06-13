# llm_factory.py
import torch
from langchain_core.callbacks import StreamingStdOutCallbackHandler
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, GenerationConfig
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_huggingface.llms import HuggingFacePipeline

class LLMFactory:
    @staticmethod
    def load_peft_model(model_name: str, device: str) -> HuggingFacePipeline:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map={ "cuda": "cuda", "mps": "mps", "cpu": "cpu" }[device]
        )
        gen_config = GenerationConfig(max_new_tokens=256, do_sample=True, eos_token_id=tokenizer.eos_token_id or 2)
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device=0 if device in ("cuda","mps") else -1,
            generation_config=gen_config
        )
        return HuggingFacePipeline(pipeline=pipe)

    @staticmethod
    def load_llm(engine: int, backend: int, device: str = "cuda"):
        """
        engine:
          1 -> "gpt-4o-mini"
          2 -> "gemma3:4b"
          3 -> "qwen3:4b"
          4 -> "사설 PEFT 모델명"
        backend:
          1 -> OpenAI ChatOpenAI
          2 -> Ollama ChatOllama
          3 -> HF Pipeline (PEFT 포함)
        """
        if engine == 1:
            model_name = "gpt-4o-mini"
        elif engine == 2:
            model_name = "gemma3:4b"
        elif engine == 3:
            model_name = "qwen3:4b"
        elif engine == 4:
            model_name = "SiniDSBA/8800QA_SET"
        else:
            raise ValueError("engine은 1~4 중 하나여야 합니다.")

        # OpenAI / Ollama / HF-Pipeline 분기
        if backend == 1:
            return ChatOpenAI(
                model=model_name,
                temperature=0,
                streaming=True,
                callbacks=[StreamingStdOutCallbackHandler()],

            )
        elif backend == 2:
            return ChatOllama(model=model_name, temperature=0, streaming=True, callbacks=[StreamingStdOutCallbackHandler()])
        elif backend == 3:
            # PEFT 모델 전용
            return LLMFactory.load_peft_model(model_name, device)
        else:
            raise ValueError("backend는 1,2,3 중 하나여야 합니다.")
