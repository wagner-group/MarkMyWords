""" VLLM Server """

from vllm import LLM, LogitProcessor, DefaultLogitProcessor, InputMetadata, SamplingParams
import torch

from .server import Server
from watermark_benchmark.utils.stats import Stats
from watermark_benchmark.utils.classes import Generation, WatermarkSpec, ConfigSpec 

from typing import Optional, List, Dict, Any 

class VLLMServer(Server, LogitProcessor):

    def __init__(self, config: Dict[str, Any], **kwargs) -> None:
        model = config.model
        self.ray = config.distributed
        if self.ray:
            self.server = LLM(model, tensor_parallel_size=torch.cuda.device_count(), **kwargs)
        else:
            self.server = LLM(model, **kwargs)

        self.devices = [i for i in range(torch.cuda.device_count())]
        self.watermark_engine = None
        self._activate_processor()
        self.max_idx = 0


    def _activate_processor(self) -> None:
        if not self.ray:
            for worker in self.server.llm_engine.workers:
                worker.model.sampler.processor = self
        else:
            self.server.llm_engine._run_workers("update_logit_processor", get_all_outputs=True, processor=self)

    def _deactivate_processor(self) -> None:
        if not self.ray:
            for worker in self.server.llm_engine.workers:
                worker.model.sampler.processor = DefaultLogitProcessor()
        else:
            self.server.llm_engine._run_workers("update_logit_processor", get_all_outputs=True, processor=DefaultLogitProcessor())

    def install(self, watermark_engine) -> None:
        self.watermark_engine = watermark_engine


    def process(self, \
            logits: torch.Tensor, \
            input_metadata: InputMetadata) -> torch.Tensor:

        prev_token_ids = [input_metadata.seq_data[seq_id].get_token_ids() \
                for seq_group in input_metadata.seq_groups\
                for seq_id in seq_group[0]]
        ids = [seq_id - self.max_idx \
                for seq_group in input_metadata.seq_groups\
                for seq_id in seq_group[0]]

        self.stats.update(logits, ids)
        if self.watermark_engine is not None:
            logits = self.watermark_engine.process(logits, prev_token_ids, ids)
        else:
            LogitProcessor._apply_temperatures(logits, input_metadata)
        return logits


    def run(self, \
            inputs: List[str], \
            config: ConfigSpec, \
            temp: float,
            keys: Optional[List[int]] = None,
            watermark_spec: Optional[WatermarkSpec] = None,
            use_tqdm = False) -> List[Generation]:

        params = SamplingParams(temperature=temp, max_tokens = config.max_new_tokens, n = config.num_return_sequences)
        self.stats = Stats(len(inputs), temp)
        outputs = self.server.generate(inputs, params, use_tqdm=use_tqdm)

        generations = [Generation(watermark_spec if watermark_spec is not None else None, keys[i] if keys is not None else None, None, int(output.request_id) - self.max_idx, output.prompt, output.outputs[0].text.strip(), None, None, None, *self.stats[int(output.request_id) - self.max_idx], temp) for i,output in enumerate(outputs)]
        self.max_idx = 1 + max([g.id + self.max_idx for g in generations])
        return generations


    def tokenizer(self):
        return self.server.llm_engine.tokenizer

