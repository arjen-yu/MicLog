from __future__ import annotations

import time

from tqdm import tqdm

from .cache import MultiLevelTemplateCache
from .config import OnlineParserConfig
from .io import PredictionCsvWriter, load_support_bank, load_target_logs, write_summary_json
from .model_runner import HFTemplateGenerator
from .prompting import build_messages
from .retriever import BM25Retriever, demos_for_prompt
from .types import ParsePrediction, ParseStats, RetrievedDemo
from .validator import extract_template_from_response, validate_template_against_log


class OnlineParserPipeline:
    def __init__(self, config: OnlineParserConfig) -> None:
        self.config = config
        self.cache = MultiLevelTemplateCache(
            exact_cache_size=config.exact_cache_size,
            signature_cache_size=config.signature_cache_size,
            pattern_cache_size=config.pattern_cache_size,
            pattern_cache_version=config.pattern_cache_version,
        )
        self.support_records = load_support_bank(config.support_root_path, config.dataset)
        self.target_records = load_target_logs(
            config.target_root_path,
            config.dataset,
            max_logs=config.max_target_logs,
        )
        self.retriever = BM25Retriever(
            self.support_records,
            retrieval_field=config.retrieval_field,
            exclude_same_content=config.exclude_same_content,
        )
        self.generator = HFTemplateGenerator(
            model_path=str(config.model_path_obj),
            adapter_dir=None if config.adapter_dir_path is None else str(config.adapter_dir_path),
            max_new_tokens=config.max_new_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
            attn_implementation=config.attn_implementation,
            trust_remote_code=config.trust_remote_code,
            force_cpu=config.force_cpu,
            seed=config.seed,
        )
        self.stats = ParseStats(dataset=config.dataset, shots=config.shots)

    def _add_timings(
        self,
        *,
        cache_lookup_ms: float,
        retrieval_ms: float,
        prompt_build_ms: float,
        llm_query_ms: float,
        validation_ms: float,
        latency_ms: float,
    ) -> None:
        self.stats.total_cache_lookup_ms += cache_lookup_ms
        self.stats.total_retrieval_ms += retrieval_ms
        self.stats.total_prompt_build_ms += prompt_build_ms
        self.stats.total_llm_query_ms += llm_query_ms
        self.stats.total_validation_ms += validation_ms
        self.stats.total_latency_ms += latency_ms

    def parse_one(self, record) -> ParsePrediction:
        started = time.perf_counter()
        self.stats.total_logs += 1
        cache_lookup_ms = 0.0
        retrieval_ms = 0.0
        prompt_build_ms = 0.0
        llm_query_ms = 0.0
        validation_ms = 0.0

        cache_started = time.perf_counter()
        cache_result = self.cache.lookup(record.content)
        cache_lookup_ms = (time.perf_counter() - cache_started) * 1000.0
        if cache_result is not None:
            latency_ms = (time.perf_counter() - started) * 1000.0
            self._add_timings(
                cache_lookup_ms=cache_lookup_ms,
                retrieval_ms=retrieval_ms,
                prompt_build_ms=prompt_build_ms,
                llm_query_ms=llm_query_ms,
                validation_ms=validation_ms,
                latency_ms=latency_ms,
            )
            if cache_result.source == "exact_cache":
                self.stats.exact_cache_hits += 1
            elif cache_result.source == "signature_cache":
                self.stats.signature_cache_hits += 1
            else:
                self.stats.pattern_cache_hits += 1
            return ParsePrediction(
                dataset=record.dataset,
                line_id=record.line_id,
                content=record.content,
                predicted_template=cache_result.template,
                decision_source=cache_result.source,
                cache_hit=True,
                exact_cache_hit=cache_result.source == "exact_cache",
                signature_cache_hit=cache_result.source == "signature_cache",
                pattern_cache_hit=cache_result.source == "pattern_cache",
                num_demos=0,
                latency_ms=latency_ms,
                cache_lookup_ms=cache_lookup_ms,
                retrieval_ms=retrieval_ms,
                prompt_build_ms=prompt_build_ms,
                llm_query_ms=llm_query_ms,
                validation_ms=validation_ms,
                model_attempts=0,
                ground_truth_template=record.event_template,
            )

        retrieval_started = time.perf_counter()
        demos = self.retriever.retrieve(record, self.config.shots) if self.config.shots > 0 else []
        prompt_demos = demos_for_prompt(demos)
        retrieval_ms = (time.perf_counter() - retrieval_started) * 1000.0

        prompt_started = time.perf_counter()
        messages = build_messages(
            query_content=record.content,
            demos=prompt_demos,
            instruction=self.config.instruction,
            system_prompt=self.config.system_prompt,
        )
        prompt_build_ms = (time.perf_counter() - prompt_started) * 1000.0

        self.stats.llm_calls += 1
        raw_response = ""
        predicted_template = ""
        valid = False
        attempts = 0
        for _ in range(self.config.max_generation_attempts):
            attempts += 1
            llm_started = time.perf_counter()
            raw_response = self.generator.generate(messages)
            llm_query_ms += (time.perf_counter() - llm_started) * 1000.0
            validation_started = time.perf_counter()
            predicted_template = extract_template_from_response(raw_response)
            if predicted_template:
                valid, _errors, _replacements = validate_template_against_log(record.content, predicted_template)
            validation_ms += (time.perf_counter() - validation_started) * 1000.0
            if valid:
                break

        if valid:
            self.stats.llm_valid_count += 1
            self.cache.update(record.content, predicted_template)
            decision_source = "llm"
        else:
            self.stats.llm_invalid_count += 1
            predicted_template = ""
            decision_source = "failed"
            self.stats.failed_count += 1

        latency_ms = (time.perf_counter() - started) * 1000.0
        self._add_timings(
            cache_lookup_ms=cache_lookup_ms,
            retrieval_ms=retrieval_ms,
            prompt_build_ms=prompt_build_ms,
            llm_query_ms=llm_query_ms,
            validation_ms=validation_ms,
            latency_ms=latency_ms,
        )
        return ParsePrediction(
            dataset=record.dataset,
            line_id=record.line_id,
            content=record.content,
            predicted_template=predicted_template,
            decision_source=decision_source,
            cache_hit=False,
            exact_cache_hit=False,
            signature_cache_hit=False,
            pattern_cache_hit=False,
            num_demos=len(prompt_demos),
            latency_ms=latency_ms,
            cache_lookup_ms=cache_lookup_ms,
            retrieval_ms=retrieval_ms,
            prompt_build_ms=prompt_build_ms,
            llm_query_ms=llm_query_ms,
            validation_ms=validation_ms,
            model_attempts=attempts,
            raw_response=raw_response,
            ground_truth_template=record.event_template,
            demo_line_ids="|".join(demo.line_id for demo in demos),
            demo_scores="|".join(f"{demo.score:.6f}" for demo in demos),
        )

    def parse_records_to_csv(self, output_path) -> None:
        iterator = self.target_records
        if self.config.show_progress:
            iterator = tqdm(
                self.target_records,
                desc=f"Parsing {self.config.dataset}",
                unit="log",
                dynamic_ncols=True,
            )
        with PredictionCsvWriter(output_path) as writer:
            for record in iterator:
                writer.write(self.parse_one(record))
                if self.config.show_progress and hasattr(iterator, "set_postfix"):
                    iterator.set_postfix(
                        cache_hits=self.stats.cache_hits,
                        signature_hits=self.stats.signature_cache_hits,
                        llm_calls=self.stats.llm_calls,
                        failed=self.stats.failed_count,
                    )

    def run(self) -> ParseStats:
        self.config.output_dir_path.mkdir(parents=True, exist_ok=True)
        self.parse_records_to_csv(self.config.output_dir_path / "predictions.csv")
        write_summary_json(
            self.config.output_dir_path / "summary.json",
            self.stats,
            extra={
                "dataset": self.config.dataset,
                "model_path": str(self.config.model_path_obj),
                "adapter_dir": None if self.config.adapter_dir_path is None else str(self.config.adapter_dir_path),
                "target_root": str(self.config.target_root_path),
                "support_root": str(self.config.support_root_path),
                "shots": self.config.shots,
                "retrieval_field": self.config.retrieval_field,
                "exact_cache_size": self.config.exact_cache_size,
                "signature_cache_size": self.config.signature_cache_size,
                "pattern_cache_size": self.config.pattern_cache_size,
                "pattern_cache_version": self.config.pattern_cache_version,
            },
        )
        return self.stats
