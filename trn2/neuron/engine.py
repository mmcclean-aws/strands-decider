"""Serving engine for the Neuron port: strands-decider requests -> bucketed device forwards.

Host side is the reference implementation's own code (strands_decider.prompting: rendering,
`fit_window`, option positions, `to_answer`), so prompts, token ids and answer semantics are
identical to `strands-decider serve`. Each question becomes one row (state + question, the
reference engine's batched path); rows are grouped into compiled (T, B) buckets, padded on the
right, and run in as few forwards as possible. The device graph returns calibrated
probabilities [B, K] directly (readout, temperature and masked softmax are on device).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

import torch

from strands_decider.prompting import fit_window, option_token_index, render_question, render_state, to_answer
from strands_decider.schema import SystemOneRequest, SystemOneResponse, Usage

Runner = Callable[[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]


@dataclass
class EngineConfig:
    buckets: list[int] = field(default_factory=lambda: [256, 512, 768, 1024, 1536, 2048, 2560, 2816, 3072, 4096])
    # compiled batch sizes per T bucket (rows per forward); the engine picks the smallest that
    # holds the remaining rows, so a single question never pays for a padded batch
    batch: dict[int, list[int]] = field(default_factory=lambda: {256: [1, 2, 4, 8], 512: [1, 2, 4], 768: [1, 2],
                                                                    1024: [1, 2], 1536: [1], 2048: [1], 2560: [1], 2816: [1],
                                                                    3072: [1], 4096: [1]})
    k_pad: int = 32
    max_length: int = 4096
    pad_id: int = 248044
    model_name: str = "strands-decider-2B-qwen3.5-v1-2610-neuron"
    # shared-prefix path for multi-question requests: state encoded once (prefix buckets), then
    # all question suffixes (<= suffix_len tokens) run from its caches in batches of suffix_batch
    use_prefix_cache: bool = True
    prefix_buckets: list[int] = field(default_factory=lambda: [256, 512, 768, 1024, 1536, 2048, 2560, 2816, 3072, 4096])
    suffix_len: int = 128
    suffix_batch: list[int] = field(default_factory=lambda: [2, 4, 6, 8])


class NeuronDeciderEngine:
    def __init__(self, tokenizer, runners: dict[tuple[int, int], Runner], temperature_by_kind: dict[str, float],
                 ordinal_smoothing: float = 0.1, cfg: EngineConfig | None = None, prefix_runner=None, suffix_runner=None):
        self.tok = tokenizer
        self.runners = runners
        self.prefix_runner = prefix_runner  # (ids [1,S_b], s_len [1]) -> device caches
        self.suffix_runner = suffix_runner  # (ids [B,L], s_len, caches, ans, opt, ns, it) -> probs [B,K]
        self.temps = temperature_by_kind
        self.ordinal_smoothing = ordinal_smoothing
        self.cfg = cfg or EngineConfig()

    def _bucket(self, L: int) -> int:
        for b in self.cfg.buckets:
            if L <= b:
                return b
        raise ValueError(f"prompt of {L} tokens exceeds the largest bucket {self.cfg.buckets[-1]}")

    def rows(self, request: SystemOneRequest):
        names = list(request.questions.keys())
        rendered = [render_question(request.questions[n]) for n in names]
        state_text = render_state(request.state)
        s, q, offsets = fit_window(self.tok.backend_tokenizer, state_text, [rq.text for rq in rendered], self.cfg.max_length)
        rows = []
        for name, rq, qi, offs in zip(names, rendered, q, offsets):
            opt = [len(s) + i for i in option_token_index(offs, rq.option_spans, 0)]
            rows.append({"name": name, "rq": rq, "ids": s + qi, "opt": opt})
        return rows, len(s)

    def evaluate(self, request: SystemOneRequest, timings: dict | None = None) -> SystemOneResponse:
        t0 = time.perf_counter()
        rows, n_state = self.rows(request)
        t1 = time.perf_counter()
        answers = {}
        n_tokens = 0
        if (self.cfg.use_prefix_cache and self.prefix_runner is not None and len(rows) > 1
                and n_state <= self.cfg.prefix_buckets[-1]
                and all(len(r["ids"]) - n_state <= self.cfg.suffix_len for r in rows)):
            n_tokens = self._prefix_path(rows, n_state, answers)
            rows = []
        by_bucket: dict[int, list[dict]] = {}
        for r in rows:
            by_bucket.setdefault(self._bucket(len(r["ids"])), []).append(r)
        for T, items in by_bucket.items():
            sizes = sorted(self.cfg.batch.get(T, [1]))
            s = 0
            while s < len(items):
                left = len(items) - s
                B = next((b for b in sizes if b >= left), sizes[-1])
                run = self.runners[(T, B)]
                chunk = items[s:s + B]
                s += B
                ids = torch.full((B, T), self.cfg.pad_id, dtype=torch.int32)
                ans = torch.zeros(B, dtype=torch.int32)
                opt = torch.zeros(B, self.cfg.k_pad, dtype=torch.int32)
                ns = torch.full((B,), 2, dtype=torch.int32)
                it = torch.ones(B, dtype=torch.float32)
                for j, r in enumerate(chunk):
                    L = len(r["ids"])
                    if len(r["opt"]) > self.cfg.k_pad:
                        raise ValueError(f"{len(r['opt'])} options exceed k_pad={self.cfg.k_pad}")
                    ids[j, :L] = torch.tensor(r["ids"], dtype=torch.int32)
                    ans[j] = L - 1
                    opt[j, : len(r["opt"])] = torch.tensor(r["opt"], dtype=torch.int32)
                    ns[j] = r["rq"].n_slots
                    it[j] = 1.0 / float(self.temps.get(r["rq"].kind, 1.0))
                    n_tokens += L
                probs = run(ids, ans, opt, ns, it)
                for j, r in enumerate(chunk):
                    rq = r["rq"]
                    answers[r["name"]] = to_answer(rq, probs[j, : rq.n_slots].tolist(), ordinal_smoothing=self.ordinal_smoothing)
        t2 = time.perf_counter()
        if timings is not None:
            timings.update(host_ms=1e3 * (t1 - t0), device_ms=1e3 * (t2 - t1), total_ms=1e3 * (t2 - t0))
        ordered = {n: answers[n] for n in request.questions}
        return SystemOneResponse(model=self.cfg.model_name, answers=ordered,
                                 usage=Usage(input_tokens=n_tokens, output_tokens=len(ordered)))

    def _prefix_path(self, rows, n_state, answers) -> int:
        S_b = next(b for b in self.cfg.prefix_buckets if n_state <= b)
        pid = torch.full((1, S_b), self.cfg.pad_id, dtype=torch.int32)
        pid[0, :n_state] = torch.tensor(rows[0]["ids"][:n_state], dtype=torch.int32)
        s_len = torch.tensor([n_state], dtype=torch.int32)
        caches = self.prefix_runner(pid, s_len)
        L = self.cfg.suffix_len
        sizes = sorted(self.cfg.suffix_batch)
        n_tokens = n_state
        s = 0
        while s < len(rows):
            left = len(rows) - s
            B = next((b for b in sizes if b >= left), sizes[-1])
            chunk = rows[s:s + B]
            s += B
            ids = torch.full((B, L), self.cfg.pad_id, dtype=torch.int32)
            ans = torch.zeros(B, dtype=torch.int32)
            opt = torch.zeros(B, self.cfg.k_pad, dtype=torch.int32)
            ns = torch.full((B,), 2, dtype=torch.int32)
            it = torch.ones(B, dtype=torch.float32)
            for j, r in enumerate(chunk):
                suf = r["ids"][n_state:]
                ids[j, : len(suf)] = torch.tensor(suf, dtype=torch.int32)
                ans[j] = len(suf) - 1
                opt[j, : len(r["opt"])] = torch.tensor([o - n_state for o in r["opt"]], dtype=torch.int32)
                ns[j] = r["rq"].n_slots
                it[j] = 1.0 / float(self.temps.get(r["rq"].kind, 1.0))
                n_tokens += len(suf)
            probs = self.suffix_runner(ids, s_len, caches, ans, opt, ns, it)
            for j, r in enumerate(chunk):
                rq = r["rq"]
                answers[r["name"]] = to_answer(rq, probs[j, : rq.n_slots].tolist(), ordinal_smoothing=self.ordinal_smoothing)
        return n_tokens
