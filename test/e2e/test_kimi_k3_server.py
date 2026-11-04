"""Kimi-K3 with --monokernel-backend amd on MI35x, TP8, across the switch point.

The runtime serves one width here: a DSpark verify step of 8 rows at one
request in flight. The server runs one request a step, so every decode step is
that width and the launches cover all of them. That is deliberate. A second
running request makes a 16-row step, which is wider than MAX_ROWS, and those
steps take the standard path -- enough of them to average away a fault the
launches have. The startup line is checked first, because without it every
other test could pass on the normal path alone. Then a short GSM8K run for
gross failures, and the tokens of covered steps scored again by the normal
prefill path (KL consistency). A fault on one rank stops the server before any
token of the failed step is sent.

The kernel checks (``test/kernels/kimi_k3``) cover the launch's own outputs;
this file covers what only a server shows: that the routing is live, that the
tokens are right, and that a failed wait is fail-stop.
"""

import os
import re
import tempfile
import time
import unittest
from pathlib import Path

import requests
import torch
from kimi_k3_server import ARGS, ENV, MODEL
from sglang.test.few_shot_gsm8k import run_eval as run_eval_few_shot_gsm8k
from sglang.test.kl_test_utils import (
    test_input_output_logprobs_match_helper as assert_logprobs_match,
)
from sglang.test.test_utils import (
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
    terminate_and_kill_process_tree,
)

from sgl_amd_mono import BACKEND

LAUNCH_TIMEOUT = 3600
# Both paths score 0.975 to 0.990 on this recipe, over three samples each, so
# the bound is for gross failures and not for a regression of a question or
# two. It does bite: a conv hand-off that was wrong for every covered layer
# scored 0.870 to 0.915 here, while the same fault at --max-running-requests 2
# scored 0.980 and passed.
GSM8K_BOUND = 0.95
# Decode against prefill on short prompts, mean KL.
KL_BOUND = 0.03
# SGLANG_AMD_MONO_CHECK arm. K1 runs on the KDA layers. Layers 1 to 92 are
# covered; the dense layer 0 keeps the model's own forward. The checkpoint
# lists 24 of them as full attention. That leaves 68 KDA layers.
# Take the count from the config, not from the cadence. linear_attn_config
# numbers its layers from 1. Its full_attn_layers [4, 8 .. 92, 93] are layers
# 3, 7 .. 91 and 92 here: 23 on the cadence, plus a 24th off it.
KDA_LAYERS = 68
# The first covered layer is layer 1. Its residual contribution is still in
# flight when K1 would take it. So covered() declines that layer on every
# step ("this costs one layer a step", k1.py). It is the only KDA layer this
# arm never sees. Every other layer logs one check a decode step, so the
# count is layers x steps. A check that stopped engaging on some layers then
# reads as too few. A bare "no violation was logged" would read as a pass.
CHECKED_LAYERS = KDA_LAYERS - 1
# The least number of decode steps this arm must cover. One DSpark verify
# step is 8 tokens, so the request below is 16 steps. Ask for 10 and leave
# the rest as room for a step the launch declines.
CHECK_STEPS = 10
CHECK_TOKENS = 128
# The outputs check.py scores, and the bounds spec.py declares for them.
# This arm is the only thing that runs them. It therefore also pins that all
# eight are reached. An output that stopped being compared logs nothing and
# violates nothing.
CHECK_NAMES = frozenset(
    f"k1.{out}.{metric}"
    for out in ("core", "bank", "ssm", "window")
    for metric in ("cosine", "rel")
)
# "monokernel check layer 3: k1.core.cosine=0.99999678 k1.core.rel=0.0026 ..."
CHECK_LINE = re.compile(r"monokernel check layer (\d+): ((?:k1\.[\w.]+=\S+ ?)+)")
# The startup line that says the routing is live. The runtime serves whole
# requests only, so DSpark's 8 tokens a request leave batch size 1.
RUNS = "runs target_verify steps of batch size [1] x 8 tokens"


def _gfx950_gpus() -> int:
    if not (torch.cuda.is_available() and torch.version.hip):
        return 0
    if "gfx950" not in torch.cuda.get_device_properties(0).gcnArchName:
        return 0
    return torch.cuda.device_count()


def _launch(extra_env=None, extra_args=(), log=None):
    env = {**os.environ, **ENV, **(extra_env or {})}
    return popen_launch_server(
        MODEL,
        DEFAULT_URL_FOR_TEST,
        timeout=LAUNCH_TIMEOUT,
        other_args=ARGS + ["--monokernel-backend", BACKEND, *extra_args],
        env=env,
        return_stdout_stderr=log,
    )


@unittest.skipIf(_gfx950_gpus() < 8, "needs 8 gfx950 GPUs")
class TestMonokernelKimiK3(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.logs = tuple(
            tempfile.NamedTemporaryFile("w+", suffix=s) for s in (".out", ".err")
        )
        cls.process = _launch(extra_args=["--watchdog-timeout", "600"], log=cls.logs)

    @classmethod
    def tearDownClass(cls):
        if getattr(cls, "process", None) is not None:
            terminate_and_kill_process_tree(cls.process)
        for log in getattr(cls, "logs", ()):
            log.close()

    def test_the_tested_batches_run_the_monokernels(self):
        # Routing is fixed at startup; without this the tests below could pass
        # on the normal path alone.
        log = "".join(Path(f.name).read_text() for f in self.logs)
        self.assertIn(RUNS, log)

    def test_gsm8k_across_the_switch(self):
        # The server runs one request a step, so every decode step is covered
        # and nothing the launches get wrong is averaged with the standard
        # path. The client still sends two at a time, which keeps the queue
        # full; the server serialises them.
        metrics = run_eval_few_shot_gsm8k(
            type(
                "Args",
                (),
                {
                    "num_shots": 5,
                    "data_path": None,
                    "num_questions": 200,
                    "max_new_tokens": 512,
                    "parallel": 2,
                    "host": "127.0.0.1",
                    "port": int(DEFAULT_URL_FOR_TEST.split(":")[-1]),
                },
            )()
        )
        self.assertGreater(metrics["accuracy"], GSM8K_BOUND, metrics)

    def test_covered_steps_match_prefill(self):
        # The launch folds the residual add in, so a wrong seam is a silent
        # numerical error: only scoring the same tokens the other way finds it.
        assert_logprobs_match(
            self,
            DEFAULT_URL_FOR_TEST,
            max_new_tokens=32,
            kl_div_threshold=KL_BOUND,
        )


@unittest.skipIf(_gfx950_gpus() < 8, "needs 8 gfx950 GPUs")
class TestMonokernelKimiK3Fault(CustomTestCase):
    def test_a_stalled_rank_stops_the_server_without_tokens(self):
        # Past startup, which is 10 steps here, rank 1 stalls at its first
        # wait. At the deadline, half the watchdog, it stops. Its peers then
        # stop on its flag. The count is steps, not launches: this model bumps
        # its epoch once a forward, where DeepSeek-V4.1 advances it once a
        # launch.
        with (
            tempfile.NamedTemporaryFile("w+", suffix=".out") as out,
            tempfile.NamedTemporaryFile("w+", suffix=".err") as err,
        ):
            process = _launch(
                extra_env={"SGLANG_AMD_MONO_FAULT_INJECT": "1:1:10"},
                extra_args=["--watchdog-timeout", "20"],
                log=(out, err),
            )
            try:
                try:
                    r = requests.post(
                        DEFAULT_URL_FOR_TEST + "/generate",
                        json={
                            "text": "Write a long essay about lighthouses.",
                            "sampling_params": {
                                "max_new_tokens": 1024,
                                "temperature": 0,
                                "ignore_eos": True,
                            },
                        },
                        timeout=300,
                    )
                    ok = r.status_code == 200 and r.json().get("text")
                except requests.RequestException:
                    ok = False
                self.assertFalse(ok, "a failed step's tokens reached the client")
                deadline = time.time() + 300
                while process.poll() is None and time.time() < deadline:
                    time.sleep(2)
                self.assertIsNotNone(process.poll(), "the server kept running")
                err.seek(0)
                out.seek(0)
                log = err.read() + out.read()
                # A launch never runs on after a failed wait: it stops with a
                # store to the null page, so every fault is at the null
                # address -- none on the words that never came.
                faults = re.findall(
                    r"Memory access fault by GPU[^\n]*on address \(?([^)\s]+)\)?",
                    log,
                )
                self.assertTrue(faults, log[-2000:])
                self.assertEqual(set(faults) - {"nil", "0x0"}, set(), faults)
            finally:
                terminate_and_kill_process_tree(process)


@unittest.skipIf(_gfx950_gpus() < 8, "needs 8 gfx950 GPUs")
class TestMonokernelCheck(CustomTestCase):
    def test_every_k1_launch_is_within_the_bounds(self):
        """SGLANG_AMD_MONO_CHECK: K1 against SGLang's own attention, a layer at
        a time, on the real weights at TP8 with one step's state carried into
        the next.

        The other arms here score tokens. A wrong recurrent state or conv
        window moves a GSM8K score by a point or two and nothing says which
        layer did it; this is what names the layer. It is also the only path
        that runs the ``k1.*`` bounds at all.

        Eager only: the comparison is Python, and a captured graph runs none on
        replay, hence --disable-cuda-graph.
        """
        with (
            tempfile.NamedTemporaryFile("w+", suffix=".out") as out,
            tempfile.NamedTemporaryFile("w+", suffix=".err") as err,
        ):
            process = _launch(
                extra_env={"SGLANG_AMD_MONO_CHECK": "1"},
                extra_args=["--disable-cuda-graph", "--watchdog-timeout", "600"],
                log=(out, err),
            )
            try:
                r = requests.post(
                    DEFAULT_URL_FOR_TEST + "/generate",
                    json={
                        "text": "Write a long essay about lighthouses.",
                        "sampling_params": {
                            "max_new_tokens": CHECK_TOKENS,
                            "temperature": 0,
                            "ignore_eos": True,
                        },
                    },
                    timeout=1800,
                )
                r.raise_for_status()
            finally:
                terminate_and_kill_process_tree(process)
            out.seek(0)
            err.seek(0)
            log = out.read() + err.read()

        checks = [
            (int(layer), dict(kv.split("=") for kv in values.split()))
            for layer, values in CHECK_LINE.findall(log)
        ]
        # check.py logs the conv window on its own line, after finish(). No
        # layer's own comparison covers that output.
        in_layer = [(i, v) for i, v in checks if "k1.core.cosine" in v]
        handoff = [(i, v) for i, v in checks if "k1.window.cosine" in v]
        self.assertGreater(
            len(in_layer), CHECKED_LAYERS * CHECK_STEPS, "too few checked launches"
        )
        self.assertGreater(
            len(handoff), CHECKED_LAYERS * CHECK_STEPS, "too few checked hand-offs"
        )
        # Every KDA layer, not just some of them. This is what says K1 is
        # live on each layer.
        self.assertEqual(
            len({i for i, _ in in_layer}),
            CHECKED_LAYERS,
            "not every KDA layer checked",
        )
        scored = {name for _i, v in checks for name in v}
        self.assertEqual(scored, set(CHECK_NAMES), scored)

        worst = {
            name: (min if name.endswith(".cosine") else max)(
                float(v[name]) for _i, v in checks if name in v
            )
            for name in sorted(CHECK_NAMES)
        }
        print(
            f"{len(in_layer)} checked launches, {len(handoff)} hand-offs; "
            + " ".join(f"{k}={v:.8g}" for k, v in worst.items())
        )
        # check.py scores each one against its spec bound. It logs a violation
        # for a miss. Report a miss, do not work around it. A miss means the
        # kernel does not reach the spec, or the spec has the wrong number.
        violations = re.findall(r"monokernel check violation: [^\n]*", log)
        self.assertEqual(violations, [], violations[:20])


if __name__ == "__main__":
    unittest.main()
