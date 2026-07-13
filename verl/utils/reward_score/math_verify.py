# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import multiprocessing
import threading
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError

try:
    from math_verify.errors import TimeoutException
except ImportError:

    class TimeoutException(Exception):
        pass

    print("To use Math-Verify, please install it first by running `pip install math-verify`.")

_pool = None
_pool_lock = threading.Lock()


def _get_pool():
    global _pool
    if _pool is None:
        with _pool_lock:
            if _pool is None:
                _pool = ProcessPoolExecutor(max_workers=4, mp_context=multiprocessing.get_context("spawn"))
    return _pool


def _verify_in_subprocess(ground_truth_boxed: str, model_output: str) -> tuple[float, str]:
    """Run math_verify in a subprocess where signal.alarm() works."""
    from math_verify.grader import verify
    from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig, parse

    gold_targets = (LatexExtractionConfig(),)
    pred_targets = (ExprExtractionConfig(), LatexExtractionConfig())

    extracted_gold = parse(ground_truth_boxed, gold_targets)
    extracted_pred = parse(model_output, pred_targets)
    pred_str = str(extracted_pred[0]) if extracted_pred else "<unparsed>"
    if extracted_gold and extracted_pred:
        score = max(1.0 if any(verify(g, p) for g in extracted_gold) else 0.0 for p in extracted_pred)
        return score, pred_str
    return 0.0, pred_str


def compute_score(model_output: str, ground_truth: str, timeout_score: float = 0, timeout: float = 30.0) -> float:
    return compute_score_with_pred(model_output, ground_truth, timeout_score, timeout)[0]


def compute_score_with_pred(
    model_output: str, ground_truth: str, timeout_score: float = 0, timeout: float = 30.0
) -> tuple[float, str]:
    """Same as compute_score, but also returns Math-Verify's extracted/normalized
    prediction string — used for exploration-diversity metrics (unique_answer_ratio_at_k
    etc.) that need to compare predictions across a group, not just correctness."""
    ret_score, pred_str = 0.0, "<unparsed>"
    ground_truth_boxed = "\\boxed{" + ground_truth + "}"
    try:
        future = _get_pool().submit(_verify_in_subprocess, ground_truth_boxed, model_output)
        ret_score, pred_str = future.result(timeout=timeout)
    except (FuturesTimeoutError, TimeoutException):
        ret_score = timeout_score
    except Exception as e:
        print(f"Error in math_verify compute_score: {e}")
    return ret_score, pred_str
