# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Pins _next_token_log_probs_from_output: the TAFR HF forwards (anchor/replay
scoring fallback and failure-SFT) must produce identical per-token log-probs
whether the model returns plain logits or the fused-kernels CausalLMOutputForPPO
(use_fused_kernels=True patches the model class forward to return log_probs
computed against roll(input_ids, -1))."""

from types import SimpleNamespace

import torch

from verl.workers.engine.fsdp.transformer_impl import _next_token_log_probs_from_output


def test_fused_and_plain_outputs_agree():
    torch.manual_seed(0)
    batch, seq, vocab = 3, 8, 32
    logits = torch.randn(batch, seq, vocab)
    input_ids = torch.randint(0, vocab, (batch, seq))

    # Reference: what the plain-logits branch computes in production
    # (logits[:, :-1] gathered at ids[:, 1:]). Computed directly here because
    # logprobs_from_logits dispatches to a GPU-only flash-attn kernel when the
    # library is installed, so the plain branch can't run on CPU in this test.
    reference_lp = (
        torch.log_softmax(logits[:, :-1, :], dim=-1).gather(-1, input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)
    )

    # Fused-kernels contract: log_probs[:, t] = log p(roll(ids, -1)[t] | ids[..t]),
    # i.e. position t scores token ids[t+1]; the last position wraps to ids[0].
    rolled_labels = torch.roll(input_ids, shifts=-1, dims=-1)
    full_lp = torch.log_softmax(logits, dim=-1).gather(-1, rolled_labels.unsqueeze(-1)).squeeze(-1)
    fused = SimpleNamespace(log_probs=full_lp, logits=None)
    fused_lp = _next_token_log_probs_from_output(fused, input_ids)

    assert fused_lp.shape == (batch, seq - 1)
    assert torch.allclose(fused_lp, reference_lp, atol=1e-6)


if __name__ == "__main__":
    test_fused_and_plain_outputs_agree()
    print("fused/plain TAFR forward parity OK")
