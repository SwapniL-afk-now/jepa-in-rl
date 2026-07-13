# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Pins the fuse_old_log_prob invariant: with a single on-policy optimizer step,
old_log_prob == log_prob.detach() (ratio == 1), so clipping never binds and the
clipped PPO surrogate's GRADIENT equals the plain policy gradient
-mean(adv * grad log_prob). (The loss VALUES differ by construction: the
surrogate evaluates to -mean(adv) at ratio 1.) If this breaks, the fused path
in RayPPOTrainer.fit() is no longer exact."""

import torch
from omegaconf import OmegaConf

from verl.trainer.ppo.core_algos import agg_loss, compute_policy_loss_vanilla


def _make_config():
    return OmegaConf.create(
        {
            "clip_ratio": 0.2,
            "clip_ratio_low": 0.2,
            "clip_ratio_high": 0.2,
            "clip_ratio_c": 3.0,
            "loss_scale_factor": None,
            "global_batch_info": {},
        }
    )


def test_fused_ratio_one_matches_plain_policy_gradient():
    torch.manual_seed(0)
    batch, seq = 4, 6
    logits = torch.randn(batch, seq, requires_grad=True)
    log_prob = torch.nn.functional.logsigmoid(logits)
    advantages = torch.randn(batch, seq)
    response_mask = (torch.rand(batch, seq) > 0.2).float()
    response_mask[:, 0] = 1.0  # every row keeps at least one token

    pg_loss, pg_metrics = compute_policy_loss_vanilla(
        old_log_prob=log_prob.detach(),  # fused path: ratio == 1 exactly
        log_prob=log_prob,
        advantages=advantages,
        response_mask=response_mask,
        loss_agg_mode="token-mean",
        config=_make_config(),
    )
    (fused_grad,) = torch.autograd.grad(pg_loss, logits, retain_graph=True)

    plain_loss = agg_loss(
        loss_mat=-advantages * log_prob, loss_mask=response_mask, loss_agg_mode="token-mean"
    )
    (plain_grad,) = torch.autograd.grad(plain_loss, logits)

    assert float(pg_metrics["actor/ppo_kl"]) == 0.0
    assert float(pg_metrics["actor/pg_clipfrac"]) == 0.0
    assert torch.allclose(fused_grad, plain_grad, atol=1e-6)


def _random_batch(seed: int = 0):
    torch.manual_seed(seed)
    batch, seq = 4, 6
    logits = torch.randn(batch, seq, requires_grad=True)
    log_prob = torch.nn.functional.logsigmoid(logits)
    advantages = torch.randn(batch, seq)
    response_mask = (torch.rand(batch, seq) > 0.2).float()
    response_mask[:, 0] = 1.0
    return logits, log_prob, advantages, response_mask


def test_fused_ratio_one_gspo():
    """GSPO's sequence-level ratio also collapses to 1 when old == new, so the
    fused path must reproduce the plain policy gradient under GSPO's hardcoded
    seq-mean-token-mean aggregation."""
    from verl.trainer.ppo.core_algos import compute_policy_loss_gspo
    from verl.workers.config.actor import ActorConfig

    logits, log_prob, advantages, response_mask = _random_batch(1)
    config = ActorConfig(strategy="fsdp", rollout_n=1, use_dynamic_bsz=True, clip_ratio=0.2, clip_ratio_low=0.2, clip_ratio_high=0.28)

    pg_loss, pg_metrics = compute_policy_loss_gspo(
        old_log_prob=log_prob.detach(),
        log_prob=log_prob,
        advantages=advantages,
        response_mask=response_mask,
        config=config,
    )
    (fused_grad,) = torch.autograd.grad(pg_loss, logits, retain_graph=True)

    plain_loss = agg_loss(
        loss_mat=-advantages * log_prob, loss_mask=response_mask, loss_agg_mode="seq-mean-token-mean"
    )
    (plain_grad,) = torch.autograd.grad(plain_loss, logits)

    assert float(pg_metrics["actor/ppo_kl"]) == 0.0
    assert torch.allclose(fused_grad, plain_grad, atol=1e-6)


def test_fused_ratio_one_gpg():
    """GPG never uses old_log_prob (pure -adv * log_prob), so the fused path is
    trivially exact; this pins the registry contract."""
    from verl.trainer.ppo.core_algos import compute_policy_loss_gpg
    from verl.workers.config.actor import ActorConfig

    logits, log_prob, advantages, response_mask = _random_batch(2)
    config = ActorConfig(strategy="fsdp", rollout_n=1, use_dynamic_bsz=True)

    pg_loss, _ = compute_policy_loss_gpg(
        old_log_prob=log_prob.detach(),
        log_prob=log_prob,
        advantages=advantages,
        response_mask=response_mask,
        loss_agg_mode="token-mean",
        config=config,
    )
    (fused_grad,) = torch.autograd.grad(pg_loss, logits, retain_graph=True)

    plain_loss = agg_loss(
        loss_mat=-advantages * log_prob, loss_mask=response_mask, loss_agg_mode="token-mean"
    )
    (plain_grad,) = torch.autograd.grad(plain_loss, logits)

    assert torch.allclose(fused_grad, plain_grad, atol=1e-6)


if __name__ == "__main__":
    test_fused_ratio_one_matches_plain_policy_gradient()
    test_fused_ratio_one_gspo()
    test_fused_ratio_one_gpg()
    print("fused ratio==1 invariant OK (vanilla, gspo, gpg)")
