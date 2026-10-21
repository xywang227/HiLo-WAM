"""Regression checks for the released hierarchy and its three training losses."""
import unittest
from types import SimpleNamespace

import torch
from torch import nn

from hilo_wam.models.wan22.action_dit import ActionDiT
from hilo_wam.models.wan22.hilo_wam import HiLoWAM
from hilo_wam.models.wan22.mot import MoT
from hilo_wam.models.wan22.wan_video_dit import WanVideoDiT


class TinyVAE(nn.Module):
    """Deterministic encoder with the production temporal compression ratio."""
    temporal_downsample_factor = 4
    spatial_downsample_factor = 16
    upsampling_factor = 16
    model = SimpleNamespace(z_dim=48)

    def encode(self, video, **kwargs):
        if isinstance(video, list):
            video = torch.stack(video)
        return video[:, :, ::4, ::16, ::16].repeat(1, 16, 1, 1, 1)


def make_model():
    common = dict(text_dim=16, freq_dim=16, eps=1e-6, num_heads=2,
                  attn_head_dim=12, num_layers=1, use_gradient_checkpointing=True)
    video = WanVideoDiT(hidden_dim=48, ffn_dim=96, in_dim=48, out_dim=48,
                        patch_size=(1, 2, 2), has_image_input=False,
                        seperated_timestep=True, **common)
    action = ActionDiT(hidden_dim=24, ffn_dim=48, action_dim=14, **common)
    mot = MoT({'video': video, 'action': action}, mot_checkpoint_mixed_attn=True)
    return HiLoWAM(video, action, mot, TinyVAE(), text_dim=16, proprio_dim=14)


class ModelTests(unittest.TestCase):
    def test_policy_inference_returns_a_finite_action_chunk(self):
        model = make_model().eval()
        result = model.infer_action(
            prompt=None, input_image=torch.randn(1, 3, 32, 32), action_horizon=32,
            proprio=torch.zeros(14), context=torch.randn(1, 8, 16),
            context_mask=torch.ones(1, 8, dtype=torch.bool),
            num_inference_steps=2, high_video_inference_steps=2,
            low_video_inference_steps=2, action_inference_steps=2,
            joint_denoise=True, seed=42)
        self.assertEqual(result['action'].shape, (32, 14))
        self.assertTrue(torch.isfinite(result['action']).all())

    def test_hierarchy_routes_history_and_keeps_prediction_branches(self):
        model = make_model()
        mask = model._build_hierarchical_training_attention_mask(
            high_frames=5, low_frames=3, tokens_per_frame=2,
            action_seq_len=33, has_proprio_token=True, history_high_frames=3,
            low_visible_high_start=0, low_visible_high_end=3, device=torch.device('cpu'))
        # High-level tokens cannot read low-level or action tokens.
        self.assertFalse(mask[:10, 10:].any())
        # Low-level and action tokens can use history plus the first high prediction.
        self.assertTrue(mask[10:16, :8].all())
        self.assertTrue(mask[17:, :8].all())
        self.assertFalse(mask[10:, 8:10].any())
        # Both video branches retain future prediction; actions read all low tokens.
        self.assertTrue(mask[:10, :10].all())
        self.assertTrue(mask[10:16, 10:16].all())
        self.assertTrue(mask[17:, 10:16].all())
        self.assertEqual(model._downstream_keyframe_indices(3), ([0, 1, 2, 3], 3))

    def test_all_three_losses_backpropagate(self):
        torch.manual_seed(42)
        model = make_model().train()
        # Freeze the encoder, as in the production trainer.
        model.vae.requires_grad_(False)
        sample = dict(video=torch.randn(1, 3, 9, 32, 32),
                      keyframe=torch.randn(1, 3, 17, 32, 32),
                      action=torch.randn(1, 32, 14), proprio=torch.randn(1, 32, 14),
                      context=torch.randn(1, 8, 16), context_mask=torch.ones(1, 8, dtype=torch.bool))
        loss, metrics = model.training_loss(sample)
        self.assertTrue(torch.isfinite(loss))
        for key in ('loss_video_high', 'loss_video_low', 'loss_action'):
            self.assertGreater(metrics[key], 0)
        loss.backward()
        for module in (model.video_expert, model.action_expert, model.proprio_encoder):
            self.assertTrue(any(p.grad is not None and torch.isfinite(p.grad).all()
                                and p.grad.abs().sum() > 0 for p in module.parameters()))


if __name__ == '__main__':
    unittest.main()
