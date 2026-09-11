from pathlib import Path
from typing import Optional

import torch

from .blending import identity_blend
from .config import DeepProtectConfig, ModelPaths, resolve_device
from .feature_bank import load_feature_bank
from .models.arcface import load_arcface
from .models.e4e import E4EEncoder
from .models.farl import FaRLTextImageEncoder
from .models.lora import inject_lora, load_lora_state_dict
from .models.stylegan import clone_generator, load_stylegan
from .optimization import optimize_generator
from .utils import pil_to_tensor, save_json, tensor_to_pil
from .watermark import adversarial_watermark, attribute_direction


class DeepProtectPipeline:
    """End-to-end identity blending, generator optimization, and watermarking."""

    def __init__(self, paths: ModelPaths, config: Optional[DeepProtectConfig] = None):
        self.paths = paths
        self.config = config or DeepProtectConfig()
        self.config.validate()
        self.device = resolve_device(self.config.device)
        self.bank = load_feature_bank(paths.feature_bank, device=self.device)
        self.generator = load_stylegan(paths.stylegan, device=self.device)
        self.identity_model = load_arcface(paths.arcface, device=self.device)
        self.text_image_model = FaRLTextImageEncoder(paths.farl, device=self.device)
        self.e4e = E4EEncoder(paths.e4e, device=self.device)

    def protect(
        self,
        image_path: Path,
        output_dir: Path,
        prompt: str = "nose",
        mode: str = "combined",
        optimize: bool = True,
        verbose: bool = True,
    ) -> dict:
        """Protect one aligned source image and write all reproducibility artifacts."""

        if mode not in {"combined", "attribute"}:
            raise ValueError("mode must be 'combined' or 'attribute'.")
        image_path = Path(image_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        from PIL import Image

        pil_image = Image.open(image_path).convert("RGB")
        source_01 = (
            pil_to_tensor(pil_image, self.config.image_size).add(1.0).div(2.0)
            .unsqueeze(0).to(self.device)
        )
        source_clip = self.text_image_model.encode_images([pil_image])

        metadata = {
            "source": str(image_path.resolve()),
            "mode": mode,
            "prompt": prompt,
            "device": self.device,
            "tau": self.config.tau,
            "candidate_count": self.config.candidate_count,
            "order_regularization_weight": self.config.order_regularization_weight,
            "middle_style_indices": list(self.config.middle_style_indices),
            "lora_blocks": list(self.config.lora_blocks),
            "lora_rank": self.config.lora_rank,
            "learning_rate": self.config.learning_rate,
            "identity_lock_weight": self.config.identity_lock_weight,
            "watermark_epsilon": self.config.watermark_epsilon,
            "watermark_steps": self.config.watermark_steps,
            "watermark_step_size": self.config.watermark_step_size,
        }
        tensor_to_pil(source_01 * 2.0 - 1.0).save(output_dir / "source.png")

        if mode == "combined":
            w_plus = self.e4e.encode(source_01)
            blend = identity_blend(
                w_plus,
                source_clip,
                self.bank,
                tau=self.config.tau,
                middle_style_indices=self.config.middle_style_indices,
                candidate_count=self.config.candidate_count,
            )
            torch.save(w_plus.detach().cpu(), output_dir / "latent_original.pt")
            torch.save(blend.latent.detach().cpu(), output_dir / "latent_blended.pt")
            torch.save(blend.candidate_indices.detach().cpu(), output_dir / "blend_candidates.pt")
            metadata["candidate_indices"] = blend.candidate_indices.cpu().tolist()
            metadata["candidate_clip_similarities"] = blend.clip_similarities.cpu().tolist()

            if optimize:
                tuned_generator = clone_generator(self.generator).to(self.device)
                optimization = optimize_generator(
                    tuned_generator,
                    blend.latent,
                    source_01,
                    self.identity_model,
                    blocks=self.config.lora_blocks,
                    rank=self.config.lora_rank,
                    lora_scale=self.config.lora_scale,
                    learning_rate=self.config.learning_rate,
                    identity_lock_weight=self.config.identity_lock_weight,
                    max_steps=self.config.max_optimization_steps,
                    lpips_threshold=self.config.lpips_threshold,
                    lpips_network=self.config.lpips_network,
                    device=self.device,
                    verbose=verbose,
                )
                base_01 = ((optimization.optimized_image + 1.0) / 2.0).clamp(0, 1)
                torch.save(optimization.lora_state, output_dir / "lora_state.pt")
                tensor_to_pil(optimization.initial_image).save(output_dir / "identity_blended.png")
                tensor_to_pil(optimization.optimized_image).save(output_dir / "generator_optimized.png")
                metadata.update(
                    {
                        "generator_optimization_steps": optimization.steps,
                        "generator_final_l2": optimization.final_l2,
                        "generator_final_lpips": optimization.final_lpips,
                        "generator_final_identity_lock": optimization.final_identity_lock,
                        "lora_modules": optimization.injected_modules,
                    }
                )
            else:
                with torch.no_grad():
                    blended_image = self.generator.synthesis(
                        blend.latent.to(self.device), noise_mode="const", force_fp32=True
                    )
                base_01 = ((blended_image + 1.0) / 2.0).clamp(0, 1)
                tensor_to_pil(blended_image).save(output_dir / "identity_blended.png")
                metadata["generator_optimization_steps"] = 0
        else:
            base_01 = source_01
            metadata["generator_optimization_steps"] = 0

        attribute_vector, lda, _ = attribute_direction(
            self.bank,
            self.text_image_model,
            prompt,
            m=self.config.candidate_count,
            lambda_reg=self.config.order_regularization_weight,
        )
        watermark = adversarial_watermark(
            base_01,
            self.identity_model,
            attribute_vector,
            epsilon=self.config.watermark_epsilon,
            step_size=self.config.watermark_step_size,
            steps=self.config.watermark_steps,
        )
        tensor_to_pil(watermark.image_01 * 2.0 - 1.0).save(output_dir / "protected.png")
        torch.save(watermark.perturbation.cpu(), output_dir / "watermark.pt")
        torch.save(attribute_vector.cpu(), output_dir / "attribute_direction.pt")
        metadata.update(
            {
                "attribute_direction_target_sign": watermark.target_sign,
                "attribute_direction_norm": float(attribute_vector.norm().item()),
                "watermark_max_abs": float(watermark.perturbation.abs().max().item()),
                "order_aware_lda_selected_indices": lda.selected_indices_.tolist(),
                "order_aware_lda_selected_ranks": lda.selected_ranks_.tolist(),
            }
        )
        save_json(output_dir / "metadata.json", metadata)
        return metadata
