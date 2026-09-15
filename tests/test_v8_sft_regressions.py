"""Focused CPU regressions for v8. The notebook itself is never executed."""
import ast
import copy
import math
from pathlib import Path
import unittest

import torch
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from transformers import PretrainedConfig


PATH = Path(__file__).parents[1] / "notebooks" / "working-molab-v8.py"
TREE = ast.parse(PATH.read_text(encoding="utf-8"))


def extract(name, **scope):
    nodes = [n for n in ast.walk(TREE) if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name == name]
    assert len(nodes) == 1, (name, len(nodes))
    node = copy.deepcopy(nodes[0])
    node.decorator_list = []
    namespace = {"torch": torch, "_math": math, **scope}
    exec(compile(ast.Module([node], []), str(PATH), "exec"), namespace)
    return namespace[name]


Smoother = extract("SelectiveLabelSmoother")
NS = extract("zeropower_via_newtonschulz5")
OrScale = extract("GrokOrScale", zeropower_via_newtonschulz5=NS)
SelectCheckpoint = extract("select_sft_resume_checkpoint")
ArrowRowIndex = extract("_arrow_row_index")


class Switch:
    for_training = staticmethod(lambda model: model.train())
    for_inference = staticmethod(lambda model: model.eval())


Preserve = extract("preserve_training_state", FastVisionModel=Switch)
Trainer = extract("JointSFTTrainer", Seq2SeqTrainer=Seq2SeqTrainer,
                  SelectiveLabelSmoother=Smoother, preserve_training_state=Preserve)


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = PretrainedConfig()
        self.proj = torch.nn.Linear(3, 5, bias=False)

    def forward(self, input_ids, labels=None, **kwargs):
        return {"logits": self.proj(input_ids.float())}


class V8SFTRegressions(unittest.TestCase):
    def train_once(self, accumulation, legacy=False):
        torch.manual_seed(17)
        model = TinyModel()
        data = [{"input_ids": torch.tensor([1., 0., 2.]), "labels": torch.tensor(1)}] * 32
        args = Seq2SeqTrainingArguments(
            output_dir=str(PATH.parent), use_cpu=True, max_steps=1,
            per_device_train_batch_size=32 // accumulation,
            gradient_accumulation_steps=accumulation,
            label_smoothing_factor=0.1, max_grad_norm=0,
            lr_scheduler_type="constant", save_strategy="no",
            logging_steps=1, report_to="none", disable_tqdm=True,
            remove_unused_columns=False,
        )
        trainer = Trainer(model=model, args=args, train_dataset=data,
                          optimizers=(torch.optim.SGD(model.parameters(), lr=0.01), None))
        if legacy:
            trainer.model_accepts_loss_kwargs = True
        result = trainer.train()
        return result.training_loss, model.proj.weight.detach()

    def test_gradient_accumulation_is_normalized_by_real_trainer(self):
        loss1, weights1 = self.train_once(1)
        loss16, weights16 = self.train_once(16)
        old_loss, old_weights = self.train_once(16, legacy=True)
        self.assertAlmostEqual(loss1, loss16, places=5)
        torch.testing.assert_close(weights1, weights16)
        self.assertAlmostEqual(old_loss / loss16, 16, places=4)
        self.assertFalse(torch.allclose(weights16, old_weights))

    def test_smoothing_is_finite_in_bfloat16_with_suppressed_inf(self):
        logits = torch.tensor([[[0.5, -float("inf"), 1.25, -0.75]]], dtype=torch.bfloat16, requires_grad=True)
        labels = torch.tensor([[2]])
        loss = Smoother(0.1, [1])({"logits": logits}, labels)
        lp = logits.detach().float().log_softmax(-1)
        reference = 0.9 * -lp[0, 0, 2] + 0.1 * -lp[..., [0, 2, 3]].mean()
        torch.testing.assert_close(loss, reference)
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        torch.testing.assert_close(Smoother(0.1, [1])((logits,), labels), loss)

    def test_zero_lora_bootstraps_and_legacy_calibration_recovers(self):
        param = torch.nn.Parameter(torch.zeros(3, 7))
        optim = OrScale([param], lr=0.01)
        param.grad = torch.ones_like(param)
        optim.step()
        self.assertGreater(param.norm().item(), 0)
        self.assertIsNone(optim.state[param]["c_denom"])
        optim.state[param]["c_denom"] = 0.0
        optim.step()
        self.assertGreater(optim.state[param]["c_denom"], 0)

    def test_newton_schulz_preserves_old_result_for_both_shapes(self):
        for shape in ((3, 19), (19, 3)):
            torch.manual_seed(3)
            matrix = torch.randn(shape)
            old = matrix / (matrix.norm() + 1e-7)
            if shape[0] < shape[1]:
                old = old.T
            for _ in range(5):
                gram = old @ old.T
                old = 3.4445 * old + (-4.7750 * gram + 2.0315 * gram @ gram) @ old
            if shape[0] < shape[1]:
                old = old.T
            torch.testing.assert_close(NS(matrix, apply_shape_scale=False), old, atol=1e-5, rtol=1e-4)

    def test_resume_skips_partial_latest_checkpoint(self):
        required = ["adapter_config.json", "adapter_model.safetensors", "trainer_state.json",
                    "optimizer.pt", "scheduler.pt", "rng_state.pth"]
        files = [f"joint/sft/checkpoint-{step}/{name}" for step in (90, 100) for name in required]
        files.append("joint/sft/checkpoint-200/adapter_config.json")
        self.assertEqual(SelectCheckpoint(files, "joint/sft"), "joint/sft/checkpoint-100")
        with self.assertRaises(RuntimeError):
            SelectCheckpoint(files[-1:], "joint/sft")

    def test_arrow_index_mapping_does_not_materialize_dataset(self):
        class Scalar:
            def __init__(self, value):
                self.value = value

            def as_py(self):
                return self.value

        class Indices:
            def column(self, index):
                self.requested_column = index
                return [Scalar(7), Scalar(2)]

        dataset = type("Dataset", (), {"_indices": Indices()})()
        self.assertEqual(ArrowRowIndex(dataset, 1), 2)
        self.assertEqual(ArrowRowIndex(type("Dataset", (), {"_indices": None})(), 4), 4)


if __name__ == "__main__":
    unittest.main()
