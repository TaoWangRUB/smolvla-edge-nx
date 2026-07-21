"""Container-start patch: constrained vision adaptation (rover M1 escape valve v2).

  1. fp32 VLM load (Maxwell has no bf16).
  2. In set_requires_grad's unfrozen branch: freeze the language tower AND the
     whole vision encoder, then UNFREEZE only the top TOPK vision layers. This
     is the milder rung after the full vision unfreeze regressed (it scrambled
     the pretrained features); adapting just the top layers lets the encoder
     re-weight task-relevant features (e.g. colour binding) while the lower
     layers stay intact. A pragmatic stand-in for vision-encoder LoRA (D5
     contingency 1) given lerobot/peft integration friction.

Run with --policy.train_expert_only=false --policy.freeze_vision_encoder=false
so this branch executes.
"""
import os

TOPK = int(os.environ.get('VISION_TOPK', '2'))

p = '/opt/conda/lib/python3.11/site-packages/lerobot/policies/smolvla/smolvlm_with_expert.py'
s = open(p).read()
s = s.replace('torch_dtype="bfloat16"', 'torch_dtype="float32"')

old = """        else:
            # To avoid unused params issue with distributed training"""
new = f"""        else:
            # Rover M1 escape valve v2: CONSTRAINED vision adaptation.
            for _prm in self.get_vlm_model().text_model.parameters():
                _prm.requires_grad = False
            _vm = self.get_vlm_model().vision_model
            for _prm in _vm.parameters():
                _prm.requires_grad = False
            for _layer in _vm.encoder.layers[-{TOPK}:]:
                for _prm in _layer.parameters():
                    _prm.requires_grad = True
            # To avoid unused params issue with distributed training"""
assert old in s, 'freeze-branch anchor not found'
open(p, 'w').write(s.replace(old, new))
print(f'patched: fp32 + top-{TOPK} vision-layer unfreeze (LM + lower vision frozen)')
