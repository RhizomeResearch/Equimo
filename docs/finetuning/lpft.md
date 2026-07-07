# LP-FT

LP-FT is a two-stage workflow:

1. Train a linear probe or head-only model.
2. Continue from that trained head while unfreezing the backbone.

```python
recipe = eqft.lpft()
stage1 = recipe.prepare_stage1(model)

# external training updates the head here
trained_stage1_model = train(stage1.model, stage1.plan)

stage2 = recipe.prepare_stage2(
    base_model=model,
    stage1_model=trained_stage1_model,
)
```

By default, stage 1 disables stochastic depth and `eqx.nn.Dropout` outside the
selected head so the head is trained on deterministic frozen-backbone features.
Stage 2 transfers the trained stage-1 head onto the original base model, which
restores the base model's configured stochastic regularization.

`LPFTRecipe.stage1_plan()` remains available for callers that only need a plan,
but it now builds that plan from the stage-1-prepared model. In other words,
`recipe.stage1_plan(model).combine()` has the same deterministic stage-1 model
policy as `recipe.prepare_stage1(model).model`.

`LPFTRecipe.stage2_plan()` remains a lower-level plan builder for callers that
already transferred or replaced the head themselves. To intentionally reset or
replace the head, do that on the model before building the stage-2 plan.
