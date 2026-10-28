# Evaluation journeys

`yewo.yaml` contains the real third-party application workflows used for
release-candidate evaluation. It currently has no reviewed lock; do not call it
a release gate until `yewo lock eval/journeys/yewo.yaml` has been generated and
reviewed.

The controlled Flutter app owns its workflows and lock beside its source in
`eval/controlled_app/`. Its existing lock covers `permissions_flow`,
`restaurant_menu_matrix_smoke`, and `delivery_checkout`. The `notifications`
and `food_delivery_suite` workflows were already absent from that lock, so a
full `--locked` run correctly refuses to start.

Generated run evidence belongs in `eval/runs/`, not beside these contracts.
