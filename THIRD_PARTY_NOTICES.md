# Third-party notices

The optional public SSL expert evaluated by `evaluate_public_expert.py` uses the
`UNet3D` implementation distributed in
[`Rogerking928/selma3d-2026-rogerking`](https://github.com/Rogerking928/selma3d-2026-rogerking)
and the SSL checkpoint distributed in
[`d119115003-rgb/selma3d-2026-ssl`](https://github.com/d119115003-rgb/selma3d-2026-ssl).
Those artifacts are licensed under the Apache License 2.0. They are kept as a
distinct expert and are not represented as weights trained by Nuvo AI.

The standard Nuvo AI checkpoints and containers do not include this optional
expert unless their ensemble manifest explicitly lists it.
