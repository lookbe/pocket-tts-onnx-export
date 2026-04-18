# Model Header Comparison: English v1 vs English v2

Comparison of weights and shapes between `english_v1` and `english_v2`. Data offsets and dtypes are ignored.

## Added Weights in v2
- `flow_lm.bos_before_voice`: shape `[1, 1, 1024]`

## Shape Changes
| Weight Name | v1 Shape | v2 Shape |
| :--- | :--- | :--- |
| `flow_lm.speaker_proj_weight` | `[1024, 512]` | `[1024, 32]` |
| `mimi.downsample.conv.conv.weight` | `[512, 512, 32]` | `[32, 512, 32]` |

