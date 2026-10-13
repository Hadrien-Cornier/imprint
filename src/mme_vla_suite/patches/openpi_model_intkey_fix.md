# openpi core patch: mem_encoder.blocks int-key coercion

**File (NOT git-tracked here):** `src/openpi/models/model.py`, method `BaseModelConfig.load()`.
**Why:** mem_encoder.blocks is a plain Python list -> nnx assigns INTEGER child keys (0,1),
but orbax/intersect_trees re-key as STRINGS ("0","1"). `check_pytree_equality` then fails:
`expected (0,1) got ('0','1')`. Blocks memory-checkpoint loading for serving/eval.

**Fix:** immediately AFTER `params = ocp.transform_utils.intersect_trees(state.to_pure_dict(), params)`
in load(), insert:

```python
        def _intify_numeric_keys(tree):
            if isinstance(tree, dict):
                return {(int(k) if isinstance(k, str) and k.isdigit() else k): _intify_numeric_keys(v)
                        for k, v in tree.items()}
            return tree
        params = _intify_numeric_keys(params)
```

Apply on any fresh openpi checkout before serving memory (cosmo_bottleneck) checkpoints.
