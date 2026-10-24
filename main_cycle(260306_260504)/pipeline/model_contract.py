"""Validate C-step model inputs without loading images or running inference."""
from pathlib import Path
import json

ENVIRONMENT_FEATURES = [
    'capped_gdd_cum', 'capped_gdd_day', 'past_7d_capped_gdd_sum',
    'heat_hours_28_ratio', 'heat_degree_hours_28', 'vpd_hours_ratio_1_8',
    'vpd_degree_hours_1_8', 'hot_dry_load_28_1_8', 'past_7d_hot_dry_load_sum',
    'ec_mean', 'water_air_gap_day', 'water_air_gap_day_abs',
]
MODEL_ENVIRONMENT_FEATURES = [
    'effective_gdd_cum', 'effective_gdd_day', 'gdd_loss_ratio',
    'daily_env_penalty', 'past_7d_env_penalty_mean',
    'heat_hours_28_ratio', 'heat_degree_hours_28',
    'vpd_hours_ratio_1_8', 'vpd_degree_hours_1_8',
    'hot_dry_load_28_1_8', 'past_7d_hot_dry_load_sum', 'ec_penalty',
    'water_air_gap_day', 'water_air_gap_day_abs',
    'past_7d_effective_gdd_sum', 'past_7d_gdd_loss_ratio_mean',
]
SHARED_ENVIRONMENT_FEATURES = [c for c in MODEL_ENVIRONMENT_FEATURES if c in ENVIRONMENT_FEATURES]
UNAVAILABLE_ENVIRONMENT_FEATURES = [c for c in MODEL_ENVIRONMENT_FEATURES if c not in ENVIRONMENT_FEATURES]
MODEL_ENVIRONMENT_POLICY = 'legacy16_nan_adapter'


def align_model_environment(frame):
    """Keep equivalent inputs and add unavailable frozen-model columns as NaN.

    Existing explicitly supplied legacy values are preserved for the separate
    validation-cycle adapters. No aliases, zeros, means or penalties are inferred.
    """
    import numpy as np
    import pandas as pd
    result = frame.copy()
    for name in MODEL_ENVIRONMENT_FEATURES:
        if name not in result:
            result[name] = np.nan
        else:
            result[name] = pd.to_numeric(result[name], errors='coerce')
            result[name] = result[name].where(np.isfinite(result[name]), np.nan)
    return result


def environment_contract():
    return dict(calculation_schema='manuscript12', model_schema='legacy16',
                policy=MODEL_ENVIRONMENT_POLICY,
                calculated_features=ENVIRONMENT_FEATURES,
                model_features=MODEL_ENVIRONMENT_FEATURES,
                shared_features=SHARED_ENVIRONMENT_FEATURES,
                unavailable_features=UNAVAILABLE_ENVIRONMENT_FEATURES,
                missing_value='NaN; handled by the unchanged frozen CatBoost models',
                scope='Inference adapter; not a model retrained with 12 inputs')


def component_contracts(root):
    """Read ordered feature lists and validate the frozen input schema for the declared NaN adapter."""
    result = {}
    for mode in ('direct_stacking', 'residual'):
        folder = Path(root) / mode
        if not (folder / 'ensemble.json').is_file():
            folder = Path(root) / ('454columns_' + mode)
        path = folder / 'ensemble.json'
        metadata = json.loads(path.read_text(encoding='utf-8-sig'))
        names = metadata['features']
        if not names or len(names) != len(set(names)):
            raise ValueError(f'{mode}: empty or duplicate model feature names')
        ordered_environment = [c for c in names if c in MODEL_ENVIRONMENT_FEATURES]
        if ordered_environment != MODEL_ENVIRONMENT_FEATURES:
            raise ValueError(f'{mode}: frozen legacy16 input names/order differ from the declared contract')
        if any(c in names for c in set(ENVIRONMENT_FEATURES) - set(MODEL_ENVIRONMENT_FEATURES)):
            raise ValueError(f'{mode}: unexpected mix of manuscript12 and legacy16 inputs')
        if metadata.get('mode') != mode:
            raise ValueError(f'Wrong component mode in {path}')
        if not metadata.get('seeds'):
            raise ValueError(f'No ensemble members in {path}')
        result[mode] = (folder, metadata)
    residual = result['residual'][1]['features']
    stacking = result['direct_stacking'][1]['features']
    if stacking != residual + ['pred_weight_cnn']:
        raise ValueError('Stacking must use the residual feature order plus pred_weight_cnn')
    return result


def inspect_artifacts(root, expected_model_tag='454columns_blend50'):
    """Check bundled file integrity and report C-step compatibility separately.

    This read-only check does not load model tensors, run inference or require
    image/sensor/survey inputs. The calculation uses manuscript12; the frozen model retains legacy16
    inputs through the explicitly declared missing-value adapter.
    """
    import hashlib
    root = Path(root).resolve()
    manifest_path = root / 'manifest.json'
    if not manifest_path.is_file():
        return dict(status='FILES_INCOMPLETE', missing=['manifest.json'],
                    mismatched=[], files_checked=0, pipeline_ready=False,
                    model_inference=False, writes=0)
    manifest = json.loads(manifest_path.read_text(encoding='utf-8-sig'))
    missing, mismatched, seen = [], [], set()
    total_bytes = 0
    for item in manifest['files']:
        name = item['path']
        path = (root / name).resolve()
        if not path.is_relative_to(root) or name in seen:
            raise ValueError('Invalid or duplicate artifact path: ' + name)
        seen.add(name)
        if not path.is_file():
            missing.append(name)
            continue
        total_bytes += path.stat().st_size
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
                digest.update(block)
        if path.stat().st_size != item['bytes'] or digest.hexdigest() != item['sha256']:
            mismatched.append(name)
    c_step = dict(packaged_model=manifest['packaged_c_model_tag'],
                  packaged_environment_schema=manifest['packaged_c_environment_schema'],
                  active_environment_schema=manifest['active_environment_schema'],
                  inference_adapter=environment_contract())
    try:
        components = component_contracts(root / 'c_step')
        c_step.update(compatible=True, feature_counts={mode: len(meta['features'])
                      for mode, (_, meta) in components.items()})
    except (ValueError, KeyError, FileNotFoundError) as exc:
        c_step.update(compatible=False, reason=str(exc))
    rule_path = root / 'harvest_rule.json'
    rule = json.loads(rule_path.read_text(encoding='utf-8-sig')) if rule_path.is_file() else {}
    rule_tag = rule.get('weight_model')
    harvest = dict(packaged_weight_model=rule_tag, expected_weight_model=expected_model_tag,
                   compatible=rule_tag == expected_model_tag)
    files_ok = bool(seen) and not missing and not mismatched
    return dict(status='FILES_VERIFIED' if files_ok else 'FILES_INCOMPLETE',
                files_checked=len(seen), total_bytes=total_bytes, missing=missing,
                mismatched=mismatched, c_step=c_step, harvest_rule=harvest,
                pipeline_ready=files_ok and c_step['compatible'] and harvest['compatible'],
                model_inference=False, writes=0)
