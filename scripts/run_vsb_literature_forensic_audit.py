#!/usr/bin/env python3
"""Resumable development-only VSB literature forensic audit."""

from __future__ import annotations

import argparse
import ast
import json
import logging
import os
import sys
import time
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from partial_discharge_adaptive_fusion.dataset import VSB_DATASET_ID, resolve_dataset, load_vsb_metadata, iter_vsb_signal_batches
from partial_discharge_adaptive_fusion.pulse_impl import PulsePolicy
from partial_discharge_adaptive_fusion.vsb_forensics import (
    AUDIT_VERSION, alignment_methods, alignment_summary, audit_provenance,
    build_development_manifest, cwt_log_power_summary, dataset_identity_audit,
    detector_features, detector_match, detector_specs, detect_forensic_detector,
    evaluate_feature_baseline, evaluate_random_signal_protocol, morphology_table,
    phase_effect_table, robust_noise_stats, runtime_record,
)

ROOT_CONFIG = ROOT / "configs/experiments/vsb-literature-forensic-audit-localraw.yaml"
AUDIT_ROOT = ROOT / "results/audits/vsb-literature-forensic"
REPORT_ROOT = ROOT / "reports/audits"
FIGURE_ROOT = ROOT / "reports/figures/vsb-literature-forensic"
PROTECTED = (ROOT / "reports/metrics/v5-pulse-aware", ROOT / "results/manifests/matlab_manifest_v3_windowed.csv")


def logger_setup(log_file: str | None) -> logging.Logger:
    logger = logging.getLogger("vsb_forensic")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("[%(asctime)s] %(message)s", "%Y-%m-%d %H:%M:%S")
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    logger.addHandler(stream)
    if log_file:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_frame(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    if path.suffix == ".parquet":
        frame.to_parquet(temporary, index=False)
    else:
        frame.to_csv(temporary, index=False)
    temporary.replace(path)


def is_complete(root: Path, stage: str) -> bool:
    return (root / (stage + ".complete")).is_file() and (root / (stage + ".json")).is_file()


def finish_stage(root: Path, stage: str, payload: dict[str, Any]) -> None:
    write_json(root / (stage + ".json"), payload)
    (root / (stage + ".complete")).write_text("complete\n", encoding="utf-8")


def read_json(path: Path, default: Any = None) -> Any:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else default


def nonwrapped_segment(values: np.ndarray, center: int, length: int) -> np.ndarray | None:
    half = length // 2
    start = int(center) - half
    stop = start + length
    if start < 0 or stop > len(values):
        return None
    return np.asarray(values[start:stop], dtype=np.float32)


def context(config_path: Path, raw_root: Path):
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    dataset = resolve_dataset(VSB_DATASET_ID, config["dataset"]["version"], raw_root=raw_root)
    metadata = load_vsb_metadata(dataset, train=True).copy()
    manifest, summary = build_development_manifest(
        metadata,
        split_seed=int(config["development"]["split_seed"]),
        oof_folds=int(config["development"]["oof_folds"]),
    )
    assignment = manifest.frame[["sample_id", "group_id", "split", "oof_fold"]].copy()
    assignment["sample_id"] = assignment["sample_id"].astype(str)
    metadata["sample_id"] = metadata["signal_id"].astype(str)
    metadata = metadata.merge(assignment, on="sample_id", how="left", validate="one_to_one")
    metadata["target"] = metadata["target"].astype(int)
    metadata["phase"] = metadata["phase"].astype(str)
    development = metadata[metadata["split"].isin(config["development"]["allowed_splits"])].copy()
    return config, dataset, metadata, development, summary


def stage_literature(root: Path, logger: logging.Logger) -> None:
    started = time.perf_counter()
    register = [
        {"item": "V3 whole-signal CWT", "status": "VERIFIED", "basis": "repository source and historical artifacts"},
        {"item": "V4 uniform windows 16384/8192 and top-10 percent", "status": "VERIFIED", "basis": "repository configs and reports"},
        {"item": "V5 alignment, alpha/beta flattening, knee, MIL, CWT", "status": "VERIFIED", "basis": "repository pulse implementation"},
        {"item": "Dual-CyCon Net high-level dual-domain and cycle-consistency method", "status": "VERIFIED", "basis": "https://arxiv.org/abs/2012.11532 abstract and bibliographic record"},
        {"item": "Dual-CyCon exact VSB preprocessing, labels, and split transfer", "status": "UNAVAILABLE", "basis": "No exact VSB-equivalent preprocessing protocol was verified for this audit."},
        {"item": "Michau detector parameters", "status": "PROJECT LITERATURE ANCHOR — NOT INDEPENDENTLY VERIFIED DURING AUDIT", "basis": "predeclared audit adaptation"},
        {"item": "Chen detector parameters", "status": "PROJECT LITERATURE ANCHOR — NOT INDEPENDENTLY VERIFIED DURING AUDIT", "basis": "predeclared audit adaptation"},
    ]
    payload = {
        "audit_version": AUDIT_VERSION,
        "repository_register_available": (ROOT / "references/papers/v5_literature_register.md").is_file(),
        "records": register,
        "external_code_used": False,
        "external_weights_used": False,
        "unresolved": [
            "Dropbox workspace unavailable; active checkout used.",
            "researchdata forbidden and not required.",
            "signal-level VSB target is not silently converted to measurement-level labels.",
        ],
    }
    finish_stage(root, "literature", runtime_record(started, 0, 0, 0) | {"findings": payload})
    logger.info("literature stage complete: %d records", len(register))


def stage_integrity(root: Path, config_path: Path, raw_root: Path, logger: logging.Logger) -> None:
    started = time.perf_counter()
    config, dataset, metadata, development, summary = context(config_path, raw_root)
    identity = dataset_identity_audit(dataset, metadata)
    identity["provenance"] = audit_provenance(dataset, metadata, config_path=config_path, historical_paths=PROTECTED)
    pattern_fn = __import__("partial_discharge_adaptive_fusion.vsb_forensics", fromlist=["phase_pattern_table"]).phase_pattern_table
    patterns = pattern_fn(metadata)
    write_frame(root / "development_metadata.csv", development.sort_values("sample_id"))
    write_frame(root / "phase_label_patterns.csv", patterns)
    write_json(root / "recomputed_vsb_manifest.json", summary)
    groups = metadata.groupby("id_measurement")["target"]
    identity["label_summary"] = {
        "signals": int(len(metadata)),
        "positive_signals": int(metadata["target"].sum()),
        "measurements": int(metadata["id_measurement"].nunique()),
        "any_positive_measurements": int(groups.max().sum()),
        "all_positive_measurements": int(groups.sum().eq(3).sum()),
        "mixed_label_measurements": int(groups.nunique().gt(1).sum()),
        "literature_discrepancy": {
            "repository_positive_phase_signals": int(metadata["target"].sum()),
            "literature_measurement_positive_anchor": 575,
            "audit_any_positive_measurements": int(groups.max().sum()),
        },
    }
    identity["split_summary"] = summary
    identity["holdout_policy"] = {
        "development_signal_rows_loaded": int(len(development)),
        "test_signal_rows_loaded": 0,
        "official_unlabeled_test_loaded": False,
        "matlab_te2_loaded": False,
    }
    identity["integrity_status"] = "PASS" if (
        identity.get("integrity_status") == "PASS"
        and summary["group_isolation_status"] == "PASS"
        and len(development) == summary["development_signal_count"]
    ) else "FAIL"
    write_json(root / "integrity_findings.json", identity)
    finish_stage(root, "integrity", runtime_record(started, len(development), 0, 0))
    logger.info("integrity stage complete: %s; dev signals=%d", identity["integrity_status"], len(development))


def feature_row(output: Any, meta: pd.Series) -> dict[str, Any]:
    row = {
        "sample_id": str(meta["sample_id"]), "id_measurement": str(meta["id_measurement"]),
        "phase": str(meta["phase"]), "target": int(meta["target"]), "split": str(meta["split"]),
        "oof_fold": int(meta["oof_fold"]), "origin": int(output.origin),
        "alignment_status": str(output.alignment_status),
    }
    row.update(detector_features(output))
    return row


def stage_scan(root: Path, config_path: Path, raw_root: Path, batch_size: int, logger: logging.Logger) -> None:
    started = time.perf_counter()
    config, dataset, metadata, development, _ = context(config_path, raw_root)
    policy = PulsePolicy()
    detectors = tuple(detector_specs())
    feature_rows, noise_rows, align_rows, match_rows, failures = [], [], [], [], []
    processed = 0
    batches = 0
    for batch in iter_vsb_signal_batches(dataset, development, batch_size=batch_size):
        batches += 1
        for index, signal in enumerate(batch.signal):
            meta = batch.metadata.iloc[index]
            method_results = alignment_methods(signal, policy)
            align = {"sample_id": str(meta["sample_id"]), "id_measurement": str(meta["id_measurement"]), "phase": str(meta["phase"])}
            for method, result in method_results.items():
                align[method + "_success"] = bool(result.success)
                align[method + "_origin"] = int(result.origin)
                align[method + "_half_period"] = int(result.half_period)
                align[method + "_frequency_hz"] = result.frequency_hz
                align[method + "_reason"] = result.reason
            for left, right in combinations(method_results, 2):
                lres, rres = method_results[left], method_results[right]
                if lres.success and rres.success:
                    delta = ((rres.origin - lres.origin) / len(signal) * 360.0 + 180.0) % 360.0 - 180.0
                    align[left + "_vs_" + right + "_disagreement_degrees"] = abs(float(delta))
                else:
                    align[left + "_vs_" + right + "_disagreement_degrees"] = np.nan
            align_rows.append(align)
            outputs = {}
            for detector in detectors:
                try:
                    output = detect_forensic_detector(signal, detector, policy)
                    outputs[detector] = output
                    row = feature_row(output, meta)
                    row["raw_median"] = float(np.median(signal))
                    row["raw_std"] = float(np.std(signal))
                    feature_rows.append(row)
                    noise = robust_noise_stats(signal)
                    noise.update({
                        "sample_id": str(meta["sample_id"]), "id_measurement": str(meta["id_measurement"]),
                        "phase": str(meta["phase"]), "target": int(meta["target"]), "split": str(meta["split"]),
                        "detector": detector, "transformed_mad_std": float(output.noise_scale),
                    })
                    noise_rows.append(noise)
                except Exception as exc:
                    failures.append({"sample_id": str(meta["sample_id"]), "detector": detector, "error": type(exc).__name__ + ": " + str(exc)})
            for left, right in combinations(detectors, 2):
                if left in outputs and right in outputs:
                    match = detector_match(outputs[left], outputs[right], int(config["detectors"]["matching_tolerance_samples"]))
                    match.update({"sample_id": str(meta["sample_id"]), "left_detector": left, "right_detector": right})
                    match_rows.append(match)
            processed += 1
            if processed % 100 == 0:
                logger.info("scan: %d/%d signals, elapsed %.1fs", processed, len(development), time.perf_counter() - started)
    write_frame(root / "detector_features.parquet", pd.DataFrame(feature_rows))
    write_frame(root / "noise_statistics.parquet", pd.DataFrame(noise_rows))
    write_frame(root / "alignment_records.csv", pd.DataFrame(align_rows))
    write_frame(root / "detector_matches.csv", pd.DataFrame(match_rows))
    write_json(root / "scan_failures.json", failures)
    align_summary = alignment_summary(pd.DataFrame(align_rows))
    write_json(root / "alignment_summary.json", align_summary)
    finish_stage(root, "scan", runtime_record(started, processed, batches, len(failures)))
    logger.info("scan stage complete: signals=%d failures=%d alignment=%s", processed, len(failures), align_summary["status"])


def stage_morphology(root: Path, config_path: Path, raw_root: Path, batch_size: int, logger: logging.Logger) -> None:
    started = time.perf_counter()
    config, dataset, metadata, development, _ = context(config_path, raw_root)
    pattern = development.groupby("id_measurement")["target"].apply(lambda x: "".join(x.astype(int).astype(str)))
    groups = []
    for value in sorted(pattern.unique()):
        groups.extend(pattern[pattern == value].index.astype(str).tolist()[:4])
    selected = development[development["id_measurement"].astype(str).isin(groups)].sort_values(["id_measurement", "phase"])
    waveforms, rows = [], []
    for batch in iter_vsb_signal_batches(dataset, selected, batch_size=batch_size):
        for index, signal in enumerate(batch.signal):
            meta = batch.metadata.iloc[index]
            try:
                output = detect_forensic_detector(signal, "chen_anchor", PulsePolicy())
                for peak, score, sign in zip(output.peaks, output.scores, output.signs):
                    segment = nonwrapped_segment(output.transformed, int(peak), 128)
                    if segment is None:
                        continue
                    waveforms.append(segment)
                    rows.append({
                        "sample_id": str(meta["sample_id"]), "id_measurement": str(meta["id_measurement"]),
                        "phase": str(meta["phase"]), "target": int(meta["target"]), "split": str(meta["split"]),
                        "peak_index": int(peak), "peak_score": float(score), "detector_polarity": int(sign),
                    })
            except Exception as exc:
                logger.warning("morphology failed for %s: %s", meta["sample_id"], exc)
    if waveforms:
        assignments, summary = morphology_table(np.asarray(waveforms), pd.DataFrame(rows), random_state=42, clusters=12)
    else:
        assignments, summary = pd.DataFrame(rows), {"n_waveforms": 0}
    write_frame(root / "morphology_assignments.parquet", assignments)
    write_json(root / "morphology_summary.json", summary)
    finish_stage(root, "morphology", runtime_record(started, len(selected), 0, 0))
    logger.info("morphology stage complete: waveforms=%d", len(waveforms))


def stage_cwt(root: Path, config_path: Path, raw_root: Path, batch_size: int, logger: logging.Logger) -> None:
    started = time.perf_counter()
    config, dataset, metadata, development, _ = context(config_path, raw_root)
    cwt = config["cwt"]
    scales = np.geomspace(float(cwt["minimum_scale_samples"]), float(cwt["maximum_scale_samples"]), int(cwt["scales"]))
    events, parents, processed = [], [], 0
    for batch in iter_vsb_signal_batches(dataset, development, batch_size=batch_size):
        for index, signal in enumerate(batch.signal):
            meta = batch.metadata.iloc[index]
            try:
                output = detect_forensic_detector(signal, "v5_current", PulsePolicy())
                chosen = 0
                for peak, score, sign in zip(output.peaks, output.scores, output.signs):
                    segment = nonwrapped_segment(output.transformed, int(peak), int(cwt["segment_length"]))
                    if segment is None:
                        continue
                    values = cwt_log_power_summary(segment, scales=scales, time_bins=int(cwt["time_bins"]), morlet_w0=float(cwt["morlet_w0"]))
                    values.update({
                        "sample_id": str(meta["sample_id"]), "id_measurement": str(meta["id_measurement"]),
                        "phase": str(meta["phase"]), "target": int(meta["target"]), "split": str(meta["split"]),
                        "peak_index": int(peak), "peak_score": float(score), "polarity": int(sign),
                    })
                    events.append(values)
                    chosen += 1
                    if chosen >= int(cwt["maximum_events_per_parent"]):
                        break
                parent = {
                    "sample_id": str(meta["sample_id"]), "id_measurement": str(meta["id_measurement"]),
                    "phase": str(meta["phase"]), "target": int(meta["target"]), "split": str(meta["split"]),
                    "cwt_event_count": chosen,
                }
                if chosen:
                    for key in events[-chosen]:
                        if key.startswith("cwt_"):
                            parent[key] = float(np.mean([item[key] for item in events[-chosen:]]))
                parents.append(parent)
            except Exception as exc:
                logger.warning("CWT failed for %s: %s", meta["sample_id"], exc)
            processed += 1
            if processed % 100 == 0:
                logger.info("CWT: %d/%d signals, events=%d, elapsed %.1fs", processed, len(development), len(events), time.perf_counter() - started)
    write_frame(root / "cwt_event_summaries.parquet", pd.DataFrame(events))
    write_frame(root / "cwt_features.parquet", pd.DataFrame(parents))
    finish_stage(root, "cwt", runtime_record(started, processed, 0, 0))
    logger.info("CWT stage complete: signals=%d events=%d", processed, len(events))


def baseline_one(frame: pd.DataFrame, columns: list[str], classifier: str, seed: int, mode: str) -> dict[str, Any]:
    try:
        return evaluate_feature_baseline(frame, columns, classifier=classifier, seed=seed, grouped=True, mode=mode)
    except Exception as exc:
        return {"classifier": classifier, "seed": seed, "grouped": True, "mode": mode, "status": "FAILED", "error": type(exc).__name__ + ": " + str(exc)}


def domain_shift(frame: pd.DataFrame, columns: list[str]) -> dict[str, Any]:
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedGroupKFold
    from partial_discharge_adaptive_fusion.vsb_forensics import _make_estimator
    x = frame[columns].replace([np.inf, -np.inf], np.nan).fillna(0).to_numpy(float)
    y = (frame["split"].to_numpy() == "validation").astype(int)
    groups = frame["id_measurement"].astype(str).to_numpy()
    p = np.full(len(frame), np.nan)
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    for train, test in splitter.split(x, y, groups):
        estimator = _make_estimator("logistic_regression", 42)
        estimator.fit(x[train], y[train])
        p[test] = estimator.predict_proba(x[test])[:, 1]
    return {"roc_auc": float(roc_auc_score(y, p)), "n_rows": int(len(frame)), "n_features": int(len(columns)), "protocol": "group-aware five-fold domain classifier"}


def stage_baseline(root: Path, logger: logging.Logger) -> None:
    started = time.perf_counter()
    frame = pd.read_parquet(root / "detector_features.parquet")
    # The scan stores one sparse row per parent signal and detector. Collapse
    # those rows before joining one CWT row and before fitting signal-level baselines.
    frame = frame.groupby("sample_id", as_index=False).first()
    cwt_path = root / "cwt_features.parquet"
    if cwt_path.is_file():
        cwt = pd.read_parquet(cwt_path)
        cwt_cols = [c for c in cwt.columns if c.startswith("cwt_")]
        frame = frame.merge(cwt[["sample_id"] + cwt_cols], on="sample_id", how="left", validate="one_to_one")
    columns = [
        c for c in frame.columns
        if (c.startswith("v5_current_") or c.startswith("michau_anchor_") or c.startswith("chen_anchor_") or c.startswith("dualcycon_strict_") or c.startswith("cwt_"))
        and frame[c].dtype.kind in "bifu"
    ]
    frame[columns] = frame[columns].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    rows = []
    for seed in (42, 43, 44):
        for classifier in ("logistic_regression", "hist_gradient_boosting"):
            for mode in ("phase_independent", "measurement_aware"):
                result = baseline_one(frame, columns, classifier, seed, mode)
                result["feature_family"] = "detectors_plus_cwt"
                rows.append(result)
            try:
                result = evaluate_random_signal_protocol(frame, columns, classifier=classifier, seed=seed)
                result["feature_family"] = "detectors_plus_cwt"
                rows.append(result)
            except Exception as exc:
                rows.append({"classifier": classifier, "seed": seed, "mode": "random_signal_level_split", "status": "FAILED", "error": type(exc).__name__ + ": " + str(exc)})
    measurement = frame.groupby("id_measurement", as_index=False).agg({**{c: "mean" for c in columns}, "target": "max", "split": "first"})
    measurement["sample_id"] = "measurement:" + measurement["id_measurement"].astype(str)
    measurement["phase"] = "measurement"
    measurement["oof_fold"] = -1
    for seed in (42, 43, 44):
        for classifier in ("logistic_regression", "hist_gradient_boosting"):
            result = baseline_one(measurement, columns, classifier, seed, "measurement_any_positive")
            result["feature_family"] = "measurement_any_positive"
            rows.append(result)
    results = pd.DataFrame(rows)
    if "metrics" in results:
        results["metrics"] = results["metrics"].apply(lambda value: json.dumps(value, sort_keys=True) if isinstance(value, dict) else value)
    write_frame(root / "baseline_results.csv", results)
    write_frame(root / "phase_effects.csv", phase_effect_table(frame, "v5_current"))
    try:
        domain = domain_shift(frame, columns)
    except Exception as exc:
        domain = {"status": "FAILED", "error": type(exc).__name__ + ": " + str(exc)}
    write_json(root / "domain_shift.json", domain)
    failed = int((results.get("status", pd.Series(dtype=str)) == "FAILED").sum())
    finish_stage(root, "baseline", runtime_record(started, len(frame), 0, failed))
    logger.info("baseline stage complete: rows=%d failed=%d domain_auc=%s", len(frame), failed, domain.get("roc_auc", "unavailable"))


def stage_report(root: Path, report_root: Path, figure_root: Path, config_path: Path, logger: logging.Logger) -> None:
    started = time.perf_counter()
    report_root.mkdir(parents=True, exist_ok=True)
    figure_root.mkdir(parents=True, exist_ok=True)
    integrity = read_json(root / "integrity_findings.json", {})
    alignment = read_json(root / "alignment_summary.json", {})
    baseline = pd.read_csv(root / "baseline_results.csv") if (root / "baseline_results.csv").is_file() else pd.DataFrame()
    patterns = pd.read_csv(root / "phase_label_patterns.csv", dtype={"pattern": str}) if (root / "phase_label_patterns.csv").is_file() else pd.DataFrame()
    domain = read_json(root / "domain_shift.json", {})
    morphology = read_json(root / "morphology_summary.json", {})
    literature_runtime = read_json(root / "literature.json", {})
    literature = literature_runtime.get("findings", literature_runtime) if isinstance(literature_runtime, dict) else {}
    rows = []
    for _, item in baseline.iterrows():
        try:
            raw = item.get("metrics", "")
            metrics = json.loads(raw) if isinstance(raw, str) and raw.startswith("{") else ast.literal_eval(raw)
            rows.append({"classifier": item.get("classifier"), "seed": int(item.get("seed")), "mode": item.get("mode"), "feature_family": item.get("feature_family"), "mcc": float(metrics.get("mcc", np.nan)), "roc_auc": float(metrics.get("roc_auc", np.nan))})
        except (ValueError, SyntaxError, TypeError, json.JSONDecodeError):
            pass
    best = pd.DataFrame(rows)
    grouped = best[(best["mode"] == "phase_independent") & (best["feature_family"] == "detectors_plus_cwt")] if len(best) else best
    best_mcc = float(grouped["mcc"].max()) if len(grouped) else float("nan")
    integrity_pass = integrity.get("integrity_status") == "PASS"
    alignment_status = alignment.get("status", "UNAVAILABLE")
    if not integrity_pass:
        verdict = "INCONCLUSIVE"
    elif alignment_status == "VALID" and np.isfinite(best_mcc) and best_mcc >= 0.60:
        verdict = "GO"
    elif integrity_pass:
        verdict = "PIVOT"
    else:
        verdict = "STOP"
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        if len(patterns):
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.bar(patterns["pattern"], patterns["measurements"])
            ax.set_title("VSB three-phase measurement patterns")
            fig.tight_layout()
            fig.savefig(figure_root / "measurement_label_patterns.png", dpi=140)
            plt.close(fig)
        if len(best):
            fig, ax = plt.subplots(figsize=(8, 4))
            best.boxplot(column="mcc", by="mode", ax=ax)
            fig.suptitle("")
            ax.set_title("Development diagnostic MCC")
            fig.tight_layout()
            fig.savefig(figure_root / "baseline_mcc_by_mode.png", dpi=140)
            plt.close(fig)
    except Exception as exc:
        logger.warning("figure generation failed: %s", exc)
    manifest = {
        "audit_version": AUDIT_VERSION,
        "config": str(config_path.relative_to(ROOT)),
        "verdict": verdict,
        "integrity": integrity,
        "literature": literature,
        "alignment": alignment,
        "baseline_best_development_mcc": best_mcc,
        "domain_shift": domain,
        "morphology": morphology,
        "stages": {stage: is_complete(root, stage) for stage in ("literature", "integrity", "scan", "morphology", "cwt", "baseline")},
        "holdouts": {"vsb_grouped_test": "LOCKED", "official_unlabeled_test": "LOCKED", "matlab_te2": "LOCKED"},
        "protected_input_paths": [str(path.relative_to(ROOT)) for path in PROTECTED],
    }
    write_json(report_root / "vsb_literature_forensic_audit.json", manifest)
    pattern_text = patterns.to_string(index=False) if len(patterns) else "Unavailable"
    best_text = best.sort_values("mcc", ascending=False).head(20).to_string(index=False) if len(best) else "Unavailable"
    report = f"""# VSB Literature Forensic Audit

## Decision

Final audit decision: {verdict}. This is a development-only diagnostic, not V6, not model search, and not confirmatory holdout evaluation.

## Observed

- Audit version: {AUDIT_VERSION}
- Data contract: local data/raw through PD_RAW_DATA_ROOT.
- Integrity status: {integrity.get("integrity_status", "UNAVAILABLE")}
- Development signals: {integrity.get("label_summary", {}).get("signals", "UNAVAILABLE")}
- Development measurements: {integrity.get("label_summary", {}).get("measurements", "UNAVAILABLE")}
- Positive phase signals: {integrity.get("label_summary", {}).get("positive_signals", "UNAVAILABLE")}
- Any-positive measurements: {integrity.get("label_summary", {}).get("any_positive_measurements", "UNAVAILABLE")}
- Mixed-label measurements: {integrity.get("label_summary", {}).get("mixed_label_measurements", "UNAVAILABLE")}
- Alignment status: {alignment_status}
- Best grouped phase-independent diagnostic MCC: {best_mcc:.4f}

### Three-phase patterns

{pattern_text}

### Baseline results

{best_text}

## Interpretation

The primary unit is the original phase signal. Grouped train/validation partitions preserve id_measurement. Measurement-aware and any-positive measurement diagnostics are secondary and do not replace the signal-level target. Random signal-level splits are intentionally optimistic and are not deployable evidence.

Phase-derived quantities are called electrical phase only when the predeclared alignment criteria pass. Otherwise they are normalized temporal positions. The detector and CWT settings marked as project literature anchors are adaptations, not exact reproductions. The CWT output is bounded event summaries, not a full scalogram cache.

## Unresolved

- The literature measurement-positive count and repository phase-signal labels use different analytical units.
- Dropbox is unavailable in the active checkout.
- The exact external provenance of Michau and Chen parameter anchors was not independently verified during this audit.
- No external weights, researchdata package, or holdout signals were used.

## Root-cause ranking

1. Measurement-unit mismatch and mixed-label groups can make literature-level and signal-level claims non-equivalent.
2. An invalid or ambiguous cycle reference prevents physical phase interpretation.
3. Low grouped baseline performance would indicate insufficient stable structure in these label-free representations, not failed raw-data identity.
4. Any random-split gain is protocol-dependent optimism if measurement groups cross partitions.

## Scientific consequence

This verdict only controls whether a later representation revision is justified. It does not confirm adaptive fusion and does not open locked holdouts. Detailed provenance is in the JSON and stage outputs.

## Figures

- reports/figures/vsb-literature-forensic/measurement_label_patterns.png
- reports/figures/vsb-literature-forensic/baseline_mcc_by_mode.png
"""
    (report_root / "vsb_literature_forensic_audit.md").write_text(report, encoding="utf-8")
    lines = [
        "# VSB Forensic Audit Manifest", "",
        "- Audit version: " + AUDIT_VERSION,
        "- Verdict: " + verdict,
        "- Configuration: " + str(config_path.relative_to(ROOT)),
        "- Raw-data contract: PD_RAW_DATA_ROOT/data/raw",
        "- Protected inputs: reports/metrics/v5-pulse-aware/ and results/manifests/matlab_manifest_v3_windowed.csv",
        "- Locked: VSB grouped test, official unlabeled test, MATLAB Te2.",
        "", "## Stages", "",
    ]
    lines.extend("- " + stage + ": " + str(is_complete(root, stage)) for stage in ("literature", "integrity", "scan", "morphology", "cwt", "baseline"))
    (report_root / "vsb_literature_forensic_manifest.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    finish_stage(root, "report", runtime_record(started, int(integrity.get("label_summary", {}).get("signals", 0)), 0, 0))
    logger.info("report stage complete: verdict=%s", verdict)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT_CONFIG)
    parser.add_argument("--raw-root", type=Path, default=ROOT / "data/raw")
    parser.add_argument("--audit-root", type=Path, default=AUDIT_ROOT)
    parser.add_argument("--report-root", type=Path, default=REPORT_ROOT)
    parser.add_argument("--figure-root", type=Path, default=FIGURE_ROOT)
    parser.add_argument("--stage", choices=("all", "literature", "integrity", "scan", "morphology", "cwt", "baseline", "report"), default="all")
    parser.add_argument("--io-batch-size", type=int, default=4)
    parser.add_argument("--log-file", type=str, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    logger = logger_setup(args.log_file)
    args.config = args.config.resolve()
    args.raw_root = args.raw_root.resolve()
    args.audit_root = args.audit_root.resolve()
    args.report_root = args.report_root.resolve()
    args.figure_root = args.figure_root.resolve()
    args.audit_root.mkdir(parents=True, exist_ok=True)
    os.environ["PD_RAW_DATA_ROOT"] = str(args.raw_root)
    stages = ("literature", "integrity", "scan", "morphology", "cwt", "baseline", "report") if args.stage == "all" else (args.stage,)
    logger.info("audit start: stage=%s raw_root=%s", args.stage, args.raw_root)
    for stage in stages:
        if is_complete(args.audit_root, stage) and not args.force:
            logger.info("stage %s already complete; skip", stage)
            continue
        if stage == "literature":
            stage_literature(args.audit_root, logger)
        elif stage == "integrity":
            stage_integrity(args.audit_root, args.config, args.raw_root, logger)
        elif stage == "scan":
            stage_scan(args.audit_root, args.config, args.raw_root, args.io_batch_size, logger)
        elif stage == "morphology":
            stage_morphology(args.audit_root, args.config, args.raw_root, args.io_batch_size, logger)
        elif stage == "cwt":
            stage_cwt(args.audit_root, args.config, args.raw_root, args.io_batch_size, logger)
        elif stage == "baseline":
            stage_baseline(args.audit_root, logger)
        elif stage == "report":
            stage_report(args.audit_root, args.report_root, args.figure_root, args.config, logger)
    logger.info("audit finished")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
