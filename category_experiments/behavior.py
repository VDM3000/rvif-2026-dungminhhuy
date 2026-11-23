from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from common import (
    FOLDS,
    FULL_RESULTS_DIR,
    category_dir,
    image_sets_from_disk,
)


META_COLUMNS = ["subject_int", "subject_id", "set_name", "label"]


def aggregate_category_features(category: str) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    source = FULL_RESULTS_DIR / "stim_behavior_features.csv"
    if not source.exists():
        raise FileNotFoundError(
            f"Missing stimulus feature table: {source}. Run the baseline notebook first."
        )
    stimulus = pd.read_csv(source)
    image_stems = stimulus["image"].astype(str).map(lambda value: Path(value).stem)
    category_images = image_sets_from_disk()[category]
    stimulus = stimulus[image_stems.isin(category_images)].copy()
    if stimulus["image"].map(lambda value: Path(str(value)).stem).nunique() != len(category_images):
        raise RuntimeError(f"Behavior table does not cover every {category} image")

    stimulus_meta = ["subject_int", "subject_id", "set_name", "label", "image"]
    numeric_columns = [column for column in stimulus.columns if column not in stimulus_meta]
    aggregate_functions = ["mean", "std", "median", "min", "max"]
    subject_aggregate = stimulus.groupby("subject_int")[numeric_columns].agg(aggregate_functions)
    subject_aggregate.columns = [
        f"{column}_{function}" for column, function in subject_aggregate.columns
    ]
    subject_aggregate = subject_aggregate.reset_index()
    subject_meta = stimulus[META_COLUMNS].drop_duplicates("subject_int")
    valid_images = (
        stimulus.groupby("subject_int")["image"].nunique().reset_index(name="num_valid_images")
    )
    total_fixations = (
        stimulus.groupby("subject_int")["fixation_count"]
        .sum()
        .reset_index(name="num_total_fixations")
    )
    mean_fixations = (
        stimulus.groupby("subject_int")["fixation_count"]
        .mean()
        .reset_index(name="mean_fixations_per_image")
    )
    subject = (
        subject_meta.merge(subject_aggregate, on="subject_int", how="left")
        .merge(valid_images, on="subject_int", how="left")
        .merge(total_fixations, on="subject_int", how="left")
        .merge(mean_fixations, on="subject_int", how="left")
        .sort_values("subject_int")
        .reset_index(drop=True)
    )
    if len(subject) != 160:
        raise RuntimeError(f"Expected 160 subjects for {category}, found {len(subject)}")

    all_features = [column for column in subject.columns if column not in META_COLUMNS]
    duration = sorted(column for column in all_features if column.startswith("duration_"))
    pupil = sorted(column for column in all_features if column.startswith("pupil_"))
    scanpath_bases = [
        "fixation_count",
        "x_mean",
        "y_mean",
        "x_std",
        "y_std",
        "x_dispersion",
        "y_dispersion",
        "xy_dispersion_area",
        "distance_to_center_mean",
        "distance_to_center_std",
        "distance_to_center_max",
        "scanpath_length",
        "saccade_amplitude_mean",
        "saccade_amplitude_std",
        "saccade_amplitude_median",
        "saccade_amplitude_max",
        "num_valid_images",
        "num_total_fixations",
        "mean_fixations_per_image",
    ]

    def belongs(column: str) -> bool:
        return any(column == base or column.startswith(base + "_") for base in scanpath_bases)

    scanpath = sorted(column for column in all_features if belongs(column))
    if any(column.startswith(("duration_", "pupil_")) for column in scanpath):
        raise RuntimeError("Scanpath feature set overlaps duration or pupil features")
    feature_sets = {
        "duration_only": duration,
        "pupil_only": pupil,
        "scanpath_only": scanpath,
        "duration_pupil": sorted(set(duration + pupil)),
        "duration_scanpath": sorted(set(duration + scanpath)),
        "pupil_scanpath": sorted(set(pupil + scanpath)),
        "all_behavior": sorted(set(duration + pupil + scanpath)),
    }
    return subject, feature_sets


def make_model() -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clf", HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.03, max_leaf_nodes=15,
            l2_regularization=0.01, random_state=42,
        )),
    ])


def behavior_cv(subject: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    predictions: list[pd.DataFrame] = []
    for fold in FOLDS:
        train = subject[subject["set_name"] != fold]
        validation = subject[subject["set_name"] == fold].copy()
        model = make_model()
        model.fit(train[columns], train["label"].astype(int))
        validation["behavior_prob"] = model.predict_proba(validation[columns])[:, 1]
        predictions.append(validation[META_COLUMNS + ["behavior_prob"]])
    return pd.concat(predictions, ignore_index=True).sort_values(["set_name", "subject_int"])


def load_msnet_predictions(category: str) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for fold in FOLDS:
        path = category_dir(category) / "msnet" / fold / "prob_val.csv"
        if not path.exists():
            raise FileNotFoundError(f"MSNet fold is incomplete: {path}")
        frames.append(pd.read_csv(path))
    result = pd.concat(frames, ignore_index=True).sort_values(["set_name", "subject_int"])
    if len(result) != 160 or result["subject_int"].nunique() != 160:
        raise RuntimeError(f"Invalid MSNet OOF predictions for {category}")
    return result
