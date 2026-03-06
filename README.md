# QuPath XGBoost Classifier Extension

A [QuPath](https://qupath.github.io) extension that brings [XGBoost](https://xgboost.readthedocs.io/) gradient-boosted tree classification to object classification workflows.
Train a model from point or area annotations across multiple project images, then run inference across any selection of images — all from a GUI, no scripting required.

---

## How it works

### Training

The extension looks for **annotated regions** (point or area ROIs) on top of existing detections to build training data.

1. For each selected project entry it reads the object hierarchy.
2. Every detection that falls inside an annotation whose class is one of the selected output classes becomes a training sample. If a detection is covered by annotations of more than one class, it is discarded to avoid label ambiguity.
3. Measurement values for the selected features are extracted from each training detection into a feature matrix.
4. An initial XGBoost model is trained on the full feature set to rank features by **gain importance**.
5. A final model is retrained using only the top-N selected features (configurable; 0 = keep all).
6. Feature names and class names are embedded directly in the saved model JSON so no external metadata file is needed at inference time.

### Inference

1. The saved model JSON is loaded; feature names and class names are read from its metadata.
2. For each selected project entry, measurements are extracted from all detections using the embedded feature list.
3. Each detection is assigned the class with the highest predicted probability.
4. Image data is saved back to the project.

---

## Requirements

- **QuPath** 0.7.0 or later
- **Java** 25
- Detections must already exist in the project images (e.g. from cell detection or tile classification)
- Training images must have point or area annotations placed on top of detections, with a `PathClass` set to one of the desired output classes

---

## Build

### Standalone shadow jar (recommended for distribution)

```bash
gradlew shadowJar
```

The output is in `build/libs/` with `-all` in the filename.
Drag this onto QuPath to install it — you will be prompted to create a user directory if you don't have one yet.

### During development (composite build)

Create a file called `include-extra` in the **root of the QuPath source directory** (not this repo):

```
[includeBuild]
../qupath-extension-xgboost

[dependencies]
io.github.qupath:qupath-extension-xgboost
```

Then run QuPath from the QuPath source directory:

```bash
gradlew run
```

When using composite build, add `implementation` alongside `shadow` for XGBoost4J in `build.gradle.kts` so the dependency is available on the runtime classpath:

```kotlin
implementation("ml.dmlc:xgboost4j_2.13:3.2.0")
shadow("ml.dmlc:xgboost4j_2.13:3.2.0")
```

---

## Usage

After installation the extension adds two commands under **Extensions → XGBoost Classifier**.

---

### Train XGBoost Classifier

Opens the training dialog.

#### Project Entries

Select which project images to use for training. Only images with annotations on top of detections will contribute training data.

- **Filter field** — type to search image names
- **Available / Selected / Both** — toggle which side of the list the filter applies to
- **With data file only** — hide images that have no saved data file

#### Features

Select which object measurements to offer to the model. Use the **↺ Refresh** button to discover available measurements from the currently open image.

- **Filter field** — type to narrow the list; toggle **Available / Selected / Both** to target either side
- All features moved to the right will be used as the initial feature set; the model then ranks and optionally reduces them further by gain importance

#### Output Classes

Select which annotation classes define the training labels. At least two classes are required.

- **Filter field** — works the same as the features filter

#### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| Model name | `my_xgboost` | Base filename; saved as `<name>.json` |
| Output dir | Project `classifiers/object_classifiers/` | Directory for the saved model |
| Rounds | 150 | Number of boosting rounds |
| Max depth | 5 | Maximum depth of each tree |
| Learning rate (η) | 0.2 | Step size shrinkage; lower = more conservative |
| Subsample | 0.8 | Fraction of training rows sampled per round |
| Top-N features | 10 | Keep only the N highest-gain features after the first pass; set to 0 to keep all |

The **Log** panel streams progress during training, including per-image object counts, the top-20 feature importances, and the final saved model path.

---

### Run XGBoost Classifier

Opens the inference dialog.

#### Project Entries

Same filter controls as the training dialog. Select the images you want to classify.

#### Model

Pick the `.json` model file produced by the training step.
The file chooser defaults to the project's `classifiers/object_classifiers/` folder.

The **Log** panel reports the number of detections classified per image.

---

## Model file format

Models are saved as XGBoost native JSON. Feature names and class names are embedded as model attributes:

- `feature_names` — the ordered list of measurement names the model expects
- `class_names` — comma-separated list of output class names in label index order

This makes a model file fully self-contained — it can be shared across projects or used in scripts as long as the same measurements exist in the target images.

---

## Tips

- **More training data is better.** Collect annotations across multiple images to cover the range of imaging conditions in your dataset.
- **Start with Top-N = 0** to see the full importance ranking in the log, then re-train with a smaller Top-N to reduce overfitting on noisy measurements.
- **Conflicting annotations** (a detection covered by two different classes) are silently discarded. If you see unexpectedly low training counts, check for overlapping annotations of different classes.
- The final model is trained from scratch on the reduced feature set, so Rounds and Max depth apply to both the ranking pass and the final pass.

---

## License

GPL v3 — consistent with QuPath's own license.

---

## Acknowledgements

- [QuPath](https://github.com/qupath/qupath) — Pete Bankhead and contributors
- [XGBoost4J](https://xgboost.readthedocs.io/en/stable/jvm/index.html) — DMLC / XGBoost contributors
