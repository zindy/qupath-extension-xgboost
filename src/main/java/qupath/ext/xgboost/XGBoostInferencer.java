package qupath.ext.xgboost;

import ml.dmlc.xgboost4j.java.Booster;
import ml.dmlc.xgboost4j.java.DMatrix;
import ml.dmlc.xgboost4j.java.XGBoost;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import qupath.lib.objects.PathObject;
import qupath.lib.objects.classes.PathClass;
import qupath.lib.projects.ProjectImageEntry;

import java.awt.image.BufferedImage;
import java.io.File;
import java.util.Arrays;
import java.util.Collection;
import java.util.List;
import java.util.function.Consumer;

/**
 * Core XGBoost inference logic.
 * <p>
 * Translated from {@code infer_xgboost.groovy}.  Call
 * {@link #infer(List, File, Consumer)} from a background thread.
 */
public class XGBoostInferencer {

    private static final Logger logger = LoggerFactory.getLogger(XGBoostInferencer.class);

    private XGBoostInferencer() {}

    // ── Public API ─────────────────────────────────────────────────────────────

    /**
     * Load a saved XGBoost model and classify all detections in the supplied
     * project entries.  Image data is saved back to the project after each entry.
     *
     * @param entries   project entries to classify
     * @param modelFile {@code _xgboost.json} file produced by {@link XGBoostTrainer}
     * @param log       consumer for progress messages
     * @throws Exception on I/O or XGBoost error
     */
    public static void infer(
            List<ProjectImageEntry<BufferedImage>> entries,
            File modelFile,
            Consumer<String> log
    ) throws Exception {

        // ── 1. Load model and embedded metadata ────────────────────────────────
        Booster booster = XGBoost.loadModel(modelFile.getAbsolutePath());

        String[] requiredFeatures = booster.getFeatureNames();
        String   classAttr        = booster.getAttr("class_names");

        if (requiredFeatures == null || requiredFeatures.length == 0) {
            throw new IllegalStateException(
                    "Model has no embedded feature names. "
                    + "Was it trained with XGBoostTrainer?");
        }
        if (classAttr == null || classAttr.isBlank()) {
            throw new IllegalStateException(
                    "Model has no embedded class_names attribute. "
                    + "Was it trained with XGBoostTrainer?");
        }

        String[] classNames = classAttr.split(",");
        int nClasses = classNames.length;

        log.accept("Model loaded:  " + requiredFeatures.length + " features, "
                + nClasses + " classes: " + Arrays.toString(classNames));

        // ── 2. Classify each entry ─────────────────────────────────────────────
        for (var entry : entries) {
            var imageData = entry.readImageData();
            var hierarchy = imageData.getHierarchy();
            Collection<PathObject> detections = hierarchy.getDetectionObjects();

            if (detections.isEmpty()) {
                log.accept("  " + entry.getImageName() + ": no detections – skipped");
                continue;
            }

            // Build prediction matrix
            int n = detections.size();
            int f = requiredFeatures.length;
            float[] predData = new float[n * f];

            int i = 0;
            for (PathObject det : detections) {
                var mlist = det.getMeasurementList();
                for (int j = 0; j < f; j++) {
                    double v = mlist.get(requiredFeatures[j]);
                    predData[i * f + j] = Double.isNaN(v) ? 0f : (float) v;
                }
                i++;
            }

            DMatrix predMat = new DMatrix(predData, n, f, Float.NaN);
            float[][] preds = booster.predict(predMat);

            // Assign predicted classes
            i = 0;
            for (PathObject det : detections) {
                int predClass;
                if (nClasses == 2) {
                    predClass = preds[i][0] >= 0.5f ? 1 : 0;
                } else {
                    int best = 0;
                    for (int c = 1; c < nClasses; c++) {
                        if (preds[i][c] > preds[i][best]) best = c;
                    }
                    predClass = best;
                }
                det.setPathClass(PathClass.fromString(classNames[predClass]));
                i++;
            }

            entry.saveImageData(imageData);

            log.accept("  " + entry.getImageName() + ": classified " + n + " detections");
        }

        log.accept("Done.  Classes: " + Arrays.toString(classNames));
    }
}
