package qupath.ext.xgboost;

import ml.dmlc.xgboost4j.java.Booster;
import ml.dmlc.xgboost4j.java.DMatrix;
import ml.dmlc.xgboost4j.java.XGBoost;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import qupath.lib.objects.PathObject;
import qupath.lib.projects.ProjectImageEntry;
import qupath.lib.objects.PathObjectTools;


import java.awt.image.BufferedImage;
import java.io.File;
import java.util.*;
import java.util.function.Consumer;
import java.util.stream.Collectors;

/**
 * Core XGBoost training logic.
 * <p>
 * Translated from {@code train_xgboost.groovy}.  Call
 * {@link #train(List, File, int, int, float, float, int, Consumer)} from a
 * background thread; pass a {@link Consumer Consumer&lt;String&gt;} to receive
 * progress messages suitable for display in a log TextArea.
 */
public class XGBoostTrainer {

    private static final Logger logger = LoggerFactory.getLogger(XGBoostTrainer.class);

    private XGBoostTrainer() {}

    // ── Public API ─────────────────────────────────────────────────────────────

    /**
     * Train an XGBoost model from point-annotation ground truth in the supplied
     * project entries and save it alongside the base classifier JSON.
     *
     * @param entries          project entries to use for training
     * @param classifierJson   existing object-classifier JSON that defines feature
     *                         names and class names/colours
     * @param numRounds        boosting rounds
     * @param maxDepth         maximum tree depth
     * @param eta              learning rate
     * @param subsample        row-subsampling ratio per round (0–1)
     * @param topNFeatures     keep only the top-N features by gain importance;
     *                         0 keeps all features
     * @param log              consumer for human-readable progress messages
     * @throws Exception       on I/O or XGBoost error
     */
    public static void train(
            List<ProjectImageEntry<BufferedImage>> entries,
            List<String> featureNames,
            List<String> classNames,
            String modelOutputPath,          // full path for the saved .json
            int numRounds,
            int maxDepth,
            float eta,
            float subsample,
            int topNFeatures,
            Consumer<String> log
    ) throws Exception {

        log.accept("Features: " + featureNames.size() + "   Classes: " + classNames);

        // ── 2. Collect training data from all selected entries ──────────────────
        List<float[]> trainingFeatures = new ArrayList<>();
        List<Integer> trainingLabels   = new ArrayList<>();

        for (var entry : entries) {
            var imageData = entry.readImageData();
            var hierarchy = imageData.getHierarchy();

            // b) Match detections to points; discard conflicts (multiple classes)
            Map<PathObject, String> detClass    = new LinkedHashMap<>();

            for (PathObject anno : hierarchy.getAnnotationObjects()) {
                if (anno.getPathClass() == null || anno.getROI() == null) continue;
                String cls = anno.getPathClass().toString();
                if (!classNames.contains(cls)) continue;

                List<PathObject> hits;
                if (anno.getROI().isPoint()) {
                    hits = new ArrayList<>();
                    for (var pt : anno.getROI().getAllPoints()) {
                        hits.addAll(PathObjectTools.getObjectsForLocation(
                                hierarchy, pt.getX(), pt.getY(),
                                anno.getROI().getZ(), anno.getROI().getT(), -1));
                    }
                } else {
                    hits = new ArrayList<>(hierarchy.getAllDetectionsForROI(anno.getROI()));
                }

                // assign class — handle conflicts: only keep detections touched by exactly one class
                for (PathObject det : hits) {
                    if (detClass.containsKey(det)) {
                        detClass.put(det, null); // conflict marker
                    } else {
                        detClass.put(det, cls);
                    }
                }
            }
            // remove conflicts and detections with unknown class
            detClass.entrySet().removeIf(e -> e.getValue() == null);
            List<PathObject> validDets = new ArrayList<>(detClass.keySet());

            if (validDets.isEmpty()) {
                log.accept("  " + entry.getImageName() + ": 0 training objects found");
                continue;
            }

            // c) Extract measurement vectors
            for (PathObject det : validDets) {
                var mlist = det.getMeasurementList();
                float[] row = new float[featureNames.size()];
                for (int i = 0; i < featureNames.size(); i++) {
                    double v = mlist.get(featureNames.get(i));
                    row[i] = Double.isNaN(v) ? 0f : (float) v;
                }
                trainingFeatures.add(row);
                trainingLabels.add(classNames.indexOf(detClass.get(det)));
            }

            log.accept("  " + entry.getImageName() + ": " + validDets.size() + " training objects");
        }

        if (trainingFeatures.isEmpty()) {
            throw new IllegalStateException(
                    "No training objects found. Make sure selected entries have point annotations "
                    + "placed on top of detections with one of the known classes.");
        }

        int nSamples  = trainingFeatures.size();
        int nFeatures = featureNames.size();
        int nClasses  = classNames.size();
        log.accept("Total: " + nSamples + " objects  |  " + nClasses + " classes");

        // ── 3. Flatten to DMatrix ───────────────────────────────────────────────
        float[] flatData   = new float[nSamples * nFeatures];
        float[] labelArray = new float[nSamples];
        for (int i = 0; i < nSamples; i++) {
            System.arraycopy(trainingFeatures.get(i), 0, flatData, i * nFeatures, nFeatures);
            labelArray[i] = trainingLabels.get(i);
        }
        DMatrix trainMat = new DMatrix(flatData, nSamples, nFeatures, Float.NaN);
        trainMat.setLabel(labelArray);

        // ── 4. XGBoost parameters ───────────────────────────────────────────────
        Map<String, Object> params = buildParams(nClasses, maxDepth, eta, subsample);
        Map<String, DMatrix> watches = new LinkedHashMap<>();
        watches.put("train", trainMat);

        // ── 5. Initial training to rank features by gain ────────────────────────
        log.accept("Training initial model for feature-importance ranking…");
        Booster booster = XGBoost.train(trainMat, params, numRounds, watches, null, null);

        record FeatureScore(String name, double score, int idx) {}

        // getScore returns Map<String,Integer> in XGBoost4J; values are usable
        // for relative ordering even when expressed as integers.
        Map<String, Double> rawScores = booster.getScore("", "gain");
        List<FeatureScore> ranked = rawScores.entrySet().stream()
                .map(e -> {
                    int idx = Integer.parseInt(e.getKey().replace("f", ""));
                    return new FeatureScore(featureNames.get(idx),
                                           e.getValue(), idx);
                })
                .sorted(Comparator.comparingDouble(FeatureScore::score).reversed())
                .collect(Collectors.toList());

        log.accept("\n── Top feature importances (gain) ──");
        ranked.stream()
              .limit(Math.min(20, ranked.size()))
              .forEach(fs -> log.accept(
                      String.format("  %-50s  %.0f", fs.name(), fs.score())));

        // ── 6. Select top-N features ────────────────────────────────────────────
        List<FeatureScore> selected = (topNFeatures > 0 && topNFeatures < nFeatures)
                ? ranked.subList(0, topNFeatures)
                : ranked;
        int[] selIdx = selected.stream().mapToInt(FeatureScore::idx).toArray();
        log.accept("\nUsing " + selIdx.length + " features for final model.");

        // ── 7. Retrain on selected features only ────────────────────────────────
        float[] reducedData = new float[nSamples * selIdx.length];
        for (int i = 0; i < nSamples; i++) {
            float[] row = trainingFeatures.get(i);
            for (int j = 0; j < selIdx.length; j++) {
                reducedData[i * selIdx.length + j] = row[selIdx[j]];
            }
        }
        DMatrix reducedMat = new DMatrix(reducedData, nSamples, selIdx.length, Float.NaN);
        reducedMat.setLabel(labelArray);

        Map<String, DMatrix> reducedWatches = new LinkedHashMap<>();
        reducedWatches.put("train", reducedMat);

        log.accept("Training final model…");
        Booster finalBooster = XGBoost.train(reducedMat, params, numRounds,
                reducedWatches, null, null);

        // ── 8. Embed metadata and save ──────────────────────────────────────────
        String[] finalNames = Arrays.stream(selIdx)
                .mapToObj(featureNames::get)
                .toArray(String[]::new);

        finalBooster.setFeatureNames(finalNames);
        finalBooster.setAttr("class_names", String.join(",", classNames));

        finalBooster.saveModel(modelOutputPath);
        log.accept("Model saved → " + modelOutputPath);
        log.accept("Training complete!");
    }

    // ── Helpers ────────────────────────────────────────────────────────────────

    private static Map<String, Object> buildParams(
            int nClasses, int maxDepth, float eta, float subsample) {

        Map<String, Object> p = new LinkedHashMap<>();
        p.put("max_depth",   maxDepth);
        p.put("eta",         (double) eta);
        p.put("subsample",   (double) subsample);
        p.put("objective",   nClasses == 2 ? "binary:logistic" : "multi:softprob");
        p.put("eval_metric", nClasses == 2 ? "logloss"         : "mlogloss");
        p.put("nthread",     Runtime.getRuntime().availableProcessors());
        p.put("seed",        42);
        if (nClasses > 2) p.put("num_class", nClasses);
        return p;
    }
}
