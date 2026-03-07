package qupath.ext.xgboost.ui;

import javafx.application.Platform;
import javafx.collections.ListChangeListener;
import javafx.concurrent.Task;
import javafx.geometry.Insets;
import javafx.geometry.Pos;
import javafx.scene.Node;
import javafx.scene.Scene;
import javafx.scene.control.*;
import javafx.scene.layout.*;
import javafx.stage.DirectoryChooser;
import javafx.stage.Modality;
import javafx.stage.Stage;
import org.controlsfx.control.ListSelectionView;
import qupath.ext.xgboost.XGBoostTrainer;
import qupath.lib.gui.QuPathGUI;
import qupath.lib.gui.panes.ProjectEntryPredicate;
import qupath.lib.objects.PathObjectFilter;
import qupath.lib.objects.PathObjectTools;
import qupath.lib.objects.classes.PathClass;
import qupath.lib.projects.ProjectImageEntry;
import qupath.lib.gui.dialogs.ProjectDialogs;
import qupath.fx.dialogs.Dialogs;

import java.awt.image.BufferedImage;
import java.io.File;
import java.util.*;
import java.util.stream.Collectors;

/**
 * Train-dialog controller.
 * <p>
 * Accordion UI: Project Entries / Features / Output Classes, each with a
 * filter row that lets you target the <em>Available</em> or <em>Selected</em>
 * side independently.  Parameters and Log sit below the accordion.
 */
public class TrainController {

    // ── Filter side ────────────────────────────────────────────────────────────
    private enum FilterSide { AVAILABLE, SELECTED, BOTH }

    // ── Master (unfiltered) lists ──────────────────────────────────────────────
    private List<ProjectImageEntry<BufferedImage>> allImages   = new ArrayList<>();
    private List<String>                           allFeatures = new ArrayList<>();
    private List<PathClass>                        allClasses  = new ArrayList<>();

    // ── Backing "true-selected" lists (survive filter changes) ────────────────
    private final List<ProjectImageEntry<BufferedImage>> selectedEntries  = new ArrayList<>();
    private final List<String>                           selectedFeatures = new ArrayList<>();
    private final List<PathClass>                        selectedClasses  = new ArrayList<>();

    // ── Re-entrancy guards ─────────────────────────────────────────────────────
    private boolean updatingEntries, updatingFeatures, updatingClasses;

    // ── Selection views ────────────────────────────────────────────────────────
    private ListSelectionView<ProjectImageEntry<BufferedImage>> entrySelector;
    private ListSelectionView<String>                           featureSelector;
    private ListSelectionView<PathClass>                        classSelector;

    // ── Filter controls ────────────────────────────────────────────────────────
    private TextField   entryFilterField;
    private CheckBox    withDataOnlyBox;
    private ToggleGroup entryFilterGroup;

    private TextField   featureFilterField;
    private ToggleGroup featureFilterGroup;

    private TextField   classFilterField;
    private ToggleGroup classFilterGroup;

    // ── Model output ───────────────────────────────────────────────────────────
    private TextField modelNameField;
    private TextField modelDirField;

    // ── Hyperparameter spinners ────────────────────────────────────────────────
    private Spinner<Integer> numRoundsSpinner;
    private Spinner<Integer> maxDepthSpinner;
    private Spinner<Double>  etaSpinner;
    private Spinner<Double>  subsampleSpinner;
    private Spinner<Integer> topNSpinner;

    // ── Output ─────────────────────────────────────────────────────────────────
    private TextArea logArea;
    private Button   trainButton;

    private final QuPathGUI qupath;
    private Stage stage;

    public TrainController(QuPathGUI qupath) {
        this.qupath = qupath;
    }

    // ── Public API ──────────────────────────────────────────────────────────────

    public void show() {
        if (stage == null) stage = buildStage();
        refreshEntries();
        refreshFeaturesAndClasses();
        stage.show();
        stage.toFront();
    }

    // ── Stage ───────────────────────────────────────────────────────────────────

    private Stage buildStage() {

        // ══ 1. PROJECT ENTRIES ════════════════════════════════════════════════
        entrySelector = new ListSelectionView<>();
        entrySelector.setCellFactory(lv -> new ListCell<>() {
            @Override protected void updateItem(ProjectImageEntry<BufferedImage> item, boolean empty) {
                super.updateItem(item, empty);
                setText(empty || item == null ? "" : item.getImageName());
            }
        });
        entrySelector.setPrefHeight(200);
        wireTargetListener(entrySelector, selectedEntries,
                () -> updatingEntries, v -> updatingEntries = v,
                this::updateEntryFilter);

        entryFilterGroup = new ToggleGroup();
        entryFilterField = new TextField();
        entryFilterField.setPromptText("Filter…");
        entryFilterField.textProperty().addListener((o, a, b) -> updateEntryFilter());
        withDataOnlyBox  = new CheckBox("With data file only");
        withDataOnlyBox.selectedProperty().addListener((o, a, b) -> updateEntryFilter());

        HBox entryFilterRow = filterRow(entryFilterGroup, entryFilterField, withDataOnlyBox);

        VBox entryContent = new VBox(6, entrySelector, entryFilterRow);
        entryContent.setPadding(new Insets(6));
        TitledPane entryPane = new TitledPane("Project Entries", entryContent);
        entryPane.setAnimated(false);

        // ══ 2. FEATURES ═══════════════════════════════════════════════════════
        featureSelector = new ListSelectionView<>();
        featureSelector.setPrefHeight(200);
        wireTargetListener(featureSelector, selectedFeatures,
                () -> updatingFeatures, v -> updatingFeatures = v,
                this::updateFeatureFilter);

        featureFilterGroup = new ToggleGroup();
        featureFilterField = new TextField();
        featureFilterField.setPromptText("Filter…");
        featureFilterField.textProperty().addListener((o, a, b) -> updateFeatureFilter());

        Button refreshBtn = new Button("↺  Refresh from current image");
        refreshBtn.setOnAction(e -> refreshFeaturesAndClasses());

        HBox featureFilterRow = filterRow(featureFilterGroup, featureFilterField, refreshBtn);

        VBox featureContent = new VBox(6, featureSelector, featureFilterRow);
        featureContent.setPadding(new Insets(6));
        TitledPane featurePane = new TitledPane("Features", featureContent);
        featurePane.setAnimated(false);

        // ══ 3. OUTPUT CLASSES ═════════════════════════════════════════════════
        classSelector = new ListSelectionView<>();
        classSelector.setCellFactory(lv -> new ListCell<>() {
            @Override protected void updateItem(PathClass item, boolean empty) {
                super.updateItem(item, empty);
                setText(empty || item == null ? "" : item.toString());
            }
        });
        classSelector.setPrefHeight(140);
        wireTargetListener(classSelector, selectedClasses,
                () -> updatingClasses, v -> updatingClasses = v,
                this::updateClassFilter);

        classFilterGroup = new ToggleGroup();
        classFilterField = new TextField();
        classFilterField.setPromptText("Filter…");
        classFilterField.textProperty().addListener((o, a, b) -> updateClassFilter());

        HBox classFilterRow = filterRow(classFilterGroup, classFilterField);

        VBox classContent = new VBox(6, classSelector, classFilterRow);
        classContent.setPadding(new Insets(6));
        TitledPane classPane = new TitledPane("Output Classes", classContent);
        classPane.setAnimated(false);

        entryFilterGroup.selectedToggleProperty().addListener((obs, o, n) -> updateEntryFilter());
        featureFilterGroup.selectedToggleProperty().addListener((obs, o, n) -> updateFeatureFilter());
        classFilterGroup.selectedToggleProperty().addListener((obs, o, n) -> updateClassFilter());

        // ══ Accordion ═════════════════════════════════════════════════════════
        Accordion accordion = new Accordion(entryPane, featurePane, classPane);
        accordion.setExpandedPane(entryPane);

        // ══ 4. PARAMETERS ═════════════════════════════════════════════════════
        modelNameField   = new TextField("my_xgboost");
        modelNameField.setPromptText("model_name  (saved as <name>.json)");
        HBox.setHgrow(modelNameField, Priority.ALWAYS);

        modelDirField = new TextField();
        modelDirField.setPromptText("Output directory (defaults to project classifiers folder)");
        HBox.setHgrow(modelDirField, Priority.ALWAYS);

        Button browseDir = new Button("Browse…");
        browseDir.setOnAction(e -> {
            DirectoryChooser dc = new DirectoryChooser();
            dc.setTitle("Select output directory");
            trySetInitialDir(dc);
            File dir = dc.showDialog(stage);
            if (dir != null) modelDirField.setText(dir.getAbsolutePath());
        });
        HBox dirRow = new HBox(6, modelDirField, browseDir);
        dirRow.setAlignment(Pos.CENTER_LEFT);

        numRoundsSpinner = intSpinner(1, 2000, 150);
        maxDepthSpinner  = intSpinner(1, 20,    5);
        etaSpinner       = dblSpinner(0.001, 1.0, 0.2, 0.01);
        subsampleSpinner = dblSpinner(0.1,   1.0, 0.8, 0.05);
        topNSpinner      = intSpinner(0, 500, 10);

        GridPane paramGrid = new GridPane();
        paramGrid.setHgap(10); paramGrid.setVgap(6);
        paramGrid.setPadding(new Insets(6));
        int row = 0;
        addParamRow(paramGrid, row++, "Model name:",     modelNameField,   "Base filename for the saved model JSON");
        addParamRow(paramGrid, row++, "Output dir:",     dirRow,           "Directory where the model JSON will be saved");
        addParamRow(paramGrid, row++, "Rounds:",         numRoundsSpinner, "Number of boosting rounds");
        addParamRow(paramGrid, row++, "Max depth:",      maxDepthSpinner,  "Maximum tree depth");
        addParamRow(paramGrid, row++, "Learning rate:",  etaSpinner,       "Step size shrinkage η (0–1)");
        addParamRow(paramGrid, row++, "Subsample:",      subsampleSpinner, "Fraction of training samples per round (0–1)");
        addParamRow(paramGrid, row++, "Top-N features:", topNSpinner,      "Keep only the N highest-gain features (0 = keep all)");

        TitledPane paramsPane = new TitledPane("Parameters", paramGrid);
        paramsPane.setCollapsible(true);
        paramsPane.setExpanded(false);
        paramsPane.setAnimated(false);

        // ══ 5. LOG ════════════════════════════════════════════════════════════
        logArea = new TextArea();
        logArea.setEditable(false);
        logArea.setPrefHeight(160);
        logArea.setStyle("-fx-font-family: monospace; -fx-font-size: 11;");
        VBox.setVgrow(logArea, Priority.ALWAYS);

        // ══ 6. TRAIN BUTTON ═══════════════════════════════════════════════════
        trainButton = new Button("Train");
        trainButton.setDefaultButton(true);
        trainButton.setPrefWidth(100);
        trainButton.setOnAction(e -> runTraining());
        HBox btnRow = new HBox(trainButton);
        btnRow.setAlignment(Pos.CENTER_RIGHT);

        // ══ Root ══════════════════════════════════════════════════════════════
        VBox root = new VBox(8, accordion, paramsPane, new Label("Log:"), logArea, btnRow);
        root.setPadding(new Insets(10));

        Stage s = new Stage();
        s.initOwner(qupath.getStage());
        s.initModality(Modality.NONE);
        s.setTitle("Train XGBoost Classifier");
        s.setScene(new Scene(root, 740, 820));
        s.setMinWidth(520);
        s.setMinHeight(500);
        return s;
    }

    // ── Filter-row builder ─────────────────────────────────────────────────────

    /**
     * Creates a filter row:  [Available] [Selected]  [__filter field__]  [extra nodes…]
     * The active toggle button gets a yellow tint.
     */
    private static HBox filterRow(ToggleGroup group, TextField filterField, Node... extras) {
        ToggleButton btnAvail = filterToggle("Available", group, true);
        ToggleButton btnSel   = filterToggle("Selected",  group, false);
        ToggleButton btnBoth  = filterToggle("Both",      group, false);

        HBox.setHgrow(filterField, Priority.ALWAYS);

        HBox row = new HBox(4, btnAvail, btnSel, btnBoth, filterField);
        for (Node n : extras) row.getChildren().add(n);
        row.setAlignment(Pos.CENTER_LEFT);
        row.setPadding(new Insets(4, 0, 0, 0));
        return row;
    }

    private static ToggleButton filterToggle(String text, ToggleGroup group, boolean selected) {
        ToggleButton btn = new ToggleButton(text);
        btn.setToggleGroup(group);
        btn.setSelected(selected);
        btn.setPrefHeight(24);
        // Yellow tint when active
        //String onStyle  = "-fx-base: #e8c840;";
        //String offStyle = "";
        //btn.setStyle(selected ? onStyle : offStyle);
        //btn.selectedProperty().addListener((obs, o, n) -> btn.setStyle(n ? onStyle : offStyle));
        // Prevent deselecting all buttons (at least one must stay selected)
        btn.addEventFilter(javafx.scene.input.MouseEvent.MOUSE_RELEASED, e -> {
            if (btn.isSelected()) e.consume();
        });
        return btn;
    }

    // ── Backing-list listener wiring ───────────────────────────────────────────

    /**
     * Attaches a {@link ListChangeListener} to {@code selector.getTargetItems()} that
     * keeps {@code backingList} in sync when the user moves items, then triggers a
     * filter rebuild via {@code onChanged}.
     */
    private <T> void wireTargetListener(
            ListSelectionView<T> selector,
            List<T> backingList,
            java.util.function.BooleanSupplier guardGet,
            java.util.function.Consumer<Boolean> guardSet,
            Runnable onChanged) {

        selector.getTargetItems().addListener((ListChangeListener<T>) change -> {
            if (guardGet.getAsBoolean()) return;
            guardSet.accept(true);
            try {
                while (change.next()) {
                    if (change.wasAdded())   backingList.addAll(change.getAddedSubList());
                    if (change.wasRemoved()) backingList.removeAll(change.getRemoved());
                }
            } finally {
                guardSet.accept(false);
            }
            // Defer so we don't modify lists mid-change-event
            Platform.runLater(onChanged);
        });
    }

    // ── Data refresh ───────────────────────────────────────────────────────────

    @SuppressWarnings("unchecked")
    private void refreshEntries() {
        var project = qupath.getProject();
        allImages = project == null ? List.of()
                : (List<ProjectImageEntry<BufferedImage>>) (List<?>) project.getImageList();
        // Prune selections that are no longer in the project
        selectedEntries.retainAll(allImages);
        updateEntryFilter();
    }

    private void refreshFeaturesAndClasses() {
        var imageData = qupath.getImageData();
        if (imageData == null) {
            logArea.appendText("[INFO] Open an image first to discover features and classes.\n");
            return;
        }
        var hierarchy = imageData.getHierarchy();

        var detections = hierarchy.getFlattenedObjectList(null).stream()
                .filter(PathObjectFilter.DETECTIONS_ALL)
                .collect(Collectors.toList());
        allFeatures = new ArrayList<>(PathObjectTools.getAvailableFeatures(detections));

        allClasses = hierarchy.getAnnotationObjects().stream()
                .map(a -> a.getPathClass())
                .filter(Objects::nonNull)
                .distinct().sorted()
                .collect(Collectors.toList());

        selectedFeatures.retainAll(allFeatures);
        selectedClasses.retainAll(allClasses);

        updateFeatureFilter();
        updateClassFilter();

        logArea.appendText("[INFO] Found " + allFeatures.size() + " features, "
                + allClasses.size() + " classes from current image.\n");
    }

    // ── Filter update methods ──────────────────────────────────────────────────

    private void updateEntryFilter() {
        if (updatingEntries) return;
        String      text     = entryFilterField  == null ? "" : entryFilterField.getText();
        boolean     withData = withDataOnlyBox   != null && withDataOnlyBox.isSelected();
        FilterSide  side     = entryFilterGroup  == null ? FilterSide.AVAILABLE : getFilterSide(entryFilterGroup);

        updatingEntries = true;
        try {
            // Source: images not already selected, matching filter when side targets available
            var sourcePredicate = ProjectEntryPredicate.createIgnoreCase(
                    side != FilterSide.SELECTED ? text : "");

            List<ProjectImageEntry<BufferedImage>> source = allImages.stream()
                    .filter(p -> !selectedEntries.contains(p))
                    .filter(p -> !withData || p.hasImageData())
                    .filter(sourcePredicate)
                    .collect(Collectors.toList());

            // Target: selected items, filtered when side targets selected
            var targetPredicate = ProjectEntryPredicate.createIgnoreCase(
                    side != FilterSide.AVAILABLE ? text : "");

            List<ProjectImageEntry<BufferedImage>> target = selectedEntries.stream()
                    .filter(p -> !withData || p.hasImageData())
                    .filter(targetPredicate)
                    .collect(Collectors.toList());

            entrySelector.getSourceItems().setAll(source);
            entrySelector.getTargetItems().setAll(target);
        } finally {
            updatingEntries = false;
        }
    }

    private void updateFeatureFilter() {
        if (updatingFeatures) return;
        String     text = featureFilterField == null ? "" : featureFilterField.getText().toLowerCase();
        FilterSide side = featureFilterGroup == null ? FilterSide.AVAILABLE : getFilterSide(featureFilterGroup);

        updatingFeatures = true;
        try {
            List<String> source = allFeatures.stream()
                    .filter(f -> !selectedFeatures.contains(f))
                    .filter(f -> side == FilterSide.SELECTED || text.isBlank()
                              || f.toLowerCase().contains(text))
                    .collect(Collectors.toList());

            List<String> target = selectedFeatures.stream()
                    .filter(f -> side == FilterSide.AVAILABLE || text.isBlank()
                              || f.toLowerCase().contains(text))
                    .collect(Collectors.toList());

            featureSelector.getSourceItems().setAll(source);
            featureSelector.getTargetItems().setAll(target);
        } finally {
            updatingFeatures = false;
        }
    }

    private void updateClassFilter() {
        if (updatingClasses) return;
        String     text = classFilterField == null ? "" : classFilterField.getText().toLowerCase();
        FilterSide side = classFilterGroup == null ? FilterSide.AVAILABLE : getFilterSide(classFilterGroup);

        updatingClasses = true;
        try {
            List<PathClass> source = allClasses.stream()
                    .filter(c -> !selectedClasses.contains(c))
                    .filter(c -> side == FilterSide.SELECTED || text.isBlank()
                              || c.toString().toLowerCase().contains(text))
                    .collect(Collectors.toList());

            List<PathClass> target = selectedClasses.stream()
                    .filter(c -> side == FilterSide.AVAILABLE || text.isBlank()
                              || c.toString().toLowerCase().contains(text))
                    .collect(Collectors.toList());

            classSelector.getSourceItems().setAll(source);
            classSelector.getTargetItems().setAll(target);
        } finally {
            updatingClasses = false;
        }
    }

    // ── Training ───────────────────────────────────────────────────────────────

    private void runTraining() {
        // Use the backing lists (full selection, not just what's visible through filter)
        if (selectedEntries.isEmpty() || selectedFeatures.isEmpty()
                || selectedClasses.size() < 2) {
            logArea.appendText("[ERROR] Select entries, ≥1 feature, and ≥2 classes.\n");
            return;
        }
        String name = modelNameField.getText().trim();
        if (name.isBlank()) {
            logArea.appendText("[ERROR] Enter a model name.\n");
            return;
        }

        String dirText = modelDirField.getText().trim();
        File outDir;
        if (!dirText.isBlank()) {
            outDir = new File(dirText);
        } else {
            var project = qupath.getProject();
            outDir = (project != null && project.getPath() != null)
                    ? project.getPath().getParent()
                              .resolve("classifiers").resolve("object_classifiers").toFile()
                    : new File(System.getProperty("user.home"));
        }
        outDir.mkdirs();
        String modelPath = new File(outDir, name + ".json").getAbsolutePath();

        List<ProjectImageEntry<BufferedImage>> entries   = List.copyOf(selectedEntries);
        List<String>                           features  = List.copyOf(selectedFeatures);
        List<String>                           classNames = selectedClasses.stream()
                .map(PathClass::toString).collect(Collectors.toList());

        int   numRounds = numRoundsSpinner.getValue();
        int   maxDepth  = maxDepthSpinner.getValue();
        float eta       = etaSpinner.getValue().floatValue();
        float subsample = subsampleSpinner.getValue().floatValue();
        int   topN      = topNSpinner.getValue();

        // Warn if any selected entry is currently open (may have unsaved changes)
        List<ProjectImageEntry<BufferedImage>> openAndSelected = selectedEntries.stream()
                .filter(ProjectDialogs.getCurrentImages(qupath)::contains)
                .collect(Collectors.toList());

        if (!openAndSelected.isEmpty()) {
            String names = openAndSelected.stream()
                    .map(ProjectImageEntry::getImageName)
                    .collect(Collectors.joining("\n  "));
            var result = Dialogs.builder()
                    .title("Unsaved changes?")
                    .content(new Label("The following images are open in a viewer and may have unsaved changes:\n\n  "
                            + names + "\n\nSave them first (Ctrl+S) to include the latest annotations.\nContinue anyway?"))
                    .buttons(ButtonType.YES, ButtonType.NO)
                    .showAndWait()
                    .orElse(ButtonType.NO);
            if (result != ButtonType.YES) return;
        }

        logArea.clear();
        trainButton.setDisable(true);

        Task<Void> task = new Task<>() {
            @Override protected Void call() throws Exception {
                XGBoostTrainer.train(entries, features, classNames, modelPath,
                        numRounds, maxDepth, eta, subsample, topN,
                        msg -> Platform.runLater(() -> {
                            logArea.appendText(msg + "\n");
                            logArea.setScrollTop(Double.MAX_VALUE);
                        }));
                return null;
            }
            @Override protected void failed() {
                Platform.runLater(() -> {
                    logArea.appendText("[ERROR] " + getException().getMessage() + "\n");
                    trainButton.setDisable(false);
                });
            }
            @Override protected void succeeded() {
                Platform.runLater(() -> trainButton.setDisable(false));
            }
        };
        new Thread(task, "xgboost-train") {{ setDaemon(true); }}.start();
    }

    // ── Static helpers ─────────────────────────────────────────────────────────

    private static FilterSide getFilterSide(ToggleGroup group) {
        var sel = group.getSelectedToggle();
        if (sel instanceof ToggleButton btn) {
            return switch (btn.getText()) {
                case "Selected"  -> FilterSide.SELECTED;
                case "Both"      -> FilterSide.BOTH;
                default          -> FilterSide.AVAILABLE;
            };
        }
        return FilterSide.AVAILABLE;
    }

    private void trySetInitialDir(DirectoryChooser dc) {
        var project = qupath.getProject();
        if (project != null && project.getPath() != null) {
            File dir = project.getPath().getParent()
                    .resolve("classifiers").resolve("object_classifiers").toFile();
            if (dir.isDirectory()) dc.setInitialDirectory(dir);
        }
    }

    private static void addParamRow(GridPane g, int row, String label,
                                    Node ctrl, String tip) {
        Label l = new Label(label);
        l.setTooltip(new Tooltip(tip));
        GridPane.setHgrow(ctrl, Priority.ALWAYS);
        g.add(l, 0, row);
        g.add(ctrl, 1, row);
    }

    private static Spinner<Integer> intSpinner(int min, int max, int initial) {
        var s = new Spinner<>(new SpinnerValueFactory.IntegerSpinnerValueFactory(min, max, initial));
        s.setEditable(true); s.setPrefWidth(90); return s;
    }

    private static Spinner<Double> dblSpinner(double min, double max, double init, double step) {
        var s = new Spinner<>(new SpinnerValueFactory.DoubleSpinnerValueFactory(min, max, init, step));
        s.setEditable(true); s.setPrefWidth(90); return s;
    }
}