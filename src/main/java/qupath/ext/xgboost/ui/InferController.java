package qupath.ext.xgboost.ui;

import javafx.application.Platform;
import javafx.collections.ListChangeListener;
import javafx.concurrent.Task;
import javafx.geometry.Insets;
import javafx.geometry.Pos;
import javafx.scene.Scene;
import javafx.scene.control.*;
import javafx.scene.layout.*;
import javafx.stage.FileChooser;
import javafx.stage.Modality;
import javafx.stage.Stage;
import org.controlsfx.control.ListSelectionView;
import qupath.ext.xgboost.XGBoostInferencer;
import qupath.lib.gui.QuPathGUI;
import qupath.lib.gui.panes.ProjectEntryPredicate;
import qupath.lib.projects.ProjectImageEntry;

import java.awt.image.BufferedImage;
import java.io.File;
import java.util.*;
import java.util.stream.Collectors;

/**
 * Inference-dialog controller.
 */
public class InferController {

    private enum FilterSide { AVAILABLE, SELECTED, BOTH }

    private final QuPathGUI qupath;
    private Stage stage;

    // ── Master + backing lists ─────────────────────────────────────────────────
    private List<ProjectImageEntry<BufferedImage>> allImages      = new ArrayList<>();
    private final List<ProjectImageEntry<BufferedImage>> selectedEntries = new ArrayList<>();
    private boolean updatingEntries = false;

    // ── UI ─────────────────────────────────────────────────────────────────────
    private ListSelectionView<ProjectImageEntry<BufferedImage>> entrySelector;
    private TextField  entryFilterField;
    private CheckBox   withDataOnlyBox;
    private ToggleGroup entryFilterGroup;

    private TextField modelJsonField;
    private TextArea  logArea;
    private Button    runButton;

    public InferController(QuPathGUI qupath) {
        this.qupath = qupath;
    }

    // ── Public API ─────────────────────────────────────────────────────────────

    public void show() {
        if (stage == null) stage = buildStage();
        refreshEntries();
        stage.show();
        stage.toFront();
    }

    // ── Stage construction ─────────────────────────────────────────────────────

    private Stage buildStage() {

        // ══ 1. PROJECT ENTRIES ════════════════════════════════════════════════
        entrySelector = new ListSelectionView<>();
        entrySelector.setCellFactory(lv -> new ListCell<>() {
            @Override protected void updateItem(ProjectImageEntry<BufferedImage> item, boolean empty) {
                super.updateItem(item, empty);
                setText(empty || item == null ? "" : item.getImageName());
            }
        });
        entrySelector.setPrefHeight(240);

        // Keep backing list in sync when the user moves items
        entrySelector.getTargetItems().addListener((ListChangeListener<ProjectImageEntry<BufferedImage>>) change -> {
            if (updatingEntries) return;
            updatingEntries = true;
            try {
                while (change.next()) {
                    if (change.wasAdded())   selectedEntries.addAll(change.getAddedSubList());
                    if (change.wasRemoved()) selectedEntries.removeAll(change.getRemoved());
                }
            } finally {
                updatingEntries = false;
            }
            Platform.runLater(this::updateEntryFilter);
        });

        // Filter row
        entryFilterGroup = new ToggleGroup();
        entryFilterField = new TextField();
        entryFilterField.setPromptText("Filter…");
        entryFilterField.textProperty().addListener((o, a, b) -> updateEntryFilter());
        withDataOnlyBox = new CheckBox("With data file only");
        withDataOnlyBox.selectedProperty().addListener((o, a, b) -> updateEntryFilter());
        entryFilterGroup.selectedToggleProperty().addListener((o, a, b) -> updateEntryFilter());

        HBox filterRow = filterRow(entryFilterGroup, entryFilterField, withDataOnlyBox);

        VBox entryContent = new VBox(6, entrySelector, filterRow);
        entryContent.setPadding(new Insets(6));
        TitledPane entryPane = new TitledPane("Project Entries", entryContent);
        entryPane.setCollapsible(false);

        // ══ 2. MODEL ══════════════════════════════════════════════════════════
        modelJsonField = new TextField();
        modelJsonField.setPromptText("my_xgboost.json");
        modelJsonField.setTooltip(new Tooltip("Saved XGBoost model JSON produced by the Train dialog"));
        HBox.setHgrow(modelJsonField, Priority.ALWAYS);

        Button browseButton = new Button("Browse…");
        browseButton.setOnAction(e -> browseForModel());

        HBox modelRow = new HBox(6, modelJsonField, browseButton);
        modelRow.setAlignment(Pos.CENTER_LEFT);

        GridPane modelGrid = new GridPane();
        modelGrid.setHgap(10);
        modelGrid.setVgap(6);
        modelGrid.setPadding(new Insets(6));
        Label modelLbl = new Label("Model:");
        modelLbl.setTooltip(new Tooltip("Saved XGBoost model JSON"));
        GridPane.setHgrow(modelRow, Priority.ALWAYS);
        modelGrid.add(modelLbl, 0, 0);
        modelGrid.add(modelRow, 1, 0);

        TitledPane modelPane = new TitledPane("Model", modelGrid);
        modelPane.setCollapsible(false);
        modelPane.setAnimated(false);

        // ══ 3. LOG ════════════════════════════════════════════════════════════
        logArea = new TextArea();
        logArea.setEditable(false);
        logArea.setPrefHeight(160);
        logArea.setStyle("-fx-font-family: monospace; -fx-font-size: 11;");
        VBox.setVgrow(logArea, Priority.ALWAYS);

        // ══ 4. RUN BUTTON ═════════════════════════════════════════════════════
        runButton = new Button("Run Inference");
        runButton.setDefaultButton(true);
        runButton.setPrefWidth(140);
        runButton.setOnAction(e -> runInference());

        HBox btnRow = new HBox(runButton);
        btnRow.setAlignment(Pos.CENTER_RIGHT);

        // ══ Root ══════════════════════════════════════════════════════════════
        VBox root = new VBox(8, entryPane, modelPane, new Label("Log:"), logArea, btnRow);
        root.setPadding(new Insets(10));

        Stage s = new Stage();
        s.initOwner(qupath.getStage());
        s.initModality(Modality.NONE);
        s.setTitle("Run XGBoost Classifier");
        s.setScene(new Scene(root, 720, 640));
        s.setMinWidth(500);
        s.setMinHeight(440);
        return s;
    }

    // ── Filter-row builder (shared pattern with TrainController) ──────────────

    private static HBox filterRow(ToggleGroup group, TextField filterField,
                                   javafx.scene.Node... extras) {
        ToggleButton btnAvail = filterToggle("Available", group, true);
        ToggleButton btnSel   = filterToggle("Selected",  group, false);
        ToggleButton btnBoth  = filterToggle("Both",      group, false);
        HBox.setHgrow(filterField, Priority.ALWAYS);
        HBox row = new HBox(4, btnAvail, btnSel, btnBoth, filterField);
        for (var n : extras) row.getChildren().add(n);
        row.setAlignment(Pos.CENTER_LEFT);
        row.setPadding(new Insets(4, 0, 0, 0));
        return row;
    }

    private static ToggleButton filterToggle(String text, ToggleGroup group, boolean selected) {
        ToggleButton btn = new ToggleButton(text);
        btn.setToggleGroup(group);
        btn.setSelected(selected);
        btn.setPrefHeight(24);
        // Prevent deselecting all buttons
        btn.addEventFilter(javafx.scene.input.MouseEvent.MOUSE_RELEASED, e -> {
            if (btn.isSelected()) e.consume();
        });
        return btn;
    }

    // ── Data refresh ───────────────────────────────────────────────────────────

    @SuppressWarnings("unchecked")
    private void refreshEntries() {
        var project = qupath.getProject();
        allImages = project == null ? List.of()
                : (List<ProjectImageEntry<BufferedImage>>) (List<?>) project.getImageList();
        selectedEntries.retainAll(allImages);
        updateEntryFilter();
    }

    // ── Filter update ──────────────────────────────────────────────────────────

    private void updateEntryFilter() {
        if (updatingEntries) return;
        String     text      = entryFilterField == null ? "" : entryFilterField.getText();
        boolean    withData  = withDataOnlyBox  != null && withDataOnlyBox.isSelected();
        FilterSide side      = getFilterSide(entryFilterGroup);

        updatingEntries = true;
        try {
            var sourcePredicate = ProjectEntryPredicate.createIgnoreCase(
                    side != FilterSide.SELECTED ? text : "");
            var targetPredicate = ProjectEntryPredicate.createIgnoreCase(
                    side != FilterSide.AVAILABLE ? text : "");

            List<ProjectImageEntry<BufferedImage>> source = allImages.stream()
                    .filter(p -> !selectedEntries.contains(p))
                    .filter(p -> !withData || p.hasImageData())
                    .filter(sourcePredicate)
                    .collect(Collectors.toList());

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

    // ── Inference ──────────────────────────────────────────────────────────────

    private void runInference() {
        // Use backing list so hidden-but-selected entries are included
        if (selectedEntries.isEmpty()) {
            logArea.appendText("[ERROR] Select at least one project entry.\n");
            return;
        }
        String modelPath = modelJsonField.getText().trim();
        if (modelPath.isBlank()) {
            logArea.appendText("[ERROR] Select a model JSON file.\n");
            return;
        }
        File modelFile = new File(modelPath);
        if (!modelFile.isFile()) {
            logArea.appendText("[ERROR] File not found: " + modelPath + "\n");
            return;
        }

        List<ProjectImageEntry<BufferedImage>> entries = List.copyOf(selectedEntries);

        logArea.clear();
        runButton.setDisable(true);

        Task<Void> task = new Task<>() {
            @Override protected Void call() throws Exception {
                XGBoostInferencer.infer(entries, modelFile,
                        msg -> Platform.runLater(() -> {
                            logArea.appendText(msg + "\n");
                            logArea.setScrollTop(Double.MAX_VALUE);
                        }));
                return null;
            }
            @Override protected void failed() {
                Platform.runLater(() -> {
                    logArea.appendText("[ERROR] " + getException().getMessage() + "\n");
                    runButton.setDisable(false);
                });
            }
            @Override protected void succeeded() {
                Platform.runLater(() -> runButton.setDisable(false));
            }
        };
        new Thread(task, "xgboost-infer") {{ setDaemon(true); }}.start();
    }

    // ── Helpers ────────────────────────────────────────────────────────────────

    private void browseForModel() {
        FileChooser fc = new FileChooser();
        fc.setTitle("Select XGBoost model JSON");
        fc.getExtensionFilters().add(
                new FileChooser.ExtensionFilter("XGBoost model JSON", "*.json"));
        var project = qupath.getProject();
        if (project != null && project.getPath() != null) {
            File dir = project.getPath().getParent()
                    .resolve("classifiers").resolve("object_classifiers").toFile();
            if (dir.isDirectory()) fc.setInitialDirectory(dir);
        }
        File chosen = fc.showOpenDialog(stage);
        if (chosen != null) modelJsonField.setText(chosen.getAbsolutePath());
    }

    private static FilterSide getFilterSide(ToggleGroup group) {
        if (group == null) return FilterSide.AVAILABLE;
        var sel = group.getSelectedToggle();
        if (sel instanceof ToggleButton btn) {
            return switch (btn.getText()) {
                case "Selected" -> FilterSide.SELECTED;
                case "Both"     -> FilterSide.BOTH;
                default         -> FilterSide.AVAILABLE;
            };
        }
        return FilterSide.AVAILABLE;
    }
}