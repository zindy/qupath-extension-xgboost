package qupath.ext.xgboost.ui;

import javafx.application.Platform;
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
import qupath.lib.projects.ProjectImageEntry;

import java.awt.image.BufferedImage;
import java.io.File;
import java.util.List;
import java.util.ResourceBundle;

/**
 * Inference-dialog controller.
 * <p>
 * Builds a JavaFX {@link Stage} that contains:
 * <ul>
 *   <li>A {@link ListSelectionView} to pick project entries</li>
 *   <li>A file-chooser for the saved {@code _xgboost.json} model</li>
 *   <li>A log {@link TextArea} updated live during inference</li>
 *   <li>A <em>Run Inference</em> button that calls {@link XGBoostInferencer} on a background thread</li>
 * </ul>
 */
public class InferController {

    private static final ResourceBundle RES =
            ResourceBundle.getBundle("qupath.ext.xgboost.ui.strings");

    private final QuPathGUI qupath;
    private Stage stage;

    private ListSelectionView<ProjectImageEntry<BufferedImage>> listSelectionView;
    private TextField modelJsonField;
    private TextArea  logArea;
    private Button    runButton;

    public InferController(QuPathGUI qupath) {
        this.qupath = qupath;
    }

    // ── Public API ─────────────────────────────────────────────────────────────

    /** Show (or bring to front) the dialog, refreshing the entry list from the current project. */
    public void show() {
        if (stage == null) {
            stage = buildStage();
        }
        refreshEntries();
        stage.show();
        stage.toFront();
    }

    // ── Stage construction ─────────────────────────────────────────────────────

    @SuppressWarnings("unchecked")
    private Stage buildStage() {
        // ── Entry selector ─────────────────────────────────────────────────────
        listSelectionView = new ListSelectionView<>();
        listSelectionView.setCellFactory(lv -> new ListCell<>() {
            @Override
            protected void updateItem(ProjectImageEntry<BufferedImage> item, boolean empty) {
                super.updateItem(item, empty);
                setText(empty || item == null ? "" : item.getImageName());
            }
        });
        listSelectionView.setPrefHeight(260);

        Label entriesLabel = new Label(RES.getString("infer.entries.label"));
        entriesLabel.setStyle("-fx-font-weight: bold;");

        // ── Model file row ─────────────────────────────────────────────────────
        modelJsonField = new TextField();
        modelJsonField.setPromptText("classifiers/object_classifiers/my_classifier_xgboost.json");
        modelJsonField.setTooltip(new Tooltip(RES.getString("infer.model.tooltip")));
        HBox.setHgrow(modelJsonField, Priority.ALWAYS);

        Button browseButton = new Button(RES.getString("train.browse")); // reuse string
        browseButton.setOnAction(e -> browseForModel());

        HBox modelRow = new HBox(6, modelJsonField, browseButton);
        modelRow.setAlignment(Pos.CENTER_LEFT);

        // ── Model grid ─────────────────────────────────────────────────────────
        GridPane modelGrid = new GridPane();
        modelGrid.setHgap(10);
        modelGrid.setVgap(6);
        modelGrid.setPadding(new Insets(4, 0, 4, 0));
        Label modelLbl = new Label(RES.getString("infer.model.label"));
        modelLbl.setTooltip(new Tooltip(RES.getString("infer.model.tooltip")));
        GridPane.setHgrow(modelRow, Priority.ALWAYS);
        modelGrid.add(modelLbl, 0, 0);
        modelGrid.add(modelRow, 1, 0);

        TitledPane modelPane = new TitledPane("Model", modelGrid);
        modelPane.setCollapsible(false);

        // ── Log area ───────────────────────────────────────────────────────────
        logArea = new TextArea();
        logArea.setEditable(false);
        logArea.setPrefHeight(160);
        logArea.setStyle("-fx-font-family: monospace; -fx-font-size: 11;");
        VBox.setVgrow(logArea, Priority.ALWAYS);

        // ── Run button ─────────────────────────────────────────────────────────
        runButton = new Button(RES.getString("infer.run.button"));
        runButton.setDefaultButton(true);
        runButton.setPrefWidth(140);
        runButton.setOnAction(e -> runInference());

        HBox buttonRow = new HBox(runButton);
        buttonRow.setAlignment(Pos.CENTER_RIGHT);

        // ── Root layout ────────────────────────────────────────────────────────
        VBox root = new VBox(10,
                entriesLabel,
                listSelectionView,
                modelPane,
                new Label("Log:"),
                logArea,
                buttonRow);
        root.setPadding(new Insets(12));

        Stage s = new Stage();
        s.initOwner(qupath.getStage());
        s.initModality(Modality.NONE);
        s.setTitle(RES.getString("infer.stage.title"));
        s.setScene(new Scene(root, 700, 620));
        s.setMinWidth(500);
        s.setMinHeight(460);
        return s;
    }

    // ── Button actions ─────────────────────────────────────────────────────────

    private void browseForModel() {
        FileChooser fc = new FileChooser();
        fc.setTitle(RES.getString("infer.model.label"));
        fc.getExtensionFilters().add(
                new FileChooser.ExtensionFilter("XGBoost model JSON", "*_xgboost.json", "*.json"));
        trySetInitialDir(fc);

        File chosen = fc.showOpenDialog(stage);
        if (chosen != null) {
            modelJsonField.setText(chosen.getAbsolutePath());
        }
    }

    private void runInference() {
        List<ProjectImageEntry<BufferedImage>> selected =
                List.copyOf(listSelectionView.getTargetItems());
        String modelPath = modelJsonField.getText().trim();

        if (selected.isEmpty() || modelPath.isBlank()) {
            logArea.appendText("[ERROR] " + RES.getString("infer.select.error") + "\n");
            return;
        }

        File modelFile = new File(modelPath);
        if (!modelFile.isFile()) {
            logArea.appendText("[ERROR] File not found: " + modelPath + "\n");
            return;
        }

        logArea.clear();
        runButton.setDisable(true);

        Task<Void> task = new Task<>() {
            @Override
            protected Void call() throws Exception {
                XGBoostInferencer.infer(
                        selected, modelFile,
                        msg -> Platform.runLater(() -> {
                            logArea.appendText(msg + "\n");
                            logArea.setScrollTop(Double.MAX_VALUE);
                        }));
                return null;
            }

            @Override
            protected void failed() {
                Throwable ex = getException();
                Platform.runLater(() -> {
                    logArea.appendText("[ERROR] " + ex.getMessage() + "\n");
                    runButton.setDisable(false);
                });
            }

            @Override
            protected void succeeded() {
                Platform.runLater(() -> runButton.setDisable(false));
            }
        };

        Thread thread = new Thread(task, "xgboost-infer");
        thread.setDaemon(true);
        thread.start();
    }

    // ── Helpers ────────────────────────────────────────────────────────────────

    @SuppressWarnings("unchecked")
    private void refreshEntries() {
        listSelectionView.getSourceItems().clear();
        listSelectionView.getTargetItems().clear();
        var project = qupath.getProject();
        if (project != null) {
            listSelectionView.getSourceItems().addAll(
                    (List<ProjectImageEntry<BufferedImage>>) (List<?>) project.getImageList());
        }
    }

    private void trySetInitialDir(FileChooser fc) {
        var project = qupath.getProject();
        if (project != null && project.getPath() != null) {
            File dir = project.getPath().getParent()
                    .resolve("classifiers")
                    .resolve("object_classifiers")
                    .toFile();
            if (dir.isDirectory()) fc.setInitialDirectory(dir);
        }
    }
}
