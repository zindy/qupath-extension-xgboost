package qupath.ext.xgboost;

import javafx.beans.property.BooleanProperty;
import javafx.scene.control.MenuItem;
import javafx.scene.control.SeparatorMenuItem;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import qupath.ext.xgboost.ui.InferController;
import qupath.ext.xgboost.ui.TrainController;
import qupath.fx.dialogs.Dialogs;
import qupath.fx.prefs.controlsfx.PropertyItemBuilder;
import qupath.lib.common.Version;
import qupath.lib.gui.QuPathGUI;
import qupath.lib.gui.extensions.QuPathExtension;
import qupath.lib.gui.prefs.PathPrefs;

import java.util.ResourceBundle;

/**
 * QuPath extension that adds XGBoost-based object classification.
 * <p>
 * Provides two commands under Extensions &gt; XGBoost Classifier:
 * <ul>
 *   <li><b>Train</b> – collect point-annotation training data from selected project entries,
 *       run feature-importance selection, and save a trained XGBoost model.</li>
 *   <li><b>Run Inference</b> – load a saved model and classify detections in selected entries.</li>
 * </ul>
 */
public class XGBoostExtension implements QuPathExtension {

    private static final ResourceBundle resources =
            ResourceBundle.getBundle("qupath.ext.xgboost.ui.strings");
    private static final Logger logger = LoggerFactory.getLogger(XGBoostExtension.class);

    private static final String EXTENSION_NAME        = resources.getString("name");
    private static final String EXTENSION_DESCRIPTION = resources.getString("description");
    private static final Version EXTENSION_QUPATH_VERSION = Version.parse("v0.6.0");

    /** Persistent preference – lets users disable the extension from the Preferences pane. */
    private static final BooleanProperty enableExtensionProperty =
            PathPrefs.createPersistentPreference("xgboost.enabled", true);

    private boolean isInstalled = false;

    // Lazy-created dialog controllers (one per session; refreshed on show())
    private TrainController trainController;
    private InferController inferController;

    // ── QuPathExtension API ────────────────────────────────────────────────────

    @Override
    public void installExtension(QuPathGUI qupath) {
        if (isInstalled) {
            logger.debug("{} is already installed", getName());
            return;
        }
        isInstalled = true;
        addPreferenceToPane(qupath);
        addMenuItems(qupath);
    }

    @Override public String getName()           { return EXTENSION_NAME; }
    @Override public String getDescription()    { return EXTENSION_DESCRIPTION; }
    @Override public Version getQuPathVersion() { return EXTENSION_QUPATH_VERSION; }

    // ── Private helpers ────────────────────────────────────────────────────────

    private void addPreferenceToPane(QuPathGUI qupath) {
        var item = new PropertyItemBuilder<>(enableExtensionProperty, Boolean.class)
                .name(resources.getString("menu.enable"))
                .category(EXTENSION_NAME)
                .description(EXTENSION_DESCRIPTION)
                .build();
        qupath.getPreferencePane().getPropertySheet().getItems().add(item);
    }

    private void addMenuItems(QuPathGUI qupath) {
        var menu = qupath.getMenu("Extensions>" + EXTENSION_NAME, true);

        MenuItem trainItem = new MenuItem(resources.getString("menu.train"));
        trainItem.setOnAction(e -> showTrainDialog(qupath));
        trainItem.disableProperty().bind(enableExtensionProperty.not());

        MenuItem inferItem = new MenuItem(resources.getString("menu.infer"));
        inferItem.setOnAction(e -> showInferDialog(qupath));
        inferItem.disableProperty().bind(enableExtensionProperty.not());

        menu.getItems().addAll(trainItem, new SeparatorMenuItem(), inferItem);
    }

    private void showTrainDialog(QuPathGUI qupath) {
        if (qupath.getProject() == null) {
            Dialogs.showErrorMessage(EXTENSION_NAME, resources.getString("train.no_project"));
            return;
        }
        if (trainController == null) {
            trainController = new TrainController(qupath);
        }
        trainController.show();
    }

    private void showInferDialog(QuPathGUI qupath) {
        if (qupath.getProject() == null) {
            Dialogs.showErrorMessage(EXTENSION_NAME, resources.getString("infer.no_project"));
            return;
        }
        if (inferController == null) {
            inferController = new InferController(qupath);
        }
        inferController.show();
    }
}
