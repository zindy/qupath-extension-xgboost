plugins {
    // To optionally create a shadow/fat jar that bundle up any non-core dependencies
    id("com.gradleup.shadow") version "8.3.5"
    // QuPath Gradle extension convention plugin
    id("qupath-conventions")
}


qupathExtension {
    name = "qupath-extension-xgboost"
    group = "io.github.qupath"
    version = "0.1.3"
    description = "XGBoost object classifier extension for QuPath"
    automaticModule = "io.github.qupath.extension.xgboost"
}

dependencies {

    // Main dependencies for most QuPath extensions
    shadow(libs.bundles.qupath)
    shadow(libs.bundles.logging)
    shadow(libs.qupath.fxtras)

    // Available at runtime for both composite-build AND shadow jar
    implementation("ml.dmlc:xgboost4j_2.13:3.2.0")
    shadow("ml.dmlc:xgboost4j_2.13:3.2.0")

    // For testing
    testImplementation(libs.bundles.qupath)
    testImplementation(libs.junit)

}

tasks.jar {
    manifest {
        attributes(
            "Implementation-Version" to project.version
        )
    }
}
