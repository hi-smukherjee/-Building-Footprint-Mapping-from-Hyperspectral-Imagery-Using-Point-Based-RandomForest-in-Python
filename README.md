🏙️ Building Footprint Mapping from Hyperspectral Imagery Using Point-Based Random Forest Classification in Python

📄 Overview:
  This project presents a robust and efficient approach to building footprint extraction using high-resolution hyperspectral imagery and a point-based Random Forest classification method implemented entirely in Python. The workflow leverages the NEON Surface Directional Reflectance product (~1 m spatial resolution), combining rich spectral information with machine learning to distinguish built-up features with high precision.
  Training and non-training points are manually digitized in QGIS and used to train a Random Forest (RF) model. The approach includes preprocessing, feature extraction, model training, classification, and post-processing, resulting in both raster and vector outputs.

📊 Project Specifications:

| **Attribute**            | **Details**                                                                               |
| ------------------------ | ----------------------------------------------------------------------------------------- |
| **Title**                | Building Footprint Mapping from Hyperspectral Imagery Using Point-Based RF Classification |
| **Imagery Source**       | NEON Surface Directional Reflectance (Hyperspectral)                                      |
| **Platform**             | NEON AOP (Airborne Observation Platform)                                                  |
| **Spatial Resolution**   | \~1 meter                                                                                 |
| **Year of Imagery**      | 2021                                                                                      |
| **Sample Type**          | Point-based (built-up and non-built-up), digitized in QGIS                                |
| **Classification Model** | Random Forest (Scikit-learn)                                                              |
| **Programming Language** | Python                                                                                    |
| **Output Formats**       | GeoTIFF (classified map), Shapefile (vectorized footprints), Pickle (trained model)       |
| **Validation Metrics**   | R² Score, RMSE, Confusion Matrix, Accuracy, Precision, Recall, F1-Score, IoU (optional)   |

⚙️ Dependencies: Install required packages via pip or conda: pip install geopandas rasterio scikit-learn numpy matplotlib shapely joblib
| **Library**           | **Purpose**                                                                |
| --------------------- | -------------------------------------------------------------------------- |
| `geopandas`           | Handling shapefiles (training points, output vectors)                      |
| `rasterio`            | Reading/writing raster datasets (hyperspectral images)                     |
| `scikit-learn`        | Model training, Random Forest classifier, and evaluation metrics           |
| `numpy`               | Array operations and numerical processing                                  |
| `matplotlib`          | Visualization of classification and metrics                                |
| `shapely`             | Geometry manipulation                                                      |
| `joblib`              | Saving and loading trained model files                                     |
| `momepy` *(optional)* | Shape regularization and polygon simplification (e.g., rectangularization) |

🚀 Workflow:
graph TD
    A[🎯 Define Objective] --> B[🛰️ Load Hyperspectral Image]
    B --> B1[📉 Select Top 10 Reflective Bands]
    B1 --> C[🧭 Digitize Training Points (Built-up/Non-built-up) in QGIS]
    C --> D[💾 Export Points to CSV with Labels]
    D --> E[📥 Extract Pixel Values from Selected Bands]
    E --> F[🧠 Train Random Forest Model]
    F --> G[🗺️ Predict Building Footprints Across Scene]
    G --> H[🧹 Post-process: Filtering & Shape Refinement]
    H --> I[📤 Export GeoTIFF and Shapefile Outputs]
    I --> J[📊 Evaluate Results Using Metrics]
    
📌 Model Parameters:
| Parameter      | Value | Description                                   |
| -------------- | ----- | --------------------------------------------- |
| `n_estimators` | 500   | Number of trees in the forest                 |
| `random_state` | 42    | Ensures reproducibility                       |
| `n_jobs`       | -1    | Utilizes all CPU cores for faster computation |

📈 Evaluation Metrics:
| Metric       | Value  | Description                                                                         |
| ------------ | ------ | ----------------------------------------------------------------------------------- |
| **R² Score** | 0.9938 | Indicates how well the model explains variance in the data (closer to 1 is better). |
| **RMSE**     | 0.0394 | Root Mean Squared Error — measures average magnitude of prediction errors.          |

📦 Final Outputs:
| Output Type           | Format           | Description                                                                             |
| --------------------- | ---------------- | --------------------------------------------------------------------------------------- |
| **Classified Map**    | `GeoTIFF`        | Full-scene raster with binary building classification (1 = Building, 0 = Non-Building). |
| **Vector Footprints** | `Shapefile`      | Polygonized building footprints extracted from post-processed classification.           |
| **Trained Model**     | `.pkl`           | Serialized Random Forest model for reuse or inference.                                  |

❓ Why Use Random Forest and Hyperspectral Data?
  🌈 Hyperspectral Data (NEON S2 Surface Reflectance)
      Contains hundreds of narrow spectral bands, capturing detailed surface characteristics.
      Enables fine discrimination between land cover types—especially helpful for differentiating buildings vs. vegetation or bare soil.
      High spatial resolution (~1 m) allows for precise boundary detection of urban structures.

  🌲 Random Forest (RF) Classifier
      A robust, ensemble-based machine learning algorithm.
      Handles high-dimensional data well (e.g., 100+ hyperspectral bands).
      Performs feature selection implicitly—automatically focuses on the most relevant spectral bands.
      Works well with small training samples (e.g., point-based CSVs), unlike deep learning which requires large datasets.
      Offers high accuracy, fast training, and built-in metrics like feature importance.

  🧠 Combined Strength
      When hyperspectral richness is fused with RF’s interpretability and power, the result is a scalable, accurate, and computationally
      efficient solution for building footprint mapping from aerial/satellite data.

🌐 Data Source & Download
  🛰️ Hyperspectral Data:
      Platform: Google Earth Engine (GEE)
      Dataset Used: NEON Surface Reflectance (e.g., NEON_D_HYPERSPECTRAL)
      Resolution: ~1 meter (varies by site)

📌 Conclusion:
This project demonstrates the potential of using Random Forest (RF) classification on high-resolution hyperspectral imagery to accurately map built-up areas. By leveraging:
  Carefully selected top 10 bands with high reflectance in urban features, Labeled point samples (built-up vs. non-built-up), A 500-tree Random Forest model, and Robust post-processing techniques, we achieved high classification accuracy with: ✅ R² Score: 0.9938 ✅ RMSE: 0.0394

The final output includes a classified GeoTIFF and a clean shapefile of predicted building footprints, which can be used for urban planning, change detection, and geospatial analysis.
