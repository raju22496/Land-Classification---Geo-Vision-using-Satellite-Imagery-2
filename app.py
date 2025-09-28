import os
import cv2
import torch
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from werkzeug.utils import secure_filename
import pandas as pd
import pytorch_lightning as pl
from torch import nn
import requests


# Define the SegmentationModel class
class SegmentationModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        # Note: The actual model implementation details will be loaded from the saved model
        # This class definition is needed just to load the model

    def forward(self, inputs, targets=None):
        outputs = self.model(inputs)
        if targets is not None:
            loss = self.criterion(outputs, targets)
            # Additional metric calculations would be here in the original model
            return loss, None, outputs
        else:
            return outputs


# Load your model and necessary components
model_path = "model.pt"
color_dict_path = "class_dict.csv"

# Load class dictionary
color_dict = pd.read_csv(color_dict_path)
CLASSES = color_dict["name"].tolist()

# Setup Flask app
app = Flask(__name__)
app.secret_key = "land_classification_secret_key"
app.config["UPLOAD_FOLDER"] = "static/uploads"
app.config["RESULT_FOLDER"] = "static/results"
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB max upload size
app.config["OPENSTREETMAP_API_ENDPOINT"] = "https://nominatim.openstreetmap.org/search"
app.config["OVERPASS_API_ENDPOINT"] = "https://overpass-api.de/api/interpreter"

# Create needed directories
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
os.makedirs(app.config["RESULT_FOLDER"], exist_ok=True)


# Image preprocessing functions
def preprocess_image(image_path, target_size=320):
    # Read and resize image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (target_size, target_size))

    # Convert to tensor and normalize
    image_tensor = torch.from_numpy(np.transpose(image, (2, 0, 1))).float() / 255.0
    image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension

    return image_tensor, image


# Convert category to RGB for visualization
def category2rgb(category_mask):
    rgb_mask = np.zeros(category_mask.shape[:2] + (3,))
    for i, row in color_dict.iterrows():
        rgb_mask[category_mask == i] = (row["r"], row["g"], row["b"])
    return np.uint8(rgb_mask)


# Load model
def load_model():
    try:
        model = torch.load(model_path, map_location=torch.device("cpu"))
        model.eval()
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None


# Create global model variable
segmentation_model = load_model()


# Function to get coordinates from a location search
def get_coordinates(location_name):
    params = {
        "q": location_name,
        "format": "json",
        "limit": 1,
    }

    headers = {"User-Agent": "LandClassificationApp/1.0"}

    try:
        response = requests.get(
            app.config["OPENSTREETMAP_API_ENDPOINT"], params=params, headers=headers
        )
        response.raise_for_status()
        data = response.json()

        if data and len(data) > 0:
            return {
                "lat": float(data[0]["lat"]),
                "lon": float(data[0]["lon"]),
                "display_name": data[0]["display_name"],
            }
        else:
            return None
    except Exception as e:
        print(f"Error fetching coordinates: {e}")
        return None


# Function to get nearby points of interest
def get_nearby_pois(lat, lon, radius=1000):
    """
    Get nearby points of interest using Overpass API

    :param lat: Latitude
    :param lon: Longitude
    :param radius: Search radius in meters
    :return: Dictionary of POIs by category
    """
    # Overpass query to find amenities within the radius
    query = f"""
    [out:json];
    (
      node["amenity"~"school|hospital|restaurant|parking|pharmacy|bank|fuel|police|fire_station"](around:{radius},{lat},{lon});
      way["amenity"~"school|hospital|restaurant|parking|pharmacy|bank|fuel|police|fire_station"](around:{radius},{lat},{lon});
      relation["amenity"~"school|hospital|restaurant|parking|pharmacy|bank|fuel|police|fire_station"](around:{radius},{lat},{lon});
      
      node["shop"](around:{radius},{lat},{lon});
      way["shop"](around:{radius},{lat},{lon});
      
      node["leisure"~"park|playground|sports_centre"](around:{radius},{lat},{lon});
      way["leisure"~"park|playground|sports_centre"](around:{radius},{lat},{lon});
      
      node["tourism"~"hotel|museum|attraction"](around:{radius},{lat},{lon});
      way["tourism"~"hotel|museum|attraction"](around:{radius},{lat},{lon});
    );
    out center;
    """

    try:
        response = requests.post(
            app.config["OVERPASS_API_ENDPOINT"],
            data={"data": query},
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        response.raise_for_status()
        data = response.json()

        # Process and categorize results
        pois = {
            "education": [],
            "healthcare": [],
            "food": [],
            "shopping": [],
            "transportation": [],
            "recreation": [],
            "other": [],
        }

        for element in data.get("elements", []):
            if "tags" in element:
                name = element["tags"].get("name", "Unnamed")
                poi_type = element["tags"].get(
                    "amenity",
                    element["tags"].get(
                        "shop",
                        element["tags"].get(
                            "leisure", element["tags"].get("tourism", "other")
                        ),
                    ),
                )

                # Get coordinates for the POI
                if element["type"] == "node":
                    poi_lat = element["lat"]
                    poi_lon = element["lon"]
                else:  # way or relation with center coordinates
                    poi_lat = element.get("center", {}).get("lat", lat)
                    poi_lon = element.get("center", {}).get("lon", lon)

                # Determine category
                category = "other"
                if poi_type in [
                    "school",
                    "university",
                    "college",
                    "kindergarten",
                    "library",
                ]:
                    category = "education"
                elif poi_type in [
                    "hospital",
                    "clinic",
                    "doctors",
                    "pharmacy",
                    "dentist",
                ]:
                    category = "healthcare"
                elif poi_type in ["restaurant", "cafe", "bar", "fast_food", "pub"]:
                    category = "food"
                elif "shop" in element["tags"]:
                    category = "shopping"
                elif poi_type in ["parking", "fuel", "bus_station", "train_station"]:
                    category = "transportation"
                elif poi_type in [
                    "park",
                    "playground",
                    "sports_centre",
                    "museum",
                    "attraction",
                ]:
                    category = "recreation"

                # Add to appropriate category
                pois[category].append(
                    {
                        "name": name,
                        "type": poi_type,
                        "lat": poi_lat,
                        "lon": poi_lon,
                        "distance": calculate_distance(lat, lon, poi_lat, poi_lon),
                    }
                )

        # Sort each category by distance
        for category in pois:
            pois[category] = sorted(pois[category], key=lambda x: x["distance"])

        return pois

    except Exception as e:
        print(f"Error fetching nearby POIs: {e}")
        return None


# Calculate distance between two coordinate points
def calculate_distance(lat1, lon1, lat2, lon2):
    """
    Calculate distance between two points in kilometers using the Haversine formula
    """
    from math import radians, sin, cos, sqrt, atan2

    # Convert to radians
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])

    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    r = 6371  # Radius of earth in kilometers

    return round(r * c * 1000)  # Distance in meters


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/documentation")
def documentation():
    return render_template("documentation.html", classes=CLASSES)


@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        # Check if a file was uploaded
        if "file" not in request.files:
            flash("No file part")
            return redirect(request.url)

        file = request.files["file"]

        # If user submits empty form
        if file.filename == "":
            flash("No selected file")
            return redirect(request.url)

        # If file looks good, process it
        if file:
            # Save the uploaded file
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(file_path)

            try:
                # Preprocess image
                image_tensor, original_image = preprocess_image(file_path)

                # Make prediction
                with torch.no_grad():
                    output = segmentation_model(image_tensor)

                # Get predicted class
                pred_mask = output.argmax(dim=1).squeeze().cpu().numpy()

                # Convert to RGB for visualization
                colored_mask = category2rgb(pred_mask)

                # Create overlay image (50% original, 50% prediction)
                resized_original = cv2.resize(
                    original_image, (pred_mask.shape[0], pred_mask.shape[1])
                )
                overlay = cv2.addWeighted(resized_original, 0.5, colored_mask, 0.5, 0)

                # Save results
                result_filename = "result_" + filename
                overlay_filename = "overlay_" + filename
                mask_path = os.path.join(app.config["RESULT_FOLDER"], result_filename)
                overlay_path = os.path.join(
                    app.config["RESULT_FOLDER"], overlay_filename
                )

                cv2.imwrite(mask_path, cv2.cvtColor(colored_mask, cv2.COLOR_RGB2BGR))
                cv2.imwrite(overlay_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

                # Calculate land cover statistics
                total_pixels = pred_mask.size
                class_stats = {}

                for i, class_name in enumerate(CLASSES):
                    class_pixels = np.sum(pred_mask == i)
                    percentage = (class_pixels / total_pixels) * 100
                    class_stats[class_name] = {
                        "pixels": int(class_pixels),
                        "percentage": round(percentage, 2),
                    }

                return render_template(
                    "result.html",
                    original_image=os.path.join("uploads", filename),
                    mask_image=os.path.join("results", result_filename),
                    overlay_image=os.path.join("results", overlay_filename),
                    class_stats=class_stats,
                )

            except Exception as e:
                flash(f"Error processing image: {str(e)}")
                return redirect(request.url)

    return render_template("predict.html")


@app.route("/batch-predict")
def batch_predict():
    return render_template("batch.html")


@app.route("/nearby", methods=["GET", "POST"])
def nearby():
    if request.method == "POST":
        location = request.form.get("location")
        radius = int(request.form.get("radius", 1000))

        if not location:
            flash("Please enter a location")
            return redirect(request.url)

        # Get coordinates from location
        coordinates = get_coordinates(location)

        if not coordinates:
            flash("Location not found. Please try another location.")
            return redirect(request.url)

        # Get nearby points of interest
        pois = get_nearby_pois(coordinates["lat"], coordinates["lon"], radius)

        if not pois:
            flash("Error fetching nearby points of interest.")
            return redirect(request.url)

        return render_template(
            "nearby_results.html",
            location=coordinates["display_name"],
            radius=radius,
            pois=pois,
            coordinates=coordinates,
        )

    return render_template("nearby.html")


@app.route("/api/predict", methods=["POST"])
def api_predict():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    try:
        # Save the uploaded file
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(file_path)

        # Preprocess image
        image_tensor, _ = preprocess_image(file_path)

        # Make prediction
        with torch.no_grad():
            output = segmentation_model(image_tensor)

        # Get predicted class
        pred_mask = output.argmax(dim=1).squeeze().cpu().numpy()

        # Calculate land cover statistics
        total_pixels = pred_mask.size
        class_stats = {}

        for i, class_name in enumerate(CLASSES):
            class_pixels = np.sum(pred_mask == i)
            percentage = (class_pixels / total_pixels) * 100
            class_stats[class_name] = {
                "pixels": int(class_pixels),
                "percentage": round(percentage, 2),
            }

        return jsonify(
            {"success": True, "filename": filename, "statistics": class_stats}
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/nearby", methods=["POST"])
def api_nearby():
    try:
        data = request.json

        if not data:
            return jsonify({"error": "No JSON data provided"}), 400

        location = data.get("location")
        radius = int(data.get("radius", 1000))

        if not location:
            return jsonify({"error": "Location is required"}), 400

        # Get coordinates from location
        coordinates = get_coordinates(location)

        if not coordinates:
            return jsonify({"error": "Location not found"}), 404

        # Get nearby points of interest
        pois = get_nearby_pois(coordinates["lat"], coordinates["lon"], radius)

        if not pois:
            return jsonify({"error": "Error fetching nearby points of interest"}), 500

        return jsonify(
            {
                "success": True,
                "location": coordinates["display_name"],
                "coordinates": {"lat": coordinates["lat"], "lon": coordinates["lon"]},
                "pois": pois,
            }
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
