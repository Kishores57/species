/**
 * TreeVisionAI — Leaf Detection & Habitat Explorer Client Logic
 */

document.addEventListener("DOMContentLoaded", () => {
  // DOM Elements
  const dropzone = document.getElementById("dropzone");
  const fileInput = document.getElementById("file-input");
  const browseBtn = document.getElementById("browse-btn");
  const previewWrapper = document.getElementById("preview-wrapper");
  const previewImg = document.getElementById("preview-img");
  const resetBtn = document.getElementById("reset-btn");
  const detectBtn = document.getElementById("detect-btn");
  const sampleChips = document.querySelectorAll(".sample-chip");

  const placeholderView = document.getElementById("placeholder-view");
  const loadingBox = document.getElementById("loading-box");
  const resultsView = document.getElementById("results-view");

  // Result Elements
  const resDisplayName = document.getElementById("res-display-name");
  const resScientificName = document.getElementById("res-scientific-name");
  const resConfidence = document.getElementById("res-confidence");
  const resFamily = document.getElementById("res-family");
  const resStatus = document.getElementById("res-status");
  const resFoundIn = document.getElementById("res-found-in");
  const resHabitats = document.getElementById("res-habitats");
  const resLeafFeatures = document.getElementById("res-leaf-features");
  const resEcological = document.getElementById("res-ecological");
  const exifLocationContainer = document.getElementById("exif-location-container");
  const resGpsLink = document.getElementById("res-gps-link");
  const probabilitiesContainer = document.getElementById("probabilities-container");

  let currentFile = null;

  // File selection via button
  browseBtn.addEventListener("click", (e) => {
    e.stopPropagation();
    fileInput.click();
  });

  dropzone.addEventListener("click", () => {
    fileInput.click();
  });

  // Drag and Drop Handling
  ["dragenter", "dragover"].forEach((eventName) => {
    dropzone.addEventListener(eventName, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropzone.classList.add("drag-active");
    });
  });

  ["dragleave", "drop"].forEach((eventName) => {
    dropzone.addEventListener(eventName, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropzone.classList.remove("drag-active");
    });
  });

  dropzone.addEventListener("drop", (e) => {
    const dt = e.dataTransfer;
    const files = dt.files;
    if (files && files.length > 0) {
      handleImageFile(files[0]);
    }
  });

  fileInput.addEventListener("change", (e) => {
    if (fileInput.files && fileInput.files.length > 0) {
      handleImageFile(fileInput.files[0]);
    }
  });

  function handleImageFile(file) {
    if (!file.type.startsWith("image/")) {
      alert("Please upload an image file (.jpg, .jpeg, .png, or .webp).");
      return;
    }
    currentFile = file;

    const reader = new FileReader();
    reader.onload = (e) => {
      previewImg.src = e.target.result;
      dropzone.style.display = "none";
      previewWrapper.classList.add("active");
      detectBtn.disabled = false;
    };
    reader.readAsDataURL(file);
  }

  // Reset button
  resetBtn.addEventListener("click", (e) => {
    e.stopPropagation();
    resetState();
  });

  function resetState() {
    currentFile = null;
    fileInput.value = "";
    previewImg.src = "";
    previewWrapper.classList.remove("active");
    dropzone.style.display = "block";
    detectBtn.disabled = true;

    // Reset results view
    resultsView.classList.remove("active");
    loadingBox.classList.remove("active");
    placeholderView.style.display = "block";
  }

  // Quick Sample Chips
  sampleChips.forEach((chip) => {
    chip.addEventListener("click", async () => {
      const src = chip.getAttribute("data-src");
      const species = chip.getAttribute("data-species");

      try {
        const response = await fetch(src);
        const blob = await response.blob();
        const file = new File([blob], `${species}_sample.jpg`, { type: "image/jpeg" });

        handleImageFile(file);
        // Automatically trigger detection for quick demo experience
        runDetection();
      } catch (err) {
        console.error("Could not load sample image:", err);
      }
    });
  });

  // Run Detection
  detectBtn.addEventListener("click", runDetection);

  async function runDetection() {
    if (!currentFile) return;

    placeholderView.style.display = "none";
    resultsView.classList.remove("active");
    loadingBox.classList.add("active");
    detectBtn.disabled = true;

    const formData = new FormData();
    formData.append("file", currentFile);

    try {
      const res = await fetch("/predict", {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        throw new Error(errData.detail || `Server error: ${res.status}`);
      }

      const data = await res.json();
      displayResults(data);
    } catch (error) {
      alert(`Detection failed: ${error.message}`);
      placeholderView.style.display = "block";
    } finally {
      loadingBox.classList.remove("active");
      detectBtn.disabled = false;
    }
  }

  function displayResults(data) {
    // Top prediction info
    resDisplayName.textContent = data.prediction || "Unknown";
    resScientificName.textContent = data.scientific_name || "";
    resConfidence.textContent = data.confidence || "0%";
    resFamily.textContent = data.family || "N/A";
    resStatus.textContent = data.conservation_status || "N/A";

    // Where it was found & Habitat
    resFoundIn.textContent = data.found_in || "No origin data available.";
    resHabitats.textContent = data.common_habitats || "No habitat data available.";

    // EXIF Location if available in uploaded image
    if (data.photo_gps && data.photo_gps.maps_url) {
      exifLocationContainer.style.display = "block";
      resGpsLink.href = data.photo_gps.maps_url;
      resGpsLink.innerHTML = `<span>🛰️ Photo Captured at: ${data.photo_gps.latitude}°N, ${data.photo_gps.longitude}°E (Open in Google Maps)</span>`;
    } else {
      exifLocationContainer.style.display = "none";
    }

    // Botanical & Ecological Details
    resLeafFeatures.textContent = data.leaf_characteristics || "N/A";
    resEcological.textContent = data.ecological_importance || "N/A";

    // Softmax probabilities breakdown
    probabilitiesContainer.innerHTML = "";
    if (data.top_predictions && Array.isArray(data.top_predictions)) {
      data.top_predictions.forEach((item) => {
        const row = document.createElement("div");
        row.className = "prob-row";

        row.innerHTML = `
          <div class="prob-header">
            <span class="prob-name">${item.display_name} <em>(${item.scientific_name})</em></span>
            <span class="prob-score">${item.confidence}%</span>
          </div>
          <div class="prob-bar-track">
            <div class="prob-bar-fill" style="width: 0%;"></div>
          </div>
        `;

        probabilitiesContainer.appendChild(row);

        // Animate fill bar on next animation frame
        requestAnimationFrame(() => {
          setTimeout(() => {
            const fill = row.querySelector(".prob-bar-fill");
            if (fill) fill.style.width = `${item.confidence}%`;
          }, 60);
        });
      });
    }

    resultsView.classList.add("active");
  }
});
