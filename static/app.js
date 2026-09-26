// Crop_AI — Application Client Script
// Clean on-page rendering for the 4-stage AI pipeline:
// 1. Crop Verification
// 2. Disease Classification
// 3. Potassium Deficiency
// 4. Plant / Weed Localization

document.addEventListener("DOMContentLoaded", () => {
    const imageInput = document.getElementById("imageInput");
    const fileNameDisplay = document.getElementById("fileName");
    const previewContainer = document.getElementById("previewContainer");
    const imagePreview = document.getElementById("imagePreview");
    const scanBtn = document.getElementById("scanBtn");
    const statusMessage = document.getElementById("statusMessage");

    const resultCard = document.getElementById("resultCard");

    // 1. Crop Verification
    const cropStatusVal = document.getElementById("cropStatusVal");
    const cropVerifiedVal = document.getElementById("cropVerifiedVal");
    const cropConfidenceVal = document.getElementById("cropConfidenceVal");

    // 2. Disease Detection
    const diseaseStatusVal = document.getElementById("diseaseStatusVal");
    const diseaseTypeVal = document.getElementById("diseaseTypeVal");
    const diseaseConfidenceVal = document.getElementById("diseaseConfidenceVal");

    // 3. Potassium Status
    const potassiumStatusVal = document.getElementById("potassiumStatusVal");
    const potassiumConfidenceVal = document.getElementById("potassiumConfidenceVal");

    // 4. Plant & Weed Detection
    const plantCountVal = document.getElementById("plantCountVal");
    const weedCountVal = document.getElementById("weedCountVal");

    // 5. AI Detection Result
    const annotatedImage = document.getElementById("annotatedImage");
    const legendContainer = document.getElementById("legendContainer");
    const noDetectionsNote = document.getElementById("noDetectionsNote");

    // 6. Summary Cards
    const mCropStatus = document.getElementById("mCropStatus");
    const mCropConf = document.getElementById("mCropConf");
    const mDiseaseStatus = document.getElementById("mDiseaseStatus");
    const mDiseaseConf = document.getElementById("mDiseaseConf");
    const mPotassiumStatus = document.getElementById("mPotassiumStatus");
    const mPotassiumConf = document.getElementById("mPotassiumConf");
    const mPlantCount = document.getElementById("mPlantCount");
    const mWeedCount = document.getElementById("mWeedCount");

    let currentFile = null;

    function showStatus(msg, type = "info") {
        statusMessage.textContent = msg;
        statusMessage.className = `status-message ${type}`;
        statusMessage.style.display = "block";
    }

    function clearStatus() {
        statusMessage.textContent = "";
        statusMessage.style.display = "none";
    }

    // 1. Image Selection & Preview
    imageInput.addEventListener("change", (e) => {
        clearStatus();
        const files = e.target.files;
        if (!files || files.length === 0) {
            currentFile = null;
            fileNameDisplay.textContent = "No image chosen";
            previewContainer.style.display = "none";
            scanBtn.disabled = true;
            return;
        }

        const file = files[0];
        const validTypes = ["image/jpeg", "image/png"];
        if (!validTypes.includes(file.type)) {
            showStatus("Please upload a JPG, JPEG or PNG image.", "error");
            currentFile = null;
            imageInput.value = "";
            fileNameDisplay.textContent = "No image chosen";
            previewContainer.style.display = "none";
            scanBtn.disabled = true;
            return;
        }

        currentFile = file;
        fileNameDisplay.textContent = file.name;

        // Display image preview immediately
        const reader = new FileReader();
        reader.onload = (loadEvt) => {
            imagePreview.src = loadEvt.target.result;
            previewContainer.style.display = "block";
            scanBtn.disabled = false;
        };
        reader.readAsDataURL(file);

        // Hide previous results until user clicks Analyze Image
        resultCard.style.display = "none";
    });

    // 2. Analyze Image Trigger
    scanBtn.addEventListener("click", async () => {
        if (!currentFile) {
            showStatus("Please select a soybean image.", "error");
            return;
        }

        scanBtn.disabled = true;
        scanBtn.textContent = "Analyzing...";
        showStatus("Analyzing image through AI models...", "info");
        resultCard.style.display = "none";

        const formData = new FormData();
        formData.append("image", currentFile);

        try {
            const response = await fetch("/predict", {
                method: "POST",
                body: formData
            });

            const data = await response.json();

            if (!response.ok || !data.success) {
                const errMsg = data.error || "Unable to analyze this image. Please try another clear soybean image.";
                showStatus(errMsg, "error");
                scanBtn.disabled = false;
                scanBtn.textContent = "Analyze Image";
                return;
            }

            clearStatus();

            // Render all outputs directly on the webpage
            renderAllResults(data);

            scanBtn.disabled = false;
            scanBtn.textContent = "Analyze Image";
        } catch (err) {
            showStatus("Unable to analyze this image. Please try another clear soybean image.", "error");
            scanBtn.disabled = false;
            scanBtn.textContent = "Analyze Image";
        }
    });

    // 3. Render all results directly to the DOM
    function renderAllResults(data) {
        // 1. Soybean Crop Verification
        cropStatusVal.textContent = "Soybean";
        cropVerifiedVal.textContent = "✓ Verified";
        cropConfidenceVal.textContent = `${(data.soybean_confidence * 100).toFixed(2)}%`;

        // 2. Disease Detection
        const dis = data.disease;
        if (dis.label === "Healthy") {
            diseaseStatusVal.textContent = "Healthy (No Disease Detected)";
            diseaseTypeVal.textContent = "Healthy";
        } else {
            diseaseStatusVal.textContent = "Disease Detected";
            diseaseTypeVal.textContent = dis.formatted_name;
        }
        diseaseConfidenceVal.textContent = `${(dis.confidence * 100).toFixed(2)}%`;

        // 3. Potassium Status
        const pot = data.potassium;
        potassiumStatusVal.textContent = pot.formatted_status;
        potassiumConfidenceVal.textContent = `${(pot.confidence * 100).toFixed(2)}%`;

        // 4. Plant & Weed Detection
        const plantsCount = data.plants ? data.plants.count : 0;
        const weedsCount = data.weeds ? data.weeds.count : 0;
        plantCountVal.textContent = plantsCount;
        weedCountVal.textContent = weedsCount;

        // 5. AI Detection Result (Annotated Image & Legend)
        if (data.visualization) {
            annotatedImage.src = data.visualization;
        } else {
            annotatedImage.src = imagePreview.src;
        }

        if (data.has_localized_detections) {
            legendContainer.style.display = "flex";
            noDetectionsNote.style.display = "none";
        } else {
            legendContainer.style.display = "none";
            noDetectionsNote.style.display = "block";
        }

        // 6. Summary Cards
        mCropStatus.textContent = "Soybean (Verified)";
        mCropConf.textContent = `${(data.soybean_confidence * 100).toFixed(2)}%`;

        mDiseaseStatus.textContent = dis.formatted_name;
        mDiseaseConf.textContent = `${(dis.confidence * 100).toFixed(2)}%`;

        mPotassiumStatus.textContent = pot.formatted_status;
        mPotassiumConf.textContent = `${(pot.confidence * 100).toFixed(2)}%`;

        mPlantCount.textContent = plantsCount;
        mWeedCount.textContent = weedsCount;

        // Display results card smoothly
        resultCard.style.display = "block";
    }
});
