// API Configuration
const API_URL = 'http://localhost:5000/predict';

// Disease-specific data mapping
const diseaseData = {
    'Early_Blight': {
        status: 'disease',
        title: 'Early Blight Disease Detected',
        description: 'The leaf exhibits symptoms of early blight, a fungal disease caused by Alternaria solani. Dark spots with concentric rings are visible. This requires both fungicide treatment and proper nutrition to boost plant immunity.',
        severity: 75,
        severityLevel: 'high',
        primaryFertilizer: {
            name: 'NPK 10-26-26 + Micronutrients',
            composition: 'Nitrogen: 10%, Phosphorus: 26%, Potassium: 26%, plus Zinc, Boron',
            dosage: '1 tablespoon per liter of water, apply as foliar spray weekly'
        },
        secondaryFertilizers: [
            {
                name: 'Calcium Nitrate',
                description: 'Strengthens cell walls to resist disease penetration'
            },
            {
                name: 'Seaweed Extract',
                description: 'Boosts plant immunity and stress resistance'
            },
            {
                name: 'Copper Fungicide',
                description: 'Controls fungal spread and prevents further infection'
            }
        ],
        schedule: [
            'Week 1: Remove affected leaves immediately, apply copper fungicide + NPK foliar spray',
            'Week 2: Continue foliar spray twice weekly, ensure proper spacing for air circulation',
            'Week 3: Apply calcium nitrate to strengthen plant structure and disease resistance',
            'Week 4: Monitor new growth closely, reduce frequency if significant improvement is seen',
            'Week 5-6: Maintain preventive spraying schedule every 10-14 days'
        ],
        careTips: [
            'Remove and destroy all infected leaves to prevent disease spread',
            'Improve air circulation by spacing plants properly',
            'Water at soil level only, avoid wetting leaves especially in evening',
            'Apply organic mulch to prevent soil splash onto lower leaves',
            'Ensure plants receive adequate sunlight (6-8 hours daily)'
            
        ],
        nutrients: [
            { name: 'Nitrogen (N)', level: 55, status: 'low' },
            { name: 'Phosphorus (P)', level: 45, status: 'low' },
            { name: 'Potassium (K)', level: 50, status: 'low' },
            { name: 'Calcium (Ca)', level: 40, status: 'deficient' },
            { name: 'Magnesium (Mg)', level: 60, status: 'low' },
            { name: 'Iron (Fe)', level: 58, status: 'low' }
        ]
    },
    'Late_Blight': {
        status: 'disease',
        title: 'Late Blight Disease Detected',
        description: 'The leaf shows signs of late blight, caused by Phytophthora infestans. This is a severe water mold disease that can rapidly destroy entire plants. Immediate action with fungicides and proper nutrition is critical.',
        severity: 90,
        severityLevel: 'high',
        primaryFertilizer: {
            name: 'NPK 12-32-16 + Copper',
            composition: 'Nitrogen: 12%, Phosphorus: 32%, Potassium: 16%, Copper: 2%',
            dosage: '2 tablespoons per liter of water, apply as foliar spray every 5 days'
        },
        secondaryFertilizers: [
            {
                name: 'Potassium Phosphite',
                description: 'Systemic protection against water mold diseases'
            },
            {
                name: 'Mancozeb Fungicide',
                description: 'Broad-spectrum protection against late blight'
            },
            {
                name: 'Calcium Chloride',
                description: 'Strengthens cell walls and improves disease resistance'
            }
        ],
        schedule: [
            'Day 1: Remove all infected plant parts, apply systemic fungicide immediately',
            'Day 3-5: Apply NPK foliar spray with copper supplement',
            'Week 2: Continue aggressive fungicide treatment every 5-7 days',
            'Week 3: Apply potassium phosphite for systemic protection',
            'Week 4-6: Maintain preventive fungicide schedule, monitor weather conditions',
            'Ongoing: Increase frequency during humid/wet conditions'
        ],
        careTips: [
            'Act immediately - late blight spreads rapidly in 24-48 hours',
            'Remove infected plants entirely if more than 50% affected',
            'Avoid working with plants when foliage is wet',
            'Increase spacing between plants to improve air circulation',
            'Apply preventive fungicides before symptoms appear in humid weather',
            'Monitor neighboring plants daily for early detection',
            'Destroy infected plant material, do not compost',
            'Consider resistant varieties for future planting'
        ],
        nutrients: [
            { name: 'Nitrogen (N)', level: 50, status: 'low' },
            { name: 'Phosphorus (P)', level: 35, status: 'deficient' },
            { name: 'Potassium (K)', level: 45, status: 'low' },
            { name: 'Calcium (Ca)', level: 30, status: 'deficient' },
            { name: 'Magnesium (Mg)', level: 55, status: 'low' },
            { name: 'Iron (Fe)', level: 52, status: 'low' }
        ]
    },
    'Healthy': {
        status: 'healthy',
        title: 'Healthy Leaf - Maintenance Required',
        description: 'The leaf appears healthy with good color and structure. Continue with regular maintenance fertilization to sustain plant health and prevent future diseases.',
        severity: 15,
        severityLevel: 'low',
        primaryFertilizer: {
            name: 'Balanced NPK 20-20-20',
            composition: 'Nitrogen: 20%, Phosphorus: 20%, Potassium: 20%',
            dosage: '1 tablespoon per gallon of water, every 2-3 weeks'
        },
        secondaryFertilizers: [
            {
                name: 'Compost Tea',
                description: 'Organic nutrient boost with beneficial microbes'
            },
            {
                name: 'Fish Emulsion',
                description: 'Gentle liquid fertilizer for regular feeding'
            },
            {
                name: 'Seaweed Extract',
                description: 'Micronutrients and growth hormones for optimal health'
            }
        ],
        schedule: [
            'Week 1-2: Apply balanced NPK fertilizer at recommended dosage',
            'Week 3-4: Supplement with organic compost tea for beneficial microbes',
            'Week 5-6: Continue regular feeding routine with fish emulsion',
            'Week 7-8: Apply seaweed extract for trace minerals',
            'Monthly: Add slow-release granular fertilizer at soil surface',
            'Quarterly: Conduct soil test to adjust fertilization program'
        ],
        careTips: [
            'Maintain consistent watering schedule - avoid drought stress',
            'Apply 2-3 inch layer of organic mulch to retain soil moisture',
            'Rotate between synthetic and organic fertilizers for balanced nutrition',
            'Monitor for any changes in leaf color or growth patterns',
            'Prune regularly to promote healthy growth and air circulation',
            'Inspect plants weekly for early signs of pests or disease',
            'Maintain proper plant spacing to prevent disease',
            'Ensure adequate sunlight exposure (6-8 hours daily)'
        ],
        nutrients: [
            { name: 'Nitrogen (N)', level: 85, status: 'optimal' },
            { name: 'Phosphorus (P)', level: 82, status: 'optimal' },
            { name: 'Potassium (K)', level: 88, status: 'optimal' },
            { name: 'Calcium (Ca)', level: 80, status: 'optimal' },
            { name: 'Magnesium (Mg)', level: 78, status: 'optimal' },
            { name: 'Iron (Fe)', level: 85, status: 'optimal' }
        ]
    }
};

let uploadedFile = null;

// DOM Elements
const uploadBox = document.getElementById('uploadBox');
const fileInput = document.getElementById('fileInput');
const imagePreview = document.getElementById('imagePreview');
const previewSection = document.getElementById('previewSection');
const analyzeBtn = document.getElementById('analyzeBtn');
const resultsSection = document.getElementById('resultsSection');
const changeImageBtn = document.getElementById('changeImageBtn');
const newAnalysisBtn = document.getElementById('newAnalysisBtn');

// Upload box click handler
uploadBox.addEventListener('click', () => {
    fileInput.click();
});

// Drag and drop handlers
uploadBox.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadBox.style.borderColor = '#667eea';
    uploadBox.style.background = '#edf2f7';
});

uploadBox.addEventListener('dragleave', () => {
    uploadBox.style.borderColor = '#cbd5e0';
    uploadBox.style.background = '#f7fafc';
});

uploadBox.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadBox.style.borderColor = '#cbd5e0';
    uploadBox.style.background = '#f7fafc';
    
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        handleFile(files[0]);
    }
});

// File input change handler
fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
        handleFile(e.target.files[0]);
    }
});

// Handle file upload
function handleFile(file) {
    if (!file.type.startsWith('image/')) {
        alert('Please upload an image file');
        return;
    }

    uploadedFile = file;
    const reader = new FileReader();
    
    reader.onload = (e) => {
        imagePreview.src = e.target.result;
        uploadBox.style.display = 'none';
        previewSection.style.display = 'block';
        analyzeBtn.disabled = false;
        resultsSection.style.display = 'none';
    };
    
    reader.readAsDataURL(file);
}

// Change image button
changeImageBtn.addEventListener('click', () => {
    fileInput.click();
});

// Analyze button
analyzeBtn.addEventListener('click', async () => {
    if (!uploadedFile) return;
    
    analyzeBtn.textContent = 'Analyzing with AI Model...';
    analyzeBtn.disabled = true;
    
    try {
        // Call the model API
        const prediction = await predictWithModel(uploadedFile);
        
        if (prediction.success) {
            displayResults(prediction);
            analyzeBtn.textContent = 'Analyze Leaf';
            analyzeBtn.disabled = false;
            
            // Scroll to results
            resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
        } else {
            throw new Error(prediction.error || 'Prediction failed');
        }
    } catch (error) {
        console.error('Error during analysis:', error);
        alert('Error analyzing image. Please make sure the backend server is running.\n\nTo start the server:\n1. Install dependencies: pip install -r requirements.txt\n2. Run: python app.py\n\nError: ' + error.message);
        analyzeBtn.textContent = 'Analyze Leaf';
        analyzeBtn.disabled = false;
    }
});

// Function to call the model API
async function predictWithModel(file) {
    const formData = new FormData();
    formData.append('image', file);
    
    try {
        const response = await fetch(API_URL, {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            throw new Error(`Server error: ${response.status}`);
        }
        
        const result = await response.json();
        return result;
    } catch (error) {
        console.error('API call failed:', error);
        return { success: false, error: error.message };
    }
}

// Display results
function displayResults(prediction) {
    // Get diagnosis data based on model prediction
    const predictedClass = prediction.prediction;
    const confidence = prediction.confidence;
    const diagnosis = diseaseData[predictedClass];
    
    if (!diagnosis) {
        console.error('Unknown prediction class:', predictedClass);
        alert('Error: Unknown disease classification');
        return;
    }
    
    // Add confidence information to description
    const confidencePercent = (confidence * 100).toFixed(1);
    
    // Update confidence display
    document.getElementById('confidenceValue').textContent = `${confidencePercent}%`;
    
    // Update status badge
    const statusBadge = document.getElementById('statusBadge');
    statusBadge.textContent = diagnosis.status.toUpperCase();
    statusBadge.className = `status-badge ${diagnosis.status}`;
    
    // Update diagnosis
    document.getElementById('diagnosisTitle').textContent = diagnosis.title;
    document.getElementById('diagnosisDescription').textContent = diagnosis.description;
    
    // Update severity
    const severityFill = document.getElementById('severityFill');
    severityFill.style.width = diagnosis.severity + '%';
    severityFill.className = `severity-fill ${diagnosis.severityLevel}`;
    document.getElementById('severityText').textContent = 
        `${diagnosis.severity}% - ${diagnosis.severityLevel.toUpperCase()}`;
    
    // Update primary fertilizer
    document.getElementById('primaryFertilizer').textContent = diagnosis.primaryFertilizer.name;
    document.getElementById('primaryComposition').textContent = diagnosis.primaryFertilizer.composition;
    document.getElementById('primaryDosage').textContent = diagnosis.primaryFertilizer.dosage;
    
    // Update secondary fertilizers
    const secondaryContainer = document.getElementById('secondaryFertilizers');
    secondaryContainer.innerHTML = diagnosis.secondaryFertilizers.map(fert => `
        <div class="secondary-item">
            <h4>${fert.name}</h4>
            <p>${fert.description}</p>
        </div>
    `).join('');
    
    // Update schedule
    const scheduleList = document.getElementById('scheduleList');
    scheduleList.innerHTML = diagnosis.schedule.map(item => `<li>${item}</li>`).join('');
    
    // Update care tips
    const careTipsList = document.getElementById('careTipsList');
    careTipsList.innerHTML = diagnosis.careTips.map(tip => `<li>${tip}</li>`).join('');
    
    // Update nutrients
    const nutrientsGrid = document.getElementById('nutrientsGrid');
    nutrientsGrid.innerHTML = diagnosis.nutrients.map(nutrient => `
        <div class="nutrient-item">
            <div class="nutrient-header">
                <span class="nutrient-name">${nutrient.name}</span>
                <span class="nutrient-status ${nutrient.status}">${nutrient.status.toUpperCase()}</span>
            </div>
            <div class="nutrient-bar">
                <div class="nutrient-bar-fill ${nutrient.status}" style="width: ${nutrient.level}%"></div>
            </div>
            <span class="nutrient-value">${nutrient.level}% of optimal level</span>
        </div>
    `).join('');
    
    // Show results
    resultsSection.style.display = 'block';
}

// New analysis button
newAnalysisBtn.addEventListener('click', () => {
    uploadedFile = null;
    fileInput.value = '';
    uploadBox.style.display = 'block';
    previewSection.style.display = 'none';
    resultsSection.style.display = 'none';
    analyzeBtn.disabled = true;
    
    // Scroll to top
    window.scrollTo({ top: 0, behavior: 'smooth' });
});
