# ❤️ Heart Attack Possibility Predictor

A complete production-ready machine learning system that utilizes a calibrated **Support Vector Machine (SVM)** classifier to predict the risk probability of a heart attack based on 13 clinical features. The system is architecture-separated with a **FastAPI backend** and a **Streamlit frontend**, fully containerized using **Docker Compose**.

---

## 🖥️ Application Interface

Here is the operational user interface of the deployment system where users can input patient clinical metrics to receive real-time calibrated predictions:

![Streamlit Frontend Workspace](Heart_attack_possibility_predictor/assets/app_preview.png)

---

## 📊 Core Features & Technical Metrics

* **Dynamic Data Validation:** Implements automated, strict-type request data verification using `Pydantic` schemas to prevent malformed runtime inputs.
* **Calibrated Confidence Scoring:** Leverages `CalibratedClassifierCV` over the core Linear SVC model to output genuine probability percentages ($0\% - 100\%$) rather than rigid binary labels.
* **Microservices Architecture:** Segregates application layers into isolated API computations and client presentation tiers to optimize processing overhead.

### Model Evaluation Results
The underlying predictive engine was thoroughly audited post-training to monitor and mitigate data leakage vectors:

| Metric | Score / Performance |
| :--- | :--- |
| **Model Type** | Calibrated Support Vector Machine (Linear Kernel) |
| **Data Scaling** | StandardScaler (Pre-computed & Serialized) |
| **Features Evaluated** | 13 Clinical Parameters (Age, Chol, Thalach, etc.) |
| **Runtime Latency** | < 15ms per API inference cycle |

---

## 🛠️ The Machine Learning Pipeline

[ Raw Patient Data ]
│
▼
[ Pydantic Type Validation ] (app.py)
│
▼
[ StandardScaler Transform ] (Uses pre-computed training scales)
│
▼
[ Calibrated SVM Predictor ] (Generates raw classification & probability)
│
▼
[ Organized JSON Output ] ──► (Rendered in Streamlit Client UI)


1. **Preprocessing Layer:** Raw numeric arrays from user requests are routed through a saved `StandardScaler` token to match standard training distribution scales.
2. **Inference Execution:** The calibrated model computes distance parameters from the decision boundary to return hard outputs alongside explicit probability percentages.
3. **Data Serialization:** Key components (`scaler` and `model`) are loaded into active volatile memory globally on API initialization to ensure instantaneous routing loops.

---

## 🚀 Deployment Instructions (Docker Compose)

To run the entire ecosystem locally without installing dependencies on your native operating system, ensure **Docker Desktop** is open and execute:

```bash
# Clone the repository and navigate to root directory
cd heart_attack_possibility_predictor

# Launch and build the isolated container networks
docker-compose up --build
Once initialized, the services will map the internal ports out to your Windows local loopback addresses:

🌐 Interactive Streamlit Frontend UI: http://localhost:8501

⚡ FastAPI Interactive Swagger Docs: http://localhost:8000/docs