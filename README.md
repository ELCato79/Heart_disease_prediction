# Heart Disease Prediction — End-to-End MLOps Pipeline

## Project Overview
This project demonstrates an end-to-end MLOps pipeline for a Heart Disease Prediction system.
The focus is not only on model building, but on **reproducibility, automation, and deployment**
using standerd MLOps practices.

---

## Project Structure
.
├── configs/ # Configuration files (YAML / env-based)
├── data/ # Data directory (ignored in git)
├── model/ # Trained models / artifacts (ignored in git)
├── reports/ # Evaluation reports & plots
├── results/ # Experiment outputs
├── scripts/ # Training and evaluation scripts
├── src/ # Application source code
├── tests/ # Unit tests (pytest)
├── Dockerfile # Containerization
├── requirements.txt
├── pytest.ini
└── README.md


---

##  MLOps Workflow
1. Code is developed locally and version-controlled using Git.
2. GitHub acts as the **single source of truth**.
3. Every push triggers automated testing via **GitHub Actions (CI)**.
4. The application is containerized using **Docker**.
5. Deployment is automated to **Render** using CI/CD.
6. Environment variables and secrets are managed outside the codebase.

---

##  Containerization
The project uses Docker to ensure environment consistency across:
- Local development
- CI pipelines
- Production deployment

---

##  Deployment
The application is deployed on **Render** using Docker-based deployment.
Any push to the `main` branch automatically triggers a new deployment.

---

## Testing
Unit tests are written using `pytest` and executed automatically in the CI pipeline
before any deployment.

---

## Configuration & Secrets
- Configuration is externalized using environment variables.
- Sensitive information is never committed to the repository.

---

## Future Enhancements
- Model monitoring & drift detection
- Experiment tracking (MLflow)
- Model versioning & rollback
- Automated retraining pipelines

---

## License
MIT License
