# AI-based Dropout Prediction and Counseling System

An AI-powered Early Warning System (EWS) designed to identify students at risk of dropping out and enable timely, personalized counseling interventions. The system is context-aware and accounts for region-specific dropout drivers, particularly relevant to states like Rajasthan.

---

## Problem Statement
Student dropouts remain a critical challenge in the Indian education system, especially in rural and socio-economically constrained regions. Traditional systems often rely solely on academic performance, missing key contextual factors that influence dropout behavior.

---

## Solution Overview
This system combines machine learning with rule-based insights to predict dropout risk and support human-led interventions.

**Key capabilities:**
- Predicts student dropout risk (High / Medium / Low)
- Merges academic, attendance, fee, and contextual indicators
- Provides personalized counseling recommendations
- Enables early intervention through dashboards and alerts

---

## Regional Context Awareness
The system is designed to incorporate region-specific risk indicators relevant to states such as Rajasthan, including:
- Gender-based educational discontinuity
- Early marriage risk as a contextual signal
- Rural accessibility and economic constraints

These factors are treated as *supporting risk indicators* and are used alongside academic data to improve prediction reliability and ensure ethically responsible, human-in-the-loop decision-making.

---

## Technical Architecture
- **Frontend:** React.js (Admin & Student portals, interactive dashboards)
- **Backend:** FastAPI (CSV uploads, model inference, API services)
- **Machine Learning:** Random Forest Classifier (91% accuracy)
- **Database:** SQLAlchemy
- **Data Processing:** Pandas, NumPy
- **Visualization:** Plotly, Seaborn (PCA / t-SNE / UMAP)
- **NLP:** AI-powered chatbot and Counselor–Student Bridge

---

## Key Features
- Early risk detection with monthly alerts
- Risk-based personalized resource recommendations
- Dual-role login (Admin / Student)
- Counselor–Student Bridge for real-time mentoring
- Scalable and open-source system design

---

## Hackathon
Built as part of **Smart India Hackathon 2025**  
Problem Statement ID: 25102  
Theme: Smart Automation  
Category: Software

---

## Future Scope
- Multilingual chatbot support
- Sentiment analysis for student feedback
- Personalized learning and counseling plans
- Mental health support integration
- Policy-level analytics for administrators

---

## Ethical Considerations
The system functions strictly as a decision-support tool. All predictions are designed to assist educators and counselors, ensuring transparency, fairness, and human oversight in intervention decisions.
