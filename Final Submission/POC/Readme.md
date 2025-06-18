SDG3 Maternal Health Risk Prediction System
This submission showcases a complete solution for maternal health risk prediction aligned with Sustainable Development Goal 3 (SDG3). The contents include:

notebook.ipynb
Contains the core machine learning pipeline: data preprocessing, model training, prediction logic, and integration with clinical recommendations.

PoC_SDG3_Health_Maternal_Health_Risk_Prediction_System.pdf
A detailed Proof of Concept document describing the problem statement, approach, model evaluation, and real-world applicability.

Readme.md
A concise guide to the project setup, usage instructions, and team contributions.

How_to_use_the_website.txt
A user-friendly walkthrough describing how to interact with the web interface and interpret the risk predictions.

livedemolink.txt
Direct URL to the live deployed version of the prediction system.

video_demonstration_of_website.txt
Link to a recorded demo highlighting the website’s features, workflows, and user interface.

Limitations and Future Work
Current Limitations
1. Data and Population Specificity

Training Scope: Models trained specifically on Telangana dataset

Population Variations: May require recalibration for other states/regions

Temporal Factors: Snapshot of current care patterns, may not reflect future changes

Missing Variables: Some important risk factors not captured in current dataset

2. Technical Constraints

Feature Dependency: Requires specific data collection protocols at ANC visits

Connectivity Requirements: While offline-capable, some features need periodic internet

Update Frequency: Models should be retrained periodically with new outcome data

3. Clinical and Implementation Limitations

Decision Support Only: Augments but cannot replace clinical judgment

ANC Setting Focus: Optimized for routine care, not emergency situations

Training Needs: Healthcare workers require initial orientation and ongoing support

Future Development Roadmap

1. Additional Risk Prediction Models

1.Unified High-Risk Score (0-100) with 90% clinical concordance — High priority
2.ANC Dropout Prediction with 85% accuracy — Critical priority
3.Maternal Mortality Risk with 85% sensitivity — Critical priority
4.Early Warning System with 90% detection rate — High priority
5.Stillbirth Risk with 80% sensitivity — High priority
6.Birth Defect Screening flagged for ultrasound — High priority
7.Preterm Model Enhancement from 88.7% to 95% precision — High priority

2. System Enhancements

Technical Improvements

1.Mobile App Development
2.App size optimized to less than 15MB
3.Load time under 3 seconds on 2G networks
4.RAM usage below 200MB
5.Voice Interface Integration
6.Telugu and Hindi voice data entry
7.Hands-free navigation
8.Voice-based alerts
9.Advanced Offline Capabilities
10.Full risk calculation available offline
11.Smart synchronization on reconnection

Integration Expansions

1.EMR System Integration
2.Direct connection to e-Sanjeevani platform
3.Automated import of health records
4.Unified patient ID management
5.Laboratory System Connection
6.Automatic import of hemoglobin and blood sugar values
7.Linkage with district diagnostic centers
8.Critical value alerting system
9.Telemedicine Platform
10.One-click specialist consultation
11.Integration with eSanjeevani OPD services
12.Automated case summaries

Expected Impact

50% reduction in ANC dropouts

30% better high-risk identification

25% fewer inappropriate referrals

90% healthcare worker adoption

