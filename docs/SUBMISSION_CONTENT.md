# Challenge Submission Draft

## Project Title

PlacementIQ — On-Device AI Placement Readiness & Career Assistant

## Problem

Students often receive academic and placement analytics without a clear explanation of why their readiness score changes or which skills they should improve next.

## Solution

PlacementIQ combines machine learning, explainable AI and a local AI assistant to transform student academic and technical-skill information into an understandable placement-readiness assessment and personalized improvement plan.

## Technical Implementation

- Random Forest for placement-readiness prediction
- K-Means for student performance grouping
- SHAP for feature-level explanation
- Skill-gap analysis based on target-role skills
- Local AI assistant layer
- Local-first processing architecture
- Designed for deployment on Snapdragon-powered HP PCs

## Innovation

The system does not stop at a prediction. It explains the prediction and converts the result into an actionable learning plan. The intended Snapdragon deployment adds an on-device AI layer so student information can be processed locally.

## Deployment & Accessibility

The prototype is packaged as a lightweight Python application and can be launched locally. The final device build should use an open-source model compatible with Qualcomm AI Hub / supported Snapdragon AI runtimes and should be demonstrated directly on the target HP Snapdragon PC.

## Presentation

Recommended 3–5 minute demo:
1. Introduce the placement problem.
2. Select a student.
3. Show readiness prediction.
4. Explain the prediction with SHAP.
5. Ask the AI assistant a question.
6. Show the resulting skill-gap recommendations.
7. Demonstrate local execution on Snapdragon hardware.
