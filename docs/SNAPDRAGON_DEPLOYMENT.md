# Snapdragon Deployment Plan

## Target

Snapdragon-powered HP PC running the PlacementIQ application locally.

## Current prototype

The ML components use scikit-learn and SHAP and are fully local. No student information is sent to a remote API.

## Generative AI upgrade for the final challenge build

The challenge rules allow open-source models. A recommended final architecture is:

React/Streamlit UI
    |
FastAPI / Python application
    |
Placement ML + SHAP
    |
Local LLM
    |
ONNX Runtime / Qualcomm AI Hub compatible deployment
    |
Snapdragon NPU / accelerator where supported

Use a small instruction-tuned open-source model suitable for local inference. Qualcomm AI Hub can be used to identify compatible models and deployment artifacts.

## Evidence to collect before submission

- HP laptop model
- Snapdragon processor/model
- Screenshot of the application running on the HP PC
- Screenshot of local model inference
- Model name and version
- Runtime used
- Inference/device configuration
- Short demo video if the challenge intake form allows it

## Do not overclaim

Only say "NPU accelerated", "Qualcomm optimized", or similar if you have actually measured or verified it on the target Snapdragon device.

## Why local AI?

Student academic and placement information can be sensitive. Local inference can reduce the need to transmit personal information to an external service and can make the assistant useful without an always-on cloud connection.
