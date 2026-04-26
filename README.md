🥬 Lettuce Project in a Vertical Farm
📌 Overview

This project focuses on developing an image-based analysis pipeline for lettuce growth and heading stage detection in a rotating vertical farming system.
Unlike conventional static cultivation environments, the rotating structure introduces unique challenges such as:

inconsistent viewing angles
occlusion between plants
non-uniform lighting conditions
difficulty in tracking individual plants over time

The goal of this project is to build a robust, scalable, and practical computer vision pipeline that can:

extract meaningful growth indicators
detect heading (core formation) stages
support real-world decision-making in smart farms


👨‍🔬 About Me

I am an agricultural researcher with ~3 years of experience in applied data analysis and “vibe coding.”
I previously completed a Ph.D. focused on multimodal imaging (RGB, Depth, Thermal) for pepper stress detection.

Now, I am exploring a smaller but highly practical project:
👉 applying image-based analysis to lettuce in vertical farms,
with a strong focus on pipeline design, generalization, and real-world applicability.


🌱 Vertical Farm Structure

This project is based on a rotating-type vertical farming system, where:

Beds rotate continuously in a circular structure
Images are captured at specific positions rather than fixed time intervals
Each bed contains multiple lettuce plants arranged in rows
Cameras are installed at different angles:
Top-view (vertical, 90°) → stable for area-based growth tracking
Front-view (angled, ~45°) → useful for structural and heading analysis, but more complex

This setup makes the problem significantly harder than standard image datasets, especially for:

segmentation consistency
spatial alignment (warp & scale)
plant-level tracking
📂 Project Structure
🟢 Top View Analysis (양상추 윗면 1작기)
This module focuses on top-view image analysis, which provides a relatively stable and reliable perspective.

Key objectives:
Extract area-based growth metrics
Track temporal growth patterns
Provide baseline indicators for plant development

Top-view data is less affected by occlusion and perspective distortion, making it suitable for:

early-stage growth analysis
time-series trend extraction
validation of overall plant health

🔵 Front View Analysis – 1st Crop (양상추 정면 1작기)
This is the initial attempt to build a full pipeline for front-view analysis.

What was done:
Bed segmentation (YOLO-based)
Image alignment (warp)
Scale normalization
Lettuce segmentation
Slot-based plant assignment
Limitations:
Severe occlusion between leaves
Over-segmentation (multiple instances per plant)
Unstable slot assignment across time
Noisy area-based metrics

👉 Conclusion:
A full pipeline was successfully constructed, but metric reliability and structural consistency were insufficient.

🟡 Front View Analysis – 1.5 Crop (양상추 정면 1.5작기)
This phase focuses on improving the weaknesses of the 1st crop pipeline.

Key idea:
👉 Shift from “whole plant area” → core (heading) 중심 분석

What was explored:
Center-point detection
Elliptical core region modeling
Core-focused feature extraction
Quality labeling (good / bad / ambiguous)
Insight:
Total area is too noisy in front view
Core region provides a more stable signal for heading stage
Hybrid approach (algorithm + deep learning) is necessary

👉 This stage represents a conceptual transition toward core-based modeling.

🔴 Front View Analysis – 2nd Crop (양상추 정면 2작기)
This is the current working stage, focusing on refinement and practical usability.

Goals:
Clean and consistent training data
Improved segmentation robustness
Stable feature extraction pipeline
Preparation for learning-based models
Ongoing work:
Pipeline simplification and stabilization
Outlier filtering and quality control
Better alignment between top-view and front-view data
Preparing datasets for CNN / CLIP-based modeling

👉 Objective:
Move from experimental pipeline → usable and generalizable system

🎯 Final Goal

The ultimate goal of this project is not just model accuracy, but:

A reproducible pipeline
A generalizable framework across crops
A system that can be integrated into real-world smart farming environments
