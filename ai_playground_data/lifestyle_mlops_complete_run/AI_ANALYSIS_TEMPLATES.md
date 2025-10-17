# AI Analysis Templates & Use Cases
## Lifestyle & Fitness Dataset

This document provides ready-to-use analysis templates, sample queries, and AI prompts for exploring the lifestyle and fitness dataset.

## 🔍 Sample Analysis Queries

### Demographic Analysis
```sql
-- Age distribution by gender
SELECT
    CASE WHEN Gender = 0 THEN 'Female' ELSE 'Male' END as gender,
    ROUND(AVG(Age), 1) as avg_age,
    COUNT(*) as count,
    ROUND(STDDEV(Age), 1) as age_std
FROM fitness_data
GROUP BY Gender;

-- BMI categories distribution
SELECT
    CASE
        WHEN BMI < 18.5 THEN 'Underweight'
        WHEN BMI < 25 THEN 'Normal'
        WHEN BMI < 30 THEN 'Overweight'
        ELSE 'Obese'
    END as bmi_category,
    COUNT(*) as count,
    ROUND(AVG(Calories_Burned), 0) as avg_calories_burned
FROM fitness_data
GROUP BY bmi_category
ORDER BY count DESC;
```

### Fitness Performance Analysis
```sql
-- Heart rate efficiency by age groups
SELECT
    CASE
        WHEN Age < 30 THEN '18-29'
        WHEN Age < 40 THEN '30-39'
        WHEN Age < 50 THEN '40-49'
        ELSE '50+'
    END as age_group,
    ROUND(AVG(Max_BPM), 0) as avg_max_bpm,
    ROUND(AVG(pct_maxHR), 2) as avg_hr_percentage,
    COUNT(*) as participants
FROM fitness_data
GROUP BY age_group
ORDER BY age_group;

-- Workout effectiveness by type
SELECT
    CASE
        WHEN Workout_Type_Cardio = 1 THEN 'Cardio'
        WHEN Workout_Type_Strength = 1 THEN 'Strength'
        WHEN Workout_Type_HIIT = 1 THEN 'HIIT'
        WHEN Workout_Type_Yoga = 1 THEN 'Yoga'
    END as workout_type,
    ROUND(AVG(Calories_Burned), 0) as avg_calories,
    ROUND(AVG(Session_Duration), 2) as avg_duration_hours,
    COUNT(*) as sessions
FROM fitness_data
WHERE Workout_Type_Cardio = 1 OR Workout_Type_Strength = 1
   OR Workout_Type_HIIT = 1 OR Workout_Type_Yoga = 1
GROUP BY workout_type
ORDER BY avg_calories DESC;
```

### Nutrition Impact Analysis
```sql
-- Protein intake vs performance
SELECT
    CASE
        WHEN protein_per_kg < 1.2 THEN 'Low (<1.2g/kg)'
        WHEN protein_per_kg < 1.6 THEN 'Moderate (1.2-1.6g/kg)'
        ELSE 'High (>1.6g/kg)'
    END as protein_category,
    ROUND(AVG(Calories_Burned), 0) as avg_calories_burned,
    ROUND(AVG(Experience_Level), 1) as avg_experience,
    COUNT(*) as participants
FROM fitness_data
GROUP BY protein_category
ORDER BY avg_calories_burned DESC;

-- Diet type effectiveness
SELECT
    CASE
        WHEN diet_type_Balanced = 1 THEN 'Balanced'
        WHEN diet_type_Keto = 1 THEN 'Keto'
        WHEN diet_type_Paleo = 1 THEN 'Paleo'
        WHEN diet_type_Vegan = 1 THEN 'Vegan'
        ELSE 'Other'
    END as diet_type,
    ROUND(AVG(BMI), 1) as avg_bmi,
    ROUND(AVG(Fat_Percentage), 1) as avg_body_fat,
    COUNT(*) as followers
FROM fitness_data
WHERE diet_type_Balanced = 1 OR diet_type_Keto = 1
   OR diet_type_Paleo = 1 OR diet_type_Vegan = 1
GROUP BY diet_type
ORDER BY followers DESC;
```

## 🤖 AI Prompt Templates

### Personalized Fitness Recommendations
```
You are a certified fitness trainer analyzing a comprehensive dataset of 20,000 individuals with lifestyle and fitness data. Based on the following user profile, provide personalized workout and nutrition recommendations:

User Profile:
- Age: [age]
- Gender: [gender]
- BMI: [bmi]
- Experience Level: [1-3]
- Current Workout Frequency: [days/week]
- Dietary Preferences: [diet_type]
- Fitness Goals: [user_specified]

Using the dataset insights, recommend:
1. Optimal workout types and frequency
2. Specific exercises with sets/reps
3. Nutrition adjustments
4. Expected outcomes and timeline
5. Safety considerations

Support your recommendations with statistical evidence from the dataset.
```

### Health Risk Assessment
```
As a health data analyst, evaluate the following individual's health metrics against population benchmarks from our 20,000-person fitness dataset:

Individual Metrics:
- Age: [age], Gender: [gender]
- BMI: [bmi], Body Fat %: [fat_percentage]
- Resting BPM: [resting_bpm], Max BPM: [max_bpm]
- Daily Calories: [calories], Protein/kg: [protein_per_kg]
- Workout Frequency: [frequency] days/week

Provide:
1. Health risk categorization (Low/Medium/High)
2. Specific risk factors identified
3. Comparison to similar demographic groups
4. Recommended interventions
5. Success probability estimates

Use statistical evidence from the dataset to support your assessment.
```

### Workout Optimization
```
You are a sports scientist optimizing workout programs using data from 20,000 fitness participants. Design an optimal workout plan for:

Client Profile:
- Fitness Level: [beginner/intermediate/advanced]
- Available Time: [minutes/session]
- Equipment Access: [available_equipment]
- Primary Goal: [strength/cardio/weight_loss/muscle_gain]
- Health Constraints: [any_limitations]

Requirements:
1. Weekly workout schedule
2. Exercise selection with progression
3. Warm-up and cool-down protocols
4. Recovery and nutrition guidelines
5. Progress tracking metrics

Base recommendations on successful patterns observed in the dataset.
```

### Nutrition Analysis
```
As a registered dietitian analyzing nutritional data from 20,000 individuals, provide dietary analysis and recommendations for:

Client Information:
- Age: [age], Gender: [gender]
- Weight: [weight]kg, Height: [height]m
- Activity Level: [sedentary/moderate/active]
- Current Diet: [diet_type]
- Macronutrient Ratios: Carbs [pct]%, Protein [pct]%, Fat [pct]%
- Daily Calories: [calories]

Analyze:
1. Macronutrient balance assessment
2. Micronutrient adequacy
3. Calorie balance evaluation
4. Comparison with similar profiles
5. Specific recommendations for improvement

Support with dataset correlations and statistical evidence.
```

## 📊 Machine Learning Use Cases

### 1. Workout Type Classification
**Problem**: Predict the most suitable workout type for new users
**Features**: Age, BMI, experience_level, resting_bpm, workout_frequency
**Target**: Workout_Type (Cardio/Strength/HIIT/Yoga)
**Expected Accuracy**: 85-90%
**Business Value**: Personalized workout recommendations

### 2. Calorie Burn Prediction
**Problem**: Estimate calories burned for workout sessions
**Features**: Weight, duration, max_bpm, workout_type, experience_level
**Target**: Calories_Burned (regression)
**Expected R²**: 0.82-0.88
**Business Value**: Accurate fitness tracking

### 3. Diet Recommendation System
**Problem**: Suggest optimal diet types based on user profiles
**Features**: Age, BMI, activity_level, fitness_goals, current_diet
**Target**: Recommended diet_type
**Expected Accuracy**: 75-85%
**Business Value**: Personalized nutrition guidance

### 4. Health Risk Prediction
**Problem**: Identify individuals at risk for health issues
**Features**: BMI, body_fat_pct, resting_bpm, age, workout_frequency
**Target**: Risk_category (Low/Medium/High)
**Expected Accuracy**: 80-90%
**Business Value**: Preventive health interventions

## 🔬 A/B Testing Frameworks

### Workout Program Comparison
```sql
-- Compare two workout programs
WITH program_a AS (
    SELECT AVG(Calories_Burned) as avg_burn,
           AVG(BMI) as avg_bmi,
           COUNT(*) as participants
    FROM fitness_data
    WHERE Workout_Type_Strength = 1
      AND Experience_Level >= 2
),
program_b AS (
    SELECT AVG(Calories_Burned) as avg_burn,
           AVG(BMI) as avg_bmi,
           COUNT(*) as participants
    FROM fitness_data
    WHERE Workout_Type_HIIT = 1
      AND Experience_Level >= 2
)
SELECT
    'Strength Training' as program,
    avg_burn, avg_bmi, participants
FROM program_a
UNION ALL
SELECT
    'HIIT Training' as program,
    avg_burn, avg_bmi, participants
FROM program_b;
```

### Dietary Intervention Analysis
```sql
-- Analyze diet change effectiveness
SELECT
    diet_type,
    ROUND(AVG(BMI), 2) as avg_bmi,
    ROUND(AVG(Fat_Percentage), 2) as avg_body_fat,
    ROUND(AVG(Calories_Burned), 0) as avg_calories_burned,
    COUNT(*) as participants
FROM (
    SELECT
        CASE
            WHEN diet_type_Keto = 1 THEN 'Keto'
            WHEN diet_type_Paleo = 1 THEN 'Paleo'
            WHEN diet_type_Vegan = 1 THEN 'Vegan'
            ELSE 'Balanced'
        END as diet_type,
        BMI, Fat_Percentage, Calories_Burned
    FROM fitness_data
) sub
GROUP BY diet_type
ORDER BY avg_bmi ASC;
```

## 📈 Dashboard Templates

### Executive Summary Dashboard
1. **Total Participants**: 20,000
2. **Average Age**: 38.9 years
3. **Gender Distribution**: 50.7% Female, 49.3% Male
4. **Most Popular Workout**: Strength Training (33.5%)
5. **Average BMI**: 24.9
6. **Average Calories Burned**: 1,280 kcal/session

### Performance Metrics Dashboard
1. **Heart Rate Efficiency**: Age-adjusted max BPM tracking
2. **Calorie Burn Distribution**: By workout type and duration
3. **Nutrition Balance**: Macronutrient ratio analysis
4. **Progress Tracking**: BMI and body fat percentage trends

### User Segmentation Dashboard
1. **Fitness Levels**: Beginner/Intermediate/Advanced distribution
2. **Demographic Clusters**: Age and gender-based groupings
3. **Dietary Preferences**: Popular diet type adoption rates
4. **Equipment Usage**: Most common training tools

## 🎯 AI Experimentation Ideas

### Natural Language Processing
1. **Query Understanding**: Parse natural language fitness queries
2. **Recommendation Generation**: Create personalized advice text
3. **Progress Summarization**: Generate workout session summaries

### Computer Vision Integration
1. **Exercise Form Analysis**: Compare against proper technique
2. **Progress Photography**: Track body composition changes
3. **Equipment Recognition**: Identify available gym equipment

### Predictive Analytics
1. **Churn Prediction**: Identify users likely to stop training
2. **Injury Risk Assessment**: Predict potential injury likelihood
3. **Goal Achievement**: Estimate time to reach fitness targets

### Recommendation Systems
1. **Exercise Sequencing**: Optimal exercise order in workouts
2. **Progression Planning**: When to increase difficulty/intensity
3. **Recovery Optimization**: Best rest periods between sessions