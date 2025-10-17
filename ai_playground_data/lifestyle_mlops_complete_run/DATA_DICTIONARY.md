# AI Playground Data Dictionary
## Lifestyle & Fitness Dataset Analysis

This dataset contains comprehensive lifestyle and fitness data with 20,000 records and 400+ features. Below is a detailed breakdown of all data categories and features.

## 📊 Dataset Overview
- **Total Records**: 20,000
- **Total Features**: 400+
- **Data Types**: Numerical, Categorical, Boolean
- **Primary Categories**: Demographics, Fitness Metrics, Nutrition, Exercise, Workout Types

## 👥 Demographic Features

### Basic Demographics
- **Age**: Participant's age in years (18-59)
- **Gender**: Binary classification (0=Female, 1=Male)
- **Weight (kg)**: Body weight in kilograms
- **Height (m)**: Height in meters
- **BMI**: Body Mass Index (calculated)
- **BMI_calc**: Alternative BMI calculation

### Body Composition
- **Fat_Percentage**: Body fat percentage
- **Water_Intake (liters)**: Daily water consumption
- **Lean_Mass (kg)**: Calculated lean body mass

## 💪 Fitness & Performance Metrics

### Heart Rate & BPM
- **Max_BPM**: Maximum heart rate achieved
- **Avg_BPM**: Average heart rate during session
- **Resting_BPM**: Resting heart rate
- **pct_HRR**: Percentage of Heart Rate Reserve
- **pct_maxHR**: Percentage of maximum heart rate

### Session Data
- **Session_Duration (hours)**: Workout duration
- **Calories_Burned**: Total calories burned
- **Burns Calories (per 30 min)**: Calorie burn rate
- **expected_burn**: Expected calorie burn based on calculations

### Performance Indicators
- **Workout_Frequency (days/week)**: Weekly workout frequency
- **Experience_Level**: Fitness experience level (1-3)
- **Physical_exercise**: Physical activity rating

## 🥗 Nutrition & Dietary Data

### Macronutrients
- **Carbs**: Carbohydrate intake (g)
- **Proteins**: Protein intake (g)
- **Fats**: Fat intake (g)
- **Calories**: Total caloric intake
- **cal_from_macros**: Calories calculated from macros

### Nutrient Ratios
- **pct_carbs**: Percentage of calories from carbs
- **protein_per_kg**: Protein intake per kg body weight
- **cal_balance**: Calorie balance (intake vs expenditure)

### Meal Information
- **meal_name**: Type of meal (Breakfast, Lunch, Dinner, Snack)
- **meal_type**: Meal category
- **diet_type**: Dietary pattern (Balanced, Keto, Paleo, Vegan, etc.)
- **cooking_method**: Preparation method (Baked, Grilled, etc.)
- **Daily meals frequency**: Number of meals per day

### Micronutrients
- **sugar_g**: Sugar content
- **sodium_mg**: Sodium content
- **cholesterol_mg**: Cholesterol content
- **serving_size_g**: Serving size
- **prep_time_min**: Preparation time
- **cook_time_min**: Cooking time
- **rating**: Meal rating

## 🏋️ Exercise & Training Data

### Exercise Parameters
- **Sets**: Number of sets performed
- **Reps**: Repetitions per set
- **Name of Exercise**: Specific exercise name (55 unique exercises)
- **Benefit**: Primary benefit of the exercise
- **Target Muscle Group**: Targeted muscle groups
- **Body Part**: Body region targeted
- **Type of Muscle**: Muscle fiber type

### Exercise Classification
- **Difficulty Level**: Beginner, Intermediate, Advanced
- **Equipment Needed**: Required equipment
- **Workout**: Exercise category

### Exercise Categories (One-hot encoded)
- **Name of Exercise_***: 55 different exercises (Bear Crawls, Bench Press, etc.)
- **Benefit_***: 49 different benefit categories
- **Target Muscle Group_***: 13 muscle group categories
- **Equipment Needed_***: 13 equipment types
- **Difficulty Level_***: 3 difficulty levels
- **Body Part_***: 7 body parts
- **Type of Muscle_***: 7 muscle types
- **Workout_***: 53 workout variations

## 🧘 Workout Type Categories

### Primary Workout Types (One-hot encoded)
- **Workout_Type_Cardio**: Cardiovascular training
- **Workout_Type_HIIT**: High-Intensity Interval Training
- **Workout_Type_Strength**: Resistance/strength training
- **Workout_Type_Yoga**: Yoga and flexibility training

### Meal Type Categories (One-hot encoded)
- **meal_type_Breakfast**
- **meal_type_Dinner**
- **meal_type_Lunch**
- **meal_type_Snack**

### Diet Type Categories (One-hot encoded)
- **diet_type_Balanced**
- **diet_type_Keto**
- **diet_type_Low-Carb**
- **diet_type_Paleo**
- **diet_type_Vegan**
- **diet_type_Vegetarian**

### Cooking Method Categories (One-hot encoded)
- **cooking_method_Baked**
- **cooking_method_Boiled**
- **cooking_method_Fried**
- **cooking_method_Grilled**
- **cooking_method_Raw**
- **cooking_method_Roasted**
- **cooking_method_Steamed**

## 📈 Derived & Calculated Features

### Calorie Burn Categories
- **Burns_Calories_Bin_High**: High calorie burn rate
- **Burns_Calories_Bin_Low**: Low calorie burn rate
- **Burns_Calories_Bin_Medium**: Medium calorie burn rate
- **Burns_Calories_Bin_Very High**: Very high calorie burn rate

### Performance Metrics
- **pct_HRR**: Heart rate reserve percentage
- **pct_maxHR**: Maximum heart rate percentage
- **cal_balance**: Energy balance calculation
- **expected_burn**: Predicted calorie expenditure

## 🔍 Key Insights & Analysis Opportunities

### Predictive Modeling
- **Regression Targets**: Calories_Burned, BMI, Fat_Percentage
- **Classification Targets**: Workout_Type, diet_type, Difficulty_Level
- **Time Series**: Session_Duration, heart rate patterns

### Correlation Analysis
- **Fitness Correlations**: Age vs Max_BPM, Weight vs Calories_Burned
- **Nutrition Impact**: Protein intake vs muscle development
- **Performance Factors**: Experience_Level vs workout intensity

### Clustering Opportunities
- **User Segmentation**: Based on demographics, fitness levels, dietary preferences
- **Exercise Patterns**: Grouping by workout types and difficulty levels
- **Health Profiles**: BMI categories, body composition clusters

### Anomaly Detection
- **Outlier Identification**: Unusual calorie burns, heart rate readings
- **Data Quality**: Missing values, inconsistent measurements
- **Performance Anomalies**: Unexpected fitness metrics

## 🤖 AI Analysis Use Cases

### RAG (Retrieval-Augmented Generation) Applications
1. **Personalized Fitness Recommendations**
2. **Nutrition Optimization Queries**
3. **Exercise Selection Guidance**
4. **Health Risk Assessment**

### Predictive Analytics
1. **Calorie Burn Prediction**
2. **Optimal Workout Timing**
3. **Dietary Impact Modeling**
4. **Progress Tracking**

### Natural Language Queries
- "What exercises should someone with high BMI avoid?"
- "How does protein intake affect muscle development?"
- "What are the best cardio exercises for beginners?"
- "How does age affect maximum heart rate?"

## 📋 Data Quality Notes

- **Completeness**: All features have 20,000 records (100% complete)
- **Data Types**: Mix of continuous, categorical, and binary features
- **Encoding**: One-hot encoding for categorical variables
- **Normalization**: Some features may need scaling for ML models
- **Outliers**: Present in physiological measurements (expected)

## 🔗 Feature Relationships

### Strong Correlations Expected
- Age ↔ Max_BPM (negative correlation)
- Weight ↔ Calories_Burned (positive correlation)
- Experience_Level ↔ Workout_Frequency (positive correlation)
- BMI ↔ Fat_Percentage (strong positive correlation)

### Causal Relationships
- Exercise intensity → Heart rate response
- Nutrition quality → Performance metrics
- Training frequency → Fitness improvements