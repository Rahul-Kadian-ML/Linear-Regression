# Linear-Regression
My linear-regression project in python and  scikit learn
# Linear Regression: House Price Prediction

This project demonstrates a **simple Linear Regression model** to predict house prices based on the size of the house (in square feet).

## 📊 Dataset
- **Input (X):** House size in square feet `[500, 1000]`
- **Output (y):** Cost per square foot `[10, 20]`
- **Tested on:** Houses of size `[1500, 2000, 2500]` for prediction.

## 🛠 Libraries Used
- Python 3
- NumPy
- Matplotlib
- Scikit-learn

## 🔹 Code Overview
1. **Data Preparation:** Reshape input data to fit scikit-learn model.  
2. **Model Training:** Fit a Linear Regression model using the training data.  
3. **Prediction:** Predict prices for new house sizes `[1500, 2000, 2500]`.  
4. **Visualization:**  
   - Regression line (red)  
   - Predicted points (green triangles)  
   - Original data points (cyan dots)

## 📈 Output
The predicted prices for houses of size 1500, 2000, and 2500 square feet are printed in the console.  
Also, a plot is generated showing the regression line, original data, and predicted points.

## 📌 Key Learning
- How Linear Regression works.  
- Predicting unseen data using a trained model.  
- Visualizing regression results using Matplotlib.

