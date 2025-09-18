import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
x=np.array([500,1000]).reshape(-1,1)#house size in square foot
y=np.array([10,20])#cost per square
model=LinearRegression()
model.fit(x,y)
x_line=np.linspace(500,2500,100).reshape(-1,1)
y_line=model.predict(x_line)
house_size=np.array([[1500],[2000],[2500]])
predicted_price=model.predict(house_size)
print("The predicted_price of the houses of 1500,2000,2500 square foot is:",predicted_price)
plt.plot(x_line,y_line,color="red",label="Regression Line")
plt.scatter(house_size,predicted_price,color="lime",marker='^',s=300,label="Predicted data")
plt.scatter(x,y,color="cyan",label="Original data")
plt.xlabel("Size of houses in square foot")
plt.ylabel("Cost of house per square foot")
plt.title("Linear Reggression model of house size and House price")
plt.legend()
plt.show()