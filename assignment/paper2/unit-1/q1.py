from sklearn.linear_model import LinearRegression
import numpy as np  
import matplotlib.pyplot as plt
#DATA
x=np.array([1,2,3,4,5]).reshape(-1,1)      #Number of hours studied
y=np.array([80,70,67,58,48])  #Marks
model=LinearRegression()
model.fit(x,y)
print("Intercept:",model.intercept_)
print("Coefficient:",model.coef_)
#PREDICTION
pred=model.predict(np.array([[6],[7],[8]]))
print("Predicted Marks for 6,7,8 hours of study:",pred)
#VISUALIZATION
plt.scatter(x,y,color='blue')
plt.plot(x,model.predict(x),color='red')
plt.xlabel('Hours Studied')
plt.ylabel('Marks')
plt.title('Hours Studied vs Marks')
plt.show()
