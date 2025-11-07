#import necessary libraries
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#make sure data folder exists (this creates the folder if it doesn't exist)
os.makedirs("../data", exist_ok=True)

#now lets create a sample dataset for 1000 students
n = 1000
np.random.seed(42)  # for reproducibility

#generate random study behavior and scores
hours = np.clip(np.random.normal(2.5, 1.2, n), 0, 8)  # average hours studied per day
attendance = np.clip(np.random.normal(88, 8, n), 50, 100)  # attendance percentage
previous_scores = np.clip(np.random.normal(65, 10, n), 30, 100)  # previous exam scores
anxiety = np.random.randint(1, 6, n)
extracurricular = np.random.choice(["Yes", "No"], size=n, p=[0.4, 0.6])  # 40% participate in extracurricular activities

#create final scores based on a combination of factors
final_score = 5*hours + 0.4*attendance + 0.3*previous_scores - 2*(anxiety - 3)

#add some random variation (noise)
final_score = np.clip(final_score + np.random.normal(0, 5, n), 0, 100)

#combine everything into a DataFrame
df = pd.DataFrame({
    "Hours_Studied": hours,
    "Attendance_Percentage": attendance,
    "Previous_Scores": previous_scores,
    "Anxiety_Level": anxiety,
    "Extracurricular_Activities": extracurricular,
    "Final_Score": final_score
})

#save the dataset to our data folder
df.to_csv("../data/student_performance.csv", index=False)

#print first few rows of the dataset
print(df.head())

#show basic statistics
print(df.describe())

# Visualize: distribution of scores
plt.figure(figsize=(6,4))
plt.hist(df["Final_Score"], bins=20, color="skyblue", edgecolor="black")
plt.title("Distribution of Final Scores")
plt.xlabel("Score")
plt.ylabel("Number of Students")
plt.show()

# Visualize: hours studied vs final score
plt.figure(figsize=(6,4))
plt.scatter(df["Hours_Studied"], df["Final_Score"], alpha=0.6, color="green")
plt.title("Study Hours vs Final Score")
plt.xlabel("Hours Studied per Day")
plt.ylabel("Final Score")
plt.show()


