import pandas as pd

df = pd.read_csv("original.csv")

df["Gender"] = df["Volunteering"].map({0: "Male", 1: "Female"})
df["Ethnicity"] = df["Ethnicity"].map({0: "Caucasian", 1: "African American", 2: "Asian", 3: "Other"})
df["ParentalEducation"] = df["ParentalEducation"].map({0: "None", 1: "High School", 2: "Some College", 3: "Bachelors Degree", 4: "Higher"})
df["Tutoring"] = df["Tutoring"].map({0: "No", 1: "Yes"})
df["ParentalSupport"] = df["ParentalSupport"].map({0: "None", 1: "Low", 2: "Moderate", 3: "High", 4: "Very High"})
df["Extracurricular"] = df["Extracurricular"].map({0: "No", 1: "Yes"})
df["Sports"] = df["Sports"].map({0: "No", 1: "Yes"})
df["Music"] = df["Music"].map({0: "No", 1: "Yes"})
df["Volunteering"] = df["Volunteering"].map({0: "No", 1: "Yes"})
df["GradeClass"] = df["GradeClass"].map({0.0: "A", 1.0: "B", 2.0: "C", 3.0: "D", 4.0: "F"})

df = df.drop(columns=["StudentID"])

df.to_csv("processed.csv", index=False)
