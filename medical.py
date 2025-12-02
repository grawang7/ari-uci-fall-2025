import kagglehub
from kagglehub import KaggleDatasetAdapter

file_path = r"Disease_symptom_and_patient_profile_dataset.csv"
df = kagglehub.load_dataset(
  KaggleDatasetAdapter.PANDAS,
  "uom190346a/disease-symptoms-and-patient-profile-dataset",
  file_path,
)

i = 100
d = df.head(i)
query = ""
for j in range(i):
    keys = d.iloc[j]
    info = keys.to_dict()

    if info["Fatigue"]:
        fatigue = "tired"
    else:
        fatigue = "not tired"

    if info["Fever"]:
        fever = "fever"
    else:
        fever = "no fever"

    if info["Difficulty Breathing"]:
        breathing = "have"
    else:
        breathing = "don't have"

    if info["Cough"]:
        cough = "have"
    else:
        cough = "don't have"
    query += f"""I am a {fatigue} {info["Age"]} year old {info["Gender"]} with {info["Cholesterol Level"]} cholesterol, {info["Blood Pressure"]} blood pressure, {info["Disease"]}, and with {fever}. I also {breathing} difficulty breathing and I {cough} a cough."""
