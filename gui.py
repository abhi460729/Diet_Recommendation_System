from tkinter import *
from tkinter import messagebox
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DietRecommendationSystem:
    def __init__(self, root):
        self.root = root
        self.root.title("DIET RECOMMENDATION SYSTEM")
        self.root.geometry("500x300")
        #self.root.iconbitmap('3/favicon.ico')  # Uncomment if icon file exists

        # GUI Elements
        Label(root, text="Age").grid(row=0, column=0, sticky=W, pady=4)
        Label(root, text="Veg/Non-Veg (0=Non-Veg, 1=Veg)").grid(row=1, column=0, sticky=W, pady=4)
        Label(root, text="Weight (kg)").grid(row=2, column=0, sticky=W, pady=4)
        Label(root, text="Height").grid(row=3, column=0, sticky=W, pady=4)
        Label(root, text="Height Unit").grid(row=4, column=0, sticky=W, pady=4)
        
        self.e1 = Entry(root)
        self.e2 = Entry(root)
        self.e3 = Entry(root)
        self.e4 = Entry(root)
        self.unit_var = StringVar(value="meters")
        
        self.e1.grid(row=0, column=1)
        self.e2.grid(row=1, column=1)
        self.e3.grid(row=2, column=1)
        self.e4.grid(row=3, column=1)
        OptionMenu(root, self.unit_var, "meters", "feet").grid(row=4, column=1)
        
        Button(root, text='Quit', command=root.quit).grid(row=6, column=0, sticky=W, pady=4)
        Button(root, text='Weight Loss', command=lambda: self.process_diet('weight_loss')).grid(row=1, column=4, sticky=W, pady=4)
        Button(root, text='Weight Gain', command=lambda: self.process_diet('weight_gain')).grid(row=2, column=4, sticky=W, pady=4)
        Button(root, text='Healthy', command=lambda: self.process_diet('healthy')).grid(row=3, column=4, sticky=W, pady=4)
        
        self.result_text = Text(root, height=5, width=40)
        self.result_text.grid(row=7, column=0, columnspan=5, pady=10)
        
        # Load and validate data
        try:
            self.data = self.load_data('input.csv')
            self.datafin = self.load_data('inputfin.csv')
            logger.info("Data loaded successfully")
            self.preprocess_food_data()
            logger.info("Food data preprocessed")
            self.cluster_food_data()
            logger.info("Food data clustered")
        except Exception as e:
            logger.error(f"Data loading error: {e}")
            messagebox.showerror("Error", f"Failed to load data: {e}")
            raise

    def load_data(self, file_path):
        try:
            df = pd.read_csv(file_path, on_bad_lines='warn')
            # Validate VegNovVeg
            if 'VegNovVeg' in df.columns:
                if not df['VegNovVeg'].isin([0, 1, pd.NA]).all():
                    logger.warning(f"Invalid VegNovVeg values in {file_path}. Setting invalid to 0.")
                    df['VegNovVeg'] = df['VegNovVeg'].replace(' ', 0).astype(int)
            # Ensure numeric columns
            numeric_cols = ['Calories', 'Fats', 'Proteins', 'Iron', 'Calcium', 'Sodium', 
                           'Potassium', 'Carbohydrates', 'Fibre', 'VitaminD', 'Sugars']
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            return df
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
            raise

    def preprocess_food_data(self):
        self.breakfast_data = self.data['Breakfast'].to_numpy()
        self.lunch_data = self.data['Lunch'].to_numpy()
        self.dinner_data = self.data['Dinner'].to_numpy()
        self.food_items = self.data['Food_items']
        self.veg_nonveg = self.data['VegNovVeg'].to_numpy()

        self.breakfast_food = [self.food_items[i] for i in range(len(self.breakfast_data)) if self.breakfast_data[i] == 1]
        self.lunch_food = [self.food_items[i] for i in range(len(self.lunch_data)) if self.lunch_data[i] == 1]
        self.dinner_food = [self.food_items[i] for i in range(len(self.dinner_data)) if self.dinner_data[i] == 1]
        
        self.breakfast_ids = [i for i in range(len(self.breakfast_data)) if self.breakfast_data[i] == 1]
        self.lunch_ids = [i for i in range(len(self.lunch_data)) if self.lunch_data[i] == 1]
        self.dinner_ids = [i for i in range(len(self.dinner_data)) if self.dinner_data[i] == 1]

        columns = ['Food_items'] + list(self.data.columns[5:15])
        self.breakfast_data_filtered = self.data.iloc[self.breakfast_ids][columns].to_numpy()
        self.lunch_data_filtered = self.data.iloc[self.lunch_ids][columns].to_numpy()
        self.dinner_data_filtered = self.data.iloc[self.dinner_ids][columns].to_numpy()

    def cluster_food_data(self):
        # Cluster Breakfast
        X = self.breakfast_data_filtered[:, 1:]  # Exclude Food_items
        if len(X) > 0:
            kmeans = KMeans(n_clusters=min(3, len(X)), random_state=0).fit(X)
            self.breakfast_labels = kmeans.labels_
        else:
            self.breakfast_labels = []

        # Cluster Lunch
        X = self.lunch_data_filtered[:, 1:]
        if len(X) > 0:
            kmeans = KMeans(n_clusters=min(3, len(X)), random_state=0).fit(X)
            self.lunch_labels = kmeans.labels_
        else:
            self.lunch_labels = []

        # Cluster Dinner
        X = self.dinner_data_filtered[:, 1:]
        if len(X) > 0:
            kmeans = KMeans(n_clusters=min(3, len(X)), random_state=0).fit(X)
            self.dinner_labels = kmeans.labels_
        else:
            self.dinner_labels = []

    def calculate_bmi(self, weight, height, unit):
        if unit == 'feet':
            height = height * 0.3048  # Convert feet to meters
        if height == 0:
            raise ValueError("Height cannot be zero")
        bmi = weight / (height ** 2)
        if not 15 <= bmi <= 50:  # Relaxed BMI range
            raise ValueError(f"BMI {bmi:.2f} out of realistic range (15–50)")
        return bmi

    def process_diet(self, goal):
        self.result_text.delete(1.0, END)
        try:
            age = int(self.e1.get())
            veg = int(self.e2.get())
            weight = float(self.e3.get())
            height = float(self.e4.get())
            unit = self.unit_var.get()
            
            if not (0 <= age <= 120):
                raise ValueError("Age must be between 0 and 120")
            if veg not in [0, 1]:
                raise ValueError("Veg/Non-Veg must be 0 (Non-Veg) or 1 (Veg)")
            if weight <= 0:
                raise ValueError("Weight must be positive")
            if height <= 0:
                raise ValueError("Height must be positive")

            bmi = self.calculate_bmi(weight, height, unit)
            age_class = min(age // 20, 4)  # Age classes: 0-20, 20-40, 40-60, 60-80, 80+
            
            if bmi < 16:
                bmi_class = 4  # Severely underweight
                status = "severely underweight"
            elif 16 <= bmi < 18.5:
                bmi_class = 3  # Underweight
                status = "underweight"
            elif 18.5 <= bmi < 25:
                bmi_class = 2  # Healthy
                status = "healthy"
            elif 25 <= bmi < 30:
                bmi_class = 1  # Overweight
                status = "overweight"
            else:
                bmi_class = 0  # Severely overweight
                status = "severely overweight"
            
            self.result_text.insert(END, f"Your BMI: {bmi:.2f} ({status})\n")
            self.result_text.insert(END, f"Age class: {age_class * 20}-{(age_class + 1) * 20}\n")
            
            # Prepare data for Random Forest
            data_tog = self.datafin.T
            if goal == 'weight_loss':
                indices = [1, 2, 7, 8]  # Calories, Fats, Carbohydrates, Fibre
                labels = self.breakfast_labels
                food_ids = self.breakfast_ids
            elif goal == 'weight_gain':
                indices = [0, 1, 2, 3, 4, 7, 9, 10]  # All nutrients
                labels = self.lunch_labels
                food_ids = self.lunch_ids
            else:  # healthy
                indices = [1, 2, 3, 4, 6, 7, 9]  # Calories, Fats, Proteins, Iron, Potassium, Carbohydrates, Vitamin D
                labels = self.dinner_labels
                food_ids = self.dinner_ids
            
            cat_data = data_tog.iloc[indices].T.to_numpy()[1:]
            X_train = np.zeros((len(cat_data) * 5, len(indices) + 2), dtype=np.float32)
            y_train = []
            bmi_classes = [0, 1, 2, 3, 4]
            age_classes = [0, 1, 2, 3, 4]
            
            t = 0
            for zz in range(5):
                for jj in range(len(cat_data)):
                    valloc = list(cat_data[jj])
                    valloc.append(bmi_classes[zz])
                    valloc.append(age_classes[zz])
                    X_train[t] = np.array(valloc)
                    y_train.append(labels[jj % len(labels)])
                    t += 1
            
            X_test = np.zeros((len(cat_data), len(indices) + 2), dtype=np.float32)
            for jj in range(len(cat_data)):
                valloc = list(cat_data[jj])
                valloc.append(age_class)
                valloc.append(bmi_class)
                X_test[jj] = np.array(valloc) * ((bmi_class + age_class) / 2)
            
            clf = RandomForestClassifier(n_estimators=100, random_state=0)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            
            self.result_text.insert(END, "SUGGESTED FOOD ITEMS:\n")
            for ii in range(len(y_pred)):
                if y_pred[ii] == 2:  # High-calorie cluster (arbitrary choice)
                    food_item = self.food_items[food_ids[ii]]
                    if veg == 1 and self.veg_nonveg[food_ids[ii]] == 1:
                        self.result_text.insert(END, f"- {food_item}\n")
                    elif veg == 0:
                        self.result_text.insert(END, f"- {food_item}\n")
        
        except ValueError as e:
            logger.error(f"Input error: {e}")
            messagebox.showerror("Error", f"Invalid input: {e}")
        except Exception as e:
            logger.error(f"Processing error: {e}")
            messagebox.showerror("Error", f"Processing error: {e}")

if __name__ == "__main__":
    main_win = Tk()
    app = DietRecommendationSystem(main_win)
    main_win.mainloop()