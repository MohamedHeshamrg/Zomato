import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import seaborn as sns
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer


def read_data():
    df = pd.read_parquet(r"H:\ML Intern\zomato\data\raw\zomato.parquet")
    print(f"The data has been fully collected ✅. \nShape: {df.shape}\n")
    return df



def show_data(df):
    # First 10 rows
    print("\n🚩First 10 rows : \n")
    print(df.head(10))
    print()

    # Last 10 rows 
    print("\n🚩last 10 rows : \n")
    print(df.tail(10))
    print()

    # Data frame info
    print('\n📌Data fram info : \n')
    print(df.info())
    print()



def describe_df(df):
    # Describe of numerical columns
    print('\n📌Describe of numerical columns : \n')
    print(df.describe().T)
    print()

    # Describe of catgorical columns
    print('\n📌Describe of catgorical columns : \n')
    print(df.describe(include = "O").T)
    print()



def missing_values(df):
    print("\n🔍 Null values ratio per column:\n\n")
    for col in df.columns :
        print(f"Column : {col}")
        print(f"Missing values count = {df[col].isna().sum()}")
        print(f"Missing % = {(df[col].isna().sum()/len(df))*100}")
        print("="*50 + '\n')



# function to calculate average reviews
def average_rating(reviews):
    """
    reviews: list of tuples like [('Rated 1.0', 'text'), ('Rated 4.5', 'text')]
    returns: float average rating
    """

    if isinstance(reviews, float) and np.isnan(reviews):
        return np.nan

    ratings = []
    
    for item in reviews:
        try:
            rate_str = item[0]  # "Rated 1.0"
            rate = float(rate_str.replace("Rated", "").strip())
            
  #Correcting incorrect values
            if rate < 0: 
                rate = 0
            if rate > 5:
                rate = 5

            ratings.append(rate)
        except:
            continue
    
    if not ratings:
        return np.nan

    return np.mean(ratings)

def outliers_Checking(df):
    # check numerical columns to know how i handling
    numeric_columns = [ 'rate', 'votes','approx_cost(for two people)']
    print('\n📌Boxplt for Numerical columns\n\n')
    for col in numeric_columns :
        sns.boxplot(data = df , x = col) # boxplot figer
        plt.show()
    print()
    print()


    # check outlier percentage in num columns
    print("\n🔍 outlier values ratio per column:\n\n")
    for col in numeric_columns:
        Q1 = df[col].quantile(0.25) 
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]

        print(f"Column: {col}")
        print(f"Number of outliers: {len(outliers)}")
        print(f'percentage of outlier = {(len(outliers)/len(df))*100}')
        print("-" * 20)
    print()
    print()
# Define Function to convert all Phone availabilty to "count of phone lines in each resturant" - For testing result
def phone(x):
    return 0 if pd.isna(x) else len(x.split('\r\n'))

# Define Function to convert all menu_item availabilty to "count of menu_item in each resturant" - For testing result
def menu(x):
    return 1 if pd.notna(x) else 0



def rest_groups(df):


# MultiLabelBinarizer Configuration
    mlb = MultiLabelBinarizer()
    rest_type_dummies = pd.DataFrame(mlb.fit_transform(df['rest_type']),
                                columns=mlb.classes_,
                                index=df.index)

# Restaurant Type Groups
    rest_type_groups = {
    'Rest_SweetOrBakery': ['Bakery', 'Confectionery', 'Dessert Parlor', 'Sweet Shop'],
    'Rest_Drink_Oriented_Establishments': ['Bar', 'Club', 'Lounge', 'Microbrewery', 'Pub'],
    'Rest_Specialty_Shops': ['Beverage Shop', 'Bhojanalya', 'Food Truck', 'Irani Cafee', 'Kiosk', 'Meat Shop'],
    'Rest_Dining_Establishments': ['Cafe', 'Casual Dining', 'Dhaba', 'Fine Dining', 'Food Court', 'Mess', 'Quick Bites'],
    'Rest_Takeaway_and_Delivery': ['Delivery', 'Takeaway']
        }

# Grouping columns by group
    grouped_df = pd.DataFrame(index=df.index)
    for group_name, types_list in rest_type_groups.items():
        existing_cols = [col for col in types_list if col in rest_type_dummies.columns]
        
# Add the items and then convert them to 0/1
        grouped_df[group_name] = (rest_type_dummies[existing_cols].sum(axis=1) > 0).astype(int)

# Merge results with original DataFrame
    df = pd.concat([df, grouped_df], axis=1)
    return df

def cuisine_groups(df):

# MultiLabelBinarizer Configuration

    mlb = MultiLabelBinarizer()
    cuisines_dummies = pd.DataFrame(mlb.fit_transform(df['cuisines']),
                                columns=mlb.classes_,
                                index=df.index)

# Kitchen Sets
    cuisine_groups = {
    'Total_North_Indian': ['Awadhi', 'Bihari', 'Kashmiri', 'Lucknowi', 'Mughlai', 'Punjabi', 'Rajasthani','North Indian'],
    'Total_South_Indian': ['Andhra', 'Chettinad', 'Hyderabadi', 'Kerala', 'Mangalorean', 'Tamil', 'Udupi','South Indian'],
    'Total_East_Indian':  ['Assamese', 'Bengali', 'Oriya','East Indian'],
    'Total_West_Indian':  ['Gujarati', 'Maharashtrian', 'Goan', 'Sindhi','West Indian'],
    'Total_International': ['American', 'British', 'French', 'German', 'Greek', 'Italian' , 'Lebanese',
                            'Mediterranean', 'Mexican', 'Portuguese', 'Spanish', 'Turkish', 'Vietnamese', 'Russian',
                              'South American', 'Sri Lankan', 'Tibetan', 'Middle Eastern', 'Asian', 'European'],
    'Total_Asian': ['Burmese', 'Cantonese', 'Chinese', 'Indonesian', 'Korean', 'Mongolian', 'Nepali', 
                          'Pan Asian', 'Singaporean', 'Thai', 'Tibetan', 'Vietnamese', 'Japanese'],
    'Total_Grill/BBQ/Bar_Food': ['BBQ', 'Bar Food', 'Charcoal Chicken', 'Grill', 'Kebab', 'Roast Chicken', 'Rolls', 'Steak'],
    'Total_Fast_Food': ['Burger', 'Fast Food', 'Finger Food', 'Hot dogs', 'Sandwich', 'Street Food', 'Wraps', 'Tex-Mex'],
    'Total_Beverages/Desserts': ['Beverages', 'Bubble Tea', 'Coffee', 'Desserts', 'Ice Cream', 'Juices', 'Mithai', 'Paan', 'Tea'],
    'Total_Healthy/Fusion': ['Healthy Food', 'Vegan', 'Modern Indian', 'Salad'],
    'Total_Bakery': ['Bakery', 'Parsi', 'Cake', 'Sweets']
            }

# Grouping columns by group

    grouped_cuisines = pd.DataFrame(index=df.index)

    for group_name, types_list in cuisine_groups.items():
    # Existing columns only
        existing_cols = [col for col in types_list if col in cuisines_dummies.columns]

# Add the items and then convert them to 0/1

        grouped_cuisines[group_name] = (cuisines_dummies[existing_cols].sum(axis=1) > 0).astype(int)

# Merge results with original DataFrame
    df = pd.concat([df, grouped_cuisines], axis=1)
    return df


def success(x): # new columns to predict restorant successful
    if x >= 3.75:
        return 1
    else:
        return 0
    

    
    
def data_split(df):
    df = df.copy()
    df.drop([
        'name','rate','votes','listed_in(city)',
        'Rest_SweetOrBakery','Rest_Drink_Oriented_Establishments',
        'Rest_Specialty_Shops','Rest_Dining_Establishments',
        'Rest_Takeaway_and_Delivery','Total_North_Indian',
        'Total_South_Indian','Total_East_Indian','Total_West_Indian',
        'Total_International','Total_Asian','Total_Grill/BBQ/Bar_Food',
        'Total_Fast_Food','Total_Beverages/Desserts',
        'Total_Healthy/Fusion','Total_Bakery'
    ], axis=1, inplace=True)
    
    X = df.drop("success_score", axis=1)
    Y = df['success_score']

    

    # Split to Train / Test
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.25, random_state=38
    )

    # Split Train → Train / Validation
    X_train, X_val, Y_train, Y_val = train_test_split(
        X_train, Y_train, test_size=0.05, random_state=38
    )

    return X_train, X_val, X_test, Y_train, Y_val, Y_test



def preprosses():


    # 1) Categorical normal columns
    cat_col = ['location', 'listed_in(type)', 'cost_category']

    # 2) Binary Yes/No columns
    yes_no_col = ["online_order", "book_table"]

    # 3) Multi-category phone columns (not just Yes/No)
    multi_cat_col = ["phone", "menu_item"]   # Example

    # 4) Numerical column
    num_col = ["approx_cost(for two people)"]

    preprocessor = ColumnTransformer(transformers=[

        # One-hot for multi-category
        ('one_hot_main', OneHotEncoder(handle_unknown='ignore'), cat_col),

        ('one_hot_multi', OneHotEncoder(handle_unknown='ignore'), multi_cat_col),

        # Ordinal for Yes/No
        ('ordinal_yes_no',
         OrdinalEncoder(categories=[["No", "Yes"]]*len(yes_no_col)),
         yes_no_col),

        # Numerical
        ('num_scaler', RobustScaler(), num_col)

    ],
    remainder='drop')

    return preprocessor
    











def save_datafile(data, name):
    folder = r"H:\ML Intern\zomato\data\preprocessed"
    os.makedirs(folder, exist_ok=True)

    parquet_file = fr"{folder}\{name}.parquet"

    # Convert DataFrame to Parquet using PyArrow
    table = pa.Table.from_pandas(data, preserve_index=False)

    # Save file
    pq.write_table(table, parquet_file, compression="snappy")

    print(f"✔ Saved {name}.parquet | Shape: {data.shape}")



def run_data_pipeline():
    read_data()
    dataframe = read_data()
    df = dataframe.copy()

    show_data(df)
    describe_df(df)
    missing_values(df)
    df.loc[df['reviews_list'] == "[]", 'reviews_list'] = np.nan
    df.loc[df['menu_item'] == "[]", "menu_item"] = np.nan
    df.loc[df['rate'] == "NEW", "rate"] = np.nan
    df["approx_cost(for two people)"] = (df["approx_cost(for two people)"].str.replace(",", "", regex=False).astype(float))
    df["rate"] = df["rate"].replace("-", np.nan)       
    df["rate"] = df["rate"].str.split("/").str[0]  

    print('Wrong values was changed✅')


    df["approx_cost(for two people)"]= pd.to_numeric(df["approx_cost(for two people)"], errors='coerce')
    df["rate"] = df["rate"].astype(float)   


    df.dropna(subset=["rest_type", "cuisines", "approx_cost(for two people)","phone"], inplace=True)
    df.drop(["url", "address","dish_liked"], axis=1, inplace=True)
    df.drop(df[(df['votes'] == 0) & (df['reviews_list'].isna())].index , inplace=True) 





    # Separate the rows that have votes = 0
    df1 = df[df['votes'] == 0].copy()   
    # Delete rows from the original df
    df = df[df['votes'] != 0].reset_index(drop=True)
    # Calculating the average rating in df1
    df1.loc[:, "rate"] = df1["reviews_list"].apply(average_rating)

    # Data Reintegration
    df = pd.concat([df, df1], ignore_index=True)

    df.reset_index(drop=True, inplace=True)

    df.loc[df['votes'] == 0, "votes"] = np.nan
    df.dropna(subset=["rate"], inplace = True)
    df['votes'] = df['votes'].fillna(df['votes'].mean()) # fill missing with mean
    df.drop_duplicates()


    outliers_Checking(df)

    df["cost_category"] = pd.qcut(df["approx_cost(for two people)"], 3, labels=["Low","Medium","High"])    
    df['success_score'] = df.rate.apply(success)
    df['phone'] = df.phone.apply(phone)
    df['menu_item'] = df['menu_item'].apply(menu)
    # Separate the types of restaurants for each class
    df['rest_type'] = df['rest_type'].str.split(', ').apply(lambda x: [c.strip() for c in x])
    df = rest_groups(df)

    # Separate the kitchens for each row and convert them into a menu
    df['cuisines'] = df['cuisines'].str.split(',').apply(lambda x: [c.strip() for c in x])
    df = cuisine_groups(df)
    df['menu_item'] = df["menu_item"].map({1: "Have menu", 0: "Doesn't have menu"})
    df['phone'] =  df['phone'].map({1: "Have one phone", 2: "Have two phone"})


    df.reset_index()
    df.drop(['reviews_list','rest_type','cuisines'],axis = 1 , inplace = True) # drop unusing columns
    Data = df.copy()

    X_train, X_val, X_test, Y_train, Y_val, Y_test = data_split(Data)

    datasets = {
    "data": Data,
    "X_train": X_train,
    "X_val": X_val,
    "X_test": X_test,
    "Y_train": Y_train,
    "Y_val": Y_val,
    "Y_test": Y_test
}

    for name, df in datasets.items():
        if isinstance(df, pd.Series):
            df = df.to_frame(name) 
        save_datafile(df, name)



if __name__ == "__main__":
    run_data_pipeline()