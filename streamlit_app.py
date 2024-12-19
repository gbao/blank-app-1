import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

#Function to process the uploaded Excel file
def load_excel(file):
    try:
        df = pd.read_excel(file)
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return None
    
    if 'Blade' not in df.columns:
        st.error("Excel file must contain a 'Blade' column")
        return None
    
    return df

# Function to extract the input list from the 'Blade' column
def get_input_list_from_df(df, column_name="Blade"):
    return df[column_name].tolist()

# Function to detect failure based on input failure rate and randomization 
def failure_detection(input_list, n_iteration, column_name="Blade"):
    
    df = pd.DataFrame(input_list, columns=[column_name])
    df["Blade"] = df["Blade"]*100
    df['Year'] = [i for i in range(1, len(df) + 1)]
    df['Random'] = None
    df['Failure_status'] = None

    df_copy = df.copy()
    for i in range(0, len(df_copy)):
        df_copy.loc[i, 'Random'] = np.random.rand() * 100
        if df_copy.loc[i, "Random"] > df_copy.loc[i, column_name]:
            df_copy.loc[i, "Failure_status"] = "No_failure"
        else:
            df_copy.loc[i, "Failure_status"] = "Failed"

            for j, k in enumerate(range(i + 1, len(df_copy))):
                df_copy.loc[k, column_name] = input_list[j % len(input_list)]

    return df_copy

# Function to check failure from Number of turbine in the project 
@st.cache_data
def run_multiple_iterations(input_list,n_iteration, column_name="Blade"):
    df_result = pd.DataFrame()
    for iteration in range(1, n_iteration + 1):
        # Run the failure detection function
        temp_df = failure_detection(input_list, n_iteration, column_name=column_name)

        # Dynamically create a new column name for failure status
        failure_status_col = f"failure_status_{iteration}"
        temp_df = temp_df.rename(columns={"Failure_status": failure_status_col})

        # Add the results to df_result for each iteration
        if df_result.empty:
            df_result = temp_df[['Year', column_name, 'Random', failure_status_col]].copy()
        else:
            df_result[f'Blade_{iteration}'] = temp_df[column_name]
            df_result[f'Random_{iteration}'] = temp_df['Random']  # Add Random values too (optional)
            df_result[failure_status_col] = temp_df[failure_status_col]

    return df_result

# Function to provide the result how many turbine failed per year 
def count_failed_turbines_per_year(df, n_iterations):
    # Create an empty list to store the failure counts for each year
    failure_count_per_year = []

    # Iterate over each row (representing each year)
    for index, row in df.iterrows():
        # Count failures in each year (row) across all failure status columns
        failed_count = 0
        for iteration in range(1, n_iterations + 1):
            failure_status_col = f"failure_status_{iteration}"
            if row[failure_status_col] == "Failed":
                failed_count += 1
        failure_count_per_year.append(failed_count)

    # Add the failure count to the dataframe as a new column
    df['Failed_Turbines'] = failure_count_per_year
    return df[['Year', 'Failed_Turbines']]

# Streamlit app starts here
def main():
    st.title("Turbine Failure Detection")

    # User inputs
    uploaded_file = st.file_uploader("Upload an Excel file", type=["xlsx"])

    if uploaded_file is not None:
        # Load the Excel file
        df = load_excel(uploaded_file)
        if df is not None:
            # Show the first few rows of the file to the user
            st.subheader("Uploaded Data Preview")
            st.write(df.head())

            # Extract the Blade column and convert it to a list
            input_list = get_input_list_from_df(df, column_name= "Blade")

            # Allow user to specify the number of turbines (iterations)
            n_iterations = st.number_input("Enter the number of turbines", min_value=1, max_value=100, value=30)


            # Run the turbine failure detection
            st.write("Running failure detection...")
            df_result = run_multiple_iterations(input_list, column_name="Blade", n_iteration=n_iterations)

            # Count failed turbines per year
            failure_per_year_df = count_failed_turbines_per_year(df_result, n_iterations)

            # Show the resulting dataframe
            st.subheader("Failure Counts Per Year")
            st.write(df_result)
            st.write(failure_per_year_df)

            # Plot the results
            fig = px.bar(failure_per_year_df, x="Year", y="Failed_Turbines", title="Failed Turbines per Year")
            st.plotly_chart(fig)

if __name__ =='__main__':
    main()
