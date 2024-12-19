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
    
    required_columns = ['Blade','Blade Bearing','Generator','Main Bearing','Transformer','Yaw Ring' ]

    if not all(col in df.columns for col in required_columns):
        st.error(f"Excel file must contain columns: {', '.join(required_columns)}")
        return None
    
    return df



# Function to detect failure based on input failure rate and randomization 
def failure_detection(input_dict, n_iteration, column):
    
    df = pd.DataFrame(input_dict)
    df['Year'] = [i for i in range(1, len(df) + 1)]

    for column_name in input_dict.keys():
        df[f'Random_{column_name}'] = None
        df[f'Failure_status_{column_name}'] = None


    df_copy = df.copy()
    for i in range(0, len(df_copy)):
        for column_name in input_dict.keys():
            random_value = np.random.rand() * 100
            df_copy.loc[i ,f'Random_{column_name}'] = random_value

            if random_value > df_copy.loc[i, column_name] * 100:
                df_copy.loc[i, f'Failure_status_{column_name}'] = "No_failure"
            else:
                df_copy.loc[i, f'Failure_status_{column_name}'] = "Failed"

                for j, k in enumerate(range(i + 1, len(df_copy))):
                    df_copy.loc[k, column_name] = input_dict[j % len(input_dict)] *100

    return df_copy

# Function to check failure from Number of turbine in the project 
@st.cache_data
def run_multiple_iterations(input_dict,n_iteration):
    df_result = pd.DataFrame()
    for iteration in range(1, n_iteration + 1):
        # Run the failure detection function
        temp_df = failure_detection(input_dict, n_iteration,column_name)

        if df_result.empty:
            df_result = temp_df.copy()
        else:
            for column_name in input_dict.keys():
                df_result[f'{column_name}_{iteration}'] = temp_df[column_name]
                df_result[f'Random_{column_name}_{iteration}'] = temp_df[f'Random_{column_name}']
                df_result[f'Failure_status_{column_name}_{iteration}'] = temp_df[f'Failure_status_{column_name}']
    return df_result

# Function to provide the result how many turbine failed per year 
def count_failed_turbines_per_year(df,input_dict, n_iterations):
    result = {}
    for column_name in input_dict.keys():
        failure_counts = []
        for index, row in df.iterrows():
            failed_count = 0
            for iteration in range(1, n_iterations + 1):
                failure_status_col = f'Failure_status_{column_name}_{iteration}'
                if row[failure_status_col] == "Failed":
                    failed_count += 1
            failure_counts.append(failed_count)

        result[column_name] = failure_counts

    # Combine results into a single dataframe
    final_df = pd.DataFrame(result)
    final_df['Year'] = df['Year']
    final_df['Failed_Turbines'] = final_df.sum(axis=1)
    return final_df

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
            input_dict = {col: df[col].tolist() for col in ['Blade', 'Blade Bearing', 'Generator', 'Main Bearing', 'Transformer', 'Yaw Ring']}

            # Allow user to specify the number of turbines (iterations)
            n_iterations = st.number_input("Enter the number of turbines", min_value=1, max_value=100, value=30)


            # Run the turbine failure detection
            st.write("Running failure detection...")
            df_result = run_multiple_iterations(input_dict, column_name="Blade", n_iteration=n_iterations)

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
