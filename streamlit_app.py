import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from plotly import graph_objects as go

#Function to process the uploaded Excel file
def load_excel(file):
    try:
        df = pd.read_excel(file)
        st.write("File read successfully")
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return None
    
    required_columns = ['Blade', 'Blade Bearing', 'Generator', 'Main Bearing', 'Transformer', 'Yaw Ring']

    if not all(col in df.columns for col in required_columns):
        st.error(f"Excel file must contain columns: {', '.join(required_columns)}")
        return None
    
    st.write("All required columns are present")
    return df



# Function to detect failure based on input failure rate and randomization 
def failure_detection(input_dict):
    # Create DataFrame and set 'Year' as the index

    input_dict_renamed = {f"{key}_1": value for key, value in input_dict.items()}
    df = pd.DataFrame(input_dict_renamed)
    df['Year'] = [i for i in range(1, len(df) + 1)]
    df.set_index('Year', inplace=True)

    # Prepare a list to track the desired column order
    desired_order = []

    # Add Random and Failure_status columns dynamically
    for column_name in input_dict_renamed.keys():
        random_col = f'Random_{column_name}'  
        status_col = f'Failure_status_{column_name}' 

        # Add new columns with default values
        df[random_col] = np.nan
        df[status_col] = None

        # Append column names in the desired order
    
        desired_order.extend([column_name, random_col, status_col])

    # Ensure the DataFrame columns match the desired order
    df = df[[col for col in desired_order if col in df.columns]]

    df_copy = df.copy()
    for i, year in enumerate(df_copy.index):  # Iterate based on Year index
        for column_name in input_dict_renamed.keys():
            random_value = np.random.rand() * 100
            random_col = f'Random_{column_name}'  # Adding _1 suffix
            status_col = f'Failure_status_{column_name}'  # Adding _1 suffix

            # Set Random value
            df_copy.loc[year, random_col] = random_value

            # Determine Failure status
            if random_value > df_copy.loc[year, column_name] :
                df_copy.loc[year, status_col] = "No_failure"
            else:
                df_copy.loc[year, status_col] = "Failed"

                # Update failure rates for subsequent years
                for j, future_year in enumerate(range(year + 1, len(df_copy) + 1)):
                    df_copy.loc[future_year, column_name] = input_dict_renamed[column_name][j % len(input_dict_renamed[column_name])] 

    return df_copy

# Function to check failure from Number of turbine in the project 
@st.cache_data
def failure_detection_multiple_turbines(input_dict, No_of_turbine):
    # Create an empty DataFrame to hold all turbines' results
    all_turbines_df = pd.DataFrame()

    # Loop over each turbine and apply the failure_detection function
    for turbine_idx in range(1, No_of_turbine + 1):
        # Rename input dictionary for each turbine (e.g., Blade_1, Blade_2, etc.)
        input_dict_turbine = {
            f"{key}_turbine_{turbine_idx}": value
            for key, value in input_dict.items()
        }

        # Call the original failure detection function for each turbine
        turbine_df = failure_detection(input_dict_turbine)

        # Rename the columns dynamically to include turbine index
        turbine_df.columns = [
            col.replace(f"_turbine_{turbine_idx}_1", f"_turbine_{turbine_idx}") for col in turbine_df.columns
        ]


        # Append the result for this turbine to the main DataFrame
        all_turbines_df = pd.concat([all_turbines_df, turbine_df], axis=1)

    return all_turbines_df

# Function to provide the result how many turbine failed per year 
def failure_summary_table(df,input_dict):
    summary = pd.DataFrame(index=df.index.unique())

    for component in input_dict:
        component_columns = [col for col in df.columns if f"Failure_status_{component}" in col]
        
        summary[component] = df[component_columns].apply(lambda row: (row == "Failed").sum(), axis = 1)
    
    return summary


# Function to provide result of total failure per year
def sum_failure_per_year(df):
    df['Total_Failure'] = df.select_dtypes(include=np.number).sum(axis=1)
    return df

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
            input_dict = {col: (df[col] * 100).tolist() for col in ['Blade', 'Blade Bearing', 'Generator', 'Main Bearing', 'Transformer', 'Yaw Ring']}

            # Allow user to specify the number of turbines (iterations)
            n_iterations = st.number_input("Enter the number of turbines of your project", min_value=1, max_value=150, value=30)


            # Run the turbine failure detection
            st.write("Running failure detection...")
            df_result = failure_detection_multiple_turbines(input_dict, No_of_turbine=n_iterations)

            # Count failed turbines per year
            failure_per_component_per_year_df = failure_summary_table(df_result,input_dict)
            failure_per_year_df = sum_failure_per_year(failure_per_component_per_year_df)

            # Show the resulting dataframe
            st.subheader("Failure Counts Per Year")
            st.write(df_result)
            st.write(failure_per_year_df)

            # Plot the results
            fig = px.bar(failure_per_year_df, x=failure_per_year_df.index, y=failure_per_year_df.columns[:-1], title="Stacked Bar Chart of Failures by Component",
                        labels={"value": "Failures", "Year": "Year", "variable": "Component"}, 
                        barmode='stack')
            fig.add_trace(
                go.Scatter(
                    x=failure_per_year_df.index,
                    y=failure_per_year_df["Total_Failure"],
                    mode="text",
                    text=failure_per_year_df["Total_Failure"],
                    textposition="top center",
                    showlegend=False
                )
            )
            st.plotly_chart(fig)


            export_option = st.checkbox("Do you want to export the results to an Excel file?")
            if export_option:
                export_file = pd.ExcelWriter("turbine_failure_analysis.xlsx", engine="openpyxl")

                # Export all relevant DataFrames to separate sheets
                failure_per_year_df.to_excel(export_file, sheet_name="Total Failures")
                
                export_file.save()
                st.success("Excel file exported successfully!")

                # Provide download link
                with open("turbine_failure_analysis.xlsx", "rb") as f:
                    st.download_button(
                        label="Download Excel file",
                        data=f,
                        file_name="turbine_failure_analysis.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )

if __name__ =='__main__':
    main()
