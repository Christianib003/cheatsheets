
## **Pandas Cheatsheet for Machine Learning**

Below is a **Pandas Cheatsheet for Machine Learning**, designed to be a comprehensive, concise reference for using Pandas in your ML projects. Pandas is a vital Python library for data science and machine learning, enabling efficient data manipulation and analysis through its core structures: Series and DataFrames. This cheatsheet covers essential topics with brief explanations, return values, usage examples, and edge cases where relevant, tailored for beginners and ML practitioners.

---

### **1. Getting Started**
- **Installing Pandas**: Install via the command line.
  ```bash
  pip install pandas
  ```
- **Importing Pandas**: Use the standard alias `pd`.
  ```python
  import pandas as pd
  ```

---

### **2. Data Structures**
Pandas revolves around two key structures:

- **`Series`**
  - **Explanation**: A one-dimensional labeled array, akin to a column or dictionary.
  - **Return**: `pd.Series`
  - **Usage**:
    ```python
    s = pd.Series([10, 20, 30], index=['a', 'b', 'c'])
    # Output: a    10
    #         b    20
    #         c    30
    ```
  - **Edge Case**: Duplicate indices are allowed (e.g., two 'a' indices), but accessing `s['a']` returns all matching values, which may confuse lookups.

- **`DataFrame`**
  - **Explanation**: A two-dimensional table with labeled rows and columns, like an Excel sheet or database table.
  - **Return**: `pd.DataFrame`
  - **Usage**:
    ```python
    df = pd.DataFrame({'Name': ['Mike', 'Bob'], 'Age': [30, 25]})
    # Output:    Name  Age
    #         0  Mike   30
    #         1   Bob   25
    ```
  - **Custom Index**: Set with `df.set_index('Name')` to use 'Name' as the index.

---

### **3. Data Import and Export**
Load and save data in various formats.

- **Reading Data**
  - **`pd.read_csv('file.csv')`**
    - **Return**: `pd.DataFrame`
    - **Usage**: `df = pd.read_csv('file.csv', index_col=0)` sets the first column as the index.
  - **`pd.read_json('file.json')`**
    - **Return**: `pd.DataFrame`
  - **`pd.read_excel('file.xlsx')`**
    - **Return**: `pd.DataFrame`
  - Other formats: `read_html`, `read_sql`, etc.

- **Writing Data**
  - **`df.to_csv('file.csv', index=False)`**
    - **Usage**: Saves without the index to avoid an extra unnamed column on import.
  - **`df.to_json('file.json')`**
  - **`df.to_excel('file.xlsx')`**
  - Other formats: `to_html`, `to_sql`, etc.

---

### **4. Data Exploration**
Inspect and summarize your data.

- **Viewing Data**
  - **`df.head(n=5)`**: First n rows.
  - **`df.tail(n=5)`**: Last n rows.
  - **`df.sample(n=5)`**: Random n rows.
  - **`df.columns`**: List of column names.

- **Getting Information**
  - **`df.info()`**
    - **Return**: Summary of data types, non-null counts.
    - **Usage**: `df.info()` shows missing values and types.
  - **`df.describe()`**
    - **Return**: Stats (count, mean, std, min, max) for numerical columns.

- **Statistical Functions**
  - **`df['column'].mean()`**: Mean.
  - **`df['column'].median()`**: Median.
  - **`df['column'].std()`**: Standard deviation.
  - **`df['column'].min()`**, **`df['column'].max()`**: Min/max values.

---

### **5. Data Access and Manipulation**
Access, filter, and modify data efficiently.

- **Indexing and Selecting**
  - **`df.loc[label]`**
    - **Return**: Row(s) by label.
    - **Usage**: `df.loc['Mike']` if 'Name' is the index.
  - **`df.iloc[position]`**
    - **Return**: Row(s) by integer position.
    - **Usage**: `df.iloc[0]` gets the first row.
  - **`df.at[label, column]`**
    - **Return**: Single value by label (fast).
    - **Usage**: `df.at['Mike', 'Age']` gets 30.
  - **`df.iat[position, column]`**
    - **Return**: Single value by position.
    - **Usage**: `df.iat[0, 1]` gets 30.
  - **Slicing**: `df.iloc[0:2, 0:1]` for rows 0-1, column 0.

- **Filtering and Querying**
  - **Boolean Indexing**: `df[df['Age'] > 30]`
    - **Return**: Rows where condition is true.
  - **`df.query('Age > 30')`**
    - **Return**: Rows matching the query.
    - **Tip**: Faster for large datasets; uses a string syntax.

- **Applying Functions**
  - **`df['column'].apply(func)`**
    - **Return**: Series with function applied.
    - **Usage**: `df['Age'].apply(lambda x: x * 2)` doubles ages.
  - **`df.apply(func, axis=1)`**
    - **Return**: Series applied to rows.
    - **Usage**: `df.apply(lambda row: f"{row['Name']}:{row['Age']}", axis=1)`.

- **Handling Missing Data**
  - **`df.dropna()`**
    - **Return**: Drops rows with NaN.
  - **`df.fillna(value)`**
    - **Return**: Fills NaN with `value`.
    - **Usage**: `df.fillna(df['Age'].mean())` fills with mean age.
  - **`df.isna()`**
    - **Return**: Boolean mask for NaN.

---

### **6. Data Transformation**
Reshape and combine data.

- **Sorting**
  - **`df.sort_values('column')`**
    - **Return**: Sorted DataFrame.
    - **Usage**: `df.sort_values('Age', ascending=False)` for descending order.
  - **`df.sort_index()`**
    - **Return**: Sorted by index.

- **Grouping and Aggregating**
  - **`df.groupby('column').mean()`**
    - **Return**: Mean per group.
    - **Usage**: `df.groupby('Job').mean()` averages ages by job.
  - **`df.groupby('column').agg({'col1': 'mean', 'col2': 'sum'})`**
    - **Return**: Multiple aggregations.

- **Merging, Joining, and Concatenating**
  - **`pd.concat([df1, df2])`**
    - **Return**: Stacks DataFrames (rows by default).
    - **Usage**: `pd.concat([df1, df2], axis=1)` stacks columns.
  - **`pd.merge(df1, df2, on='key')`**
    - **Return**: Merged on a column.
    - **Usage**: `pd.merge(df1, df2, on='Item', how='outer')` includes all rows.
  - **`df1.join(df2)`**
    - **Return**: Joined on index.
    - **Usage**: `df1.join(df2, how='inner')` keeps common indices.

---

### **7. Advanced Topics**
Specialized data handling.

- **Working with Dates and Times**
  - **`pd.to_datetime('date_string')`**
    - **Return**: Datetime object.
  - **`df['date'].dt.year`**
    - **Return**: Extracts year.
    - **Usage**: `df['Birthday'].dt.year > 1950`.

- **Categorical Data**
  - **`df['column'].astype('category')`**
    - **Return**: Converts to categorical type for memory efficiency.

- **String Operations**
  - **`df['column'].str.lower()`**
    - **Return**: Lowercase strings.
  - **`df['column'].str.contains('substring')`**
    - **Return**: Boolean mask for substring presence.

---

### **8. Performance Tips and Edge Cases**
- **Efficient Operations**:
  - Use vectorized operations (e.g., `df['Age'] * 2`) over loops.
  - Prefer `apply` for functions instead of iterating.

- **Common Pitfalls**:
  - **Views vs. Copies**: Chained indexing (e.g., `df['Age'][0]`) may return a copy; use `loc` or `iloc` to modify directly.
  - **Duplicate Indices**: Can lead to multiple rows returned unexpectedly.
  - **Data Types**: Mixed types in a column become `object`, impacting performance.

---

### Resources
- Youtube: [Pandas Full Python Course - NeuralNine](https://youtu.be/EhYC02PD_gc?si=mRvGHUttPf0BTJ3v)

*~Happy Learning!!~*