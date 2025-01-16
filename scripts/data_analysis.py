import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("../data/tourism_dataset.csv")

df = df.drop(columns=['Location'])

sns.boxplot(data=df, x='Category', y='Revenue')
plt.show()

sns.boxplot(data=df, x='Category', y='Visitors')
plt.show()

# map countries to numeric values
country_mapping = {
    'India': 1,
    'USA': 2,
    'Brazil': 3,
    'France': 4,
    'Egypt': 5,
    'China': 6,
    'Australia': 7
}
df['Country'] = df['Country'].map(country_mapping)

# map activity categories
category_mapping = {
    'Nature': 1,
    'Historical': 2,
    'Cultural': 3,
    'Beach': 4,
    'Adventure': 5,
    'Urban': 6
}
df['Category'] = df['Category'].map(category_mapping)

accommodation_mapping = {
    'Yes': 1,
    'No': 0
}
df['Accommodation_Available'] = df['Accommodation_Available'].map(accommodation_mapping)

df['Revenue_per_Visitor'] = df['Revenue'] / df['Visitors']

# Pearson correlation matrix
correlation_matrix = df.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Matrix (Pearson)")
plt.show()

# Spearman correlation matrix
correlation_spearman = df.corr(method='spearman')

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_spearman, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Matrix (Spearman)")
plt.show()


# plot revenue by country and category

reverse_country_mapping = {v: k for k, v in country_mapping.items()}
df['Country_Name'] = df['Country'].map(reverse_country_mapping)

revenue_per_country = df.groupby(['Country_Name', 'Category'])['Revenue'].sum().reset_index()

reverse_category_mapping = {v: k for k, v in category_mapping.items()}
revenue_per_country['Category_Name'] = revenue_per_country['Category'].map(reverse_category_mapping)

plt.figure(figsize=(12, 8))
sns.lineplot(
    x='Country_Name',
    y='Revenue',
    hue='Category_Name',
    data=revenue_per_country,
    marker='o'
)

plt.title('Revenue by Country and Category', fontsize=16)
plt.xlabel('Country', fontsize=14)
plt.ylabel('Revenue', fontsize=14)
plt.xticks(rotation=45, fontsize=12)
plt.yticks(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(title='Category', title_fontsize=12, fontsize=10)
plt.tight_layout()
plt.show()
