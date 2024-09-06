import re
import pandas as pd
import matplotlib.pyplot as plt

# Read the log file path
log_file_path = './text_eeg_multimodal_model.log'

# Define regular expressions to capture fusion_strategy and final test results
fusion_strategy_pattern = r"fusion_strategy='([\w\-]+)'"
test_acc_pattern = r"Test Acc: ([\d\.]+)"
test_f1_pattern = r"Test F1_weighted: ([\d\.]+)"

# Lists to store the extracted data
strategies = []
test_accs = []
test_f1s = []

# Read and process the log file
with open(log_file_path, 'r') as file:
    current_strategy = None
    for line in file:
        # Capture fusion_strategy
        strategy_match = re.search(fusion_strategy_pattern, line)
        if strategy_match:
            current_strategy = strategy_match.group(1)

        # Capture Test Acc
        test_acc_match = re.search(test_acc_pattern, line)
        if test_acc_match:
            test_accs.append(float(test_acc_match.group(1)))
            strategies.append(current_strategy)  # Record the current strategy

        # Capture Test F1_weighted
        test_f1_match = re.search(test_f1_pattern, line)
        if test_f1_match:
            test_f1s.append(float(test_f1_match.group(1)))

# Create a DataFrame to store the extracted data
df = pd.DataFrame({
    'Fusion Strategy': strategies,
    'Test Accuracy (%)': test_accs,
    'Test F1 Weighted (%)': test_f1s
})

# Visualize the final test metrics, categorized by fusion_strategy
plt.figure(figsize=(12, 8))

# Plot grouped bar chart for Test Accuracy and Test F1 Weighted
ax = df.groupby('Fusion Strategy').mean().plot(kind='bar', figsize=(12, 8))

plt.title('Final Test Accuracy and F1 Weighted Scores by Fusion Strategy')
plt.xlabel('Fusion Strategy')
plt.ylabel('Percentage (%)')
plt.xticks(rotation=45)
plt.legend(loc='best')

# Add specific values on top of the bars
for p in ax.patches:
    ax.annotate(f'{p.get_height():.2f}',
                (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center',
                xytext=(0, 9),
                textcoords='offset points')
plt.show()

# Create a summary table of the average results grouped by fusion strategy, and calculate the maximum values
summary_df = df.groupby('Fusion Strategy').agg({
    'Test Accuracy (%)': ['mean', 'max'],
    'Test F1 Weighted (%)': ['mean', 'max']
}).reset_index()

# Rename columns for readability
summary_df.columns = ['Fusion Strategy', 'Mean Test Accuracy (%)', 'Max Test Accuracy (%)',
                      'Mean Test F1 Weighted (%)', 'Max Test F1 Weighted (%)']

# Print the table to view the results
print(summary_df.to_string())
