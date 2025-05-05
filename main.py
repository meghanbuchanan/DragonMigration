import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from numpy.linalg import matrix_power

def load_data():
    return pd.read_csv('data.csv')

def desirability(temp, pop_density, temp_pref=80, w_temp=1.0, w_pop=0.5):
    temp_score = w_temp * (1 / (1 + (temp - temp_pref)**2))
    pop_score = w_pop * (1 / (1 + pop_density))
    return temp_score + pop_score

def get_state_neighbors():
    return {
        'Alabama': ['Mississippi', 'Tennessee', 'Florida', 'Georgia'],
        # 'Alaska': [],
        'Arizona': ['Nevada', 'Colorado', 'Utah', 'New Mexico', 'California'],
        'Arkansas': ['Oklahoma', 'Tennessee', 'Texas', 'Louisiana', 'Mississippi', 'Missouri'],
        'California': ['Oregon', 'Arizona', 'Nevada'],
        'Colorado': ['New Mexico', 'Oklahoma', 'Utah', 'Wyoming', 'Arizona', 'Kansas', 'Nebraska'],
        'Connecticut': ['New York', 'Rhode Island', 'Massachusetts'],
        'Delaware': ['New Jersey', 'Pennsylvania', 'Maryland'],
        'Florida': ['Georgia', 'Alabama'],
        'Georgia': ['North Carolina', 'South Carolina', 'Tennessee', 'Alabama', 'Florida'],
        # 'Hawaii': [],
        'Idaho': ['Utah', 'Washington', 'Wyoming', 'Montana', 'Nevada', 'Oregon'],
        'Illinois': ['Kentucky', 'Missouri', 'Wisconsin', 'Indiana', 'Iowa', 'Michigan'],
        'Indiana': ['Michigan', 'Ohio', 'Illinois', 'Kentucky'],
        'Iowa': ['Nebraska', 'South Dakota', 'Wisconsin', 'Illinois', 'Minnesota', 'Missouri'],
        'Kansas': ['Nebraska', 'Oklahoma', 'Colorado', 'Missouri'],
        'Kentucky': ['Tennessee', 'Virginia', 'West Virginia', 'Illinois', 'Indiana', 'Missouri', 'Ohio'],
        'Louisiana': ['Texas', 'Arkansas', 'Mississippi'],
        'Maine': ['New Hampshire'],
        'Maryland': ['Virginia', 'West Virginia', 'Delaware', 'Pennsylvania'],
        'Massachusetts': ['New York', 'Rhode Island', 'Vermont', 'Connecticut', 'New Hampshire'],
        'Michigan': ['Ohio', 'Wisconsin', 'Illinois', 'Indiana', 'Minnesota'],
        'Minnesota': ['North Dakota', 'South Dakota', 'Wisconsin', 'Iowa', 'Michigan'],
        'Mississippi': ['Louisiana', 'Tennessee', 'Alabama', 'Arkansas'],
        'Missouri': ['Nebraska', 'Oklahoma', 'Tennessee', 'Arkansas', 'Illinois', 'Iowa', 'Kansas', 'Kentucky'],
        'Montana': ['South Dakota', 'Wyoming', 'Idaho', 'North Dakota'],
        'Nebraska': ['Missouri', 'South Dakota', 'Wyoming', 'Colorado', 'Iowa', 'Kansas'],
        'Nevada': ['Arizona', 'Oregon', 'Utah', 'Idaho', 'California'],
        'New Hampshire': ['Massachusetts', 'Vermont', 'Maine'],
        'New Jersey': ['Pennsylvania', 'Delaware', 'New York'],
        'New Mexico': ['Oklahoma', 'Texas', 'Utah', 'Arizona', 'Colorado'],
        'New York': ['Pennsylvania', 'Rhode Island', 'Vermont', 'Connecticut', 'Massachusetts', 'New Jersey'],
        'North Carolina': ['Tennessee', 'Virginia', 'Georgia', 'South Carolina'],
        'North Dakota': ['South Dakota', 'Minnesota', 'Montana'],
        'Ohio': ['Michigan', 'Pennsylvania', 'West Virginia', 'Indiana', 'Kentucky'],
        'Oklahoma': ['Missouri', 'New Mexico', 'Texas', 'Arkansas', 'Colorado', 'Kansas'],
        'Oregon': ['Nevada', 'Washington', 'California', 'Idaho'],
        'Pennsylvania': ['New York', 'Ohio', 'West Virginia', 'Delaware', 'Maryland', 'New Jersey'],
        'Rhode Island': ['Massachusetts', 'New York', 'Connecticut'],
        'South Carolina': ['North Carolina', 'Georgia'],
        'South Dakota': ['Nebraska', 'North Dakota', 'Wyoming', 'Iowa', 'Minnesota', 'Montana'],
        'Tennessee': ['Mississippi', 'Missouri', 'North Carolina', 'Virginia', 'Alabama', 'Arkansas', 'Georgia', 'Kentucky'],
        'Texas': ['New Mexico', 'Oklahoma', 'Arkansas', 'Louisiana'],
        'Utah': ['Nevada', 'New Mexico', 'Wyoming', 'Arizona', 'Colorado', 'Idaho'],
        'Vermont': ['New Hampshire', 'New York', 'Massachusetts'],
        'Virginia': ['North Carolina', 'Tennessee', 'West Virginia', 'Kentucky', 'Maryland'],
        'Washington': ['Oregon', 'Idaho'],
        'West Virginia': ['Pennsylvania', 'Virginia', 'Kentucky', 'Ohio', 'Maryland'],
        'Wisconsin': ['Michigan', 'Minnesota', 'Illinois', 'Iowa'],
        'Wyoming': ['Nebraska', 'South Dakota', 'Utah', 'Colorado', 'Idaho', 'Montana']
    }

def build_transition_matrix(data, state_neighbors, temp_pref=85, w_temp=1.0, w_pop=0.5):
    states = list(state_neighbors.keys())
    n = len(states)
    transition_matrix = pd.DataFrame(0.0, columns=states, index=states)

    for state in states:
        neighbors = state_neighbors[state]
        scores = []
        for neighbor in neighbors:
            temp = data[data['state'] == neighbor]['temp'].values[0]
            pop_density = data[data['state'] == neighbor]['pop_density'].values[0]
            score = desirability(temp, pop_density, temp_pref, w_temp, w_pop)
            scores.append((neighbor, score))

        total = sum(score for _, score in scores)
        for neighbor, score in scores:
            transition_matrix.at[state, neighbor] = score / total

    return transition_matrix

def compute_steady_state_matrix(transition_matrix, steps=100000):
    matrix = transition_matrix.values
    result_matrix = matrix_power(matrix, steps)
    return pd.DataFrame(result_matrix, index=transition_matrix.index, columns=transition_matrix.columns)

def plot_heatmap(matrix_df, title="Transition Matrix Heatmap"):
    plt.figure(figsize=(18, 15))
    sns.heatmap(matrix_df.values, annot=False, cmap='YlGnBu', xticklabels=True, yticklabels=True, linewidths=0.2)
    plt.title(title, fontsize=16)
    plt.xlabel("To State", fontsize=12)
    plt.ylabel("From State", fontsize=12)
    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    plt.show()

def plot_top_destinations(df, title="Top 10 Destination States"):
    top_states = df.head(10)
    plt.figure(figsize=(10, 6))
    sns.barplot(x=top_states['probability'], y=top_states['state'], palette="viridis")
    plt.title(title, fontsize=14)
    plt.xlabel("Probability")
    plt.ylabel("State")
    plt.tight_layout()
    plt.show()

# Load data
data = load_data()
state_neighbors = get_state_neighbors()

# Get Transition Matrix
transition_matrix = build_transition_matrix(data, state_neighbors)
transition_matrix.to_csv("transition_matrix.csv")
print(transition_matrix)

# Compute steady state matrix
steady_matrix = compute_steady_state_matrix(transition_matrix)
steady_matrix.to_csv("steady_matrix.csv")
print(steady_matrix)

# Plot the steady matrix as a heat map
plot_heatmap(steady_matrix)

# Compute and plot average final distribution across all starting states
avg_distribution = steady_matrix.mean(axis=0).sort_values(ascending=False)
avg_df = pd.DataFrame({'state': avg_distribution.index, 'probability': avg_distribution.values})
plot_top_destinations(avg_df, title="Top 10 Destination States (Stead State)")
