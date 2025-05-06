import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from numpy.linalg import matrix_power

def load_data():
    return pd.read_csv('data.csv')

def desirability(temp, pop_density, temp_std, pop_std, temp_pref, pop_pref, w_temp, w_pop):
    temp_score = w_temp * (1 / (1 + ((temp - temp_pref) / temp_std)**2))
    pop_score = w_pop * (1 / (1 + ((pop_density - pop_pref) / pop_std)**2))
    return temp_score + pop_score

def get_state_neighbors():
    return {
        'Alabama': ['Mississippi', 'Tennessee', 'Florida', 'Georgia', 'Alabama'],
        # 'Alaska': [],
        'Arizona': ['Nevada', 'Colorado', 'Utah', 'New Mexico', 'California', 'Arizona'],
        'Arkansas': ['Oklahoma', 'Tennessee', 'Texas', 'Louisiana', 'Mississippi', 'Missouri', 'Arkansas'],
        'California': ['Oregon', 'Arizona', 'Nevada', 'California'],
        'Colorado': ['New Mexico', 'Oklahoma', 'Utah', 'Wyoming', 'Arizona', 'Kansas', 'Nebraska', 'Colorado'],
        'Connecticut': ['New York', 'Rhode Island', 'Massachusetts', 'Connecticut'],
        'Delaware': ['New Jersey', 'Pennsylvania', 'Maryland', 'Delaware'],
        'Florida': ['Georgia', 'Alabama', 'Florida'],
        'Georgia': ['North Carolina', 'South Carolina', 'Tennessee', 'Alabama', 'Florida', 'Georgia'],
        # 'Hawaii': [],
        'Idaho': ['Utah', 'Washington', 'Wyoming', 'Montana', 'Nevada', 'Oregon', 'Idaho'],
        'Illinois': ['Kentucky', 'Missouri', 'Wisconsin', 'Indiana', 'Iowa', 'Michigan', 'Illinois'],
        'Indiana': ['Michigan', 'Ohio', 'Illinois', 'Kentucky', 'Indiana'],
        'Iowa': ['Nebraska', 'South Dakota', 'Wisconsin', 'Illinois', 'Minnesota', 'Missouri', 'Iowa'],
        'Kansas': ['Nebraska', 'Oklahoma', 'Colorado', 'Missouri', 'Kansas'],
        'Kentucky': ['Tennessee', 'Virginia', 'West Virginia', 'Illinois', 'Indiana', 'Missouri', 'Ohio', 'Kentucky'],
        'Louisiana': ['Texas', 'Arkansas', 'Mississippi', 'Louisiana'],
        'Maine': ['New Hampshire', 'Maine'],
        'Maryland': ['Virginia', 'West Virginia', 'Delaware', 'Pennsylvania', 'Maryland'],
        'Massachusetts': ['New York', 'Rhode Island', 'Vermont', 'Connecticut', 'New Hampshire', 'Massachusetts'],
        'Michigan': ['Ohio', 'Wisconsin', 'Illinois', 'Indiana', 'Minnesota', 'Michigan'],
        'Minnesota': ['North Dakota', 'South Dakota', 'Wisconsin', 'Iowa', 'Michigan', 'Minnesota'],
        'Mississippi': ['Louisiana', 'Tennessee', 'Alabama', 'Arkansas', 'Mississippi'],
        'Missouri': ['Nebraska', 'Oklahoma', 'Tennessee', 'Arkansas', 'Illinois', 'Iowa', 'Kansas', 'Kentucky', 'Missouri'],
        'Montana': ['South Dakota', 'Wyoming', 'Idaho', 'North Dakota', 'Montana'],
        'Nebraska': ['Missouri', 'South Dakota', 'Wyoming', 'Colorado', 'Iowa', 'Kansas', 'Nebraska'],
        'Nevada': ['Arizona', 'Oregon', 'Utah', 'Idaho', 'California', 'Nevada'],
        'New Hampshire': ['Massachusetts', 'Vermont', 'Maine', 'New Hampshire'],
        'New Jersey': ['Pennsylvania', 'Delaware', 'New York', 'New Jersey'],
        'New Mexico': ['Oklahoma', 'Texas', 'Utah', 'Arizona', 'Colorado', 'New Mexico'],
        'New York': ['Pennsylvania', 'Rhode Island', 'Vermont', 'Connecticut', 'Massachusetts', 'New Jersey', 'New York'],
        'North Carolina': ['Tennessee', 'Virginia', 'Georgia', 'South Carolina', 'North Carolina'],
        'North Dakota': ['South Dakota', 'Minnesota', 'Montana', 'North Dakota'],
        'Ohio': ['Michigan', 'Pennsylvania', 'West Virginia', 'Indiana', 'Kentucky', 'Ohio'],
        'Oklahoma': ['Missouri', 'New Mexico', 'Texas', 'Arkansas', 'Colorado', 'Kansas', 'Oklahoma'],
        'Oregon': ['Nevada', 'Washington', 'California', 'Idaho', 'Oregon'],
        'Pennsylvania': ['New York', 'Ohio', 'West Virginia', 'Delaware', 'Maryland', 'New Jersey', 'Pennsylvania'],
        'Rhode Island': ['Massachusetts', 'New York', 'Connecticut', 'Rhode Island'],
        'South Carolina': ['North Carolina', 'Georgia', 'South Carolina'],
        'South Dakota': ['Nebraska', 'North Dakota', 'Wyoming', 'Iowa', 'Minnesota', 'Montana', 'South Dakota'],
        'Tennessee': ['Mississippi', 'Missouri', 'North Carolina', 'Virginia', 'Alabama', 'Arkansas', 'Georgia', 'Kentucky', 'Tennessee'],
        'Texas': ['New Mexico', 'Oklahoma', 'Arkansas', 'Louisiana', 'Texas'],
        'Utah': ['Nevada', 'New Mexico', 'Wyoming', 'Arizona', 'Colorado', 'Idaho', 'Utah'],
        'Vermont': ['New Hampshire', 'New York', 'Massachusetts', 'Vermont'],
        'Virginia': ['North Carolina', 'Tennessee', 'West Virginia', 'Kentucky', 'Maryland', 'Virginia'],
        'Washington': ['Oregon', 'Idaho', 'Washington'],
        'West Virginia': ['Pennsylvania', 'Virginia', 'Kentucky', 'Ohio', 'Maryland', 'West Virginia'],
        'Wisconsin': ['Michigan', 'Minnesota', 'Illinois', 'Iowa', 'Wisconsin'],
        'Wyoming': ['Nebraska', 'South Dakota', 'Utah', 'Colorado', 'Idaho', 'Montana', 'Wyoming'],
    }

def build_transition_matrix(data, state_neighbors, temp_std, pop_std, temp_pref, pop_pref, w_temp, w_pop):
    states = list(state_neighbors.keys())
    transition_matrix = pd.DataFrame(0.0, columns=states, index=states)

    for state in states:
        neighbors = state_neighbors[state]
        scores = []
        for neighbor in neighbors:
            temp = data[data['state'] == neighbor]['temp'].values[0]
            pop_density = data[data['state'] == neighbor]['pop_density'].values[0]
            score = desirability(temp, pop_density, temp_std, pop_std, temp_pref, pop_pref, w_temp, w_pop)
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
temp_std = data['temp'].std()
pop_std = data['pop_density'].std()

# Set Preferences
temperature = 100
population_density = 1000
temp_weight = 3.0
pop_weight = 0.5

print(f"Temperature Std: {temp_std}")
print(f"Population Std: {pop_std}")

# Get Transition Matrix
transition_matrix = build_transition_matrix(
    data=data,
    state_neighbors=state_neighbors,
    temp_std=temp_std,
    pop_std=pop_std,
    temp_pref=temperature,
    pop_pref=population_density,
    w_temp=temp_weight,
    w_pop=pop_weight
)
transition_matrix.to_csv("transition_matrix.csv")
print(f"Transition Matrix: {transition_matrix}")

# Compute steady state matrix
steady_matrix = compute_steady_state_matrix(transition_matrix)
steady_matrix.to_csv("steady_matrix.csv")
print(f"Steady Matrix: {steady_matrix}")

# Plot the steady matrix as a heat map
# plot_heatmap(steady_matrix)

# Compute and plot average final distribution across all starting states
avg_distribution = steady_matrix.mean(axis=0).sort_values(ascending=False)
avg_df = pd.DataFrame({'state': avg_distribution.index, 'probability': avg_distribution.values})
plot_top_destinations(avg_df, title=f"Top 10 Destination States (pop_weight={pop_weight}, temp_weight={temp_weight})")
print(f"Average Distribution:\n {avg_distribution}")