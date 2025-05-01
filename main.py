import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('data.csv');
state_neighbors = {
    'Alabama': ['Mississippi', 'Tennessee', 'Florida', 'Georgia'],
    'Alaska': [],
    'Arizona': ['Nevada', 'Colorado', 'Utah', 'New Mexico', 'California'],
    'Arkansas': ['Oklahoma', 'Tennessee', 'Texas', 'Louisiana', 'Mississippi', 'Missouri'],
    'California': ['Oregon', 'Arizona', 'Nevada'],
    'Colorado': ['New Mexico', 'Oklahoma', 'Utah', 'Wyoming', 'Arizona', 'Kansas', 'Nebraska'],
    'Connecticut': ['New York', 'Rhode Island', 'Massachusetts'],
    'Delaware': ['New Jersey', 'Pennsylvania', 'Maryland'],
    'Florida': ['Georgia', 'Alabama'],
    'Georgia': ['North Carolina', 'South Carolina', 'Tennessee', 'Alabama', 'Florida'],
    'Hawaii': [],
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

def desirability(temp, pop_density, temp_pref=85, w_temp=1.0, w_pop=0.5):
    # Gaussian preference for temperature centered at temp_pref
    temp_score = w_temp * (1 / (1 + (temp - temp_pref)**2))
    pop_score = w_pop * (1 / (1 + pop_density))  # Prefer less dense areas
    return temp_score + pop_score

# Create transition matrix
states = list(state_neighbors.keys())
n = len(states)
transition_matrix = pd.DataFrame(0.0, columns=states, index=states)

for state in states:
    neighbors = state_neighbors[state]
    scores = []
    for neighbor in neighbors:
        temp = data[data['state'] == neighbor]['temp'].values[0]
        pop_density = data[data['state'] == neighbor]['pop_density'].values[0]
        score = desirability(temp, pop_density)
        scores.append((neighbor, score))

    total = sum(score for _, score in scores)
    for neighbor, score in scores:
        transition_matrix.at[state, neighbor] = score / total

print(transition_matrix)
transition_matrix.to_csv('transition_matrix.csv')

# Visualize the transition_matrix

# NOTE: This is how we could get just a few states
# subset_states = states[:10]
# sub_matrix = transition_matrix.loc[subset_states, subset_states]

plt.figure(figsize=(18, 15))

sns.heatmap(
    transition_matrix,
    annot=False,
    cmap='YlGnBu',
    cbar=True,
    linewidths=0.2,
    xticklabels=True,
    yticklabels=True
)

plt.title("Dragon Migration Transition Matrix (All 50 States)", fontsize=16)
plt.xlabel("To State", fontsize=12)
plt.ylabel("From State", fontsize=12)
plt.xticks(rotation=90, fontsize=8)
plt.yticks(rotation=0, fontsize=8)
plt.tight_layout()
plt.show()