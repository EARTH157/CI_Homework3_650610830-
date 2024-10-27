import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

class MLPFeatureExtractor:
    """Feature extraction using dimensionality reduction technique"""
    def __init__(self, input_dim, map_size=(5, 5)):
        self.map_size = map_size
        self.input_dim = input_dim
        self.weights = np.random.randn(map_size[0], map_size[1], input_dim)
        
    def find_best_matching_unit(self, x):
        distances = np.sum((self.weights - x) ** 2, axis=2)
        return np.unravel_index(np.argmin(distances), self.map_size)
    
    def calculate_neighborhood(self, bmu, sigma):
        y, x = np.ogrid[0:self.map_size[0], 0:self.map_size[1]]
        distances = ((x - bmu[1]) ** 2 + (y - bmu[0]) ** 2) / (2 * sigma ** 2)
        return np.exp(-distances)
    
    def train_feature_extractor(self, data, epochs=100, initial_learning_rate=0.1):
        for epoch in range(epochs):
            sigma = self.map_size[0] / 2 * (1 - epoch/epochs)
            learning_rate = initial_learning_rate * (1 - epoch/epochs)
            
            for x in data:
                bmu = self.find_best_matching_unit(x)
                neighborhood = self.calculate_neighborhood(bmu, sigma)
                for i in range(self.map_size[0]):
                    for j in range(self.map_size[1]):
                        self.weights[i, j] += learning_rate * neighborhood[i, j] * (x - self.weights[i, j])
    
    def extract_features(self, data):
        features = np.zeros((len(data), 2))
        for i, x in enumerate(data):
            bmu = self.find_best_matching_unit(x)
            features[i] = np.array(bmu)
        return features

class GeneticMLP:
    """Multilayer Perceptron trained with modified genetic algorithm"""
    def __init__(self, input_size, hidden_size, output_size):
        self.network_architecture = [input_size, hidden_size, output_size]
        self.activation_function = lambda x: 1 / (1 + np.exp(-x))
        
    def initialize_population(self, population_size):
        self.population_size = population_size
        self.chromosome_positions = []
        self.velocity_vectors = []
        self.best_chromosomes = []
        self.best_fitness_scores = np.zeros(population_size)
        
        for _ in range(population_size):
            weights = []
            velocity = []
            for i in range(len(self.network_architecture) - 1):
                w = np.random.randn(self.network_architecture[i], self.network_architecture[i+1]) * 0.1
                v = np.zeros_like(w)
                weights.append(w)
                velocity.append(v)
            self.chromosome_positions.append(weights)
            self.velocity_vectors.append(velocity)
            self.best_chromosomes.append([w.copy() for w in weights])
    
    def forward_propagation(self, X, weights):
        current = X
        for w in weights:
            current = self.activation_function(np.dot(current, w))
        return current
    
    def calculate_fitness(self, weights, X, y):
        predictions = self.forward_propagation(X, weights)
        return accuracy_score(y, predictions.round())
    
    def evolve_population(self, X, y, generations=100):
        inertia_weight = 0.7
        cognitive_rate = 1.5
        social_rate = 1.5
        
        global_best_chromosome = None
        global_best_fitness = -np.inf
        
        fitness_history = []
        
        for _ in range(generations):
            for i in range(self.population_size):
                fitness = self.calculate_fitness(self.chromosome_positions[i], X, y)
                
                if fitness > self.best_fitness_scores[i]:
                    self.best_fitness_scores[i] = fitness
                    self.best_chromosomes[i] = [w.copy() for w in self.chromosome_positions[i]]
                
                if fitness > global_best_fitness:
                    global_best_fitness = fitness
                    global_best_chromosome = [w.copy() for w in self.chromosome_positions[i]]
            
            for i in range(self.population_size):
                r1, r2 = np.random.rand(2)
                
                for j in range(len(self.network_architecture) - 1):
                    cognitive_component = cognitive_rate * r1 * (self.best_chromosomes[i][j] - self.chromosome_positions[i][j])
                    social_component = social_rate * r2 * (global_best_chromosome[j] - self.chromosome_positions[i][j])
                    
                    self.velocity_vectors[i][j] = (inertia_weight * self.velocity_vectors[i][j] + 
                                                 cognitive_component + social_component)
                    self.chromosome_positions[i][j] += self.velocity_vectors[i][j]
            
            fitness_history.append(global_best_fitness)
        
        self.best_network_weights = global_best_chromosome
        return fitness_history
    
    def predict(self, X):
        return self.forward_propagation(X, self.best_network_weights).round()

def load_cancer_data(file_path):
    features = []
    labels = []
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            features.append([float(x) for x in parts[2:]])
            labels.append(1 if parts[1] == 'M' else 0)
    return np.array(features), np.array(labels)

def plot_training_progress(fitness_history):
    plt.figure(figsize=(10, 6))
    plt.plot(fitness_history)
    plt.title('Genetic MLP Training Progress')
    plt.xlabel('Generation')
    plt.ylabel('Best Fitness Score')
    plt.grid(True)
    plt.show()

def plot_feature_space(reduced_features, labels):
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(reduced_features[:, 0], reduced_features[:, 1], 
                         c=labels, cmap='viridis', alpha=0.6)
    plt.colorbar(scatter)
    plt.title('Feature Space Visualization')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.show()

def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Confusion Class')
    plt.ylabel('Actual Class')
    plt.show()

def main():
    # Load and preprocess cancer data
    X, y = load_cancer_data('wdbc.data.txt')
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Initialize and train feature extractor
    feature_extractor = MLPFeatureExtractor(input_dim=X.shape[1], map_size=(5, 5))
    print("Training Feature Extractor...")
    feature_extractor.train_feature_extractor(X_scaled, epochs=100)
    X_reduced = feature_extractor.extract_features(X_scaled)
    
    # Visualize reduced feature space
    plot_feature_space(X_reduced, y)
    
    # Perform cross-validation
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(X_reduced, y), 1):
        X_train, X_val = X_reduced[train_idx], X_reduced[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Initialize and train Genetic MLP
        genetic_mlp = GeneticMLP(input_size=2, hidden_size=5, output_size=1)
        genetic_mlp.initialize_population(population_size=20)
        
        print(f"\nTraining Fold {fold}...")
        fitness_history = genetic_mlp.evolve_population(X_train, y_train, generations=50)
        
        # Evaluate performance
        val_predictions = genetic_mlp.predict(X_val)
        fold_score = accuracy_score(y_val, val_predictions)
        cv_scores.append(fold_score)
        
        print(f"Fold {fold} Validation Accuracy: {fold_score:.4f}")
        
        # Plot training progress
        plot_training_progress(fitness_history)
    
    print("\nCross-validation Scores:", cv_scores)
    print(f"Mean CV Accuracy: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
    
    # Train final model
    final_model = GeneticMLP(input_size=2, hidden_size=5, output_size=1)
    final_model.initialize_population(population_size=20)
    fitness_history = final_model.evolve_population(X_reduced, y, generations=100)
    
    # Generate and visualize final predictions
    final_predictions = final_model.predict(X_reduced)
    plot_confusion_matrix(y, final_predictions)

if __name__ == '__main__':
    main()