import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd
import os
import time
import json
from typing import Dict, List, Tuple, Optional

class NeutronSimulatorConfig:
    """Classe per gestire la configurazione del simulatore"""
    
    def __init__(self):
        self.reset_to_defaults()
    
    def reset_to_defaults(self):
        """Ripristina i valori di default"""
        self.humidity_levels = [2, 7, 12, 17, 23, 28, 34, 40]
        self.n_neutrons = 500000
        self.porosity = 0.5
        self.soil_density = 1.43
        self.thermal_energy = 0.025
        self.max_depth = 200.0
        self.max_interactions = 1000
        self.spectrum_file = "SpectrumCurrent.csv"
        self.output_dir = "simulation_results"
        
        # Composizione del suolo (frazioni massiche)
        self.soil_composition = {
            'O': 0.5231, 
            'Si': 0.4265, 
            'Al': 0.0329, 
            'Fe': 0.0012,
            'Ca': 0.0033, 
            'Na': 0.0000, 
            'K': 0.0086, 
            'Mg': 0.0000, 
            'H': 0.0000, 
            'P': 0.0024, 
            'Ti': 0.0020
        }
        
        # Masse atomiche (u)
        self.atomic_masses = {
            'O': 15.999, 'Si': 28.085, 'Al': 26.982, 'Fe': 55.845,
            'Ca': 40.078, 'Na': 22.990, 'K': 39.098, 'Mg': 24.305,
            'H': 1.008, 'P': 30.97, 'Ti': 47.867
        }
        
        # Sezioni d'urto di assorbimento
        self.absorption_cross_sections = {
            'H': 0.332, 'O': 0.00019, 'Si': 0.171, 'Al': 0.231,
            'Fe': 2.56, 'Ca': 0.43, 'Na': 0.53, 'K': 2.1, 
            'Mg': 0.063, 'P': 0.172, 'Ti': 6.09
        }
    
    def save_config(self, filename: str):
        """Salva la configurazione su file JSON"""
        config_dict = {
            'humidity_levels': self.humidity_levels,
            'n_neutrons': self.n_neutrons,
            'porosity': self.porosity,
            'soil_density': self.soil_density,
            'thermal_energy': self.thermal_energy,
            'max_depth': self.max_depth,
            'max_interactions': self.max_interactions,
            'spectrum_file': self.spectrum_file,
            'output_dir': self.output_dir,
            'soil_composition': self.soil_composition
        }
        
        with open(filename, 'w') as f:
            json.dump(config_dict, f, indent=4)
        print(f"Configurazione salvata in: {filename}")
    
    def load_config(self, filename: str):
        """Carica la configurazione da file JSON"""
        try:
            with open(filename, 'r') as f:
                config_dict = json.load(f)
            
            for key, value in config_dict.items():
                if hasattr(self, key):
                    setattr(self, key, value)
            
            print(f"Configurazione caricata da: {filename}")
            return True
        except FileNotFoundError:
            print(f"File {filename} non trovato.")
            return False
        except Exception as e:
            print(f"Errore nel caricamento: {e}")
            return False

class NeutronThermalizer:
    """Classe principale per la simulazione di termalizzazione dei neutroni"""
    
    def __init__(self, config: NeutronSimulatorConfig):
        self.config = config
        self.NA = 6.022e23
        self.results = {}
        self._reset_results()
    
    def _reset_results(self):
        """Reset dei risultati"""
        self.results = {
            'initial_neutrons': 0,
            'thermalized_neutrons': 0,
            'backscattered_neutrons': 0,
            'absorbed_neutrons': 0,
            'energy_history': [],
            'depth_history': [],
            'backscatter_energies': [],
            'thermalization_depths': [],
            'fates': {}
        }
    
    def setup_simulation(self, humidity_percent: float):
        """Configura la simulazione per una specifica umidità"""
        # Calcola nuova composizione con acqua
        self.soil_composition, self.soil_density = self._add_water(
            self.config.soil_composition.copy(),
            humidity_percent,
            self.config.soil_density,
            self.config.porosity
        )
        
        # Calcola densità numeriche
        self.number_densities = self._calculate_number_densities()
        
        # Carica spettro energetico
        try:
            self.energy_spectrum, self.energy_pdf = self._load_energy_spectrum(
                self.config.spectrum_file
            )
        except Exception as e:
            print(f"Errore nel caricamento dello spettro: {e}")
            # Usa spettro di default se il file non è disponibile
            self.energy_spectrum = np.logspace(3, 7, 1000)  # 1 keV - 10 MeV
            self.energy_pdf = np.ones_like(self.energy_spectrum)
            self.energy_pdf /= np.sum(self.energy_pdf)
    
    def _add_water(self, soil_composition: Dict[str, float], water_percent: float, 
                   dry_density: float, porosity: float) -> Tuple[Dict[str, float], float]:
        """Aggiunge acqua alla composizione del suolo"""
        if not 0 <= water_percent <= 100:
            raise ValueError("La percentuale di acqua deve essere tra 0 e 100.")
        
        water_density = 1.0
        total_volume = 1.0
        water_volume = (water_percent / 100.0) * total_volume
        
        soil_mass = dry_density
        water_mass = water_volume * water_density
        total_mass = soil_mass + water_mass
        
        soil_fraction = soil_mass / total_mass
        water_fraction = water_mass / total_mass
        
        # Composizione dell'acqua (H2O)
        water_composition = {'O': 0.8889, 'H': 0.1111}
        
        # Nuova composizione
        new_composition = {}
        for element, fraction in soil_composition.items():
            new_composition[element] = fraction * soil_fraction
        
        new_composition['O'] = new_composition.get('O', 0) + water_composition['O'] * water_fraction
        new_composition['H'] = new_composition.get('H', 0) + water_composition['H'] * water_fraction
        
        # Normalizzazione
        total = sum(new_composition.values())
        for element in new_composition:
            new_composition[element] /= total
        
        new_density = total_mass / total_volume
        return new_composition, new_density
    
    def _load_energy_spectrum(self, csv_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """Carica lo spettro energetico da file CSV"""
        df = pd.read_csv(csv_path)
        energies = df["Energy [MeV]"].to_numpy() * 1e6  # MeV → eV
        weights = df["neutron current [a.u.]"].to_numpy()
        pdf = weights / np.sum(weights)
        return energies, pdf
    
    def _calculate_number_densities(self) -> Dict[str, float]:
        """Calcola le densità numeriche per ogni elemento"""
        densities = {}
        for element, fraction in self.soil_composition.items():
            densities[element] = (fraction * self.soil_density * self.NA) / self.config.atomic_masses[element]
        return densities
    
    def _elastic_cross_section(self, energy: float, element: str) -> float:
        """Calcola la sezione d'urto elastica (barn)"""
        if element == 'H':
            return 20.0 * (1 + 0.1 * np.log(max(energy, 0.001)))
        elif element == 'O':
            return 3.8 * (1 + 0.05 * np.log(max(energy, 0.001)))
        elif element == 'Si':
            return 2.2 * (1 + 0.03 * np.log(max(energy, 0.001)))
        elif element == 'Al':
            return 1.5 * (1 + 0.02 * np.log(max(energy, 0.001)))
        elif element == 'Fe':
            return 2.8 * (1 + 0.04 * np.log(max(energy, 0.001)))
        else:
            A = self.config.atomic_masses[element]
            return 2.0 * A**(1/3) * (1 + 0.02 * np.log(max(energy, 0.001)))
    
    def _absorption_cross_section(self, energy: float, element: str) -> float:
        """Calcola la sezione d'urto di assorbimento (barn)"""
        base_abs = self.config.absorption_cross_sections.get(element, 0.1)
        
        if energy < 1.0:  # Neutroni termici/epitermici
            return base_abs * np.sqrt(0.025 / max(energy, 0.001))
        else:
            return base_abs * 0.1
    
    def _total_cross_section(self, energy: float) -> Tuple[float, float]:
        """Calcola le sezioni d'urto totali elastica e di assorbimento"""
        total_elastic = 0
        total_absorption = 0
        
        for element in self.soil_composition:
            n_density = self.number_densities[element]
            elastic_cs = self._elastic_cross_section(energy, element)
            absorption_cs = self._absorption_cross_section(energy, element)
            
            total_elastic += n_density * elastic_cs * 1e-24  # barn → cm²
            total_absorption += n_density * absorption_cs * 1e-24
        
        return total_elastic, total_absorption
    
    def _sample_collision_element(self, energy: float) -> str:
        """Campiona l'elemento con cui avviene la collisione"""
        weights = []
        elements = []
        
        for element in self.soil_composition:
            n_density = self.number_densities[element]
            elastic_cs = self._elastic_cross_section(energy, element)
            weights.append(n_density * elastic_cs)
            elements.append(element)
        
        total_weight = sum(weights)
        if total_weight == 0:
            return random.choice(elements)
        
        rand = random.random() * total_weight
        cumulative = 0
        
        for i, weight in enumerate(weights):
            cumulative += weight
            if rand <= cumulative:
                return elements[i]
        
        return elements[-1]
    
    def _elastic_scatter(self, energy: float, element: str) -> float:
        """Calcola l'energia dopo scattering elastico"""
        A = self.config.atomic_masses[element]
        cos_theta_cm = 2.0 * random.random() - 1.0
        alpha = ((A - 1.0) / (A + 1.0))**2
        energy_new = energy * (1.0 + alpha + (1.0 - alpha) * cos_theta_cm) / 2.0
        return max(energy_new, 0.001)
    
    def _generate_isotropic_direction(self) -> np.ndarray:
        """Genera una direzione isotropica nell'emisfero superiore"""
        phi = np.random.uniform(0, 2 * np.pi)
        cos_theta = np.random.uniform(0, 1)
        sin_theta = np.sqrt(1 - cos_theta**2)
        
        return np.array([
            sin_theta * np.cos(phi),
            sin_theta * np.sin(phi),
            cos_theta
        ])
    
    def simulate_neutron(self, initial_energy: float) -> Dict:
        """Simula il trasporto di un singolo neutrone"""
        energy = float(initial_energy)
        position = np.array([0.0, 0.0, 0.0])
        direction = self._generate_isotropic_direction()
        
        energy_history = [energy]
        depth_history = [0.0]
        interactions = 0
        
        while (energy > 0.001 and position[2] < self.config.max_depth and 
               interactions < self.config.max_interactions):
            interactions += 1
            
            # Calcola cammino libero medio
            elastic_cs, absorption_cs = self._total_cross_section(energy)
            total_cs = elastic_cs + absorption_cs
            
            if total_cs <= 0:
                break
            
            mfp = 1.0 / total_cs
            distance = -mfp * np.log(random.random())
            
            # Aggiorna posizione
            position += direction * distance
            depth = position[2]
            
            # Controlla se esce dal suolo
            if depth < 0:
                return {
                    'fate': 'backscattered',
                    'final_energy': energy,
                    'energy_history': energy_history,
                    'depth_history': depth_history,
                    'final_depth': depth
                }
            
            if depth > self.config.max_depth:
                return {
                    'fate': 'deep_absorption',
                    'final_energy': energy,
                    'energy_history': energy_history,
                    'depth_history': depth_history,
                    'final_depth': depth
                }
            
            # Determina tipo di interazione
            if random.random() < absorption_cs / total_cs:
                return {
                    'fate': 'absorbed',
                    'final_energy': energy,
                    'energy_history': energy_history,
                    'depth_history': depth_history,
                    'final_depth': depth
                }
            
            # Scattering elastico
            element = self._sample_collision_element(energy)
            energy = self._elastic_scatter(energy, element)
            
            # Nuova direzione isotropa
            theta = np.arccos(2.0 * random.random() - 1.0)
            phi = 2.0 * np.pi * random.random()
            direction = np.array([
                np.sin(theta) * np.cos(phi),
                np.sin(theta) * np.sin(phi),
                np.cos(theta)
            ])
            
            energy_history.append(energy)
            depth_history.append(depth)
            
            # Controlla termalizzazione
            if energy <= self.config.thermal_energy:
                return {
                    'fate': 'thermalized',
                    'final_energy': energy,
                    'energy_history': energy_history,
                    'depth_history': depth_history,
                    'final_depth': depth
                }
        
        return {
            'fate': 'timeout',
            'final_energy': energy,
            'energy_history': energy_history,
            'depth_history': depth_history,
            'final_depth': depth
        }
    
    def run_simulation(self, humidity_percent: float, progress_callback=None) -> Dict:
        """Esegue la simulazione completa per una data umidità"""
        print(f"Simulazione per umidità {humidity_percent}%...")
        
        # Setup simulazione
        self.setup_simulation(humidity_percent)
        self._reset_results()
        
        self.results['initial_neutrons'] = self.config.n_neutrons
        fates = {'thermalized': 0, 'backscattered': 0, 'absorbed': 0, 'deep_absorption': 0, 'timeout': 0}
        
        all_energy_histories = []
        all_depth_histories = []
        
        start_time = time.time()
        
        for i in range(self.config.n_neutrons):
            if progress_callback and i % 10000 == 0:
                progress = (i / self.config.n_neutrons) * 100
                progress_callback(progress)
            
            # Campiona energia dallo spettro
            initial_energy = np.random.choice(self.energy_spectrum, p=self.energy_pdf)
            
            # Simula neutrone
            result = self.simulate_neutron(initial_energy)
            fate = result['fate']
            fates[fate] += 1
            
            all_energy_histories.append(result['energy_history'])
            all_depth_histories.append(result['depth_history'])
            
            if fate == 'thermalized':
                self.results['thermalization_depths'].append(result['final_depth'])
            elif fate == 'backscattered':
                self.results['backscatter_energies'].append(result['final_energy'])
        
        # Salva risultati
        self.results.update({
            'thermalized_neutrons': fates['thermalized'],
            'backscattered_neutrons': fates['backscattered'],
            'absorbed_neutrons': fates['absorbed'] + fates['deep_absorption'],
            'energy_history': all_energy_histories,
            'depth_history': all_depth_histories,
            'fates': fates,
            'simulation_time': time.time() - start_time,
            'humidity': humidity_percent,
            'soil_density': self.soil_density,
            'soil_composition': self.soil_composition.copy()
        })
        
        return self.results

class SimulationPlotter:
    """Classe per la creazione dei grafici"""
    
    @staticmethod
    def plot_thermalization_efficiency(results_dict: Dict, output_dir: str):
        """Crea grafico dell'efficienza di termalizzazione"""
        humidity_levels = sorted(results_dict.keys())
        efficiencies = []
        
        for humidity in humidity_levels:
            result = results_dict[humidity]
            efficiency = result['thermalized_neutrons'] / result['initial_neutrons']
            efficiencies.append(efficiency)
        
        plt.figure(figsize=(12, 8))
        bars = plt.bar(range(len(humidity_levels)), efficiencies, 
                      color='steelblue', alpha=0.8, width=0.6)
        
        # Aggiungi valori sopra le barre
        for i, (bar, eff) in enumerate(zip(bars, efficiencies)):
            height = bar.get_height()
            plt.text(i, height + height*0.01, f'{eff:.3f}', 
                    ha='center', va='bottom', fontweight='bold')
        
        plt.xlabel('Soil Moisture (%)', fontsize=12)
        plt.ylabel('Thermalized Neutrons / Simulated Neutrons', fontsize=12)
        plt.title('Thermalization Efficiency', fontsize=14)
        plt.xticks(range(len(humidity_levels)), [f'{h}%' for h in humidity_levels])
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        
        filepath = os.path.join(output_dir, 'thermalization_efficiency.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        return filepath
    
    @staticmethod
    def plot_depth_distributions(results_dict: Dict, output_dir: str):
        """Crea grafico delle distribuzioni di profondità"""
        plt.figure(figsize=(12, 8))
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(results_dict)))
        
        for i, (humidity, result) in enumerate(sorted(results_dict.items())):
            if result['thermalization_depths']:
                depths = result['thermalization_depths']
                counts, bins = np.histogram(depths, bins=50, range=(0, 100))
                bin_centers = (bins[:-1] + bins[1:]) / 2
                
                # Normalizza
                if len(depths) > 0:
                    norm_counts = counts / len(depths)
                    plt.plot(bin_centers, norm_counts, color=colors[i], 
                           linewidth=2, label=f'{humidity}%')
        
        plt.xlabel('Depth (cm)', fontsize=12)
        plt.ylabel('Relative Frequency', fontsize=12)
        plt.title('Thermalization Depth Distributions', fontsize=14)
        plt.xlim(0, 100)
        plt.legend(title='Humidity', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        filepath = os.path.join(output_dir, 'depth_distributions.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        return filepath
    
    @staticmethod
    def plot_energy_degradation(results_dict: Dict, output_dir: str, max_trajectories: int = 50):
        """Crea grafico della degradazione energetica"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        # Seleziona alcune umidità rappresentative
        selected_humidities = [2, 12, 28, 40]
        colors = ['red', 'orange', 'green', 'blue']
        
        for idx, (humidity, color) in enumerate(zip(selected_humidities, colors)):
            if humidity in results_dict:
                result = results_dict[humidity]
                ax = axes[idx]
                
                energy_histories = result['energy_history'][:max_trajectories]
                
                for trajectory in energy_histories:
                    if len(trajectory) > 1:
                        ax.plot(trajectory, alpha=0.6, color=color, linewidth=1)
                
                ax.set_xlabel('Collision Number')
                ax.set_ylabel('Energy (eV)')
                ax.set_yscale('log')
                ax.set_title(f'Energy Degradation - Humidity {humidity}%')
                ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        filepath = os.path.join(output_dir, 'energy_degradation.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        return filepath

class MenuSystem:
    """Sistema di menu interattivo"""
    
    def __init__(self):
        self.config = NeutronSimulatorConfig()
        self.simulator = NeutronThermalizer(self.config)
        self.plotter = SimulationPlotter()
        self.results_cache = {}
    
    def clear_screen(self):
        """Pulisce lo schermo"""
        os.system('cls' if os.name == 'nt' else 'clear')
    
    def wait_for_enter(self):
        """Aspetta che l'utente prema Invio"""
        input("\nPremi Invio per continuare...")
    
    def show_main_menu(self):
        """Mostra il menu principale"""
        while True:
            self.clear_screen()
            print("="*60)
            print("    NEUTRON THERMALIZATION SIMULATOR")
            print("="*60)
            print("1.  Configura parametri simulazione")
            print("2.  Mostra configurazione attuale")
            print("3.  Esegui simulazione completa")
            print("4.  Esegui simulazione singola")
            print("5.  Crea grafici dai risultati")
            print("6.  Salva/Carica configurazione")
            print("7.  Reset configurazione ai valori di default")
            print("8.  Informazioni e aiuto")
            print("0.  Esci")
            print("="*60)
            
            choice = input("Scegli un'opzione (0-8): ").strip()
            
            if choice == '0':
                print("Arrivederci!")
                break
            elif choice == '1':
                self.configuration_menu()
            elif choice == '2':
                self.show_current_config()
            elif choice == '3':
                self.run_full_simulation()
            elif choice == '4':
                self.run_single_simulation()
            elif choice == '5':
                self.create_plots_menu()
            elif choice == '6':
                self.save_load_config_menu()
            elif choice == '7':
                self.reset_config()
            elif choice == '8':
                self.show_help()
            else:
                print("Opzione non valida!")
                self.wait_for_enter()
    
    def configuration_menu(self):
        """Menu di configurazione"""
        while True:
            self.clear_screen()
            print("="*60)
            print("    CONFIGURAZIONE PARAMETRI")
            print("="*60)
            print("1.  Modifica livelli di umidità")
            print("2.  Modifica numero di neutroni")
            print("3.  Modifica porosità del suolo")
            print("4.  Modifica densità del suolo")
            print("5.  Modifica composizione del suolo")
            print("6.  Modifica parametri fisici")
            print("7.  Modifica percorsi file")
            print("8.  Modifica parametri di simulazione")
            print("0.  Torna al menu principale")
            print("="*60)
            
            choice = input("Scegli un'opzione (0-8): ").strip()
            
            if choice == '0':
                break
            elif choice == '1':
                self.modify_humidity_levels()
            elif choice == '2':
                self.modify_neutron_count()
            elif choice == '3':
                self.modify_porosity()
            elif choice == '4':
                self.modify_soil_density()
            elif choice == '5':
                self.modify_soil_composition()
            elif choice == '6':
                self.modify_physical_parameters()
            elif choice == '7':
                self.modify_file_paths()
            elif choice == '8':
                self.modify_simulation_parameters()
            else:
                print("Opzione non valida!")
                self.wait_for_enter()
    
    def modify_humidity_levels(self):
        """Modifica i livelli di umidità"""
        self.clear_screen()
        print("MODIFICA LIVELLI DI UMIDITÀ")
        print("-" * 40)
        print(f"Livelli attuali: {self.config.humidity_levels}")
        print("\nOpzioni:")
        print("1. Inserisci nuovi livelli manualmente")
        print("2. Usa preset comune (2, 7, 12, 17, 23, 28, 34, 40)")
        print("3. Usa range automatico")
        
        choice = input("Scegli (1-3): ").strip()
        
        if choice == '1':
            try:
                levels_str = input("Inserisci livelli separati da virgola (es: 5,10,15,20): ")
                levels = [float(x.strip()) for x in levels_str.split(',')]
                levels = [l for l in levels if 0 <= l <= 100]
                if levels:
                    self.config.humidity_levels = sorted(levels)
                    print(f"Livelli aggiornati: {self.config.humidity_levels}")
                else:
                    print("Nessun livello valido inserito!")
            except ValueError:
                print("Formato non valido!")
        elif choice == '2':
            self.config.humidity_levels = [2, 7, 12, 17, 23, 28, 34, 40]
            print("Livelli impostati al preset comune")
        elif choice == '3':
            try:
                min_h = float(input("Umidità minima (%): "))
                max_h = float(input("Umidità massima (%): "))
                n_points = int(input("Numero di punti: "))
                self.config.humidity_levels = list(np.linspace(min_h, max_h, n_points))
                print(f"Livelli generati: {[f'{h:.1f}' for h in self.config.humidity_levels]}")
            except ValueError:
                print("Valori non validi!")
        
        self.wait_for_enter()
    
    def modify_neutron_count(self):
        """Modifica il numero di neutroni"""
        self.clear_screen()
        print("MODIFICA NUMERO DI NEUTRONI")
        print("-" * 40)
        print(f"Numero attuale: {self.config.n_neutrons:,}")
        print("\nPreset comuni:")
        print("1. 10,000 (test veloce)")
        print("2. 100,000 (standard)")
        print("3. 500,000 (alta precisione)")
        print("4. 1,000,000 (massima precisione)")
        print("5. Personalizzato")
        
        choice = input("Scegli (1-5): ").strip()
        
        preset_values = {
            '1': 10000,
            '2': 100000,
            '3': 500000,
            '4': 1000000
        }
        
        if choice in preset_values:
            self.config.n_neutrons = preset_values[choice]
            print(f"Numero di neutroni impostato a: {self.config.n_neutrons:,}")
        elif choice == '5':
            try:
                n = int(input("Inserisci numero di neutroni: "))
                if n > 0:
                    self.config.n_neutrons = n
                    print(f"Numero di neutroni impostato a: {self.config.n_neutrons:,}")
                else:
                    print("Il numero deve essere positivo!")
            except ValueError:
                print("Valore non valido!")
        
        self.wait_for_enter()
    
    def modify_porosity(self):
        """Modifica la porosità del suolo"""
        self.clear_screen()
        print("MODIFICA POROSITÀ DEL SUOLO")
        print("-" * 40)
        print(f"Porosità attuale: {self.config.porosity:.3f}")
        print("La porosità deve essere tra 0.0 e 1.0")
        
        try:
            porosity = float(input("Nuova porosità: "))
            if 0.0 <= porosity <= 1.0:
                self.config.porosity = porosity
                print(f"Porosità impostata a: {self.config.porosity:.3f}")
            else:
                print("La porosità deve essere tra 0.0 e 1.0!")
        except ValueError:
            print("Valore non valido!")
        
        self.wait_for_enter()
    
    def modify_soil_density(self):
        """Modifica la densità del suolo"""
        self.clear_screen()
        print("MODIFICA DENSITÀ DEL SUOLO")
        print("-" * 40)
        print(f"Densità attuale: {self.config.soil_density:.3f} g/cm³")
        
        try:
            density = float(input("Nuova densità (g/cm³): "))
            if density > 0:
                self.config.soil_density = density
                print(f"Densità impostata a: {self.config.soil_density:.3f} g/cm³")
            else:
                print("La densità deve essere positiva!")
        except ValueError:
            print("Valore non valido!")
        
        self.wait_for_enter()
    
    def modify_soil_composition(self):
        """Modifica la composizione del suolo"""
        while True:
            self.clear_screen()
            print("MODIFICA COMPOSIZIONE DEL SUOLO")
            print("-" * 40)
            print("Composizione attuale (frazioni massiche):")
            
            total = sum(self.config.soil_composition.values())
            for element, fraction in self.config.soil_composition.items():
                print(f"  {element:>2}: {fraction:.4f} ({100*fraction:.2f}%)")
            print(f"Totale: {total:.4f}")
            
            print("\nOpzioni:")
            print("1. Modifica singolo elemento")
            print("2. Reset a composizione standard")
            print("3. Carica composizione da file")
            print("4. Normalizza composizione")
            print("0. Torna indietro")
            
            choice = input("Scegli (0-4): ").strip()
            
            if choice == '0':
                break
            elif choice == '1':
                self.modify_single_element()
            elif choice == '2':
                self.reset_soil_composition()
            elif choice == '3':
                self.load_soil_composition()
            elif choice == '4':
                self.normalize_soil_composition()
            else:
                print("Opzione non valida!")
                self.wait_for_enter()
    
    def modify_single_element(self):
        """Modifica un singolo elemento della composizione"""
        print("\nElementi disponibili:")
        elements = list(self.config.soil_composition.keys())
        for i, element in enumerate(elements):
            print(f"{i+1:2d}. {element}")
        
        try:
            choice = int(input("Scegli elemento (numero): ")) - 1
            if 0 <= choice < len(elements):
                element = elements[choice]
                current = self.config.soil_composition[element]
                print(f"Valore attuale per {element}: {current:.4f}")
                
                new_value = float(input(f"Nuovo valore per {element}: "))
                if new_value >= 0:
                    self.config.soil_composition[element] = new_value
                    print(f"Valore aggiornato: {element} = {new_value:.4f}")
                else:
                    print("Il valore deve essere non negativo!")
            else:
                print("Scelta non valida!")
        except (ValueError, IndexError):
            print("Input non valido!")
        
        self.wait_for_enter()
    
    def reset_soil_composition(self):
        """Reset della composizione del suolo"""
        self.config.soil_composition = {
            'O': 0.5231, 'Si': 0.4265, 'Al': 0.0329, 'Fe': 0.0012,
            'Ca': 0.0033, 'Na': 0.0000, 'K': 0.0086, 'Mg': 0.0000, 
            'H': 0.0000, 'P': 0.0024, 'Ti': 0.0020
        }
        print("Composizione ripristinata ai valori standard")
        self.wait_for_enter()
    
    def load_soil_composition(self):
        """Carica composizione da file"""
        filename = input("Nome del file JSON: ").strip()
        try:
            with open(filename, 'r') as f:
                composition = json.load(f)
            
            # Verifica che tutti gli elementi necessari siano presenti
            for element in self.config.soil_composition.keys():
                if element in composition:
                    self.config.soil_composition[element] = float(composition[element])
            
            print("Composizione caricata dal file")
        except FileNotFoundError:
            print("File non trovato!")
        except Exception as e:
            print(f"Errore nel caricamento: {e}")
        
        self.wait_for_enter()
    
    def normalize_soil_composition(self):
        """Normalizza la composizione del suolo"""
        total = sum(self.config.soil_composition.values())
        if total > 0:
            for element in self.config.soil_composition:
                self.config.soil_composition[element] /= total
            print(f"Composizione normalizzata (totale era {total:.4f})")
        else:
            print("Impossibile normalizzare: somma è zero!")
        
        self.wait_for_enter()
    
    def modify_physical_parameters(self):
        """Modifica parametri fisici"""
        while True:
            self.clear_screen()
            print("MODIFICA PARAMETRI FISICI")
            print("-" * 40)
            print(f"1. Energia termica: {self.config.thermal_energy:.3f} eV")
            print(f"2. Profondità massima: {self.config.max_depth:.1f} cm")
            print("0. Torna indietro")
            
            choice = input("Scegli (0-2): ").strip()
            
            if choice == '0':
                break
            elif choice == '1':
                try:
                    energy = float(input("Nuova energia termica (eV): "))
                    if energy > 0:
                        self.config.thermal_energy = energy
                        print(f"Energia termica impostata a: {self.config.thermal_energy:.3f} eV")
                    else:
                        print("L'energia deve essere positiva!")
                except ValueError:
                    print("Valore non valido!")
                self.wait_for_enter()
            elif choice == '2':
                try:
                    depth = float(input("Nuova profondità massima (cm): "))
                    if depth > 0:
                        self.config.max_depth = depth
                        print(f"Profondità massima impostata a: {self.config.max_depth:.1f} cm")
                    else:
                        print("La profondità deve essere positiva!")
                except ValueError:
                    print("Valore non valido!")
                self.wait_for_enter()
    
    def modify_file_paths(self):
        """Modifica percorsi dei file"""
        while True:
            self.clear_screen()
            print("MODIFICA PERCORSI FILE")
            print("-" * 40)
            print(f"1. File spettro energetico: {self.config.spectrum_file}")
            print(f"2. Directory output: {self.config.output_dir}")
            print("0. Torna indietro")
            
            choice = input("Scegli (0-2): ").strip()
            
            if choice == '0':
                break
            elif choice == '1':
                new_path = input("Nuovo percorso file spettro: ").strip()
                if new_path:
                    self.config.spectrum_file = new_path
                    print(f"Percorso file spettro aggiornato: {self.config.spectrum_file}")
                self.wait_for_enter()
            elif choice == '2':
                new_dir = input("Nuova directory output: ").strip()
                if new_dir:
                    self.config.output_dir = new_dir
                    print(f"Directory output aggiornata: {self.config.output_dir}")
                self.wait_for_enter()
    
    def modify_simulation_parameters(self):
        """Modifica parametri di simulazione"""
        self.clear_screen()
        print("MODIFICA PARAMETRI DI SIMULAZIONE")
        print("-" * 40)
        print(f"Massimo numero di interazioni per neutrone: {self.config.max_interactions}")
        
        try:
            max_int = int(input("Nuovo valore: "))
            if max_int > 0:
                self.config.max_interactions = max_int
                print(f"Parametro aggiornato: {self.config.max_interactions}")
            else:
                print("Il valore deve essere positivo!")
        except ValueError:
            print("Valore non valido!")
        
        self.wait_for_enter()
    
    def show_current_config(self):
        """Mostra la configurazione attuale"""
        self.clear_screen()
        print("="*60)
        print("    CONFIGURAZIONE ATTUALE")
        print("="*60)
        
        print(f"Livelli di umidità: {self.config.humidity_levels}")
        print(f"Numero di neutroni: {self.config.n_neutrons:,}")
        print(f"Porosità del suolo: {self.config.porosity:.3f}")
        print(f"Densità del suolo: {self.config.soil_density:.3f} g/cm³")
        print(f"Energia termica: {self.config.thermal_energy:.3f} eV")
        print(f"Profondità massima: {self.config.max_depth:.1f} cm")
        print(f"Max interazioni: {self.config.max_interactions}")
        print(f"File spettro: {self.config.spectrum_file}")
        print(f"Directory output: {self.config.output_dir}")
        
        print("\nComposizione del suolo:")
        total = sum(self.config.soil_composition.values())
        for element, fraction in self.config.soil_composition.items():
            if fraction > 0:
                print(f"  {element:>2}: {fraction:.4f} ({100*fraction:.2f}%)")
        print(f"Totale: {total:.4f}")
        
        self.wait_for_enter()
    
    def run_full_simulation(self):
        """Esegue la simulazione completa per tutti i livelli di umidità"""
        self.clear_screen()
        print("ESECUZIONE SIMULAZIONE COMPLETA")
        print("-" * 40)
        print(f"Livelli di umidità da simulare: {self.config.humidity_levels}")
        print(f"Neutroni per simulazione: {self.config.n_neutrons:,}")
        
        confirm = input("\nProcedere con la simulazione? (s/n): ").strip().lower()
        if confirm != 's':
            return
        
        # Crea directory output
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        # Progress callback
        def progress_callback(progress):
            print(f"Progresso: {progress:.1f}%", end='\r')
        
        self.results_cache = {}
        total_start_time = time.time()
        
        for i, humidity in enumerate(self.config.humidity_levels):
            print(f"\n--- Simulazione {i+1}/{len(self.config.humidity_levels)}: Umidità {humidity}% ---")
            
            # Aggiorna simulatore
            self.simulator = NeutronThermalizer(self.config)
            
            # Esegui simulazione
            start_time = time.time()
            results = self.simulator.run_simulation(humidity, progress_callback)
            simulation_time = time.time() - start_time
            
            # Salva risultati
            self.results_cache[humidity] = results
            
            # Stampa riassunto
            total = results['initial_neutrons']
            therm = results['thermalized_neutrons']
            backs = results['backscattered_neutrons']
            absorbed = results['absorbed_neutrons']
            
            print(f"\nRisultati per umidità {humidity}%:")
            print(f"  Termalizzati: {therm:,} ({100*therm/total:.1f}%)")
            print(f"  Backscattered: {backs:,} ({100*backs/total:.1f}%)")
            print(f"  Assorbiti: {absorbed:,} ({100*absorbed/total:.1f}%)")
            print(f"  Tempo simulazione: {simulation_time:.1f}s")
            
            if results['thermalization_depths']:
                avg_depth = np.mean(results['thermalization_depths'])
                print(f"  Profondità media termalizzazione: {avg_depth:.1f} cm")
        
        total_time = time.time() - total_start_time
        print(f"\n=== SIMULAZIONE COMPLETA TERMINATA ===")
        print(f"Tempo totale: {total_time:.1f}s")
        
        # Chiedi se creare i grafici
        create_plots = input("\nCreare i grafici automaticamente? (s/n): ").strip().lower()
        if create_plots == 's':
            self.create_all_plots()
        
        self.wait_for_enter()
    
    def run_single_simulation(self):
        """Esegue una simulazione per un singolo livello di umidità"""
        self.clear_screen()
        print("SIMULAZIONE SINGOLA")
        print("-" * 40)
        
        try:
            humidity = float(input("Inserisci livello di umidità (%): "))
            if not 0 <= humidity <= 100:
                print("L'umidità deve essere tra 0 e 100%!")
                self.wait_for_enter()
                return
        except ValueError:
            print("Valore non valido!")
            self.wait_for_enter()
            return
        
        print(f"Simulazione per umidità {humidity}% con {self.config.n_neutrons:,} neutroni")
        confirm = input("Procedere? (s/n): ").strip().lower()
        if confirm != 's':
            return
        
        # Progress callback
        def progress_callback(progress):
            print(f"Progresso: {progress:.1f}%", end='\r')
        
        # Crea directory output
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        # Esegui simulazione
        self.simulator = NeutronThermalizer(self.config)
        start_time = time.time()
        results = self.simulator.run_simulation(humidity, progress_callback)
        simulation_time = time.time() - start_time
        
        # Salva nei risultati
        self.results_cache[humidity] = results
        
        # Mostra risultati dettagliati
        self.show_detailed_results(results, simulation_time)
        self.wait_for_enter()
    
    def show_detailed_results(self, results: Dict, simulation_time: float):
        """Mostra risultati dettagliati di una simulazione"""
        print(f"\n{'='*60}")
        print("    RISULTATI SIMULAZIONE")
        print(f"{'='*60}")
        
        total = results['initial_neutrons']
        therm = results['thermalized_neutrons']
        backs = results['backscattered_neutrons']
        absorbed = results['absorbed_neutrons']
        
        print(f"Umidità: {results['humidity']}%")
        print(f"Densità suolo: {results['soil_density']:.3f} g/cm³")
        print(f"Neutroni iniziali: {total:,}")
        print(f"Termalizzati: {therm:,} ({100*therm/total:.2f}%)")
        print(f"Backscattered: {backs:,} ({100*backs/total:.2f}%)")
        print(f"Assorbiti: {absorbed:,} ({100*absorbed/total:.2f}%)")
        print(f"Tempo simulazione: {simulation_time:.1f}s")
        
        if results['thermalization_depths']:
            depths = results['thermalization_depths']
            avg_depth = np.mean(depths)
            std_depth = np.std(depths)
            median_depth = np.median(depths)
            print(f"\nStatistiche profondità termalizzazione:")
            print(f"  Media: {avg_depth:.1f} ± {std_depth:.1f} cm")
            print(f"  Mediana: {median_depth:.1f} cm")
            print(f"  Min-Max: {min(depths):.1f} - {max(depths):.1f} cm")
        
        if results['backscatter_energies']:
            energies = results['backscatter_energies']
            avg_energy = np.mean(energies)
            print(f"\nEnergia media backscattered: {avg_energy:.3f} eV")
        
        # Mostra composizione finale del suolo
        print(f"\nComposizione suolo finale:")
        for element, fraction in results['soil_composition'].items():
            if fraction > 0.001:
                print(f"  {element}: {100*fraction:.2f}%")
    
    def create_plots_menu(self):
        """Menu per la creazione dei grafici"""
        if not self.results_cache:
            print("Nessun risultato disponibile! Esegui prima una simulazione.")
            self.wait_for_enter()
            return
        
        while True:
            self.clear_screen()
            print("CREAZIONE GRAFICI")
            print("-" * 40)
            print("Risultati disponibili:")
            for humidity in sorted(self.results_cache.keys()):
                print(f"  - Umidità {humidity}%")
            
            print(f"\nDirectory output: {self.config.output_dir}")
            print("\nOpzioni:")
            print("1. Crea tutti i grafici")
            print("2. Grafico efficienza termalizzazione")
            print("3. Grafici distribuzioni di profondità")
            print("4. Grafici degradazione energetica")
            print("5. Grafico confronto 2 umidità")
            print("0. Torna indietro")
            
            choice = input("Scegli (0-5): ").strip()
            
            if choice == '0':
                break
            elif choice == '1':
                self.create_all_plots()
            elif choice == '2':
                self.create_efficiency_plot()
            elif choice == '3':
                self.create_depth_plots()
            elif choice == '4':
                self.create_energy_plots()
            elif choice == '5':
                self.create_comparison_plot()
            else:
                print("Opzione non valida!")
                self.wait_for_enter()
    
    def create_all_plots(self):
        """Crea tutti i grafici disponibili"""
        print("Creazione di tutti i grafici...")
        
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        try:
            # Efficienza di termalizzazione
            filepath1 = self.plotter.plot_thermalization_efficiency(
                self.results_cache, self.config.output_dir)
            print(f"✓ Creato: {filepath1}")
            
            # Distribuzioni di profondità
            filepath2 = self.plotter.plot_depth_distributions(
                self.results_cache, self.config.output_dir)
            print(f"✓ Creato: {filepath2}")
            
            # Degradazione energetica
            filepath3 = self.plotter.plot_energy_degradation(
                self.results_cache, self.config.output_dir)
            print(f"✓ Creato: {filepath3}")

            # Confronto OOOOOOOOOOOOOOOO self.create_two_humidity_comparison(h1, h2)
            filepath4 = self.create_two_humidity_comparison(self.config.humidity_levels[0], self.config.humidity_levels[-1])
            print(f"✓ Creato: {filepath4}")
            
            print("Tutti i grafici sono stati creati con successo!")
            
        except Exception as e:
            print(f"Errore nella creazione dei grafici: {e}")
        
        self.wait_for_enter()
    
    def create_efficiency_plot(self):
        """Crea solo il grafico dell'efficienza"""
        try:
            filepath = self.plotter.plot_thermalization_efficiency(
                self.results_cache, self.config.output_dir)
            print(f"Grafico creato: {filepath}")
        except Exception as e:
            print(f"Errore: {e}")
        self.wait_for_enter()
    
    def create_depth_plots(self):
        """Crea solo i grafici delle profondità"""
        try:
            filepath = self.plotter.plot_depth_distributions(
                self.results_cache, self.config.output_dir)
            print(f"Grafico creato: {filepath}")
        except Exception as e:
            print(f"Errore: {e}")
        self.wait_for_enter()
    
    def create_energy_plots(self):
        """Crea solo i grafici energetici"""
        try:
            filepath = self.plotter.plot_energy_degradation(
                self.results_cache, self.config.output_dir)
            print(f"Grafico creato: {filepath}")
        except Exception as e:
            print(f"Errore: {e}")
        self.wait_for_enter()
    
    def create_comparison_plot(self):
        """Crea grafico di confronto tra due umidità"""
        humidities = list(self.results_cache.keys())
        if len(humidities) < 2:
            print("Servono almeno 2 simulazioni per il confronto!")
            self.wait_for_enter()
            return
        
        print("Umidità disponibili:", humidities)
        try:
            h1 = float(input("Prima umidità da confrontare: "))
            h2 = float(input("Seconda umidità da confrontare: "))
            
            if h1 in self.results_cache and h2 in self.results_cache:
                self.create_two_humidity_comparison(h1, h2)
            else:
                print("Una o entrambe le umidità non sono disponibili!")
        except ValueError:
            print("Valori non validi!")
        
        self.wait_for_enter()
    
    def create_two_humidity_comparison(self, h1: float, h2: float):
        """Crea confronto tra due specifiche umidità"""
        plt.figure(figsize=(15, 5))
        
        # Sottografico 1: Efficienza
        plt.subplot(1, 3, 1)
        humidities = [h1, h2]
        efficiencies = []
        for h in humidities:
            result = self.results_cache[h]
            eff = result['thermalized_neutrons'] / result['initial_neutrons']
            efficiencies.append(eff)
        
        plt.bar(range(len(humidities)), efficiencies, color=['red', 'blue'])
        plt.xlabel('Humidity')
        plt.ylabel('Thermalization Efficiency')
        plt.title('Efficiency Comparison')
        plt.xticks(range(len(humidities)), [f'{h}%' for h in humidities])
        
        # Sottografico 2: Profondità
        plt.subplot(1, 3, 2)
        colors = ['red', 'blue']
        for i, h in enumerate(humidities):
            result = self.results_cache[h]
            if result['thermalization_depths']:
                depths = result['thermalization_depths']
                plt.hist(depths, bins=30, alpha=0.7, color=colors[i], 
                        label=f'{h}%', density=True)
        plt.xlabel('Depth (cm)')
        plt.ylabel('Density')
        plt.title('Depth Distribution')
        plt.legend()
        
        # Sottografico 3: Energie (sample)
        plt.subplot(1, 3, 3)
        for i, h in enumerate(humidities):
            result = self.results_cache[h]
            energy_histories = result['energy_history'][:20]  # Primi 20
            for trajectory in energy_histories:
                plt.plot(trajectory, alpha=0.5, color=colors[i])
        plt.xlabel('Collision Number')
        plt.ylabel('Energy (eV)')
        plt.yscale('log')
        plt.title('Energy Degradation')
        
        plt.tight_layout()
        filepath = os.path.join(self.config.output_dir, f'comparison_{h1}%_vs_{h2}%.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Confronto creato: {filepath}")
    
    def save_load_config_menu(self):
        """Menu per salvare/caricare configurazioni"""
        while True:
            self.clear_screen()
            print("SALVA/CARICA CONFIGURAZIONE")
            print("-" * 40)
            print("1. Salva configurazione attuale")
            print("2. Carica configurazione da file")
            print("3. Salva risultati simulazione")
            print("4. Carica risultati simulazione")
            print("0. Torna indietro")
            
            choice = input("Scegli (0-4): ").strip()
            
            if choice == '0':
                break
            elif choice == '1':
                self.save_configuration()
            elif choice == '2':
                self.load_configuration()
            elif choice == '3':
                self.save_results()
            elif choice == '4':
                self.load_results()
            else:
                print("Opzione non valida!")
                self.wait_for_enter()
    
    def save_configuration(self):
        """Salva la configurazione"""
        filename = input("Nome file configurazione (senza estensione): ").strip()
        if filename:
            filename += '.json'
            self.config.save_config(filename)
        self.wait_for_enter()
    
    def load_configuration(self):
        """Carica una configurazione"""
        filename = input("Nome file configurazione: ").strip()
        if self.config.load_config(filename):
            print("Configurazione caricata con successo!")
        self.wait_for_enter()
    
    def save_results(self):
        """Salva i risultati delle simulazioni"""
        if not self.results_cache:
            print("Nessun risultato da salvare!")
            self.wait_for_enter()
            return
        
        filename = input("Nome file risultati (senza estensione): ").strip()
        if filename:
            filename += '.json'
            try:
                # Prepara dati per il salvataggio (rimuovi array numpy)
                save_data = {}
                for humidity, results in self.results_cache.items():
                    save_data[str(humidity)] = {
                        'humidity': results['humidity'],
                        'initial_neutrons': results['initial_neutrons'],
                        'thermalized_neutrons': results['thermalized_neutrons'],
                        'backscattered_neutrons': results['backscattered_neutrons'],
                        'absorbed_neutrons': results['absorbed_neutrons'],
                        'simulation_time': results.get('simulation_time', 0),
                        'soil_density': results['soil_density'],
                        'soil_composition': results['soil_composition'],
                        'avg_thermalization_depth': np.mean(results['thermalization_depths']) if results['thermalization_depths'] else 0,
                        'std_thermalization_depth': np.std(results['thermalization_depths']) if results['thermalization_depths'] else 0,
                        'num_thermalized': len(results['thermalization_depths'])
                    }
                
                with open(filename, 'w') as f:
                    json.dump(save_data, f, indent=4)
                
                print(f"Risultati salvati in: {filename}")
            except Exception as e:
                print(f"Errore nel salvataggio: {e}")
        
        self.wait_for_enter()
    
    def load_results(self):
        """Carica risultati precedenti"""
        filename = input("Nome file risultati: ").strip()
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
            
            print(f"Risultati caricati per {len(data)} simulazioni:")
            for humidity_str, result_data in data.items():
                humidity = float(humidity_str)
                print(f"  - Umidità {humidity}%: {result_data['thermalized_neutrons']} termalizzati")
            
            print("Nota: Solo le statistiche di base sono state caricate.")
            print("Per grafici completi, riesegui le simulazioni.")
            
        except FileNotFoundError:
            print("File non trovato!")
        except Exception as e:
            print(f"Errore nel caricamento: {e}")
        
        self.wait_for_enter()
    
    def reset_config(self):
        """Reset configurazione ai valori di default"""
        confirm = input("Sei sicuro di voler ripristinare i valori di default? (s/n): ").strip().lower()
        if confirm == 's':
            self.config.reset_to_defaults()
            self.simulator = NeutronThermalizer(self.config)
            print("Configurazione ripristinata ai valori di default")
        self.wait_for_enter()
    
    def show_help(self):
        """Mostra informazioni e aiuto"""
        self.clear_screen()
        print("="*60)
        print("    INFORMAZIONI E AIUTO")
        print("="*60)
        
        print("DESCRIZIONE:")
        print("Questo programma simula la termalizzazione dei neutroni nel suolo")
        print("considerando diversi livelli di umidità e composizioni del terreno.")
        print()
        
        print("FUNZIONALITÀ PRINCIPALI:")
        print("• Simulazione Monte Carlo del trasporto neutronico")
        print("• Calcolo delle sezioni d'urto elastiche e di assorbimento")
        print("• Modellazione dell'effetto dell'umidità del suolo")
        print("• Generazione di grafici e analisi statistiche")
        print("• Sistema di configurazione flessibile")
        print()
        
        print("PARAMETRI PRINCIPALI:")
        print("• Livelli di umidità: percentuali di acqua nel suolo da simulare")
        print("• Numero neutroni: maggiore = più preciso ma più lento")
        print("• Porosità: frazione di volume vuoto nel suolo (0-1)")
        print("• Composizione: frazioni massiche degli elementi nel suolo")
        print("• Energia termica: soglia per considerare un neutrone termalizzato")
        print()
        
        print("CONSIGLI D'USO:")
        print("• Inizia con pochi neutroni (10k-100k) per test veloci")
        print("• Usa 500k+ neutroni per risultati di alta qualità")
        print("• Salva le configurazioni per riutilizzarle")
        print("• Controlla che il file dello spettro sia accessibile")
        print("• I risultati vengono salvati nella directory output")
        print()
        
        print("FORMATI FILE:")
        print("• Spettro energetico: CSV con colonne 'Energy [MeV]' e 'neutron current [a.u.]'")
        print("• Configurazioni: JSON")
        print("• Grafici: PNG ad alta risoluzione")
        print()
        
        print("TROUBLESHOOTING:")
        print("• File spettro non trovato: controlla il percorso")
        print("• Simulazione lenta: riduci il numero di neutroni")
        print("• Errori di memoria: riduci neutroni o livelli di umidità")
        print("• Composizione non normalizzata: usa l'opzione di normalizzazione")
        
        self.wait_for_enter()

def main():
    """Funzione principale"""
    try:
        menu = MenuSystem()
        menu.show_main_menu()
    except KeyboardInterrupt:
        print("\n\nInterruzione da tastiera. Programma terminato.")
    except Exception as e:
        print(f"\nErrore critico: {e}")
        print("Il programma verrà terminato.")

if __name__ == "__main__":
    main()