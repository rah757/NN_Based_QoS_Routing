import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

fig, ax = plt.subplots(figsize=(10, 8))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')

# Title
ax.text(5, 9.5, 'System Architecture', fontsize=18, fontweight='bold', ha='center')

# Define box drawing function
def draw_box(x, y, w, h, text, color='lightblue'):
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1", 
                         edgecolor='black', facecolor=color, linewidth=2)
    ax.add_patch(box)
    ax.text(x + w/2, y + h/2, text, ha='center', va='center', 
            fontsize=10, fontweight='bold', wrap=True)

# Layer 1: Simulation Engine
draw_box(0.5, 7.5, 3, 1, 'SimPy\nEvent Simulation', 'lightcoral')
draw_box(4, 7.5, 3, 1, 'NetworkX\nTopology Manager', 'lightgreen')
draw_box(7.5, 7.5, 2, 1, 'Metrics\nCollector', 'lightyellow')

# Layer 2: Routing Algorithms
draw_box(0.5, 5.5, 2, 1.2, 'Dijkstra\nBaseline', 'lightblue')
draw_box(3, 5.5, 2, 1.2, 'Min-Delay\nBaseline', 'lightblue')
draw_box(5.5, 5.5, 2, 1.2, 'NN Router\n(PyTorch)', 'orange')
draw_box(8, 5.5, 1.5, 1.2, 'Admission\nControl', 'pink')

# Layer 3: Network Components
draw_box(1, 3.5, 2.5, 1, 'Link Manager\n(BW, Delay, Jitter)', 'lavender')
draw_box(4, 3.5, 2.5, 1, 'Queue Manager\n(Priority Scheduling)', 'lavender')
draw_box(7, 3.5, 2.5, 1, 'Feature Extractor\n(13 features)', 'wheat')

# Layer 4: Traffic Generation
draw_box(2, 1.5, 2, 1, 'Traffic Generator\n(Poisson Process)', 'lightcyan')
draw_box(5, 1.5, 3, 1, 'Flow Classes\n(Video, VoIP, Industrial)', 'lightcyan')

# Layer 5: Data
draw_box(3, 0.2, 4, 0.8, 'Training Data (3,608 examples) → Trained NN Model', 'lightgray')

# Arrows showing flow
def draw_arrow(x1, y1, x2, y2):
    arrow = FancyArrowPatch((x1, y1), (x2, y2), 
                           arrowstyle='->', mutation_scale=20, 
                           linewidth=2, color='darkblue', alpha=0.6)
    ax.add_patch(arrow)

# Traffic gen -> Network components
draw_arrow(3, 2.5, 2.5, 3.5)
draw_arrow(6.5, 2.5, 5.5, 3.5)

# Network components -> Routing
draw_arrow(2.5, 4.5, 1.5, 5.5)
draw_arrow(5.5, 4.5, 4, 5.5)
draw_arrow(8.5, 4.5, 6.5, 5.5)

# Routing -> Simulation
draw_arrow(1.5, 6.7, 2, 7.5)
draw_arrow(4, 6.7, 4.5, 7.5)
draw_arrow(6.5, 6.7, 6.5, 7.5)

# NN training data flow
draw_arrow(5, 1.0, 6.5, 3.5)

# Add legend
legend_elements = [
    mpatches.Patch(color='lightcoral', label='Core Simulation'),
    mpatches.Patch(color='orange', label='ML Components'),
    mpatches.Patch(color='lightblue', label='Baseline Algorithms'),
    mpatches.Patch(color='lavender', label='Network Layer'),
    mpatches.Patch(color='lightcyan', label='Traffic Layer')
]
ax.legend(handles=legend_elements, loc='upper right', fontsize=9)

plt.tight_layout()
plt.savefig('system_architecture.png', dpi=300, bbox_inches='tight')
print("System architecture diagram saved!")
plt.show()