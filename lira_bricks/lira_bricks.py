import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# Настройка шрифтов и LaTeX-рендеринга математики
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Liberation Sans']
plt.rcParams['mathtext.fontset'] = 'dejavusans'  

# Конфигурация шагов
N_values = [1, 2, 3, 8]
L = 1.0       # Длина блока
h = 0.11      # Толщина блока (подобрана, чтобы N=8 смотрелся аккуратно)

# Цветовая палитра "Дерево и Металл"
wood_face = '#d7a15c'   # Теплый древесный оттенок
wood_edge = '#5c3a21'   # Темный контур
table_color = '#7f8c8d' # Край стола
cm_color = '#e74c3c'    # Линия центра масс
blue_line = '#2980b9'   # Вынос и стрелки

fig, axs = plt.subplots(2, 2, figsize=(14, 11), dpi=150)
axs = axs.flatten()

for idx, N in enumerate(N_values):
    ax = axs[idx]
    
    # Расчет положения правых краев для текущего N
    R = np.zeros(N)
    for i in range(N):
        R[i] = sum(L / (2 * j) for j in range(i + 1, N + 1))
        
    total_overhang = R[0]
    
    # 1. Отрисовка стола (фиксирован на x=0, y=0)
    table_top = patches.Rectangle((-1.5, -0.5), 1.5, 0.5, linewidth=0, facecolor='#f2f4f4', zorder=1)
    ax.add_patch(table_top)
    ax.plot([-1.5, 0], [0, 0], color=table_color, linewidth=3.5, zorder=2)
    ax.plot([0, 0], [0, -0.5], color=table_color, linewidth=3.5, zorder=2)
    
    # 2. Отрисовка деревянных блоков
    for i in range(N):
        x_right = R[i]
        x_left = x_right - L
        y_bottom = (N - 1 - i) * h
        
        # Основное тело кирпича
        rect = patches.Rectangle((x_left, y_bottom), L, h, linewidth=1.8, 
                                 edgecolor=wood_edge, facecolor=wood_face, alpha=0.9, zorder=3)
        ax.add_patch(rect)
        
        # Текстура волокон дерева (горизонтальные линии чуть темнее фона)
        ax.plot([x_left + 0.1, x_right - 0.1], [y_bottom + h*0.3, y_bottom + h*0.3], color='#c68a4c', linewidth=0.8, zorder=4)
        ax.plot([x_left + 0.2, x_right - 0.15], [y_bottom + h*0.7, y_bottom + h*0.7], color='#c68a4c', linewidth=0.8, zorder=4)
        
        # Нумерация критически важных блоков, чтобы не перегружать N=8
        if N <= 3 or i == 0 or i == N-1 or i == 1:
            lbl = f"№1" if i == 0 else (f"№{N}" if i == N-1 else f"№{i+1}")
            if N == 8 and i == 1: lbl = "..."
            ax.text(x_left + 0.05, y_bottom + h/2, lbl, 
                    va='center', ha='left', fontsize=9, color='#4a2e1b', weight='bold', zorder=5)

    # 3. Вертикаль центра масс всей системы (всегда на x=0 над краем стола)
    ax.plot([0, 0], [0, N * h], color=cm_color, linestyle='--', linewidth=1.5, zorder=6)
    ax.scatter([0], [0], color=cm_color, marker='o', s=35, edgecolors='black', zorder=7)
    
    # 4. Стрелка полного выноса конструкции S_N
    y_arrow = N * h + 0.05
    ax.plot([0, 0], [N * h, y_arrow + 0.05], color=blue_line, linestyle=':', linewidth=1.2)
    ax.plot([total_overhang, total_overhang], [N * h, y_arrow + 0.05], color=blue_line, linestyle=':', linewidth=1.2)
    ax.annotate('', xy=(total_overhang, y_arrow), xytext=(0, y_arrow),
                arrowprops=dict(arrowstyle='<->', color=blue_line, linewidth=1.3))
    
    # Математические подписи в формате LaTeX
    if N == 1:
        latex_formula = r'$S_1 = \frac{L}{2} = 0.5L$'
    elif N == 2:
        latex_formula = r'$S_2 = \frac{L}{2} + \frac{L}{4} = 0.75L$'
    elif N == 3:
        latex_formula = r'$S_3 = \frac{L}{2}\left(1 + \frac{1}{2} + \frac{1}{3}\right) \approx 0.92L$'
    elif N == 8:
        latex_formula = r'$S_8 = \frac{L}{2}\sum_{k=1}^{8}\frac{1}{k} \approx 1.36L > L$'

    ax.text(total_overhang / 2, y_arrow + 0.03, latex_formula, 
            ha='center', va='bottom', fontsize=11, color=blue_line, weight='bold')
    
    # Заголовки панелей
    ax.set_title(f"Панель {idx+1}: {N} " + ("блок" if N==1 else ("блока" if N<5 else "блоков")), 
                 fontsize=12, weight='bold', color='#2c3e50', pad=8)
    
    # Жесткие единые границы осей для сопоставимости картинок
    ax.set_xlim(-1.2, 1.6)
    ax.set_ylim(-0.25, 8 * h + 0.3)
    ax.set_aspect('equal')
    ax.axis('off')

plt.suptitle("Динамика сборки башни Лиры (Гармонический вынос за край стола)\n"
             "С каждым новым нижним блоком вся конструкция выдвигается всё дальше вправо", 
             fontsize=15, weight='bold', color='#2c3e50', y=0.98)

plt.tight_layout()
plt.savefig('harmonic_dynamics.png', bbox_inches='tight', dpi=150)
plt.show()
