import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D      
from scipy.integrate import solve_ivp
import matplotlib.animation as animation
from matplotlib.widgets import Button
from matplotlib.lines import Line2D

print("=" * 55)
print("  Proton & Antiproton -- 3D Simulation")
print("=" * 55)

e   = 1.602e-19
m_p = 1.673e-27

B0       = float(input("  Magnetfeldstaerke B0 [T]              [Standard: 0.1]: ") or 0.1)
v0       = float(input("  Startgeschwindigkeit v0 [x10^6 m/s]  [Standard: 1.0]: ") or 1.0) * 1e6
vz0      = float(input("  z-Geschwindigkeit vz0 [x10^6 m/s]    [Standard: 0.5]: ") or 0.5) * 1e6
n_cycles = int(float(input("  Anzahl Umlaeufe                       [Standard: 3  ]: ") or 3))

omega_c = e * B0 / m_p
r_c     = m_p * v0 / (e * B0)
T_c     = 2 * np.pi / omega_c
pitch   = vz0 * T_c

t_end  = n_cycles * T_c
t_eval = np.linspace(0, t_end, 4000)

print()
print("=" * 55)
print(f"  omega_c = {omega_c:.4e}  rad/s")
print(f"  r_c     = {r_c*100:.4f}  cm")
print(f"  T_c     = {T_c:.4e}  s")
print(f"  h       = {pitch*100:.4f}  cm  (Ganghoehe/Umlauf)")
print("=" * 55)

PARTICLES = [
    ("Proton",     +e, "royalblue", "-"),
    ("Antiproton", -e, "tomato",    "--"),
]

y0_state = [0.0, 0.0, 0.0,
            0.0, v0,  vz0]

def make_dgl(charge):
    def dgl(t, y):
        x, y_pos, z, vx, vy, vz = y
        alpha = charge * B0 / m_p
        return [vx, vy, vz,
                 alpha * vy,
                -alpha * vx,
                 0.0]
    return dgl

solutions = {}
for pname, charge, color, ls in PARTICLES:
    sol = solve_ivp(
        make_dgl(charge),
        (0, t_end), y0_state,
        method='RK45', t_eval=t_eval,
        rtol=1e-10, atol=1e-12
    )
    solutions[pname] = sol
    print(f"  {pname:<12} -> Integration: {'OK' if sol.success else 'FEHLER'}")

t = solutions["Proton"].t

def analytical(charge, t):
    omega = charge * B0 / m_p
    r     = m_p * v0 / (abs(charge) * B0)
    return (r * (1 - np.cos(omega * t)),
            r * np.sin(omega * t),
            vz0 * t)

for pname, charge, color, ls in PARTICLES:
    sol = solutions[pname]
    x_n, y_n, z_n = sol.y[0], sol.y[1], sol.y[2]
    x_a, y_a, z_a = analytical(charge, t)
    err = np.max(np.sqrt((x_n-x_a)**2 + (y_n-y_a)**2 + (z_n-z_a)**2))
    print(f"  {pname:<12} -> Max. Abweichung num/ana = {err:.2e} m")

print("=" * 55)

z_all     = np.concatenate([solutions[pn].y[2]*100 for pn,_,_,_ in PARTICLES])
z_fl_min  = z_all.min() - abs(z_all.max() - z_all.min()) * 0.05
z_fl_max  = z_all.max() + abs(z_all.max() - z_all.min()) * 0.05

n_grid    = 3   
scale     = 1.6 * r_c * 100  
coords    = np.linspace(-scale, scale, n_grid)
field_xy  = [(xi, yi) for xi in coords for yi in coords]

fig1 = plt.figure(figsize=(16, 7))
fig1.canvas.manager.set_window_title("Figur 1 -- Bahnen (statisch)")
fig1.suptitle(
    f"Proton & Antiproton im homogenen Magnetfeld  |  "
    f"B0={B0:.3f}T   v0={v0/1e6:.2f}e6 m/s   vz0={vz0/1e6:.2f}e6 m/s",
    fontsize=11, fontweight='bold'
)
plt.subplots_adjust(left=0.05, right=0.97, top=0.91, bottom=0.08, wspace=0.32)

ax3d = fig1.add_subplot(1, 2, 1, projection='3d')
ax_xy = fig1.add_subplot(1, 2, 2)

n_arrows = 4
for (xi, yi) in field_xy:
    z_pts = np.linspace(z_fl_min, z_fl_max, 50)
    ax3d.plot([xi]*len(z_pts), [yi]*len(z_pts), z_pts,
              color='deepskyblue', alpha=0.20, linewidth=0.9, zorder=1)
    dz_arrow = (z_fl_max - z_fl_min) * 0.08
    for k in range(1, n_arrows + 1):
        frac    = k / (n_arrows + 1)
        z_arrow = z_fl_min + frac * (z_fl_max - z_fl_min)
        ax3d.quiver(xi, yi, z_arrow,
                    0, 0, dz_arrow,
                    color='deepskyblue', alpha=0.40,
                    arrow_length_ratio=0.45, linewidth=0.8)

for pname, charge, color, ls in PARTICLES:
    sol = solutions[pname]
    ax3d.plot(sol.y[0]*100, sol.y[1]*100, sol.y[2]*100,
              color=color, ls=ls, lw=1.8,
              label=f'{pname} (num.)', zorder=5)
    x_a, y_a, z_a = analytical(charge, t)
    ax3d.plot(x_a*100, y_a*100, z_a*100,
              color=color, ls=':', lw=0.9, alpha=0.5,
              label=f'{pname} (ana.)', zorder=4)

ax3d.scatter([0], [0], [0], color='green', s=80, zorder=10, label='Start')

proxy_B = Line2D([0], [0], color='deepskyblue', lw=1.5,
                 label='B-Feldlinien (ez)')
handles, labels = ax3d.get_legend_handles_labels()
ax3d.legend(handles=handles + [proxy_B], fontsize=7, loc='upper left')

ax3d.set_xlabel('x [cm]', labelpad=5, fontsize=8)
ax3d.set_ylabel('y [cm]', labelpad=5, fontsize=8)
ax3d.set_zlabel('z [cm]', labelpad=5, fontsize=8)
ax3d.set_title('3D-Helixbahn  +  Magnetfeldlinien (B0)', fontsize=9, pad=4)
ax3d.tick_params(labelsize=7)

for pname, charge, color, ls in PARTICLES:
    sol = solutions[pname]
    ax_xy.plot(sol.y[0]*100, sol.y[1]*100,
               color=color, ls=ls, lw=1.5,
               label=f'{pname} (num.)')
    x_a, y_a, _ = analytical(charge, t)
    ax_xy.plot(x_a*100, y_a*100,
               color=color, ls=':', lw=0.9, alpha=0.5,
               label=f'{pname} (ana.)')

ax_xy.scatter([0], [0], color='green', s=80, zorder=5, label='Start')

xy_num = np.concatenate(
    [solutions[pn].y[j]*100 for pn,_,_,_ in PARTICLES for j in (0, 1)]
)
xy_ana = np.concatenate(
    [analytical(c, t)[i]*100 for _,c,_,_ in PARTICLES for i in (0, 1)]
)
xy_all = np.concatenate([xy_num, xy_ana])
lim    = max(abs(xy_all.min()), abs(xy_all.max())) * 1.15

ax_xy.set_xlim(-lim, lim)
ax_xy.set_ylim(-lim, lim)
ax_xy.set_aspect('equal', adjustable='box')   

tick_step = max(1.0, round(lim / 5 / 5) * 5)
ax_xy.xaxis.set_major_locator(plt.MultipleLocator(tick_step))
ax_xy.yaxis.set_major_locator(plt.MultipleLocator(tick_step))

ax_xy.set_xlabel('x [cm]', fontsize=9)
ax_xy.set_ylabel('y [cm]', fontsize=9)
ax_xy.set_title('Projektion: xy-Ebene\n(Kreisbewegung durch Lorentz-Kraft)',
                fontsize=9)
ax_xy.legend(fontsize=7)
ax_xy.grid(True, alpha=0.35, linestyle='--')
ax_xy.tick_params(labelsize=8)

plt.savefig('bahnen_3D.png', dpi=150, bbox_inches='tight')
print("  Plot gespeichert: bahnen_3D.png")

fig2 = plt.figure(figsize=(9, 8))
fig2.canvas.manager.set_window_title("Figur 2 -- 360 Grad Kamera-Animation")
fig2.subplots_adjust(bottom=0.17, top=0.90)

ax_anim = fig2.add_subplot(111, projection='3d')

for (xi, yi) in field_xy:
    z_pts = np.linspace(z_fl_min, z_fl_max, 40)
    ax_anim.plot([xi]*len(z_pts), [yi]*len(z_pts), z_pts,
                 color='deepskyblue', alpha=0.15, lw=0.8)
    ax_anim.quiver(xi, yi, (z_fl_min + z_fl_max) / 2,
                   0, 0, (z_fl_max - z_fl_min) * 0.1,
                   color='deepskyblue', alpha=0.40,
                   arrow_length_ratio=0.40, lw=0.8)

for pname, charge, color, ls in PARTICLES:
    sol = solutions[pname]
    ax_anim.plot(sol.y[0]*100, sol.y[1]*100, sol.y[2]*100,
                 color=color, ls=ls, lw=1.8, label=pname)

ax_anim.scatter([0], [0], [0], color='green', s=80, zorder=5, label='Start')

ax_anim.set_xlabel('x [cm]', labelpad=8, fontsize=9)
ax_anim.set_ylabel('y [cm]', labelpad=8, fontsize=9)
ax_anim.set_zlabel('z [cm]', labelpad=8, fontsize=9)
ax_anim.set_title(
    f'360 Grad Kamera-Animation\n'
    f'Proton (blau) & Antiproton (rot)  |  B0={B0:.2f}T  '
    f'(hellblau = B-Feldlinien)',
    fontsize=9, pad=6
)
ax_anim.legend(fontsize=8, loc='upper left')
ax_anim.tick_params(labelsize=7)

fig2.text(
    0.01, 0.94,
    f"B0={B0:.3f}T  |  v0={v0/1e6:.2f}e6 m/s  |  "
    f"vz0={vz0/1e6:.2f}e6 m/s  |  "
    f"r_c={r_c*100:.2f}cm  |  T_c={T_c:.2e}s",
    fontsize=7.5, va='top', family='monospace',
    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.85)
)

ax_btn    = fig2.add_axes([0.30, 0.04, 0.40, 0.07])
btn       = Button(ax_btn, '[ II ] Pause  ->  3D frei drehbar',
                   color='lightcoral', hovercolor='salmon')
is_running    = [True]
current_azim  = [0]

def toggle(event):
    is_running[0] = not is_running[0]
    if is_running[0]:
        btn.label.set_text('[ II ] Pause  ->  3D frei drehbar')
        btn.ax.set_facecolor('lightcoral')
    else:
        btn.label.set_text('[  >  ] Play  ->  Rotation startet')
        btn.ax.set_facecolor('lightgreen')
    fig2.canvas.draw_idle()

btn.on_clicked(toggle)

def animate(frame):
    if is_running[0]:
        current_azim[0] = (current_azim[0] + 2) % 360
        ax_anim.view_init(elev=20, azim=current_azim[0])
    return []

anim = animation.FuncAnimation(
    fig2, animate,
    frames=range(36000),
    interval=40,
    blit=False,
    cache_frame_data=False
)

print()
print("  Figur 1: statische Plots  (gespeichert: bahnen_3D.png)")
print("  Figur 2: Animation        [II Pause] -> 3D mit Maus drehbar")
print("=" * 55)

plt.show()
