

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D          
from scipy.integrate import solve_ivp
import matplotlib.animation as animation
from matplotlib.widgets import Button


E_CHG = 1.602e-19
M_P   = 1.673e-27
K_E   = 8.988e9


print("=" * 60)
print("  Proton & Antiproton -- 3D Simulation mit Coulomb-Kraft")
print("=" * 60)
print("  [Enter] = Standardwert uebernehmen")
print("-" * 60)

B0       = float(input("  Magnetfeldstaerke   B0  [T]            [Standard: 0.10]: ") or 0.10)
v0       = float(input("  Geschwindigkeit     v0  [x10^6 m/s]    [Standard: 1.00]: ") or 1.00) * 1e6
vz0      = float(input("  z-Geschwindigkeit   vz0 [x10^6 m/s]    [Standard: 0.50]: ") or 0.50) * 1e6
n_cycles = int(float(input("  Anzahl Umlaeufe                        [Standard: 3   ]: ") or 3))

print()
print("  Coulomb-Kraft: F = k * e^2 / r^2")
coulomb_str = input("  Coulomb-Kraft aktivieren? (j/n)        [Standard: n   ]: ") or "n"
coulomb_on  = [coulomb_str.lower() == 'j']

coulomb_exp = 0.0
if coulomb_on[0]:
    coulomb_exp = float(
        input("  Coulomb-Verstaerkung n (k_eff=k * 3.3 *10^n)  [Standard: 13  ]: ") or 13.0
    )



omega_c = E_CHG * B0 / M_P
r_c     = M_P * v0 / (E_CHG * B0)
T_c     = 2 * np.pi / omega_c
pitch   = vz0 * T_c
k_eff   = K_E * 3.3* (10.0 ** coulomb_exp)
F_lor   = E_CHG * v0 * B0
F_cou   = K_E * E_CHG**2 / (2 * r_c)**2

print()
print("=" * 60)
print(f"  omega_c  = {omega_c:.4e} rad/s")
print(f"  r_c      = {r_c*100:.4f} cm")
print(f"  T_c      = {T_c:.4e} s")
print(f"  h        = {pitch*100:.4f} cm  (Ganghoehe/Umlauf)")
print(f"  F_Lor    ~ {F_lor:.2e} N")
print(f"  F_Cou    ~ {F_cou:.2e} N  (bei r=2*r_c, ohne Verstaerkung)")
print(f"  Coulomb  : {'AN  (k_eff = k * 3.3 * 10^' + str(coulomb_exp) + ')' if coulomb_on[0] else 'AUS'}")
print("=" * 60)



def run_simulation(B0, v0, vz0, coulomb_on, coulomb_exp, n_cycles):
    omega_c = E_CHG * B0 / M_P
    T_c     = 2 * np.pi / omega_c
    r_c     = M_P * v0 / (E_CHG * B0)
    t_end   = n_cycles * T_c
    t_eval  = np.linspace(0, t_end, 5000)

    h  = vz0 * T_c
    y0 = [-r_c, 0.0, -h/2,  0.0, +v0, +vz0,
          +r_c, 0.0, +h/2,  0.0, +v0, -vz0]

    k_eff_loc = K_E * (10.0 ** coulomb_exp)

    def dgl(t, y):
        xp, yp, zp, vxp, vyp, vzp = y[0:6]
        xa, ya, za, vxa, vya, vza = y[6:12]

        alpha_p = +(E_CHG * B0 / M_P)
        alpha_a = -(E_CHG * B0 / M_P)

        if coulomb_on[0]:
            dx = xp - xa
            dy = yp - ya
            dz = zp - za
            r  = max(np.sqrt(dx**2 + dy**2 + dz**2), r_c * 1e-3)
            F_betrag = k_eff_loc * E_CHG**2 / r**2
            Fx = -F_betrag * dx / r
            Fy = -F_betrag * dy / r
            Fz = -F_betrag * dz / r
        else:
            Fx = Fy = Fz = 0.0

        axp =  alpha_p * vyp  +  Fx / M_P
        ayp = -alpha_p * vxp  +  Fy / M_P
        azp =                    Fz / M_P
        axa =  alpha_a * vya  -  Fx / M_P
        aya = -alpha_a * vxa  -  Fy / M_P
        aza =                   -Fz / M_P

        return [vxp, vyp, vzp, axp, ayp, azp,
                vxa, vya, vza, axa, aya, aza]

    sol = solve_ivp(dgl, (0, t_end), y0, method='RK45',
                    t_eval=t_eval, rtol=1e-8, atol=1e-10)
    return sol, T_c, r_c



print("  Berechnung laeuft...")
sol, T_c_sol, r_c_sol = run_simulation(
    B0, v0, vz0, coulomb_on, coulomb_exp, n_cycles
)
print(f"  Integration: {'OK' if sol.success else 'FEHLER'}")
print("=" * 60)



PARTICLES = [
    ("Proton (+e)",     0, "royalblue", "-"),
    ("Antiproton (-e)", 6, "tomato",    "--"),
]



def get_axis_limits(sol):
    all_x = np.concatenate([sol.y[0]*100, sol.y[6]*100])
    all_y = np.concatenate([sol.y[1]*100, sol.y[7]*100])
    all_z = np.concatenate([sol.y[2]*100, sol.y[8]*100])
    def lims(arr):
        lo, hi = arr.min(), arr.max()
        margin = max((hi - lo) * 0.15, 0.01)
        return lo - margin, hi + margin
    return lims(all_x), lims(all_y), lims(all_z)

xlims, ylims, zlims = get_axis_limits(sol)


coulomb_label = (
    f"Coulomb AN  (k_eff = k * 3.3 * 10^{coulomb_exp:.1f})"
    if coulomb_on[0] else "Coulomb AUS"
)

fig1 = plt.figure(figsize=(22, 9))
fig1.canvas.manager.set_window_title("Figur 1 - Bahnen (statisch)")
fig1.suptitle(
    f"Proton (+e, blau) & Antiproton (-e, rot)  |  "
    f"B0={B0:.3f}T   v0={v0/1e6:.2f}e6 m/s   "
    f"vz0={vz0/1e6:.2f}e6 m/s   {coulomb_label}",
    fontsize=10, fontweight='bold', y=0.99
)


gs = GridSpec(
    2, 3,
    figure=fig1,
    left=0.03, right=0.99,
    top=0.93, bottom=0.07,
    wspace=0.35, hspace=0.48,
    width_ratios=[2.0, 1.4, 0.9]   
)

ax3d_f1 = fig1.add_subplot(gs[:, 0], projection='3d') 
ax_xz   = fig1.add_subplot(gs[0, 1])                   
ax_yz   = fig1.add_subplot(gs[1, 1])                   
ax_info = fig1.add_subplot(gs[:, 2])                  
ax_info.axis('off')


for name, i, col, ls in PARTICLES:
    xc = sol.y[i  ] * 100
    yc = sol.y[i+1] * 100
    zc = sol.y[i+2] * 100

    ax3d_f1.plot(xc, yc, zc, color=col, ls=ls, lw=1.5, label=name)
    ax3d_f1.scatter([xc[0]], [yc[0]], [zc[0]],
                    color=col, s=80, zorder=5, marker='^')
    ax_xz.plot(xc, zc, color=col, ls=ls, lw=1.3, label=name)
    ax_yz.plot(yc, zc, color=col, ls=ls, lw=1.3, label=name)

ax3d_f1.set_xlabel('x [cm]', labelpad=6, fontsize=8)
ax3d_f1.set_ylabel('y [cm]', labelpad=6, fontsize=8)
ax3d_f1.set_zlabel('z [cm]', labelpad=6, fontsize=8)
ax3d_f1.set_xlim(xlims)
ax3d_f1.set_ylim(ylims)
ax3d_f1.set_zlim(zlims)
ax3d_f1.set_title('3D-Helixbahn  (Dreieck = Startpunkt)', fontsize=9, pad=4)
ax3d_f1.legend(fontsize=7, loc='upper left')
ax3d_f1.tick_params(labelsize=7)


ax_xz.set_xlabel('x [cm]', fontsize=8)
ax_xz.set_ylabel('z [cm]', fontsize=8)
ax_xz.set_title('Seitenansicht: xz-Ebene\n(Helix-Steigung durch vz0)', fontsize=8)
ax_xz.set_xlim(xlims)
ax_xz.set_ylim(zlims)
ax_xz.grid(True, alpha=0.3)
ax_xz.legend(fontsize=7)
ax_xz.tick_params(labelsize=7)


ax_yz.set_xlabel('y [cm]', fontsize=8)
ax_yz.set_ylabel('z [cm]', fontsize=8)
ax_yz.set_title('Frontansicht: yz-Ebene\n(Kreis + Translation)', fontsize=8)
ax_yz.set_xlim(ylims)
ax_yz.set_ylim(zlims)
ax_yz.grid(True, alpha=0.3)
ax_yz.legend(fontsize=7)
ax_yz.tick_params(labelsize=7)

k_e_disp = K_E * (3.3 * 10.0 ** coulomb_exp)
Fc_disp  = k_e_disp * E_CHG**2 / (r_c * 0.01)**2
Fl_disp  = E_CHG * v0 * B0
coulomb_status = ('AN (x 3.3 * 10^' + f'{coulomb_exp:.1f})') if coulomb_on[0] else 'AUS'

info_txt = (
    "-- Physikalische\n"
    "   Kenngroessen --------\n"
    f" B0    = {B0:.3f} T\n"
    f" v0    = {v0/1e6:.2f} x10^6 m/s\n"
    f" vz0   = {vz0/1e6:.2f} x10^6 m/s\n"
    f" w_c   = {omega_c:.3e} rad/s\n"
    f" r_c   = {r_c*100:.4f} cm\n"
    f" T_c   = {T_c:.3e} s\n"
    f" h     = {pitch*100:.4f} cm\n"
    "-- Coulomb -------------\n"
    " F = k * e^2 / r^2\n"
    f" k_eff = {k_e_disp:.2e}\n"
    f"        Nm^2/C^2\n"
    f" |F_C| = {Fc_disp:.2e} N\n"
    f" |F_L| = {Fl_disp:.2e} N\n"
    f" Verh. = {Fc_disp/Fl_disp:.1e}\n"
    "-- Status --------------\n"
    f" Coulomb:\n"
    f"  {coulomb_status}\n"
    " n=12-13:\n"
    "  Ablenk. sichtbar"
)

ax_info.text(
    0.08, 0.97,
    info_txt,
    transform=ax_info.transAxes,
    fontsize=7.5, va='top', family='monospace',
    bbox=dict(
        boxstyle='round,pad=0.6',
        facecolor='#f0f4ff',
        edgecolor='#aaaacc',
        alpha=0.97,
        linewidth=1.2
    )
)

plt.savefig('bahnen_coulomb.png', dpi=150, bbox_inches='tight')
print("  Plot gespeichert: bahnen_coulomb.png")


is_running   = [True]
current_azim = [0]

fig2 = plt.figure(figsize=(9, 8))
fig2.canvas.manager.set_window_title("Figur 2 - 360 Grad Kamera-Animation")
fig2.subplots_adjust(bottom=0.17, top=0.90, left=0.05, right=0.95)

ax_anim = fig2.add_subplot(111, projection='3d')

for name, i, col, ls in PARTICLES:
    xc = sol.y[i  ] * 100
    yc = sol.y[i+1] * 100
    zc = sol.y[i+2] * 100
    ax_anim.plot(xc, yc, zc, color=col, ls=ls, lw=1.8, label=name)
    ax_anim.scatter([xc[0]], [yc[0]], [zc[0]],
                    color=col, s=100, zorder=6, marker='^')

ax_anim.set_xlabel('x [cm]', labelpad=8, fontsize=9)
ax_anim.set_ylabel('y [cm]', labelpad=8, fontsize=9)
ax_anim.set_zlabel('z [cm]', labelpad=8, fontsize=9)
ax_anim.set_xlim(xlims)
ax_anim.set_ylim(ylims)
ax_anim.set_zlim(zlims)
ax_anim.set_title(
    '360 Grad Kamera-Animation\n'
    'Proton (blau) & Antiproton (rot)  |  B = B0 * e_z',
    fontsize=9, pad=8
)
ax_anim.legend(fontsize=8, loc='upper left')
ax_anim.tick_params(labelsize=7)

fig2.text(
    0.01, 0.87, (
        f"B0  = {B0:.3f} T\n"
        f"v0  = {v0/1e6:.2f} x10^6 m/s\n"
        f"vz0 = {vz0/1e6:.2f} x10^6 m/s\n"
        f"r_c = {r_c*100:.3f} cm\n"
        f"T_c = {T_c:.2e} s\n"
        f"Coulomb: {coulomb_status}"
    ),
    transform=fig2.transFigure,
    fontsize=8, va='top', family='monospace',
    bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.88)
)

fig2.text(
    0.75, 0.87, (
        "Dreieck = Startpunkt\n"
        "----  Proton (+e)\n"
        " --   Antiproton (-e)\n"
        "Achsen in [cm]"
    ),
    transform=fig2.transFigure,
    fontsize=8.5, va='top',
    bbox=dict(boxstyle='round,pad=0.4', facecolor='#fff8e7', alpha=0.92)
)


ax_bPP = fig2.add_axes([0.30, 0.04, 0.40, 0.07])
btn_pp = Button(ax_bPP,
    '[ II ] Pause  ->  3D frei drehbar',
    color='lightcoral', hovercolor='salmon')

def on_playpause(event):
    is_running[0] = not is_running[0]
    if is_running[0]:
        btn_pp.label.set_text('[ II ] Pause  ->  3D frei drehbar')
        btn_pp.ax.set_facecolor('lightcoral')
    else:
        btn_pp.label.set_text('[  >  ] Play  ->  Rotation startet')
        btn_pp.ax.set_facecolor('lightgreen')
    fig2.canvas.draw_idle()

btn_pp.on_clicked(on_playpause)

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
print("  Figur 1: statische Plots  (gespeichert: bahnen_coulomb.png)")
print("  Figur 2: Animation        [II Pause] -> 3D mit Maus drehbar")
print("=" * 60)

plt.show()
